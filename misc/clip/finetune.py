import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sam import SAM

import open_clip
from open_clip.zero_shot_classifier import build_zero_shot_classifier
from open_clip.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES, IMAGENET_CLASSNAMES

from dataset import get_transfer_dataset, get_classname_from_torchvision_dataset
from clip_train import (
    set_seed,
    adam_step,
    sam_step,
    evaluate_prompt_classifier,
)
import tqdm

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


def encode_text_from_embeddings(model, token_embeddings, tokenized_prompts):
    """
    token_embeddings: (C, T, D) float tensor (already includes token embedding lookup)
    tokenized_prompts: (C, T) int tensor (for EOT positions)
    Returns: normalized text features (C, D_proj)
    """
    if hasattr(model, "positional_embedding"):
        x = token_embeddings + model.positional_embedding
    else:
        x = token_embeddings

    # transformer expects (T, C, D) in CLIP-style code
    x = x.permute(1, 0, 2)  # (T, C, D)

    # attention mask if present
    if hasattr(model, "attn_mask") and model.attn_mask is not None:
        x = model.transformer(x, attn_mask=model.attn_mask)
    else:
        x = model.transformer(x)

    x = x.permute(1, 0, 2)  # (C, T, D)
    x = model.ln_final(x)

    # take features at EOT token position (CLIP convention)
    eot = tokenized_prompts.argmax(dim=-1)  # (C,)
    x = x[torch.arange(x.shape[0], device=x.device), eot]  # (C, D)

    # projection to joint embedding space if present
    if hasattr(model, "text_projection"):
        x = x @ model.text_projection  # (C, D_proj)

    x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    return x


class PromptLearner(nn.Module):
    def __init__(
        self,
        model,
        tokenizer,
        classnames,
        n_ctx=8,
        ctx_init="a photo of a",
        device="cuda",
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.classnames = classnames
        self.n_ctx = n_ctx
        self.device = device

        # Initialize context vectors from ctx_init if possible
        with torch.no_grad():
            init_tok = tokenizer([ctx_init]).to(device)  # (1, T)
            init_emb = model.token_embedding(init_tok)  # (1, T, D)
            # take tokens after SOS; if insufficient length, random init
            if init_emb.size(1) >= 1 + n_ctx:
                ctx = init_emb[0, 1 : 1 + n_ctx, :].clone()
            else:
                ctx = torch.randn(n_ctx, init_emb.size(-1), device=device) * 0.02

        self.ctx = nn.Parameter(ctx)  # (n_ctx, D)

        # Tokenize class prompts using a standard template
        # You can change the template; keep fixed across optimizers.
        templates = [f"a photo of a {name}" for name in classnames]
        tokenized = tokenizer(templates).to(device)  # (C, T)
        self.register_buffer("tokenized_prompts", tokenized)

    def forward(self):
        tokenized = self.tokenized_prompts  # (C, T)
        x = self.model.token_embedding(tokenized)  # (C, T, D)

        # Replace positions 1..n_ctx (right after SOS) with learned context vectors
        # This assumes token 0 is SOS, consistent with CLIP tokenization.
        x[:, 1 : 1 + self.n_ctx, :] = self.ctx.unsqueeze(0)

        # Encode through the (frozen) text tower
        text_features = encode_text_from_embeddings(self.model, x, tokenized)
        return text_features


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name=args.model_name,
        pretrained=args.pretrained,
        precision=args.precision,
    )
    model = model.to(device)

    for p in model.parameters():
        p.requires_grad_(False)

    if args.freeze_logit_scale and hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad_(False)

    prompt_learner = PromptLearner(
        model=model,
        tokenizer=open_clip.get_tokenizer(args.model_name),
        classnames=[],  # will set later
        n_ctx=args.n_ctx,
        ctx_init=args.ctx_init,
        device=device,
    ).to(device)

    tok = open_clip.get_tokenizer(args.model_name)

    train_ds = get_transfer_dataset(
        args.dataset, args.data_root, args.train_split, preprocess_train
    )
    val_ds = get_transfer_dataset(
        args.dataset, args.data_root, args.val_split, preprocess_val
    )
    test_ds = get_transfer_dataset(
        args.dataset, args.data_root, args.test_split, preprocess_val
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    classnames = get_classname_from_torchvision_dataset(train_ds)
    num_classes = len(classnames)

    # Optimizer
    params = list(prompt_learner.parameters())
    opt_kwargs = {
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    base_optimizer = torch.optim.AdamW

    if args.prompt_lr is not None:
        opt_kwargs["lr"] = args.prompt_lr

    if args.optimizer_name == "Adam":
        optimizer = torch.optim.AdamW(model.parameters(), **opt_kwargs)
    elif args.optimizer_name == "SAM":
        optimizer = SAM(
            model.parameters(), base_optimizer, rho=args.sam_rho, **opt_kwargs
        )
    elif args.optimizer_name == "ASAM":
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
            adaptive=True,
            rho=args.sam_rho,
            **opt_kwargs,
        )
    else:
        raise ValueError(f"Unknown optimizer_name: {args.optimizer_name}")

    warmup_steps = min(args.warmup_steps, args.max_steps)
    cosine_steps = max(1, args.max_steps - warmup_steps)

    warmup = LinearLR(
        optimizer,
        start_factor=args.warmup_start_factor,
        total_iters=max(1, warmup_steps),
    )
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=args.min_lr)

    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )

    model.train()
    step = 0
    running_loss = 0.0

    train_iter = iter(train_loader)

    while step < args.max_steps:
        try:
            images, captions = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, captions = next(train_iter)

        images = images.to(device, non_blocking=True)
        text_tokens = tok(captions).to(device, non_blocking=True)

        if args.optimizer_name == "Adam":
            loss = adam_step(optimizer, model, images, text_tokens)
            running_loss += float(loss)
        else:
            _, loss2 = sam_step(optimizer, model, images, text_tokens)
            running_loss += float(loss2)

        scheduler.step()
        step += 1

        if args.log_every > 0 and (step % args.log_every == 0):
            avg = running_loss / args.log_every
            lr_now = optimizer.param_groups[0]["lr"]
            tqdm.tqdm.write(
                f"Step {step}/{args.max_steps} | lr={lr_now:.3e} | train_loss={avg:.4f}"
            )
            running_loss = 0.0

        if args.test_every > 0 and (step % args.test_every == 0 or step == 1):
            model.eval()
            val_acc = evaluate_prompt_classifier(
                model, prompt_learner, val_loader, device
            )
            print(f"Step {step}: {args.dataset} val top-1 = {val_acc:.2f}%")
            model.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Finetune CLIP using prompt learning")

    parser.add_argument(
        "--dataset", type=str, required=True, choices=["flowers102", "cars", "food101"]
    )
    parser.add_argument("--data-root", type=str, default="./data/transfer")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument(
        "--val-split", type=str, default="val"
    )  # for cars/food you can map val->test
    parser.add_argument("--test-split", type=str, default="test")

    # prompt learning
    parser.add_argument(
        "--n-ctx", type=int, default=8, help="Number of learnable context tokens."
    )
    parser.add_argument(
        "--ctx-init",
        type=str,
        default="a photo of a",
        help="Text to initialize context from.",
    )
    parser.add_argument("--freeze-logit-scale", action="store_true")
    parser.add_argument(
        "--prompt-lr",
        type=float,
        default=None,
        help="If set, overrides learning-rate for prompt params only.",
    )

    # paths to data
    parser.add_argument(
        "--coco-train-images",
        type=str,
        default="./data/coco/images/train2017",
        help="COCO train2017 image directory (contains JPEGs).",
    )
    parser.add_argument(
        "--coco-train-ann",
        type=str,
        default="./data/coco/annotations/captions_train2017.json",
        help="COCO captions train annotations JSON.",
    )
    parser.add_argument(
        "--imagenet-val-dir",
        type=str,
        default="./data/imagenet/val_flat",
        help="ImageNet-1k validation directory in ImageFolder format (class subfolders).",
    )
    parser.add_argument(
        "--imagenet-gt",
        type=str,
        default="./data/imagenet/ILSVRC2012_validation_ground_truth.txt",
        help="ILSVRC2012 validation ground-truth labels file (50k lines, 1..1000).",
    )
    parser.add_argument(
        "--imagenet-classnames",
        type=str,
        default="./data/imagenet/ilsvrc2012_classnames_1k.txt",
        help="Devkit-derived class names in ILSVRC2012 index order (1000 lines).",
    )

    # model
    # pretrained model: "laion2b_s34b_b79k"
    parser.add_argument("--model-name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument(
        "--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16", "amp"]
    )

    # optimizer
    parser.add_argument(
        "--optimizer-name",
        type=str,
        required=True,
        choices=["Adam", "SAM", "ASAM", "AdamSAM"],
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--sam-rho", type=float, default=0.05)

    # training
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20000,
        help="Total number of optimizer steps to run (step-based training budget).",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=500, help="Number of linear warmup steps."
    )
    parser.add_argument(
        "--warmup-start-factor",
        type=float,
        default=1e-3,
        help="Warmup start factor for LinearLR (lr starts at start_factor * base_lr).",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
        help="Minimum LR for cosine decay (eta_min).",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--log-every", type=int, default=50, help="Log training loss every N steps."
    )
    parser.add_argument(
        "--test-every",
        type=int,
        default=1000,
        help="Run zero-shot ImageNet eval every N steps.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=128)

    # zero-shot classifier
    parser.add_argument("--num-classes-per-batch", type=int, default=10)

    args = parser.parse_args()
    main(args)
