import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from sam import SAM

import open_clip
from open_clip.zero_shot_classifier import build_zero_shot_classifier
from open_clip.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES, IMAGENET_CLASSNAMES

from dataset import get_cc3m_loader, ImageNetVal
from clip_train import (
    set_seed,
    adam_step,
    sam_step,
    evaluate_zeroshot_imagenet,
    SigLIPLoss,
)
import tqdm
import os

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import wandb


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model

    def forward(self, image, text):
        # Standard OpenCLIP forward
        image_features, text_features, logit_scale = self.module(image, text)
        return image_features, text_features, logit_scale

    def __getattr__(self, name):
        # Delegate missing attributes (e.g., encode_text) to the wrapped OpenCLIP model.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, local_rank, dist.get_rank(), dist.get_world_size()
    return False, 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def main(args):
    is_ddp, local_rank, rank, world_size = setup_ddp()
    is_main = rank == 0
    set_seed(args.seed + rank)
    device = torch.device("cuda", local_rank) if is_ddp else torch.device(args.device)

    if is_main:
        wandb.init(
            project="clip-cc3m-pretrain",
            name=f"{args.model_name}-{args.optimizer_name}-bs{args.batch_size*world_size}",
            config=vars(args),
        )

    num_samples = 2_800_000
    global_batch_size = args.batch_size * world_size

    args.max_steps = (num_samples // global_batch_size) * args.epoch_num

    model_precision = args.precision
    if args.precision == "amp":
        model_precision = "fp32"

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name=args.model_name,
        pretrained=args.pretrained,
        precision=model_precision,
    )
    model = Wrapper(model).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    elif torch.cuda.device_count() > 1 and is_main:
        print(f"using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    use_autocast = args.precision in ("amp", "bf16")
    autocast_dtype = torch.float16 if args.precision == "amp" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler() if args.precision == "amp" else None

    loss_fn = None
    if args.loss_type == "siglip":
        loss_fn = SigLIPLoss(temperature_init=10.0, bias_init=-10.0).to(device)
        if is_ddp:
            loss_fn = DDP(loss_fn, device_ids=[local_rank], output_device=local_rank)

    tok = open_clip.get_tokenizer(args.model_name)

    train_loader = get_cc3m_loader(
        args,
        preprocess_fn=preprocess_train,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        world_size=world_size,
    )

    imagenet_val = ImageNetVal(
        val_dir=args.imagenet_val_dir,
        gt_path=args.imagenet_gt,
        transform=preprocess_val,
        strict=True,
    )
    val_loader = DataLoader(
        imagenet_val,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    try:
        with open(args.imagenet_classnames, "r", encoding="utf-8") as f:
            classnames = [line.strip() for line in f if line.strip()]
        assert len(classnames) == 1000
    except FileNotFoundError:
        print(
            f"{args.imagenet_classnames} does not exist, moving to default classnames"
        )
        classnames = IMAGENET_CLASSNAMES

    params = list(model.parameters())
    if loss_fn is not None:
        params += list(loss_fn.parameters())

    # Optimizer
    optimizer_kwargs = {
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
        "betas": (args.beta1, args.beta2),
        "eps": args.epsilon,
    }
    base_optimizer = torch.optim.AdamW
    if args.optimizer_name == "AdamW":
        optimizer = base_optimizer(params, **optimizer_kwargs)
    elif args.optimizer_name == "SAM":
        optimizer = SAM(params, base_optimizer, rho=args.sam_rho, **optimizer_kwargs)
    elif args.optimizer_name == "ASAM":
        optimizer = SAM(
            params, base_optimizer, rho=args.sam_rho, adaptive=True, **optimizer_kwargs
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
    classifier = None

    train_iter = iter(train_loader)

    while step < args.max_steps:
        try:
            images, captions = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, captions = next(train_iter)

        images = images.to(device, non_blocking=True)
        text_tokens = tok(captions).to(device, non_blocking=True)

        if args.optimizer_name == "AdamW":
            loss = adam_step(
                optimizer,
                model,
                images,
                text_tokens,
                loss_fn=loss_fn,
                autocast_dtype=autocast_dtype,
                scaler=scaler,
            )
            running_loss += float(loss)
        else:
            _, loss2 = sam_step(
                optimizer,
                model,
                images,
                text_tokens,
                loss_fn=loss_fn,
                autocast_dtype=autocast_dtype,
                scaler=scaler,
            )
            running_loss += float(loss2)

        scheduler.step()
        step += 1

        if args.log_every > 0 and (step % args.log_every == 0) and is_main:
            avg = running_loss / args.log_every
            lr_now = optimizer.param_groups[0]["lr"]
            wandb.log(
                {
                    "train/loss": avg,
                    "train/lr": lr_now,
                    "train/step": step,
                    "global_step": step,
                }
            )
            tqdm.tqdm.write(
                f"Step {step}/{args.max_steps} | lr={lr_now:.3e} | train_loss={avg:.4f}"
            )
            running_loss = 0.0

        if (
            args.test_every > 0
            and (step % args.test_every == 0 or step == 1)
            and is_main
        ):
            model.eval()
            model_for_eval = model.module if hasattr(model, "module") else model
            if classifier is None:
                classifier = build_zero_shot_classifier(
                    model=model_for_eval,
                    tokenizer=tok,
                    classnames=classnames,
                    templates=OPENAI_IMAGENET_TEMPLATES,
                    num_classes_per_batch=args.num_classes_per_batch,
                    device=device,
                    use_tqdm=True,
                )
            else:
                classifier = classifier.to(device)

            # enforce (D, 1000) convention
            if classifier.shape[0] == 1000 and classifier.shape[1] != 1000:
                classifier = classifier.t()

            top1, top5 = evaluate_zeroshot_imagenet(
                model_for_eval, classifier, val_loader, device
            )
            print(
                f"Step {step}: Zero-shot ImageNet-1k Top-1={top1:.2f}% Top-5={top5:.2f}%"
            )
            wandb.log(
                {
                    "val/imagenet_top1": top1,
                    "val/imagenet_top5": top5,
                    "global_step": step,
                }
            )
            model.train()

    if is_main:
        wandb.finish()
    cleanup_ddp()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "Pretrain CLIP on DataComp; evaluate zero-shot on ImageNet-1k val"
    )

    # ImageNet val dir must contain ILSVRC2012_val_*.JPEG files.
    parser.add_argument(
        "--imagenet-val-dir",
        type=str,
        default="./data/imagenet/val_flat",
        help="ImageNet-1k validation directory with ILSVRC2012_val_*.JPEG files.",
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

    # dataset
    parser.add_argument(
        "--cc3m-path",
        type=str,
        default="./data/cc3m",
        help="Path to folder containing .tar shards",
    )

    # Model
    parser.add_argument("--model-name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument(
        "--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16", "amp"]
    )

    # Optimizer
    parser.add_argument(
        "--optimizer-name",
        type=str,
        required=True,
        choices=["AdamW", "SAM", "ASAM", "AdamSAM"],
    )
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--sam-rho", type=float, default=0.05)

    # Training
    parser.add_argument(
        "--loss-type", type=str, default="clip", choices=["clip", "siglip"]
    )
    parser.add_argument(
        "--epoch_num",
        type=int,
        default=35,
        help="Total number of optimizer steps to run (step-based training budget).",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=500, help="Number of linear warmup steps."
    )
    parser.add_argument(
        "--warmup-start-factor",
        type=float,
        default=1e-4,
        help="Warmup start factor for LinearLR (lr starts at start_factor * base_lr).",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
        help="Minimum LR for cosine decay (eta_min).",
    )

    parser.add_argument("--batch-size", type=int, default=1024)
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

    # Zero-shot classifier construction
    parser.add_argument("--num-classes-per-batch", type=int, default=10)

    # Eval
    parser.add_argument("--eval-batch-size", type=int, default=128)

    args = parser.parse_args()
    main(args)
