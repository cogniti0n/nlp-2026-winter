import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clip_contrastive_loss(image_features, text_features, logit_scale):
    """
    DDP-correct CLIP symmetric cross entropy loss using global negatives.
    image_features/text_features: (B_local, D), normalized
    logit_scale: scalar (already exp'ed), positive
    """
    # Gather global features for negatives
    img_all, txt_all = gather_features(image_features, text_features)

    # Build targets that point to the matching pair positions in the global batch
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        B_local = image_features.size(0)

        # This assumes every rank has identical B_local (true with partial=False)
        targets = torch.arange(B_local, device=image_features.device) + rank * B_local
    else:
        targets = torch.arange(image_features.size(0), device=image_features.device)

    logits_per_image = logit_scale * (image_features @ txt_all.t())  # (B_local, B_global)
    logits_per_text  = logit_scale * (text_features @ img_all.t())  # (B_local, B_global)

    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)
    return 0.5 * (loss_i + loss_t)

import torch
import torch.nn.functional as F

class SigLIPLoss(nn.Module):
    """
    Sigmoid Loss for Language Image Pre-Training.
     decoupling batch size from loss geometry.
     Ideally suited for optimizer benchmarking on smaller hardware.
    """
    def __init__(self, temperature_init: float = 10.0, bias_init: float = -10.0):
        super().__init__()
        self.t_prime = nn.Parameter(torch.tensor(temperature_init))
        self.b = nn.Parameter(torch.tensor(bias_init))

    def forward(self, image_features, text_features):
        # pairwise cosine similarity
        logits = torch.matmul(image_features, text_features.transpose(0, 1)) 
        
        # SigLIP scaling: logits * exp(t) + b
        t = self.t_prime.exp()
        logits = logits * t + self.b
        
        # Create labels: 1 for diagonal (positives), -1 for off-diagonal (negatives)
        labels = 2 * torch.eye(logits.shape[0], device=logits.device) - 1
        
        # Stable Sigmoid Loss: -log(sigmoid(logits * labels))
        loss = -F.logsigmoid(logits * labels).sum() / image_features.shape[0]
        return loss

import torch.distributed as dist

def gather_features(image_features: torch.Tensor, text_features: torch.Tensor):
    """
    All-gather image/text features across ranks.
    Returns concatenated (global_B, D) tensors on every rank.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return image_features, text_features

    world_size = dist.get_world_size()
    # allocate per-rank buffers
    img_list = [torch.zeros_like(image_features) for _ in range(world_size)]
    txt_list = [torch.zeros_like(text_features) for _ in range(world_size)]
    dist.all_gather(img_list, image_features.contiguous())
    dist.all_gather(txt_list, text_features.contiguous())
    return torch.cat(img_list, dim=0), torch.cat(txt_list, dim=0)

@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, target: torch.Tensor, topk=(1,)):
    # logits: (B, C), target: (B,)
    maxk = max(topk)
    pred = logits.topk(maxk, dim=1, largest=True, sorted=True).indices  # (B, maxk)
    correct = pred.eq(target.unsqueeze(1))  # (B, maxk)
    out = []
    for k in topk:
        out.append(correct[:, :k].any(dim=1).float().sum().item())
    return out  # counts

def forward_features(model, images, text_tokens):
    image_features, text_features, logit_scale = model(images, text_tokens)

    image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
    text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)

    if logit_scale.dim() > 0: 
        logit_scale = logit_scale[0]

    logit_scale = logit_scale.clamp(max=100.0)

    return image_features, text_features, logit_scale

def sam_step(optimizer, model, images, text_tokens, loss_fn=None, autocast_dtype=None, scaler=None):
    assert hasattr(optimizer, "first_step") and hasattr(optimizer, "second_step")
    if loss_fn is None:
        loss_fn = clip_contrastive_loss

    optimizer.zero_grad(set_to_none=True)
    use_autocast = (autocast_dtype is not None)
    with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
        image_features, text_features, logit_scale = forward_features(model, images, text_tokens)
        loss = loss_fn(image_features, text_features, logit_scale)
    
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        optimizer.first_step(zero_grad=True)
    else:
        loss.backward()
        optimizer.first_step(zero_grad=True)

    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
        image_features_2, text_features_2, logit_scale_2 = forward_features(model, images, text_tokens)
        loss_2 = loss_fn(image_features_2, text_features_2, logit_scale_2)

    if scaler is not None:
        scaler.scale(loss_2).backward()
        scaler.unscale_(optimizer)
        optimizer.second_step(zero_grad=True)
    else:
        loss_2.backward()
        optimizer.second_step(zero_grad=True)

    return loss.item(), loss_2.item()

def adam_step(optimizer, model, images, text_tokens, loss_fn=None, autocast_dtype=None, scaler=None):
    if loss_fn is None:
        loss_fn = clip_contrastive_loss
    optimizer.zero_grad()

    use_autocast = (autocast_dtype is not None)

    with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
        image_features, text_features, logit_scale = forward_features(model, images, text_tokens)
        loss = loss_fn(image_features, text_features, logit_scale)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return loss.item()

def prompt_train_step(optimizer, model, prompt_learner, images, targets, scaler=None):
    use_amp = scaler is not None
    optimizer.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast(enabled=use_amp):
        image_features = model.encode_image(images)
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)

        text_features = prompt_learner()  # (C, D), normalized

        # logit_scale handling
        if hasattr(model, "logit_scale"):
            logit_scale = model.logit_scale
            if logit_scale.dim() > 0:
                logit_scale = logit_scale[0]
            logit_scale = logit_scale.exp().clamp(max=100.0)
        else:
            logit_scale = 100.0

        logits = logit_scale * image_features @ text_features.t()  # (B, C)
        loss = F.cross_entropy(logits, targets)

    if use_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return float(loss.item())

@torch.no_grad()
def evaluate_prompt_classifier(model, prompt_learner, loader, device):
    model.eval()
    prompt_learner.eval()

    text_features = prompt_learner()  # (C, D)
    top1 = 0
    n = 0

    # logit scale
    if hasattr(model, "logit_scale"):
        logit_scale = model.logit_scale
        if logit_scale.dim() > 0:
            logit_scale = logit_scale[0]
        logit_scale = logit_scale.exp().clamp(max=100.0)
    else:
        logit_scale = 100.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        image_features = model.encode_image(images)
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)

        logits = logit_scale * image_features @ text_features.t()
        pred = logits.argmax(dim=1)
        top1 += (pred == targets).sum().item()
        n += targets.numel()

    return 100.0 * top1 / n

@torch.no_grad()
def evaluate_zeroshot_imagenet(model, classifier, val_loader, device):
    model.eval()
    top1 = 0.0
    top5 = 0.0
    n = 0

    # Make classifier shape robust: want (D, C)
    if classifier.dim() != 2:
        raise ValueError(f"classifier must be 2D, got {classifier.shape}")
    if classifier.shape[0] != model.text_projection.shape[1] and classifier.shape[1] == model.text_projection.shape[1]:
        classifier = classifier.t()

    for images, target in val_loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = model.logit_scale
        if logit_scale.dim() > 0:
            logit_scale = logit_scale[0]
        logit_scale = logit_scale.exp().clamp(max=100.0)
        logits = (logit_scale * image_features) @ classifier  # (B, 1000)

        c1, c5 = accuracy_topk(logits, target, topk=(1, 5))
        top1 += c1
        top5 += c5
        n += images.size(0)

    return (top1 / n) * 100.0, (top5 / n) * 100.0