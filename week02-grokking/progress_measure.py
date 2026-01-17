import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import cast

def gini_coefficient(v: torch.Tensor, eps: float = 1e-12) -> float:
    v = v.detach().float().abs().flatten()
    n = v.numel()
    if n == 0:
        return 0.0
    s = v.sum().item()
    if s < eps:
        return 0.0
    v_sorted, _ = torch.sort(v)
    i = torch.arange(1, n + 1, device=v.device, dtype=torch.float32)
    g = (2.0 * (i * v_sorted).sum() / (n * v_sorted.sum())) - (n + 1.0) / n
    return float(g.item())

@torch.no_grad()
def compute_last_token(model, x):
    model.eval()
    B, L = x.shape
    assert L == model.seq_len

    pos = torch.arange(L, device=x.device)
    h = model.tok_embd(x) + model.pos_embd(pos)
    attn_out, _ = model.mha(h, h, h, need_weights=False)
    h = h + attn_out

    h_last = h[:,-1,:]

    mlp_in = cast(nn.Linear, model.mlp[0])
    a = F.relu(mlp_in(h_last))

    return a

@torch.no_grad()
def eval_with_wl(model, x, y, wl):
    a = compute_last_token(model, x)
    logits = a @ wl.T
    loss = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(dim=1) == y).float().mean().item()

    return loss, acc

@torch.no_grad()
def total_squared_weight(model):
    s = 0.0
    for p in model.parameters():
        s += float((p.detach().float() ** 2).sum().item())
    return s

@torch.no_grad()
def fourier_energy_over_logit_axis(M, dim: int=0):
    Mf = torch.fft.fft(M.to(torch.float32), dim=dim, norm="ortho")
    Mf = torch.movedim(Mf, dim, 0)
    energy = torch.linalg.norm(Mf, dim=tuple(range(1, Mf.ndim))).real
    return energy.abs()


def compute_progress_measure(model, k_key, x_train, y_train, x_full, y_full):
    
    device = next(model.parameters()).device
    P = model.P

    k_key = k_key.detach().to(device).long().flatten()
    k_key_partner = (P - k_key) % P
    k_restricted = torch.unique(torch.cat([k_key, k_key_partner], dim=0))
    k_restricted = k_restricted[k_restricted != 0]

    mask_restricted = torch.zeros(P, device=device, dtype=torch.bool)
    mask_restricted[k_restricted] = True
    mask_excluded = ~mask_restricted
    mask_excluded[0] = True

    wl = model.wl().detach()
    wl_f = torch.fft.fft(wl.to(torch.float32), dim=0, norm="ortho")

    wl_f_restricted = torch.zeros_like(wl_f)
    wl_f_restricted[mask_restricted] = wl_f[mask_restricted]

    wl_f_excluded = torch.zeros_like(wl_f)
    wl_f_excluded[mask_excluded] = wl_f[mask_excluded]

    wl_restricted = torch.fft.ifft(wl_f_restricted, dim=0, norm="ortho").real
    wl_excluded = torch.fft.ifft(wl_f_excluded, dim=0, norm="ortho").real

    restricted_loss, _ = eval_with_wl(model, x_full, y_full, wl_restricted)
    excluded_loss, _ = eval_with_wl(model, x_train, y_train, wl_excluded)

    wl_energy = fourier_energy_over_logit_axis(wl, dim=0)
    gini_wl = gini_coefficient(wl_energy)

    we = model.we().detach()
    we_core = we[:P] # exclude "="
    we_energy = fourier_energy_over_logit_axis(we_core, dim=0)
    gini_we = gini_coefficient(we_energy)

    w2 = total_squared_weight(model)

    return {
        "k_key": k_key.detach().cpu().numpy(),
        "k_restricted": k_restricted.detach().cpu().numpy(),
        "restricted_loss": restricted_loss,
        "excluded_loss": excluded_loss,
        "gini_we": gini_we,
        "gini_wl": gini_wl,
        "total_squared_weight": w2,
    }