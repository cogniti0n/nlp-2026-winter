import random
import os
import numpy as np

from typing import cast
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F

from progress_measure import compute_progress_measure


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataset(P: int, train_frac: float, seed: int=0):
    eq_id = P

    a = torch.arange(P, dtype=torch.long)
    b = torch.arange(P, dtype=torch.long)
    aa, bb = torch.meshgrid(a, b, indexing="ij")

    aa = aa.reshape(-1)  # (P^2, )
    bb = bb.reshape(-1)
    y = (aa + bb) % P

    eq = torch.full_like(aa, fill_value=eq_id)
    x = torch.stack([aa, bb, eq], dim=1)  # (P^2, 3)

    N = x.shape[0]
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(N, generator=g)

    n_train = int(train_frac * N)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    return x_train, y_train, x_test, y_test

class TransformerModAdd(nn.Module):

    def __init__(self, P=113, d_model=128, n_head=4, d_fc=512, seq_len=3):
        super().__init__()
        self.P = P
        self.seq_len = seq_len
        self.eq_id = P  # "=" -> P
        vocab_size = P + 1  # 0 ~ P-1 + "="

        self.tok_embd = nn.Embedding(vocab_size, d_model)
        self.pos_embd = nn.Embedding(seq_len, d_model)

        self.mha = nn.MultiheadAttention(d_model, n_head, batch_first=True, bias=False, dropout=0.0)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_fc),
            nn.ReLU(),
            nn.Linear(d_fc, d_model),
        )
        self.unembd = nn.Linear(d_model, P, bias=False)

    def forward(self, x):
        B, L = x.shape
        assert L == self.seq_len

        pos = torch.arange(L, device=x.device)
        h = self.tok_embd(x) + self.pos_embd(pos)
        attn_out, _ = self.mha(h, h, h, need_weights=False)
        h = h + attn_out  # residual

        h = h + self.mlp(h)
        logits = self.unembd(h[:, -1, :])
        return logits

    def we(self):
        return self.tok_embd.weight

    def wu(self):
        return self.unembd.weight

    def wout(self):
        layer = cast(nn.Linear, self.mlp[2])
        return layer.weight, layer.bias

    def wl(self):
        """
        logits are similar to wu @ wout @ mlp, as empirically the networks do not significantly use the skip connection around the mlp
        """
        wu = self.wu()
        wout, _ = self.wout()
        return wu @ wout


def save_checkpoint(path, model, optimizer, epoch, extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_hparams": {
            "P": model.P,
            "seq_len": model.seq_len,
            "eq_id": model.eq_id,
            "vocab_size": model.P + 1,
            "d_model": model.tok_embd.embedding_dim,
            "n_head": model.mha.num_heads,
            "d_fc": model.mlp[0].out_features,
        },
        "epoch": epoch,
    }

    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()

    if extra is not None:
        ckpt["extra"] = extra

    torch.save(ckpt, path)


# no scheduler, no gradient clipping, full batch
@torch.no_grad()
def eval_epoch(model, loss_fn, x, y):
    model.eval()
    logits = model(x)
    loss = loss_fn(logits, y).item()
    acc = (logits.argmax(dim=1) == y).float().mean().item()
    return loss, acc


def train_epoch(model, optimizer, loss_fn, x, y):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()
    acc = (logits.argmax(dim=1) == y).float().mean().item()
    return loss.item(), acc


def main(args):
    set_seed(args.seed)
    device = args.device

    # store the results
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    ckpt_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    prog_dir = os.path.join(results_dir, "progress")
    os.makedirs(prog_dir, exist_ok=True)
    
    model = TransformerModAdd(
        P=args.P,
        d_model=args.d_model,
        n_head=args.n_head,
        d_fc=args.d_fc,
        seq_len=args.seq_len,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = F.cross_entropy

    print(f"creating dataset, P={args.P}")
    x_train, y_train, x_test, y_test = create_dataset(args.P, args.train_frac, args.seed)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    train_loss_tot = np.zeros((args.epochs,))
    train_acc_tot = np.zeros((args.epochs,))
    test_loss_tot = np.zeros((args.epochs,))
    test_acc_tot = np.zeros((args.epochs,))

    restricted_loss_tot = []
    excluded_loss_tot = []
    train_loss_tot_ = []
    test_loss_tot_ = []
    gini_we = []
    gini_wl = []
    total_squared_weight = []


    print(f"Beginning training for {args.epochs} epochs")
    model.train()
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, optimizer, loss_fn, x_train, y_train)
        test_loss, test_acc = eval_epoch(model, loss_fn, x_test, y_test)

        train_loss_tot[epoch] = train_loss
        train_acc_tot[epoch] = train_acc
        test_loss_tot[epoch] = test_loss
        test_acc_tot[epoch] = test_acc

        if (epoch + 1) % args.log_interval == 0:
            print(
                f"After epoch: {epoch + 1}/{args.epochs} | "
                f"train loss: {train_loss:.4f} | test loss: {test_loss:.4f} | "
                f"train acc: {train_acc:.4f} | test acc: {test_acc:.4f}"
            )

        if (epoch + 1) % args.save_interval == 0 and args.k is not None and len(args.k) > 0:
            k_key = torch.tensor(args.k, device=device, dtype=torch.long)
            pm = compute_progress_measure(model, k_key, x_train, y_train, x_test, y_test)

            restricted_loss_tot.append(pm["restricted_loss"])
            excluded_loss_tot.append(pm["excluded_loss"])
            train_loss_tot_.append(train_loss)
            test_loss_tot_.append(test_loss)
            gini_we.append(pm["gini_we"])
            gini_wl.append(pm["gini_wl"])
            total_squared_weight.append(pm["total_squared_weight"])

            # prog_path = os.path.join(prog_dir, f"progress_seed{args.seed}_epoch{epoch+1}.npz")
            # np.savez(prog_path, **pm)

            # model_path_tmp = os.path.join(ckpt_dir, f"modadd_seed{args.seed}_epoch{epoch+1}.pt")
            # save_checkpoint(
            #     model_path_tmp,
            #     model,
            #     optimizer=optimizer,
            #     epoch=epoch + 1,
            #     extra={"seed": args.seed, "train_frac": args.train_frac, "k": args.k},
            # )

    print("Training complete")
    
    we, wu, wout, wl = model.we(), model.wu(), model.wout(), model.wl()
    wout_w, wout_b = wout

    print(f"Computed model weights:")
    print(
        f"WE {we.shape}, WU {wu.shape}, Wout {wout_w.shape}/{wout_b.shape}, WL {wl.shape}"
    )
    
    metrics_path = os.path.join(results_dir, "metrics")
    os.makedirs(metrics_path, exist_ok=True)
    results_path = os.path.join(metrics_path, f"seed_{args.seed}.npz")
    np.savez(
        results_path,
        train_loss=train_loss_tot,
        train_acc=train_acc_tot,
        test_loss=test_loss_tot,
        test_acc=test_acc_tot,
    )

    model_path = os.path.join(ckpt_dir, f"modadd_seed{args.seed}.pt")
    save_checkpoint(
        model_path,
        model,
        optimizer=optimizer,
        epoch=args.epochs,
        extra={"seed": args.seed, "train_frac": args.train_frac},
    )

    restricted_loss_tot = np.array(restricted_loss_tot)
    excluded_loss_tot = np.array(excluded_loss_tot)
    train_loss_tot_ = np.array(train_loss_tot_)
    test_loss_tot_ = np.array(test_loss_tot_)
    gini_we = np.array(gini_we)
    gini_wl = np.array(gini_wl)
    total_squared_weight = np.array(total_squared_weight)

    prog_path = os.path.join(prog_dir, f"progress_seed{args.seed}.npz")
    np.savez(
        prog_path,
        restricted_loss=restricted_loss_tot,
        excluded_loss=excluded_loss_tot,
        train_loss=train_loss_tot_,
        test_loss=test_loss_tot_,
        gini_we=gini_we,
        gini_wl=gini_wl,
        total_squared_weight=total_squared_weight
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--P", type=int, default=113)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--d-fc", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--train-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--k", type=int, nargs="+", required=True)

    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
