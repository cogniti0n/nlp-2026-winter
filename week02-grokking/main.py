import random
import os
import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dataset(P: int, train_frac: float, device: str="cuda"):
    eq_id = P

    a = torch.arange(P, dtype=torch.long)
    b = torch.arange(P, dtype=torch.long)
    aa, bb = torch.meshgrid(a, b, indexing="ij")

    aa = aa.reshape(-1) # (P^2, )
    bb = bb.reshape(-1)
    y = (aa + bb) % P

    eq = torch.full_like(aa, fill_value=eq_id)
    x = torch.stack([aa, bb, eq], dim=1) # (P^2, 3)

    N = x.shape[0]
    perm = torch.randperm(N)

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
        self.eq_id = P # "=" -> P
        vocab_size = P + 1 # 0 ~ P-1 + "="

        self.tok_embd = nn.Embedding(vocab_size, d_model)
        self.pos_embd = nn.Embedding(seq_len, d_model)

        self.mha = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_fc),
            nn.ReLU(),
            nn.Linear(d_fc, d_model),
        )
        self.unembd = nn.Linear(d_model, P, bias=False)

    def forward(self, x):
        B, L = x.shape
        assert L == self.seq_len

        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.tok_embd(x) + self.pos_embd(pos)

        attn_out, _ = self.mha(h, h, h, need_weights=False)
        h = h + attn_out # residual
        
        h = h + self.mlp(h)
        logits = self.unembd(h[:, -1, :])
        return logits
    
    def we(self):
        return self.tok_embd.weight
    
    def wu(self):
        return self.unembd.weight
    
    def wout(self):
        return self.mlp[2].weight, self.mlp[2].bias
    
    def wl(self):
        """
        logits are similar to wu @ wout @ mlp, as empirically the networks do not significantly use the skip connection around the mlp
        """
        wu = self.wu()
        wout, _ = self.wout()
        return wu @ wout

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
    model = TransformerModAdd(
        P=args.P,
        d_model=args.d_model,
        n_head=args.n_head,
        d_fc=args.d_fc,
        seq_len=args.seq_len,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.CrossEntropyLoss()

    print(f"creating dataset, P={args.P}")
    x_train, y_train, x_test, y_test = create_dataset(args.P, args.train_frac, device)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    train_loss_tot = np.zeros((args.epochs,))
    train_acc_tot = np.zeros((args.epochs,))
    test_loss_tot = np.zeros((args.epochs,))
    test_acc_tot = np.zeros((args.epochs,))

    print(f"Beginning training for {args.epochs} epochs")
    model.train()
    for epoch in range(args.epochs):
        print(f"Running epoch: {epoch + 1}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, optimizer, loss_fn, x_train, y_train)
        test_loss, test_acc = eval_epoch(model, loss_fn, x_test, y_test)

        train_loss_tot[epoch] = train_loss
        train_acc_tot[epoch] = train_acc
        test_loss_tot[epoch] = test_loss
        test_acc_tot[epoch] = test_acc
    print("Training complete")

    # store the results
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"metrics_seed{args.seed}.npz")
    we, wu, wout, wl = model.we(), model.wu(), model.wout(), model.wl()
    wout_w, wout_b = wout

    print(f"Computed model weights:")
    print(f"WE {we.shape}, WU {wu.shape}, Wout {wout_w.shape}/{wout_b.shape}, WL {wl.shape}")

    np.savez(
        results_path,
        train_loss=train_loss_tot,
        train_acc=train_acc_tot,
        test_loss=test_loss_tot,
        test_acc=test_acc_tot,
        we=we.detach().cpu().numpy(),
        wu=wu.detach().cpu().numpy(),
        wout_w=wout_w.detach().cpu().numpy(),
        wout_b=wout_b.detach().cpu().numpy(),
        wl=wl.detach().cpu().numpy(),
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

    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
