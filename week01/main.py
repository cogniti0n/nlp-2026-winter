import re
import random

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
import pandas as pd

from attention import TransformerEncoder, TransformerConfig
from gru import GRUEncoder

from dataclasses import dataclass
from typing import List, Dict
from collections import Counter

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

_word_re = re.compile(r"[A-Za-z0-9']+")

def tokenize(text: str) -> List[str]:
    return _word_re.findall(text.lower())

@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    
    @property
    def pad_idx(self):
        return self.stoi[self.pad_token]
    
    @property
    def unk_idx(self):
        return self.stoi[self.unk_token]
    
def build_vocab(texts, max_vocab_size, min_freq=1, pad_token ="<PAD>", unk_token ="<UNK>"):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))

    itos = [pad_token, unk_token]
    stoi = {pad_token: 0, unk_token: 1}

    for token, freq in counter.most_common():
        if freq < min_freq: break
        if token in stoi: continue
        stoi[token] = len(itos)
        itos.append(token)
        if len(itos) >= max_vocab_size: break
    
    return Vocab(stoi, itos, pad_token, unk_token)

def encode_text(text, vocab, max_len):
    tokens = tokenize(text)
    ids = [vocab.stoi.get(tok, vocab.unk_idx) for tok in tokens]
    ids = ids[:max_len]

    if len(ids) < max_len:
        ids = ids + [vocab.pad_idx] * (max_len - len(ids))
    
    return torch.tensor(ids, dtype=torch.long)

def load_csv(path, text_col="review", label_col="sentiment"):
    df = pd.read_csv(path)
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()
    labels = [1 if r == "positive" else 0 for r in labels]
    return texts, labels

class IMDBDataset(Dataset):

    def __init__(self, texts, labels, vocab: Vocab, max_len):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = encode_text(self.texts[idx], self.vocab, self.max_len)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def run_epoch(model, dataloader, loss_fn, optimizer, device):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(x)
        loss = loss_fn(logits, y)

        if is_train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    texts, labels = load_csv(args.path)
    vocab = build_vocab(texts, args.max_vocab_size, args.min_freq, args.pad_token, args.unk_token)

    n_split = int(args.split * len(texts))
    texts_train = texts[:n_split]; texts_test = texts[n_split:]
    labels_train = labels[:n_split]; labels_test = labels[n_split:]

    train_dataset = IMDBDataset(texts_train, labels_train, vocab, args.max_len)
    test_dataset = IMDBDataset(texts_test, labels_test, vocab, args.max_len)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.arch == "GRU":
        model = GRUEncoder(
            vocab_size=len(vocab.itos),
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            output_size=args.num_classes,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )
    else:
        config = TransformerConfig(
            block_size=args.max_len,
            vocab_size=len(vocab.itos),
            pad_idx=vocab.pad_idx,
            n_layers=args.n_layers,
            n_head=args.n_head,
            n_embd=args.embed_size,
            n_ff=args.n_ff,
            dropout=args.dropout,
        )
        model = TransformerEncoder(config)

    model = model.to(device)

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        print(f"running epoch: {epoch + 1}/{args.epochs}")
        train_loss, train_acc = run_epoch(
            model, train_dataloader, loss_fn, optimizer, device
        )
        test_loss, test_acc = run_epoch(
            model, test_dataloader, loss_fn, None, device
        )
        print(
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default="IMDB-Dataset.csv")
    parser.add_argument("--arch", type=str, default="GRU", choices=["GRU", "Transformer"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--max_vocab_size", type=int, default=20000)
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--pad_token", type=str, default="<PAD>")
    parser.add_argument("--unk_token", type=str, default="<UNK>")
    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_ff", type=int, default=1024)
    parser.add_argument("--num_classes", type=int, default=2)

    args = parser.parse_args()
    main(args)
