import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class GRUEncoder(nn.Module):
    
    def __init__(self, vocab_size, padding_idx, embed_size, hidden_size, output_size, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, idx, lengths):
        # lengths: size (B,) -> actual lengths of seq before padding
        B, T = idx.size()
        embedded = self.dropout(self.embedding(idx))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        last = h_n[-1]
        logits = self.fc(last)
        return logits