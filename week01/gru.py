import torch.nn as nn

class GRUEncoder(nn.Module):
    
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        gru_out, _ = self.gru(embedded)
        out = self.fc(gru_out[:, -1, :])
        return out