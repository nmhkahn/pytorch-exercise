import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, TEXT,
		         hidden_dim=512, num_layers=1,
                 num_class=4):
        super().__init__()

        vocab_size = TEXT.vocab.vectors.size(0)
        embed_dim = TEXT.vocab.vectors.size(1)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.GRU(embed_dim, hidden_dim, 
                              num_layers=num_layers, 
                              dropout=0.5, bidirectional=True)

        self.embedding.weight.data.copy_(TEXT.vocab.vectors)
        self.embedding.weight.requires_grad=False

        self.linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim*2, num_class)
        )
    
    def forward(self, x):
        embed = self.embedding(x)
        hidden, _ = self.encoder(embed)
        
        out = self.linear(hidden[-1])
        return out
