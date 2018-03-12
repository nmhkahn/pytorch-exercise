import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,
                 vocab_size, embed_dim=300,
		         hidden_dim=512, num_layers=1,
                 num_class=4):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.GRU(embed_dim, hidden_dim, 
                              num_layers=num_layers, 
                              dropout=0.9)

        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_class)
        )
    
    def forward(self, x):
        embed = self.embedding(x)
        hidden, _ = self.encoder(embed)
        
        out = self.linear(hidden[-1])
        return out
