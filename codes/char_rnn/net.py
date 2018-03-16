import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,
                 vocab_size, embed_dim=300,
		         hidden_dim=512, num_layers=1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, 
                               num_layers=num_layers)

        self.decoder = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)

        embed = self.embedding(x)
        out, hidden = self.encoder(embed.view(1, batch_size, -1), hidden)
        out = self.decoder(out.view(out.size(0)*out.size(1), -1))
        
        return out, hidden
