import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self,
                 vocab_size, embed_dim=300,
		         hidden_dim=512, num_layers=1):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, 
                               num_layers=num_layers)

        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        batch_size = x.size(0)

        embed = self.embedding(x)
        out, hidden = self.encoder(embed)

        out = self.decoder(out)
        out = out.view(-1, out.size(2))

        return out, hidden
    
    def sample(self, prime, length):
        # NOTE:
        # For char-RNN, behaviors of train and sample phase are different.
        # e.g. The model can't see the the ground-truth sentence in the sample phase.
        #      So, the input of each time-step has to be the output of previous time.
        # To handle it, RNN model is implemented by for-loop to unroll it.
        indices = list()
        
        # prepare the first hidden state
        out, hidden = self.forward(prime)

        # hidden state of last step
        h_0 = hidden[0][:,-1,:].contiguous().view(-1, 1, self.hidden_dim)
        c_0 = hidden[1][:,-1,:].contiguous().view(-1, 1, self.hidden_dim)
        hidden = (h_0, c_0)

        x = prime[:, -1]
        for t in range(length):
            embed = self.embedding(x)
            embed = embed.view(1, 1, -1)

            out, hidden = self.encoder(embed, hidden) 
            out = self.decoder(out)
            out = out.view(-1, out.size(2))
            
            _, argmax = torch.max(out, 1)
            indices.append(argmax)

            x = argmax # previous output is current output

        return indices
