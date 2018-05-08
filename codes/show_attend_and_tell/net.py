import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Net(nn.Module):
    def __init__(self, TEXT,
                 hidden_dim=512, attn_dim=512,
                 num_layers=1):
        super().__init__()

        vocab_size = TEXT.vocab.vectors.size(0)
        embed_dim = TEXT.vocab.vectors.size(1)
        
        self.encoder = Encoder(embed_dim)
        self.decoder = AttentionDecoder(TEXT,
            vocab_size, embed_dim,
            hidden_dim, attn_dim,
            num_layers)

    def forward(self, image, caption, lengths):
        feature = self.encoder(image)
        out = self.decoder(feature, caption, lengths)

        return out

    def sample(self, image):
        feature = self.encoder(image)
        out = self.decoder.sample(feature)
        
        return out


class Encoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        # exclude last max_pool to maintain output shape as [512, 14, 14]
        self.body = models.vgg19_bn(pretrained=True).features[:-1]

        for param in self.body.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        return self.body(x)


class AttentionDecoder(nn.Module):
    def __init__(self, TEXT,
                 vocab_size, embed_dim=300,
                 hidden_dim=512, attn_dim=512, 
                 num_layers=1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # RNN layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim+attn_dim, hidden_dim, 
                           num_layers=num_layers,
                           batch_first=True)
        self.exit = nn.Linear(hidden_dim, vocab_size)

        self.embedding.weight.data.copy_(TEXT.vocab.vectors)
        self.embedding.weight.requires_grad_(False)

        # projection and attention layers
        self.proj_feature = nn.Linear(512, attn_dim)
        self.proj_hidden  = nn.Linear(hidden_dim, attn_dim)
        self.attn_hidden  = nn.Linear(attn_dim, 1)
        
    def _attention(self, feature, feature_proj, hx):
        # (num_layers, N, 512) -> (N, 1, num_layers*512)
        hx = hx.permute(1, 0, 2).view(-1, 1, self.num_layers*self.hidden_dim)
        hx_attn = self.proj_hidden(hx)

        hx_attn = F.relu(feature_proj + hx_attn)       # (N, 196, 512)
        hx_attn = self.attn_hidden(hx_attn).squeeze(2) # (N, 196)
        
        alpha = F.softmax(hx_attn, dim=1)
        context = torch.sum(feature * alpha.unsqueeze(2), 1)

        return context, alpha

    def forward(self, feature, caption, lengths):
        batch_size = feature.size(0)

        # initial hidden state
        hx = feature.new_zeros(
            self.num_layers, batch_size, self.hidden_dim,
            requires_grad=False)
        cx = hx.clone()
        
        # (N, 512, 14, 14) -> (N, 196, 512)
        feature = feature.permute(0, 2, 3, 1).view(-1, 196, 512)
        feature_proj = self.proj_feature(feature)
        
        predicts = feature.new_zeros((batch_size, lengths[0], self.vocab_size))
        for t in range(lengths[0]):
            # do not operate the sequence which shorter then current length
            # sequences have to be sorted as descending order
            batch_size = sum(i >= t for i in lengths)
            
            embed = self.embedding(caption[:batch_size, t])
            context, alpha = self._attention(
                feature[:batch_size],
                feature_proj[:batch_size], 
                hx[:, :batch_size])
            
            joint_embed = torch.cat([context, embed], 1).unsqueeze(1)
            out, (hx, cx) = self.rnn(
                joint_embed, 
                (hx[:, :batch_size], cx[:, :batch_size]))

            out = self.exit(out)
            out = out.view(-1, out.size(2))
            
            predicts[:batch_size, t, :] = out
        
        return predicts.view(-1, predicts.size(2))

    def sample(self, feature):
        batch_size = feature.size(0)
      
        # initial hidden state
        hx = feature.new_zeros(
            self.num_layers, batch_size, self.hidden_dim,
            requires_grad=False)
        cx = hx.clone()
        
        # (N, 512, 14, 14) -> (N, 196, 512)
        feature = feature.permute(0, 2, 3, 1).view(-1, 196, 512)
        feature_proj = self.proj_feature(feature)
        
        # initial embed (<start> token)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embed = torch.full((batch_size,), fill_value=2, dtype=torch.int64).to(device)

        alphas, indices = list(), list()
        for t in range(30):
            embed = self.embedding(embed)
            context, alpha = self._attention(feature, feature_proj, hx)
            
            joint_embed = torch.cat([context, embed], 1).unsqueeze(1)
            out, (hx, cx) = self.rnn(joint_embed, (hx, cx))

            out = self.exit(out)
            out = out.view(-1, out.size(2))
            _, argmax = torch.max(out, 1)

            alphas.append(alpha)
            indices.append(argmax)
                        
            # previous output is current input
            embed = argmax
                  
        alphas  = torch.stack(alphas, 1).cpu().numpy()
        indices = torch.stack(indices, 1).cpu().numpy()

        return alphas, indices
