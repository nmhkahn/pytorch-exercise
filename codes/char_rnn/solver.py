import os
import numpy as np
import torch
from net import Net
from utils import *

class Solver():
    def __init__(self, args):
        # prepare shakespeare dataset
        train_iter, data_info = load_shakespeare(args.batch_size, args.bptt_len)
        self.vocab_size = data_info["vocab_size"]
        self.TEXT = data_info["TEXT"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.net = Net(self.vocab_size, args.embed_dim, 
                       args.hidden_dim, args.num_layers).to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=1) # <pad>: 1
        self.optim   = torch.optim.Adam(self.net.parameters(), args.lr)
        
        self.args = args
        self.train_iter = train_iter
        
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        
    def fit(self):
        args = self.args

        for epoch in range(args.max_epochs):
            text = self.sample(length=300)
            self.net.train()
            for step, inputs in enumerate(self.train_iter):
                X = inputs.text.to(self.device)
                y = inputs.target.to(self.device)

                out, _ = self.net(X)
                loss = self.loss_fn(out, y.view(-1))

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            
            if (epoch+1) % args.print_every == 0:
                text = self.sample(args.sample_length, args.sample_prime)
                print("Epoch [{}/{}] loss: {:.3f}"
                    .format(epoch+1, args.max_epochs, loss.item()/args.bptt_len))
                print(text, "\n")
                self.save(args.ckpt_dir, args.ckpt_name, epoch+1)

    def sample(self, length, prime="The"):
        args = self.args

        self.net.eval()
        samples = list(prime)

        # convert prime string to torch.LongTensor type
        prime = self.TEXT.process(prime, device=self.device, train=False)
        
        # sample character indices
        indices = self.net.sample(prime, length)

        # convert char indices to string type
        for index in indices:
            out = self.TEXT.vocab.itos[index.item()]
            samples.append(out.replace("<eos>", "\n"))
        
        self.TEXT.sequential = True

        return "".join(samples)
                
    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)
