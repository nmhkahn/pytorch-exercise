import os
import numpy as np
import torch
from net import Net
from utils import *

class Solver():
    def __init__(self, args):
        
        # prepare SST dataset
        train_iter, val_iter, test_iter, sst_info = load_sst(args.batch_size, args.max_vocab)
        vocab_size = sst_info["vocab_size"]
        num_class  = sst_info["num_class"]
        TEXT = sst_info["TEXT"]
        print("[!] vocab_size: {}, num_class: {}".format(vocab_size, num_class))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = Net(TEXT, 
                       args.hidden_dim,
                       args.num_layers, num_class).to(self.device)
        self.optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            args.lr, weight_decay=args.weight_decay)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        self.args = args
        self.train_iter = train_iter
        self.val_iter   = val_iter
        self.test_iter  = test_iter
        
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        
    def fit(self):
        args = self.args

        for epoch in range(args.max_epochs):
            self.net.train()
            for step, inputs in enumerate(self.train_iter):
                X = inputs.text.to(self.device)
                y = inputs.label.to(self.device)

                pred_y = self.net(X)
                loss = self.loss_fn(pred_y, y)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            
            if (epoch+1) % args.print_every == 0:
                train_acc = self.evaluate(self.train_iter)
                val_acc   = self.evaluate(self.val_iter)

                print("Epoch [{}/{}] train_acc: {:.3f}, val_acc: {:.3f}"
                    .format(epoch+1, args.max_epochs, train_acc, val_acc))
                self.save(args.ckpt_dir, args.ckpt_name, epoch+1)

    def evaluate(self, iters):
        args = self.args

        self.net.eval()
        num_correct, num_total = 0, 0

        with torch.no_grad():
            for step, inputs in enumerate(iters):
                X = inputs.text.to(self.device)
                y = inputs.label.to(self.device)

                pred_y = self.net(X)
                _, pred_y = torch.max(pred_y.detach(), 1)
                
                num_correct += (pred_y == y.detach()).sum().item()
                num_total += y.size(0)

        return num_correct / num_total

    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)
