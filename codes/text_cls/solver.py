import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from net import Net
from utils import *

class Solver():
    def __init__(self, args):
        
        # load SST dataset
        train_iter, val_iter, test_iter, sst_info = load_sst(args.batch_size)
        vocab_size = sst_info["vocab_size"]
        num_class  = sst_info["num_class"]

        self.net = Net(vocab_size, 
                       args.embed_dim, args.hidden_dim,
                       args.num_layers, num_class)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optim   = torch.optim.Adam(self.net.parameters(), args.lr)
        
        self.net     = self.net.cuda()
        self.loss_fn = self.loss_fn.cuda()
        
        self.args = args
        self.train_iter = train_iter
        self.val_iter   = val_iter
        self.test_iter  = test_iter
        
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        
    def fit(self):
        args = self.args

        for epoch in range(args.max_epochs):
            self.net.train()
            for step, inputs in enumerate(self.train_iter):
                X = inputs.text.cuda()
                y = inputs.label.cuda()

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
        for step, inputs in enumerate(iters):
            X = inputs.text.cuda()
            y = inputs.label.cuda()

            pred_y = self.net(X)
            _, pred_y = torch.max(pred_y.data, 1)

            num_correct += (pred_y == y.data).sum()
            num_total += y.size(0)

        return num_correct / num_total

    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)
