import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from net import Net
from dataset import Dataset

class Solver():
    def __init__(self, args):
        self.net     = Net()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optim   = torch.optim.Adam(self.net.parameters(), args.lr)
        
        self.train_data = Dataset(train=True,
                                  data_root=args.data_root,
                                  size=args.image_size)
        self.test_data  = Dataset(train=False,
                                  data_root=args.data_root,
                                  size=args.image_size)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=args.batch_size,
                                       num_workers=1,
                                       shuffle=True, drop_last=True)
        
        self.net     = self.net.cuda()
        self.loss_fn = self.loss_fn.cuda()
        
        self.args = args
        
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        
    def fit(self):
        args = self.args

        for epoch in range(args.max_epochs):
            self.net.train()
            for step, inputs in enumerate(self.train_loader):
                X = Variable(inputs[0], requires_grad=False).cuda()
                y = Variable(inputs[1], requires_grad=False).cuda()
                
                pred_y = self.net(X)
                loss = self.loss_fn(pred_y, y)
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            
            if (epoch+1) % args.print_every == 0:
                train_acc = self.evaluate(self.train_data)
                test_acc = self.evaluate(self.test_data)
                print("Epoch [{}/{}] Loss: {:.3f} Train Acc: {:.3f}, Test Acc: {:.3f}".
                    format(epoch+1, args.max_epochs, loss.data[0], train_acc, test_acc))
                self.save(args.ckpt_dir, args.ckpt_name, epoch+1)

    def evaluate(self, data):
        args = self.args
        loader = DataLoader(data,
                            batch_size=1,
                            num_workers=1,
                            shuffle=False, drop_last=False)

        num_correct, num_total = 0, 0
        self.net.eval()
        for inputs in loader:
            X  = Variable(inputs[0], volatile=True).cuda()
            y  = inputs[1].cuda()

            pred_y = self.net(X)
            _, preds = torch.max(pred_y.data, 1)

            num_correct += (preds == y).sum()
            num_total += y.size(0)

        return num_correct / num_total

    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)
