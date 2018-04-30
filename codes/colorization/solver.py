import os
import numpy as np
from skimage.color import lab2rgb
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from net import Net
from dataset import Dataset

class Solver():
    def __init__(self, args):      
        # prepare a datasets
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
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net     = Net().to(self.device)
        self.loss_fn = torch.nn.L1Loss()
        self.optim   = torch.optim.Adam(self.net.parameters(), args.lr)
        
        self.args = args
        
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        
    def fit(self):
        args = self.args

        for epoch in range(args.max_epochs):
            self.net.train()
            for step, inputs in enumerate(self.train_loader):
                gt_gray = inputs[0].to(self.device)
                gt_ab   = inputs[1].to(self.device)
                
                pred_ab = self.net(gt_gray)
                loss = self.loss_fn(pred_ab, gt_ab)
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            if (epoch+1) % args.print_every == 0:
                print("Epoch [{}/{}] loss: {:.6f}".format(epoch+1, args.max_epochs, loss.item()))
                self.save(args.ckpt_dir, args.ckpt_name, epoch+1)

    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)
