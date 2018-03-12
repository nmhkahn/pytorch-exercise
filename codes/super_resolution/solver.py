import os
import numpy as np
import scipy.misc as misc
import skimage.measure as measure
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from net import Net
from dataset import Dataset

class Solver():
    def __init__(self, args):

        self.net     = Net(args.scale)
        self.loss_fn = nn.L1Loss()
        self.optim   = torch.optim.Adam(self.net.parameters(), args.lr)
        
        self.train_data = Dataset(args.scale, train=True,
                                  data_root=args.data_root,
                                  size=args.image_size)
        self.test_data  = Dataset(args.scale, train=False,
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
                image_hr = Variable(inputs[0], requires_grad=False).cuda()
                image_lr = Variable(inputs[1], requires_grad=False).cuda()
                
                image_sr = self.net(image_lr)
                loss = self.loss_fn(image_sr, image_hr)
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            
            if (epoch+1) % args.print_every == 0:
                psnr = self.evaluate(epoch+1)
                print("Epoch [{}/{}] PSNR: {:.3f}".format(epoch+1, args.max_epochs, psnr))
                self.save(args.ckpt_dir, args.ckpt_name, epoch+1)

    def evaluate(self, global_step):
        args = self.args
        loader = DataLoader(self.test_data,
                            batch_size=1,
                            num_workers=1,
                            shuffle=False, drop_last=False)

        self.net.eval()
        mean_psnr = 0
        for step, inputs in enumerate(loader):
            image_hr = Variable(inputs[0], requires_grad=False).cuda()
            image_lr = Variable(inputs[1], requires_grad=False).cuda()
                
            image_sr = self.net(image_lr)

            image_hr = image_hr[0].cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).data.numpy()
            image_lr = image_lr[0].cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).data.numpy()
            image_sr = image_sr[0].cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).data.numpy()

            mean_psnr += psnr(image_hr, image_sr) / len(self.test_data)
            
            # save images
            hr_path = os.path.join(args.result_dir, "epoch_{}_{}_HR.jpg".format(global_step, step))
            lr_path = os.path.join(args.result_dir, "epoch_{}_{}_LR.jpg".format(global_step, step))
            sr_path = os.path.join(args.result_dir, "epoch_{}_{}_SR.jpg".format(global_step, step))
            misc.imsave(hr_path, image_hr)
            misc.imsave(lr_path, image_lr)
            misc.imsave(sr_path, image_sr)

        return mean_psnr

    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)


def psnr(im1, im2):
    def im2double(im):
        min_val, max_val = 0, 255
        out = (im.astype(np.float64)-min_val) / (max_val-min_val)
        return out
        
    im1 = im2double(im1)
    im2 = im2double(im2)
    psnr = measure.compare_psnr(im1, im2, data_range=1)
    return psnr
