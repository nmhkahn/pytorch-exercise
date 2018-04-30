import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from net import *

class Solver():
    def __init__(self, args):
        # define normalize transformation
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                     std=(0.5, 0.5, 0.5))
        ])

        # prepare fashion MNIST dataset
        self.train_dataset = datasets.FashionMNIST(
            root=args.data_root,
            train=True, 
            transform=transform, 
            download=True)
 
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=args.batch_size, 
            shuffle=True)
        
        self.G = Generator(z_dim=args.z_dim)
        self.D = Discriminator()
        
        # cudafy if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.G = self.G.to(self.device)
        self.D = self.D.to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optim_G = torch.optim.Adam(self.G.parameters(), args.lr)
        self.optim_D = torch.optim.Adam(self.D.parameters(), args.lr)
        
        self.args = args
        
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        
    def fit(self):
        args = self.args

        for epoch in range(args.max_epochs):
            self.G.train()
            self.D.train()
            for step, inputs in enumerate(self.train_loader):
                batch_size = inputs[0].size(0)

                images = inputs[0].to(self.device)
                labels = inputs[1].to(self.device)
                
                # create the labels used to distingush real or fake
                real_labels = torch.ones(batch_size, dtype=torch.int64).to(self.device)
                fake_labels = torch.zeros(batch_size, dtype=torch.int64).to(self.device)
                
                # train the discriminator
                
                # discriminator <- real image
                D_real, D_real_cls = self.D(images)
                D_loss_real = self.loss_fn(D_real, real_labels)
                D_loss_real_cls = self.loss_fn(D_real_cls, labels)
                
                # noise vector
                z = torch.randn(batch_size, args.z_dim).to(self.device)

                # make label to onehot vector
                y_onehot = torch.zeros((batch_size, 10)).to(self.device)
                y_onehot.scatter_(1, labels.unsqueeze(1), 1)
                y_onehot.requires_grad_(False)
                
                # discriminator <- fake image
                G_fake = self.G(y_onehot, z)
                D_fake, D_fake_cls = self.D(G_fake)
                D_loss_fake = self.loss_fn(D_fake, fake_labels)
                D_loss_fake_cls = self.loss_fn(D_fake_cls, labels)
                
                D_loss = D_loss_real + D_loss_fake + \
                         D_loss_real_cls + D_loss_fake_cls
                self.D.zero_grad()
                D_loss.backward()
                self.optim_D.step()
                
                # train the generator

                z = torch.randn(batch_size, args.z_dim).to(self.device)
                G_fake = self.G(y_onehot, z)
                D_fake, D_fake_cls = self.D(G_fake)
                
                G_loss = self.loss_fn(D_fake, real_labels) + \
                         self.loss_fn(D_fake_cls, labels)
                self.G.zero_grad()
                G_loss.backward()
                self.optim_G.step()

            if (epoch+1) % args.print_every == 0:
                print("Epoch [{}/{}] Loss_D: {:.3f}, Loss_G: {:.3f}".
                    format(epoch+1, args.max_epochs, D_loss.item(), G_loss.item()))
                self.save(args.ckpt_dir, epoch+1)
                self.sample(epoch+1)

                
    def sample(self, global_step=0):
        self.G.eval()
        self.D.eval()
        
        args = self.args
        batch_size = args.batch_size
                
        # produce the samples among 10-classes
        with torch.no_grad():
            for i in range(10):
                z = torch.randn(batch_size, args.z_dim).to(self.device)
                labels = torch.full((batch_size,), i, dtype=torch.int64).to(self.device)
            
                # make label to onehot vector
                y_onehot = torch.zeros((batch_size, 10)).to(self.device)
                y_onehot.scatter_(1, labels.unsqueeze(1), 1)
                y_onehot.requires_grad_(False)

                G_fake = self.G(y_onehot, z)

                # save the results
                save_image(denormalize(G_fake.detach()),
                    os.path.join(args.result_dir, "fake_{}_{}.png".format(global_step, i)))

    def save(self, ckpt_dir, global_step):
        D_path = os.path.join(
            ckpt_dir, "discriminator_{}.pth".format(global_step))
        G_path = os.path.join(
            ckpt_dir, "generator_{}.pth".format(global_step))

        torch.save(self.D.state_dict(), D_path)
        torch.save(self.G.state_dict(), G_path)


def denormalize(tensor):
    out = (tensor + 1) / 2
    return out.clamp(0, 1)

