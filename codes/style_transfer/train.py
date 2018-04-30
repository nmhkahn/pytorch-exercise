import os
import numpy as np
import torch
from utils import *
from vgg import *

def mse(feat1, feat2):
    return torch.mean((feat1-feat2)**2)


def gram_matrix(matrix):
    _, c, h, w = matrix.size()
    matrix = matrix.view(c, h*w)

    return torch.mm(matrix, matrix.t())


def single_layer_style_loss(X_feat, style_feat):
    _, c, h, w = X_feat.size()

    X_gram = gram_matrix(X_feat)
    style_gram = gram_matrix(style_feat)

    return mse(X_gram, style_gram) / (c*h*w)


def single_layer_content_loss(X_feat, content_feat):
    return mse(X_feat, content_feat)


def total_variance_loss(image):
    w_variance = torch.sum((image[:,:,:,1:] - image[:,:,:,:-1])**2)
    h_variance = torch.sum((image[:,:,1:,:] - image[:,:,:-1,:])**2)

    return w_variance + h_variance


def fit(X, content, style, device, args):
    style_weights = [4.0, 3.0, 1.5, 1.0, 0.5]

    print("[!] Prepare the pretrained VGGNet")
    vgg = VGGNet().to(device)
    optim = torch.optim.Adam([X], args.lr, betas=[0.5, 0.999])

    print("[!] Start training")        
    for step in range(args.max_steps):
        style_loss, content_loss = 0, 0
        
        style_feats     = vgg(style, phase="style")
        X_style_feats   = vgg(X, phase="style")
        content_feats   = vgg(content, phase="content")
        X_content_feats = vgg(X, phase="content")
        
        for feat in zip(X_content_feats, content_feats):
            X_feat, content_feat = feat
            content_loss += single_layer_content_loss(X_feat, content_feat)

        for i, feat in enumerate(zip(X_style_feats, style_feats)):
            X_feat, style_feat = feat
            style_loss += single_layer_style_loss(X_feat, style_feat)

        tv_loss = total_variance_loss(X)
       
        loss = args.alpha * content_loss + \
               args.beta * style_loss + \
               args.tv * tv_loss

        optim.zero_grad()
        loss.backward()
        optim.step()
        
        if (step+1) % args.print_every == 0:
            print("[{}/{}] Style Loss: {:.3f} Content Loss: {:.3f}"
                .format(step+1, args.max_steps, style_loss.item(), content_loss.item()))

            save_image(X.detach()[0], os.path.join(args.result_dir, "result_{}.png".format(step+1)))


def main(args):
    print("[!] Prepare the content and style images")
    content, style = prepare_images(
        args.content, args.style, args.resize_side_max)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # cudafy if available
    content = content.to(device)
    style   = style.to(device)
    
    X = content.clone()
    X.requires_grad_(True)

    content.requires_grad_(False)
    style.requires_grad_(False)

    fit(X, content, style, device, args)
        
 
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="./result")
    parser.add_argument("--content", type=str, default="./images/content.jpg")
    parser.add_argument("--style", type=str, default="./images/style.jpg")
    parser.add_argument("--resize_side_max", type=int, default=600)

    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--print_every", type=int, default=100)
    
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=100.0)
    parser.add_argument("--tv", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    main(args)
