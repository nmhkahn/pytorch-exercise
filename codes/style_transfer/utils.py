import numpy as np
import PIL
from PIL import Image
import torch
import torchvision
from torchvision import transforms

def prepare_images(content_path, style_path,
                   resize_side_max=None, 
                   noise_ratio=0.6):
    content = Image.open(content_path)
    style   = Image.open(style_path)
        
    w, h = content.size
    # resize content image by maintaining aspect ratio
    if resize_side_max:
        scale = max(1, resize_side_max / max([w, h]))
        w, h = int(w*scale), int(h*scale)
        content = content.resize((w, h), PIL.Image.BICUBIC)

    # match style image size to content image
    style = style.resize((w, h), PIL.Image.BICUBIC)

    # transform images and add batch dimension
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    content = transform(content).unsqueeze(0)
    style = transform(style).unsqueeze(0)

    return content, style


def save_image(tensor, filename):
    transform = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    image = tensor.clone()
    image = transform(image).clamp_(0, 1)
    
    torchvision.utils.save_image(image, filename)
