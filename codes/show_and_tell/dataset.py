import os
import csv
import numpy as np
from PIL import Image
import torch
import torchtext
import torchvision.transforms as transforms

class CaptionDataset(torch.utils.data.Dataset):
    # to preprocess and numerlicalize text
    TEXT = torchtext.data.Field(sequential=True, tokenize="spacy",
                                init_token="<start>", eos_token="<end>",
                                include_lengths=True,
                                batch_first=True)
    
    def __init__(self, train, **kwargs):
        super().__init__()

        self.image_size = kwargs.get("image_size", None)
        self.data_root = kwargs.get("data_root", "./data")
        
        phase = "train" if train else "test"
        
        self.path, self.text = list(), list()
        with open(os.path.join(self.data_root, "{}.csv".format(phase))) as f:
            reader = csv.reader(f)
            for line in reader:
                self.path.append(line[0])
                self.text.append(line[1])
        
        # preprocess (tokenize) text
        for i, t in enumerate(self.text):
            self.text[i] = CaptionDataset.TEXT.preprocess(t)
        
        # build vocab with GLOVE
        # NOTE: only performed in training phase
        if phase == "train":
            CaptionDataset.TEXT.build_vocab(self.text, vectors="glove.6B.300d")
        
        # image transform function
        self.transform = transforms.Compose([ 
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225))])
    
    def __getitem__(self, index):
        path, text = self.path[index], self.text[index]
        
        image = Image.open(os.path.join(self.data_root, path))
        # some grayscale images are in the dataset
        image = image.convert("RGB")

        if self.image_size:
            image = image.resize(
                (self.image_size, self.image_size), 
                Image.BICUBIC)
        
        return self.transform(image), text
    
    def __len__(self):
        return len(self.path)
    
    def indices_to_string(self, indices, words=False):
        """Convert word indices (torch.Tensor) to sentence (string).

        Args:
            indices: torch.tensor or numpy.array of shape (T) or (T, 1)
            words: boolean, wheter return list of words
        Returns:
            sentence: string type of converted sentence
            words: (optional) list[string] type of words list
        """
        sentence = list()    
        for idx in indices:
            word = CaptionDataset.TEXT.vocab.itos[idx.item()]

            if word in ["<pad>"]: continue

            # no needs of space between the special symbols
            if len(sentence) and word in ["'", ".", "?", "!", ","]:
                sentence[-1] += word
            else:
                sentence.append(word)
                
            if word in ["<end>"]: break

        if words:
            return " ".join(sentence), sentence
        return " ".join(sentence)
    
    
def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    Reference: https://github.com/yunjey/pytorch-tutorial

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, size, size).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (N, 3, image_size, image_size).
        captions: torch tensor of shape (N, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    
    # merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    
    # add padding to match max length and numericalize caption
    captions, lengths = CaptionDataset.TEXT.process(captions, 
        device=-1, train=True)
    
    return images, captions, lengths


def get_caption_dataset(train, 
                        data_root="./data", 
                        batch_size=32, image_size=224,
                        num_workers=1, shuffle=True,
                        text_field=False):
    dataset = CaptionDataset(train=train, 
        image_size=image_size, data_root=data_root)

    loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn)
    
    if text_field:
        return loader, dataset, CaptionDataset.TEXT
    return loader, dataset
