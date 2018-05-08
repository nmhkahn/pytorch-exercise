import os
import csv
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
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
        self.max_vocab = kwargs.get("max_vocab", "10000")

        if self.max_vocab == -1:
            self.max_vocab = None
        
        self.phase = "train" if train else "val"
        json_path = "{}/annotations/captions_{}2014.json".format(self.data_root, self.phase)
        
        # load data with COCO API
        print("[!] Prepare COCO {} dataset".format(self.phase))

        coco = COCO(json_path)
        self.path, self.text = list(), list()
        for key, value in coco.anns.items():
            image_idx = value["image_id"]
            path = coco.loadImgs(image_idx)[0]["file_name"]
            
            self.text.append(value["caption"])
            self.path.append(path)

        # preprocess (tokenize) text
        for i, t in enumerate(self.text):
            self.text[i] = CaptionDataset.TEXT.preprocess(t)
        
        # build vocab with GLOVE
        # NOTE: only performed in training phase
        if self.phase == "train":
            CaptionDataset.TEXT.build_vocab(
                self.text, 
                vectors="glove.6B.300d", 
                max_size=self.max_vocab)
        
        print("[!] Dataset preparation done!")
        print("\t# of data: {}".format(len(self.text)))
        print("\tVocab size: {}\n".format(len(CaptionDataset.TEXT.vocab)))
        
        # image transform function
        self.transform = transforms.Compose([ 
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225))])
    
    def __getitem__(self, index):
        path, text = self.path[index], self.text[index]
        path = os.path.join(self.data_root, "{}2014".format(self.phase), path)
        
        # some grayscale images are in the dataset
        image = Image.open(path).convert("RGB")

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
            if word in ["<end>"]: break

            # no needs of space between the special symbols
            if len(sentence) and word in ["'", ".", "?", "!", ","]:
                sentence[-1] += word
            else:
                sentence.append(word)

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
                        max_vocab=10000,
                        batch_size=32, image_size=224,
                        num_workers=4, shuffle=True,
                        text_field=False):
    dataset = CaptionDataset(train=train,
        max_vocab=max_vocab,
        image_size=image_size, 
        data_root=data_root)

    loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn)
    
    if text_field:
        return loader, dataset, CaptionDataset.TEXT
    return loader, dataset
