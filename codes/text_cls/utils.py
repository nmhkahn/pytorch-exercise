import torch
import torchtext.data as data
import torchtext.datasets as datasets

def _iters(batch_size, 
           fine_grained, 
           repeat):
    TEXT = data.Field(sequential=True, tokenize="spacy", lower=True)
    LABEL = data.Field(sequential=False)

    train, val, test = datasets.SST.splits(
        TEXT, LABEL,
        root="./data", fine_grained=fine_grained)

    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    return TEXT, LABEL, data.BucketIterator.splits(
        (train, val, test),
        root="./data",
        batch_size=batch_size, 
        repeat=repeat)


def load_sst(batch_size, fine_grained=False):
    TEXT, LABEL, (train_iter, val_iter, test_iter) = _iters(
        batch_size=batch_size,
        fine_grained=fine_grained,
        repeat=False)

    sst_info = {
        "vocab_size": len(TEXT.vocab),
        "num_class": 6 if fine_grained else 4
    }

    return train_iter, val_iter, test_iter, sst_info
