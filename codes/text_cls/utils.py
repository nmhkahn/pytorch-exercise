import torch
import torchtext.data as data
import torchtext.datasets as datasets

def _iters(batch_size,
           max_vocab,
           fine_grained, 
           repeat):
    TEXT = data.Field(sequential=True, tokenize="spacy", lower=True)
    LABEL = data.Field(sequential=False)

    train, val, test = datasets.SST.splits(
        TEXT, LABEL,
        root="./data", fine_grained=fine_grained)

    if max_vocab == -1:
        max_vocab = None

    TEXT.build_vocab(train, vectors="glove.6B.300d", max_size=max_vocab)
    LABEL.build_vocab(train, max_size=max_vocab)

    return TEXT, LABEL, data.BucketIterator.splits(
        (train, val, test),
        batch_size=batch_size, 
        repeat=repeat)


def load_sst(batch_size, max_vocab, fine_grained=True):
    TEXT, LABEL, (train_iter, val_iter, test_iter) = _iters(
        batch_size=batch_size,
        max_vocab=max_vocab,
        fine_grained=fine_grained,
        repeat=False)

    sst_info = {
        "vocab_size": len(TEXT.vocab),
        "num_class": 6 if fine_grained else 4,
        "TEXT": TEXT
    }

    return train_iter, val_iter, test_iter, sst_info
