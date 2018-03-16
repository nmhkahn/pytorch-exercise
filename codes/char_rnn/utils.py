import torch
import torchtext.data as data
import torchtext.datasets as datasets

class Shakespeare(datasets.LanguageModelingDataset):
    urls = ["https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"]
    name = "tinyshakespeare"
    dirname = "./"

    @classmethod
    def splits(cls, 
               text_field, 
               root="./data", train="input.txt",
               **kwargs):
        return super(Shakespeare, cls).splits(
            root=root, train=train,
            text_field=text_field, **kwargs)

    @classmethod
    def iters(cls, 
              batch_size=32, bptt_len=35, 
              root="./data", 
              repeat=False,
              **kwargs):
        TEXT = data.Field(sequential=True, tokenize=list)

        train, = cls.splits(TEXT, root=root, **kwargs)
        TEXT.build_vocab(train)

        return TEXT, data.BPTTIterator.splits(
            (train, ), batch_size=batch_size, bptt_len=bptt_len, repeat=repeat)


def load_shakespeare(batch_size, bptt_len):
    TEXT, (train_iter,) = Shakespeare.iters(
        batch_size=batch_size,
        bptt_len=bptt_len,
        repeat=False)

    data_info = {
        "vocab_size": len(TEXT.vocab),
        "TEXT": TEXT
    }

    return train_iter, data_info
