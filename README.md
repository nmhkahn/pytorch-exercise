**Under construction**

# pytorch-exercise
This repository provides some exercise codes to learn PyTorch. Since this repo doesn't provide the basic tutorial, please see after reading [pytorch-exercise](https://github.com/yunjey/pytorch-tutorial) or [Official Tutorial](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html). NOTE: All the codes work on PyTorch 0.4.0.

## Contents
#### 1. Basics
- [MNIST Classification](codes/mnist)
- [Flower Classification (Custom Dataset)](codes/flower_cls)

#### 2. CNN Applications
- [Image Super-resolution](codes/super_resolution)
- [Image Colorization](codes/colorization)
- [Style Transfer](codes/style_transfer)
- [Conditional DCGAN](codes/cdcgan)

#### 3. RNN Applications
- [Char-RNN](codes/char_rnn)
- [Text Classification](codes/text_cls)
- [Image Captioning](codes/caption)

#### 4. Utilities
- [Visdom & torchsummary](codes/utilities)

## Installation
Make sure you have Python 3.5 or newer version. Installing the requirements are as follow:

```shell
pip install -r requirements.txt
```

#### (Optional)
For the RNN Applications codes, Some tokenizer pacakages such as [SpaCy](http://spacy.io/) or [NLTK](http://nltk.org/) are needed. You need to install these packages and its English model and data.

```shell
# install SpaCy
pip install spacy
python -m spacy download en

# install NLTK
pip install nltk
python -m nltk.downloader perluniprops nonbreaking_prefixes
```

[Visdom](https://github.com/facebookresearch/visdom) and [torchsummary](https://github.com/sksq96/pytorch-summary) are used in utilities exercise code. Please install these packages before run it.

```shell
pip install visdom torchsummary
```

## Getting Started
Simply run
```shell
python train.py
```
Input arguments are vary among the codes, so please check the `train.py` for more details.

## Known issues
1. `UnicodeDecodeError: 'ascii' codec can't decode byte 0xc3` when run the text classification.<br/>
: Please refer [this issue](https://github.com/pytorch/text/issues/77).

## Suggested Readings
- [pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) by yunjey
- [practical-torchtext](https://github.com/keitakurita/practical-torchtext) by keitakurita
