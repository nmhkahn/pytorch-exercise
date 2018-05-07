import os
import torch
from nltk.translate.bleu_score import sentence_bleu
from net import Net
from dataset import get_caption_dataset
from torch.nn.utils.rnn import pack_padded_sequence
from visdomX import VisdomX

class Solver():
    def __init__(self, args):
        self.train_loader, self.train_data, TEXT = get_caption_dataset(
            train=True,
            data_root=args.data_root,
            batch_size=args.batch_size, image_size=args.image_size, 
            text_field=True)
        self.test_loader, _ = get_caption_dataset(
            train=False,
            data_root=args.data_root,
            batch_size=args.batch_size, image_size=args.image_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = Net(TEXT, args.hidden_dim, args.num_layers).to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=1) # <pad>: 1
        self.optim   = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            args.lr)

        self.vis = VisdomX()

        self.args = args

        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)

    def fit(self):
        args = self.args

        for epoch in range(args.max_epochs):
            self.net.train()
            for step, inputs in enumerate(self.train_loader):
                image   = inputs[0].to(self.device)
                caption = inputs[1].to(self.device)
                lengths = inputs[2].to(self.device)

                out = self.net(image, caption, lengths)
                loss = self.loss_fn(out, caption.view(-1))

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            if (epoch+1) % args.print_every == 0:
                perplexity = torch.exp(loss).item()
                self.vis.add_scalars(perplexity, epoch,
                                     title="Perplexity",
                                     ylabel="Perplexity", xlabel="Epoch")

                print("Epoch [{}/{}] Perplexity: {:5.3f}"
                    .format(epoch+1, args.max_epochs, perplexity))

                self.save(args.ckpt_dir, args.ckpt_name, epoch+1)

    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)
