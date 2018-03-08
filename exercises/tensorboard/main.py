import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from net import Net

def eval(net, loader):
    net.eval()
    num_correct, num_total = 0, 0
    for inputs in loader:
        images  = Variable(inputs[0]).cuda()
        labels  = inputs[1].cuda()

        outputs = net(images)
        _, preds = torch.max(outputs.data, 1)

        num_correct += (preds == labels).sum()
        num_total += labels.size(0)

    return num_correct / num_total


def train(args):
    net = Net().cuda()
    writer = SummaryWriter()
    
    # MNIST dataset
    train_dataset = datasets.MNIST(root="./data/",
                                   train=True, 
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.MNIST(root="./data/",
                                  train=False, 
                                  transform=transforms.ToTensor())

    # data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size, 
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size, 
                                              shuffle=False)
    
    # create loss operation and optimizer
    loss_op = nn.CrossEntropyLoss()
    optim   = torch.optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(args.max_epochs):
        net.train()
        for step, inputs in enumerate(train_loader):
            images = Variable(inputs[0]).cuda()
            labels = Variable(inputs[1]).cuda()

            optim.zero_grad()
            outputs = net(images)
            loss = loss_op(outputs, labels)
            loss.backward()
            optim.step()

        acc = eval(net, test_loader)
        writer.add_scalar("acc", acc, epoch+1)
        print("Epoch [{}/{}] loss: {:.5f} test acc: {:.3f}"
              .format(epoch+1, args.max_epochs, loss.data[0], acc))

    torch.save(net.state_dict(), "mnist-final.pth")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
