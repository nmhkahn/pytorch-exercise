import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from net import Net

def evaluate(net, loader, device):
    net.eval()
    num_correct, num_total = 0, 0

    # same as volatile=True of the v0.3
    with torch.no_grad():
        for inputs in loader:
            images = inputs[0].to(device)
            labels = inputs[1].to(device)

            outputs = net(images)
            _, preds = torch.max(outputs.detach(), 1)

            num_correct += (preds == labels).sum().item()
            num_total += labels.size(0)

    return num_correct / num_total


def train(args):
    # prepare the MNIST dataset
    train_dataset = datasets.MNIST(root="./data/",
                                   train=True, 
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.MNIST(root="./data/",
                                  train=False, 
                                  transform=transforms.ToTensor())

    # create the data loader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size, 
                              shuffle=True, drop_last=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size, 
                             shuffle=False)

    
    # turn on the CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net = Net().to(device)
    loss_op = nn.CrossEntropyLoss()
    optim   = torch.optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(args.max_epochs):
        net.train()
        for step, inputs in enumerate(train_loader):
            images = inputs[0].to(device)
            labels = inputs[1].to(device)
            
            # forward-propagation
            outputs = net(images)
            loss = loss_op(outputs, labels)
            
            # back-propagation
            optim.zero_grad()
            loss.backward()
            optim.step()

        acc = evaluate(net, test_loader, device)
        print("Epoch [{}/{}] loss: {:.5f} test acc: {:.3f}"
              .format(epoch+1, args.max_epochs, loss.item(), acc))

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
