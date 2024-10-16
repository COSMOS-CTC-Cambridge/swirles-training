import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import ResNet, Bottleneck

import argparse
import os
import random
import numpy as np



def set_random_seed(random_seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        random_seed (int, optional): The random seed to set. Defaults to 42.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


class ResNet50(ResNet):
    """
    Initialize a ResNet50 object, based on the ResNet base class.
    Args:
        ResNet (_type_): _description_
    """
    def __init__(self, *args, **kwargs):
        super(ResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], *args, **kwargs)
    

def evaluate(model, device, test_loader):

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=100)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.", default=1024)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=1.e-2)
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=42)
    parser.add_argument("-d", "--device", type=str, help="Compute device", default='cpu')
    args = parser.parse_args()

    return args

def load_cifar10(data_root, batch_size=1024, test_batch_size=128, num_workers=8):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
    num_images = len(train_set)

    return train_loader, test_loader, num_images

def train(model, train_loader, test_loader, optimizer, criterion, device, num_images, epochs=20, evalulate_every_n_epochs=10):
    for epoch in range(epochs):

        if epoch % evalulate_every_n_epochs == 0:
            accuracy = evaluate(model=model, device=device, test_loader=test_loader)
            print("-" * 75)
            print("Epoch: {}, Accuracy: {}".format(epoch, accuracy))
            print("-" * 75)

        model.train()

        with tqdm(total=num_images) as pbar:
            for step, data in enumerate(train_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                pbar.update(len(data[0]))


def main():
    # Parse command line arguments 
    args = parse_arguments()

    # set the random seed to make results reproducible 
    set_random_seed(random_seed=args.random_seed)

    # Load the CIFAR10 dataset
    train_loader, test_loader, num_images = load_cifar10(data_root='../datasets', batch_size=args.batch_size)

    # Initialize the model
    model = ResNet50(num_classes=10)
    print(model)

    # Send the model to the device
    device = torch.device(args.device)
    model = model.to(device)

    # Define training criterion (i.e., loss function) and optimizer 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)


    train(model, 
          train_loader=train_loader,
          test_loader=test_loader,
          optimizer=optimizer,
          criterion=criterion,
          device=device,
          num_images=num_images,
          epochs=args.num_epochs,
          evalulate_every_n_epochs=10
          )

if __name__ == "__main__":

    main()
