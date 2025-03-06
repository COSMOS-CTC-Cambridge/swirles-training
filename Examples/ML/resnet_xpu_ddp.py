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

import intel_extension_for_pytorch as ipex 
import oneccl_bindings_for_pytorch


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

def setup_ddp():
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    init_method = 'tcp://127.0.0.1:29305'
    # init_method = 'file:///home/hpccai1/pytorch_tests/sync_file'
    # init_method = 'file:///tmp/sync_file'
    # init_method = 'env://'
    local_rank = int(os.environ.get('PMI_RANK', 0))
    mpi_world_size = int(os.environ.get('PMI_SIZE', -1))
    print('world size', mpi_world_size)
    torch.distributed.init_process_group(backend="ccl", init_method=init_method, world_size=mpi_world_size, rank=local_rank)
    return mpi_world_size, local_rank 

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=100)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.", default=1024)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=1.e-2)
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=42)
    parser.add_argument("-d", "--device", type=str, help="Compute device", default='xpu')
    args = parser.parse_args()

    return args

def load_cifar10(data_root, batch_size=1024, test_batch_size=128, num_workers=8, distributed=False):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_set = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform)

    if not distributed:
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_set)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)

    test_loader = DataLoader(dataset=test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
    num_images = len(train_set)

    return train_loader, test_loader, num_images

def train(model, train_loader, test_loader, optimizer, criterion, device, num_images, mpi_world_size, local_rank, epochs=20, evalulate_every_n_epochs=10):
    for epoch in range(epochs):

        if epoch % evalulate_every_n_epochs == 0:
            accuracy = evaluate(model=model, device=device, test_loader=test_loader)
            print("-" * 75)
            print("Epoch: {}, Accuracy: {}".format(epoch, accuracy))
            print("-" * 75)

        model.train()

        with tqdm(total=num_images, disable=local_rank>0) as pbar:
            for step, data in enumerate(train_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                pbar.update(len(data[0]) * mpi_world_size)


def main():
    # Parse command line arguments 
    args = parse_arguments()

    # Initialize DDP environment
    mpi_world_size, local_rank = setup_ddp()

    # set the random seed to make results reproducible 
    set_random_seed(random_seed=args.random_seed)

    # Load the CIFAR10 dataset
    train_loader, test_loader, num_images = load_cifar10(data_root='../datasets', batch_size=args.batch_size, distributed=True if mpi_world_size > 1 else False)

    # Initialize the model
    model = ResNet50(num_classes=10)
    print(model)

    # Wrap the model with DDP
    # device = torch.device("{}:{}".format(args.device, local_rank))
    device = torch.device("xpu:{}".format(local_rank))
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)



    # Send the model to the device
    # device = torch.device(args.device)
    

    # Define training criterion (i.e., loss function) and optimizer 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)

    model, optimizer = ipex.optimize(model, optimizer=optimizer)

    train(model, 
          train_loader=train_loader,
          test_loader=test_loader,
          optimizer=optimizer,
          criterion=criterion,
          device=device,
          num_images=num_images,
          mpi_world_size=mpi_world_size,
          local_rank=local_rank,
          epochs=args.num_epochs,
          evalulate_every_n_epochs=10
          )

if __name__ == "__main__":

    main()
