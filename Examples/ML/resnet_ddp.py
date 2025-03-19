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

HAS_IPEX = True
try:
    import intel_extension_for_pytorch as ipex
    import oneccl_bindings_for_pytorch
except ModuleNotFoundError:
    HAS_IPEX = False


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

def get_mpi_info():
    """
    Gets the MPI world size and local rank from environment variables,
    handling different MPI implementations.

    Returns:
        tuple: (world_size, local_rank, global_rank) or (1, 0, 0) if not running under MPI.
               Returns -1 for values that cannot be determined.
    """
    world_size = -1
    local_rank = -1
    global_rank = -1

    # --- Try OpenMPI variables ---
    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        global_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])

    # --- Try MPICH variables ---
    elif "PMI_SIZE" in os.environ:
        world_size = int(os.environ["PMI_SIZE"])
        global_rank = int(os.environ["PMI_RANK"])
        # MPICH doesn't have a standard local rank variable.  Try to infer it.
        if "MV2_COMM_WORLD_LOCAL_RANK" in os.environ:  # For MVAPICH2 (a variant of MPICH)
             local_rank = int(os.environ["MV2_COMM_WORLD_LOCAL_RANK"])
        elif "ALPS_APP_PE" in os.environ and 'SLURM_STEP_NUM_TASKS' in os.environ: # Try to infer using Slurm
            try:
                node_id = int(os.environ.get("SLURM_NODEID", 0))  # default node id is 0
                local_rank = global_rank - node_id * (int(os.environ['SLURM_STEP_NUM_TASKS']) // int(os.environ['SLURM_JOB_NUM_NODES']))
            except (ValueError, KeyError):
                pass
        else:
            local_rank = global_rank # In the worst case, assume local_rank = global_rank.

    # --- Try Intel MPI variables ---
    elif "I_MPI_HYDRA_HOST_NUM" in os.environ:  # This is a less reliable indicator of Intel MPI.
        world_size = int(os.environ["PMI_SIZE"]) if "PMI_SIZE" in os.environ else int(os.environ.get("I_MPI_HYDRA_HOST_NUM", -1)) # default to -1 if variable is missing.
        global_rank = int(os.environ["PMI_RANK"]) if "PMI_RANK" in os.environ else int(os.environ.get("I_MPI_RANK", -1))
        local_rank = int(os.environ.get("I_MPI_LOCALRANKID", -1))
        if "I_MPI_HYDRA_TOPOLIB" in os.environ and os.environ["I_MPI_HYDRA_TOPOLIB"] == "pmi":
            if "PMI_LOCAL_RANK" in os.environ: # Check another Intel MPI environment variable
                local_rank = int(os.environ["PMI_LOCAL_RANK"])

    # --- Try other common variables (less specific) ---
    elif "MPI_LOCALRANKID" in os.environ: # try generic local rank.
        local_rank = int(os.environ["MPI_LOCALRANKID"])
        if "MPI_UNIVERSE_SIZE" in os.environ:
            world_size = int(os.environ['MPI_UNIVERSE_SIZE'])
        if 'RANK' in os.environ:
            global_rank = int(os.environ['RANK'])

    elif 'SLURM_NTASKS' in os.environ: # Slurm
        world_size = int(os.environ['SLURM_NTASKS'])
        global_rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ.get('SLURM_LOCALID', -1))

    # --- If not running under MPI ---
    else:
        world_size = 1
        local_rank = 0
        global_rank = 0

    return world_size, local_rank, global_rank

def setup_ddp(device):
    if 'xpu' in device:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        init_method = 'tcp://127.0.0.1:29305'
        # init_method = 'file:///tmp/sync_file'
        # init_method = 'env://'
        local_rank = int(os.environ.get('PMI_RANK', 0))
        mpi_world_size = int(os.environ.get('PMI_SIZE', -1))
        print('world size', mpi_world_size)
        torch.distributed.init_process_group(backend="ccl",
                                             init_method=init_method,
                                             world_size=mpi_world_size,
                                             rank=local_rank)
    elif 'cuda' in device:
        torch.distributed.init_process_group(backend="nccl")
        mpi_world_size = torch.distributed.get_world_size()
        local_rank = torch.distributed.get_rank()

    return mpi_world_size, local_rank

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=100)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.", default=1024)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=1.e-2)
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=42)
    parser.add_argument("-d", "--device", type=str, help="Compute device", default='cuda')
    args = parser.parse_args()

    return args

def load_cifar10(data_root, batch_size=1024, test_batch_size=128, num_workers=16, distributed=False):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
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
    mpi_world_size, local_rank = setup_ddp(args.device)

    # set the random seed to make results reproducible
    set_random_seed(random_seed=args.random_seed)

    # Load the CIFAR10 dataset
    train_loader, test_loader, num_images = load_cifar10(data_root='./data', batch_size=args.batch_size, distributed=True if mpi_world_size > 1 else False)

    # Initialize the model
    model = torchvision.models.resnet50(pretrained=False)
    # model = ResNet50(num_classes=10)
    print(model)

    # Wrap the model with DDP
    # device = torch.device("{}:{}".format(args.device, local_rank))
    if HAS_IPEX and args.device == 'xpu':
        device = torch.device("xpu:{}".format(local_rank))
    elif args.device == 'cuda':
        device = torch.device("cuda:{}".format(local_rank))
        model = model.to(device)



    # Send the model to the device
    # device = torch.device(args.device)


    # Define training criterion (i.e., loss function) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)

    if HAS_IPEX and args.device == 'xpu':
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)

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
