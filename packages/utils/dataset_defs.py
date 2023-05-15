import numpy as np
from itertools import tee

import torch
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

def mnist(batch_size):
    # flatten 28*28 images to a 784 vector for each image
    transform = transforms.Compose([
        transforms.ToTensor(),  # convert to tensor
        transforms.Lambda(lambda x: x.view(-1))  # flatten into vector
    ])

    trainset = MNIST("/scratch/bes1g19/DeepLearning/CW/data", train=True, download=False, transform=transform)
    testset = MNIST("/scratch/bes1g19/DeepLearning/CW/data", train=False, download=False, transform=transform)

    # create data loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    return trainset, testset, trainloader, testloader

def cifar(batch_size):
    # apply transforms
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])

    trainset = CIFAR10("/ECShome/ECSdata/data_sets", train=True, download=False, transform=transform)
    testset = CIFAR10("/ECShome/ECSdata/data_sets", train=False, download=False, transform=transform)

    # create data loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    return trainset, testset, trainloader, testloader

class WarAndPeaceDataset(Dataset):
    def __init__(self, num_seqs, num_steps):
        # open text file and read in data as `text`
        with open('/scratch/bes1g19/DeepLearning/CW/data/WarAndPeace/war_peace_plain.txt', 'r') as f:
            self.text = f.read()

        # get the size of the dataset
        self.num_chars = len(self.text)
        self.chars = sorted(list(set(self.text)))
        self.num_vocab = len(self.chars)
        
        # get the padded batch size
        self.num_seqs, self.num_steps = num_seqs, num_steps
        self.batch_size = num_seqs * num_steps
        
        # create char2int and int2char dictionaries
        self.chars = tuple(set(self.text))
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        # define encoded dataset
        self.encoded = np.array([self.char2int[ch] for ch in self.text])

    def get_batches(self, dataset):
        # pad batches
        num_batches = len(dataset) // self.batch_size
        data = dataset[:num_batches * self.batch_size].reshape((self.num_seqs, -1))
        
        for n in range(0, data.shape[1], self.num_steps):
            # The features
            x = data[:, n:n+self.num_steps]
            
            # The targets, shifted by one
            y = np.zeros_like(x)
            
            try:
                y[:, :-1], y[:, -1] = x[:, 1:], data[:, n+self.num_steps]
            except IndexError:
                y[:, :-1], y[:, -1] = x[:, 1:], data[:, 0]
            yield x, y

    def __len__(self):
        return self.num_chars

    def __getitem__(self, i):
        return self.encoded[i]
    
def WarAndPeace(batch_size, num_steps, logging=False):
    # dataset init
    dataset = WarAndPeaceDataset(num_seqs=batch_size, num_steps=num_steps)
    trainset, testset = random_split(dataset, [len(dataset) - len(dataset)//3, len(dataset)//3])

    # dataloader init
    train_loader = dataset.get_batches(trainset)

    dataloader_iterators = {'get_batches': dataset.get_batches}

    if logging:
        x, y = next(iter(train_loader))

        x_decoded = [[dataset.int2char[x[index, pos]] for pos in range(len(x[index, :10]))] for index in range(len(x[:10, :10]))]
        y_decoded = [[dataset.int2char[y[index, pos]] for pos in range(len(y[index, :10]))] for index in range(len(y[:10, :10]))]

        print('x\n', x_decoded)
        print('\ny\n', y_decoded)

    return trainset, testset, dataloader_iterators

