from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
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