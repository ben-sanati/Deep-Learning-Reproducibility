import torch
import subprocess
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from gradient_descent_the_ultimate_optimizer import gdtuo

from packages.utils.dataset_defs import *
from packages.utils.train_val import train
from packages.models.NN import NeuralNetwork


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # optimizer definitions
    adam_opt = lambda : gdtuo.Adam
    sgd_opt = lambda : gdtuo.SGD

    # function mapping definitions 
    MODEL_MAP = {'NN': NeuralNetwork(784, 128, 10)}
    LOSS_MAP = {'CrossEntropyLoss': nn.CrossEntropyLoss()}
    OPT_MAP = {'Adam': adam_opt, 'SGD': sgd_opt}
    DATASET_MAP = {'MNIST': mnist}

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODEL_MAP.keys(), required=True)
    parser.add_argument('--optimizer', choices=OPT_MAP.keys(), required=True)
    parser.add_argument('--hyperoptimizer', choices=OPT_MAP.keys(), required=True)
    parser.add_argument('--loss_fn', choices=LOSS_MAP.keys(), required=True)
    parser.add_argument('--dataset', choices=DATASET_MAP.keys(), required=True)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--kappa', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # definitions
    device = torch.device(args.device)
    model = MODEL_MAP[args.model].to(device)
    loss_function = LOSS_MAP[args.loss_fn]
    optim_func = OPT_MAP[args.optimizer]()
    hyperoptim_func = OPT_MAP[args.hyperoptimizer]()
    optimizer = optim_func(alpha=args.alpha, beta1=0.9, beta2=0.99, optimizer=hyperoptim_func(args.kappa))
    trainset, testset, trainloader, testloader = DATASET_MAP[args.dataset](args.batch_size)

    print(f"Args:")
    for arg, value in vars(args).items():
        print(f"\t{arg}: {value}", flush=True)

    # train 
    train(model, loss_function, optimizer, args.num_epochs, trainloader, device)
