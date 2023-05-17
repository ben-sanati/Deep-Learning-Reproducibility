import math
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
from math import floor, log10, inf

from packages.models.CNN import *
from packages.models.NN import NeuralNetwork
from packages.models.CharRNN import CharRNN

from packages.utils.dataset_defs import *
from packages.hyperoptimizer.optimize import *
from packages.utils.train_val import Experimentation


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split('=')
            my_dict[k] = float(v)
        setattr(namespace, self.dest, my_dict)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # make the output deterministic
    seed = 7
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # optimizer definitions
    ada_opt = lambda : gdtuoAdaGrad
    adam_opt = lambda : gdtuoAdam
    adam_baydin = lambda : gdtuoAdamBaydin
    sgd_opt = lambda : gdtuoSGD
    rms_opt = lambda : gdtuoRMSProp
    noop = lambda : NoOpOptimizer

    # function mapping definitions 
    MODEL_MAP = {'NN': NeuralNetwork(784, 128, 10), 'stacked_NN': NeuralNetwork(784, 128, 10), 'CNN': ResNet(ResidualBlock, [3, 3, 3]), 'CharRNN': None}
    LOSS_MAP = {'CrossEntropyLoss': nn.CrossEntropyLoss()}
    OPT_MAP = {'AdaGrad': ada_opt, 'Adam': adam_opt, 'Adam_alpha': adam_baydin, 'SGD': sgd_opt, 'RMSProp': rms_opt, 'NoOp': noop}
    DATASET_MAP = {'MNIST': mnist, 'CIFAR': cifar, 'WarAndPeace': WarAndPeace}

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODEL_MAP.keys(), required=True)
    parser.add_argument('--optimizer', choices=OPT_MAP.keys(), required=True)
    parser.add_argument('--optimizer_args', nargs='*', action=StoreDictKeyPair)
    parser.add_argument('--hyperoptimizer', choices=OPT_MAP.keys(), required=True)
    parser.add_argument('--hyperoptimizer_args', nargs='*', action=StoreDictKeyPair)
    parser.add_argument('--num_hyperoptimizers', type=int, default=0)
    parser.add_argument('--loss_fn', choices=LOSS_MAP.keys(), required=True)
    parser.add_argument('--dataset', choices=DATASET_MAP.keys(), required=True)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--kappa', type=float, default=None)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--baseline', type=str, default='False')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # definitions
    device = torch.device(args.device)
    loss_function = LOSS_MAP[args.loss_fn] 
    optim_func = OPT_MAP[args.optimizer]()
    hyperoptim_func = OPT_MAP[args.hyperoptimizer]()

    if args.model == 'NN':
        # optimizer definition
        model = MODEL_MAP[args.model].to(device)
        if args.baseline == 'True':
            optimizer = optim_func(alpha=args.alpha, **args.optimizer_args)
        else:
            optimizer = optim_func(alpha=args.alpha, **args.optimizer_args, optimizer=hyperoptim_func(args.kappa, **args.hyperoptimizer_args))
        trainset, testset, trainloader, testloader = DATASET_MAP[args.dataset](args.batch_size)

        # logging
        print(f"Args:")
        for arg, value in vars(args).items():
            print(f"\t{arg}: {value}", flush=True)

        # perform experiments
        experiments = Experimentation(model, args.model, loss_function, optimizer, args.num_epochs, trainloader, testloader, args.batch_size, device)

        experiments.train()
        experiments.test()
        if args.baseline == 'True':
            experiments.plot(args.alpha, args.kappa, args.optimizer, args.optimizer_args, args.hyperoptimizer, args.hyperoptimizer_args, args.model, f"{args.optimizer}")
        else:
            experiments.plot(args.alpha, args.kappa, args.optimizer, args.optimizer_args, args.hyperoptimizer, args.hyperoptimizer_args, args.model, f"{args.optimizer}/{args.hyperoptimizer}")
    elif args.model == 'CNN':
        # optimizer definition
        model = MODEL_MAP[args.model].to(device)    
        kappa = (args.alpha ** 2) * 1e-3
        hyper_mu = args.optimizer_args['mu']
        hyper_mu = (1 / (1-hyper_mu)) * 1e-6

        hyperoptimizer_args = {'mu': hyper_mu}

        if args.baseline == 'True':
            optimizer = optim_func(alpha=args.alpha, **args.optimizer_args)
        else:
            optimizer = optim_func(alpha=args.alpha, **args.optimizer_args, optimizer=hyperoptim_func(args.kappa, **args.hyperoptimizer_args))

        trainset, testset, trainloader, testloader = DATASET_MAP[args.dataset](args.batch_size)

        # logging
        print(f"Args:")
        for arg, value in vars(args).items():
            print(f"\t{arg}: {value}", flush=True)

        # perform experiments
        experiments = Experimentation(model, args.model, loss_function, optimizer, args.num_epochs, trainloader, testloader, args.batch_size, device)

        experiments.train()
        experiments.test()
        if args.baseline == 'True':
            experiments.plot(args.alpha, args.kappa, args.optimizer, args.optimizer_args, args.hyperoptimizer, args.hyperoptimizer_args, args.model, f"baseline")
        else:
            experiments.plot(args.alpha, kappa, args.optimizer, args.optimizer_args, args.hyperoptimizer, hyperoptimizer_args, args.model, f"{args.alpha}/{args.optimizer_args['mu']}")
    elif args.model == 'CharRNN':
        # define model with tokens
        with open('/scratch/bes1g19/DeepLearning/CW/data/WarAndPeace/war_peace_plain.txt', 'r') as f:
            text = f.read()
        chars = tuple(set(text))
        model = CharRNN(chars, batch_size=args.batch_size, n_hidden=128, n_layers=2, drop_prob=0.5).to(device)

        # optimizer definition
        if args.baseline == 'True':
            optimizer = optim_func(alpha=args.alpha, **args.optimizer_args)
        else:
            optimizer = optim_func(alpha=args.alpha, **args.optimizer_args, optimizer=hyperoptim_func(args.kappa, **args.hyperoptimizer_args))
        
        trainset, testset, dataloader_iterators = DATASET_MAP[args.dataset](args.batch_size, num_steps=100, logging=False)
        recall_dataloaders = {'call': DATASET_MAP[args.dataset](args.batch_size, num_steps=100)}
        
        # logging
        print(f"Args:")
        for arg, value in vars(args).items():
            print(f"\t{arg}: {value}", flush=True)

        # perform experiments
        experiments = Experimentation(model, args.model, loss_function, optimizer, args.num_epochs, None, None, args.batch_size, device, num_steps=100, dataloader_iterators=dataloader_iterators, train_set=trainset, test_set=testset)

        experiments.train(recall_dataloaders)
        experiments.test()
        if args.baseline == 'True':
            experiments.plot(args.alpha, args.kappa, args.optimizer, args.optimizer_args, args.hyperoptimizer, args.hyperoptimizer_args, args.model, f"baseline")
        else:
            experiments.plot(args.alpha, args.kappa, args.optimizer, args.optimizer_args, args.hyperoptimizer, args.hyperoptimizer_args, args.model, f"{args.alpha}")
    elif args.model == 'stacked_NN':
        # optimizer definition
        model = MODEL_MAP[args.model].to(device)
        alpha_values = []

        if args.num_hyperoptimizers > 0:
            scaling = lambda layer : args.alpha / (100 * (1+layer))
            alpha_final = scaling(args.num_hyperoptimizers)

            alpha_values.insert(0, alpha_final)
        
            hyperoptimizers = hyperoptim_func(alpha_final, **args.hyperoptimizer_args)
            for layer in range(args.num_hyperoptimizers-1, 0, -1):
                hyperoptimizers = hyperoptim_func(scaling(layer), **args.hyperoptimizer_args, optimizer=hyperoptimizers)
                alpha_values.insert(0, scaling(layer))

            optimizer = optim_func(args.alpha, **args.optimizer_args, optimizer=hyperoptimizers)
            alpha_values.insert(0, args.alpha)
        else:
            optimizer = optim_func(args.alpha, **args.optimizer_args)

        trainset, testset, trainloader, testloader = DATASET_MAP[args.dataset](args.batch_size)

        # logging
        print(f"Args:")
        for arg, value in vars(args).items():
            print(f"\t{arg}: {value}", flush=True)

        alphas = {index: value for index, value in enumerate(alpha_values)}
        print(f"\tAlpha Values: {alphas}")

        # perform experiments
        experiments = Experimentation(model, args.model, loss_function, optimizer, args.num_epochs, trainloader, testloader, args.batch_size, device)

        experiments.train()
        experiments.test()
        experiments.plot(args.alpha, args.kappa, args.optimizer, args.optimizer_args, args.hyperoptimizer, args.hyperoptimizer_args, args.model, f"{args.alpha}/{args.num_hyperoptimizers}")
