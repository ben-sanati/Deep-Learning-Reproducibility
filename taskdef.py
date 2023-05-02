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

from packages.utils.dataset_defs import *
from packages.utils.train_val import *
from packages.models.NN import NeuralNetwork
from packages.hyperoptimizer.optimize import *

class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split('=')
            my_dict[k] = float(v)
        setattr(namespace, self.dest, my_dict)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # optimizer definitions
    ada_opt = lambda : gdtuoAdaGrad
    adam_opt = lambda : gdtuoAdam
    sgd_opt = lambda : gdtuoSGD
    rms_opt = lambda : gdtuoRMSProp

    # function mapping definitions 
    MODEL_MAP = {'NN': NeuralNetwork(784, 128, 10)}
    LOSS_MAP = {'CrossEntropyLoss': nn.CrossEntropyLoss()}
    OPT_MAP = {'AdaGrad': ada_opt, 'Adam': adam_opt, 'SGD': sgd_opt, 'RMSProp': rms_opt}
    DATASET_MAP = {'MNIST': mnist}

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODEL_MAP.keys(), required=True)
    parser.add_argument('--optimizer', choices=OPT_MAP.keys(), required=True)
    parser.add_argument('--optimizer_args', nargs='*', action=StoreDictKeyPair)
    parser.add_argument('--hyperoptimizer', choices=OPT_MAP.keys(), required=True)
    parser.add_argument('--hyperoptimizer_args', nargs='*', action=StoreDictKeyPair)
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
    optimizer = optim_func(alpha=args.alpha, **args.optimizer_args, optimizer=hyperoptim_func(args.kappa, **args.hyperoptimizer_args))
    trainset, testset, trainloader, testloader = DATASET_MAP[args.dataset](args.batch_size)

    print(f"Args:")
    for arg, value in vars(args).items():
        print(f"\t{arg}: {value}", flush=True)

    # train 
    trained_model, optimizer_plots_dict, Epochs = train(model, loss_function, optimizer, args.num_epochs, trainloader, device)

    # test
    test(trained_model, testloader, device)

    # plot
    plot(optimizer_plots_dict, Epochs, args.alpha, args.kappa, args.optimizer, args.optimizer_args, args.hyperoptimizer, args.hyperoptimizer_args, args.model, f"{args.optimizer}/{args.hyperoptimizer}")
