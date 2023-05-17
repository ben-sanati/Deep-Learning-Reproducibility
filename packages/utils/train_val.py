import gc
import math
import torch
import warnings
import operator
import numpy as np
from itertools import tee
from functools import reduce
import matplotlib.pyplot as plt
from packages.hyperoptimizer.optimize import *
from packages.utils.plotting_utils import *

from torch.nn.parallel import DataParallel
from torch.cuda.amp import autocast, GradScaler


class Experimentation:
    def __init__(self, model, model_name, loss_fn, optimizer, num_epochs, trainloader, testloader, batch_size, device, num_steps=None, dataloader_iterators=None, train_set=None, test_set=None):
        self.model = model
        self.model_name = model_name
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.trainloader = trainloader
        self.testloader = testloader 
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.device = device 
        self.dataloader_iterators = dataloader_iterators
        self.trainset = train_set
        self.testset = test_set
        
        # initialise logging for plots
        self.optimizer_plots_dict = {key: [value.item()] for key, value in optimizer.parameters.items()}
        self.Epochs = [0]

        if torch.cuda.device_count() > 1:
            print("\tUsing", torch.cuda.device_count(), "GPUs")
            self.model = DataParallel(self.model)
        self.model.to(self.device)

    def train(self, recall=None):
        warnings.filterwarnings("ignore")

        print(f"\n***Beginning Training***")

        # use gdtuo.ModuleWrapper to allow nn.Module be optimized by hyperoptimizers
        mw = ModuleWrapper(self.model, optimizer=self.optimizer)
        mw.initialize()

        # get the initial loss
        trainset_length = len(self.trainloader.dataset) if self.model_name != 'CharRNN' else None
        running_loss = 0.0
        if self.model_name == 'CharRNN':
            num_layers = 2
            hidden_size = 128
            h = torch.zeros(num_layers, int(self.batch_size / torch.cuda.device_count()), hidden_size)
            c = torch.zeros(num_layers, int(self.batch_size / torch.cuda.device_count()), hidden_size)
            h = (h, c)
            self.trainloader = self.dataloader_iterators['get_batches'](self.trainset)

        n = 0
        for index, (features_, labels_) in enumerate(self.trainloader):
            # forward pass
            if self.model_name == 'CharRNN':
                with autocast():
                    features_ = self.one_hot_encode(features_, len(self.model.module.chars))
                    features_, labels_ = torch.from_numpy(features_), torch.from_numpy(labels_)
                    features, labels = features_.to(self.device), labels_.to(self.device)
                    h = tuple([each.data for each in h])
                    pred, h = mw.forward(features)
                    labels = labels.view(self.batch_size*self.num_steps).type(torch.LongTensor).to(self.device)
            else:
                features, labels = features_.to(self.device), labels_.to(self.device)
                pred = mw.forward(features)

            loss = self.loss_fn(pred, labels)
            if torch.cuda.device_count() > 1:
                loss = loss.mean() # take mean of loss across multiple GPUs

            running_loss += loss.item()
            n += 1

        if self.model_name != 'CharRNN':
            train_loss = running_loss / trainset_length
        else:
            train_loss = running_loss / n
        
        self.optimizer_plots_dict['Loss'] = [train_loss]
        print(f"\tInitial Train Loss: {train_loss}", flush=True)

        # clear memory
        if self.model_name == 'CharRNN':
            del train_loss
            del running_loss
            del n
            del features_
            del features
            del labels_
            del labels
            del h
            del pred
            torch.cuda.empty_cache()
            gc.collect()
            
        for epoch in range(1, self.num_epochs+1):
            # training loop
            if self.model_name == 'CharRNN':
                num_layers = 2
                hidden_size = 128
                h = torch.zeros(num_layers, int(self.batch_size / torch.cuda.device_count()), hidden_size)
                c = torch.zeros(num_layers, int(self.batch_size / torch.cuda.device_count()), hidden_size)
                h = (h, c)
                self.trainloader = self.dataloader_iterators['get_batches'](self.trainset)
            
            running_loss = 0.0
            n = 0
            for index, (features_, labels_) in enumerate(self.trainloader):
                mw.begin() # call this before each step, enables gradient tracking on desired params
                
                # forward pass
                if self.model_name == 'CharRNN':
                    with autocast():
                        features_ = self.one_hot_encode(features_, len(self.model.module.chars))
                        features_, labels_ = torch.from_numpy(features_), torch.from_numpy(labels_)
                        features, labels = features_.to(self.device), labels_.to(self.device)
                        h = tuple([each.data for each in h])
                        pred, h = mw.forward(features)
                        labels = labels.view(self.batch_size*self.num_steps).type(torch.LongTensor).to(self.device)
                else:
                    features, labels = features_.to(self.device), labels_.to(self.device)
                    pred = mw.forward(features)

                loss = self.loss_fn(pred, labels)
                if torch.cuda.device_count() > 1:
                    loss = loss.mean() # take mean of loss across multiple GPUs

                # backward pass
                mw.zero_grad()
                loss.backward(create_graph=True) # important! use create_graph=True
                mw.step()

                running_loss += loss.item() * features_.size(0) if self.model_name != 'CharRNN' else loss.item()
                n += 1

                # clear memory
                if self.model_name == 'CharRNN':
                    del features_
                    del features
                    del labels_
                    del labels
                    del pred
                    torch.cuda.empty_cache()
                    gc.collect()

            if self.model_name != 'CharRNN':
                train_loss = running_loss / trainset_length
            else:
                train_loss = running_loss / n


            print(f"\tEpoch[{epoch}/{self.num_epochs}]: Training Loss = {train_loss:.5f}", flush=True)

            for key, value in self.optimizer.parameters.items():
                self.optimizer_plots_dict[key].append(value.item())
            self.optimizer_plots_dict['Loss'].append(train_loss)
            self.Epochs.append(epoch)

            # clear memory
            if self.model_name == 'CharRNN':
                del train_loss
                del running_loss
                del n
                del h
                del self.trainloader
                torch.cuda.empty_cache()
                gc.collect()

        print(f"***Training Complete***\n")
        print(f"Final Optimizer Parameters")
        for key, value in self.optimizer.parameters.items():
            print(f"\t{key} : {value}")

    def test(self):
        print(f"\n***Testing Results***")
        print("="*30)
        self.model.eval()
        with torch.no_grad():
            total, correct = 0, 0
            # validation 
            if self.model_name == 'CharRNN':
                num_layers = 2
                hidden_size = 128
                h = torch.zeros(num_layers, int(self.batch_size / torch.cuda.device_count()), hidden_size)
                c = torch.zeros(num_layers, int(self.batch_size / torch.cuda.device_count()), hidden_size)
                h = (h, c)
                self.testloader = self.dataloader_iterators['get_batches'](self.testset)

            for index, (features_, labels_) in enumerate(self.testloader):
                # forward pass
                if self.model_name == 'CharRNN':
                    with autocast():
                        features_ = self.one_hot_encode(features_, len(self.model.module.chars))
                        features_, labels_ = torch.from_numpy(features_), torch.from_numpy(labels_)
                        features, labels = features_.to(self.device), labels_.to(self.device)
                        h = tuple([each.data for each in h])
                        outputs, h = self.model.forward(features)
                        labels = labels.view(self.batch_size*self.num_steps).type(torch.LongTensor).to(self.device)
                else:
                    features, labels = features_.to(self.device), labels_.to(self.device)
                    outputs = self.model.forward(features)
                
                # top-1.0.0.0.0.0 accuracy
                _, prediction = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += (prediction == labels).sum().item()

                # clear memory
                if self.model_name == 'CharRNN':
                    del features_
                    del features
                    del labels_
                    del labels
                    del outputs
                    torch.cuda.empty_cache()
                    gc.collect()

            test_accuracy = (100.0 * correct) / total

            # clear memory
            if self.model_name == 'CharRNN':
                del h
                del self.testloader
                torch.cuda.empty_cache()
                gc.collect()
        
        print(f"Test Accuracy = {test_accuracy:.3f} %", flush=True)
        print(f"Test Error = {100-test_accuracy:.3f} %", flush=True)
        print("="*30)

    def plot(self, alpha, kappa, opt, opt_args, hyp, hyp_args, src, path):
        num_plots = len(self.optimizer_plots_dict)
        num_cols = 2
        num_rows = math.ceil(num_plots / num_cols)

        print("\nPlotted Lists:")

        # Set the size of each subplot
        width_per_subplot = 7.5
        height_per_subplot = 6

        # Calculate the figure size
        fig_width = num_cols * width_per_subplot
        fig_height = num_rows * height_per_subplot

        # Create the figure and axes with the calculated number of rows and columns
        mpl_style(dark=True, minor_ticks=False)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))

        # Flatten the 2D axes array into a 1D array
        axes = axes.flatten()

        # Plot the data
        print(f"Epochs: {self.Epochs}")
        for index, key in enumerate(self.optimizer_plots_dict):
            ax = axes[index]
            ax.plot(self.Epochs, self.optimizer_plots_dict[key])
            ax.scatter(self.Epochs, self.optimizer_plots_dict[key])
            print(f"{key}: {self.optimizer_plots_dict[key]}")

            if key == 'Loss':
                # Calculate the interquartile range (IQR)
                q1, q3 = np.percentile(self.optimizer_plots_dict[key], [25, 75])
                iqr = q3 - q1

                # Set the y-axis limits to be 1.0.0.0.0.0.5 times the IQR below and above the first and third quartiles
                if iqr > 0:
                    lower_limit = q1 - 1.5 * iqr
                    upper_limit = q3 + 1.5 * iqr
                    ax.set_ylim(max(0, min(round(min(self.optimizer_plots_dict[key]), 4), lower_limit)), upper_limit)

            ax.set_xlabel("Epochs")
            ax.set_ylabel(f"{key}")
            ax.set_title(f"{key} Optimization Plot")

        plt.suptitle(f"Plots Showing the Optimization of Loss and Hyperparameters against Epochs\nOptimizer [{opt} alpha={alpha} {opt_args}] : Hyperoptimizer [{hyp} kappa={kappa} {hyp_args}]", fontsize=16)
        plt.tight_layout()
        plt.savefig(f'../plots/{src}/{path}.png')

    @staticmethod
    def one_hot_encode(arr, n_labels):
        # Initialize the the encoded array
        one_hot = np.zeros((reduce(operator.mul, arr.shape), n_labels), dtype=np.float32)
        
        # Fill the appropriate elements with ones
        one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
        
        # Finally reshape it to get back to the original array
        one_hot = one_hot.reshape((*arr.shape, n_labels))
        
        return one_hot
