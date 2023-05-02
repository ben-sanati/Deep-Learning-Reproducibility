import math
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt
from packages.hyperoptimizer.optimize import *
from packages.utils.plotting_utils import *

def train(model, loss_fn, optimizer, num_epochs, trainloader, device):
    warnings.filterwarnings("ignore")

    print(f"\n***Beginning Training***")

    # use gdtuo.ModuleWrapper to allow nn.Module be optimized by hyperoptimizers
    mw = ModuleWrapper(model, optimizer=optimizer)
    mw.initialize()

    # initialise logging for plots
    optimizer_plots_dict = {key: [value.item()] for key, value in optimizer.parameters.items()}
    Epochs = [0]

    # get the initial loss
    running_loss = 0.0
    for index, (features_, labels_) in enumerate(trainloader):
        features, labels = features_.to(device), labels_.to(device)

        # forward pass
        pred = mw.forward(features)
        loss = loss_fn(pred, labels)

        running_loss += loss.item() * features_.size(0)
    train_loss = running_loss / len(trainloader.dataset)
    optimizer_plots_dict['Loss'] = [train_loss]

    # training loop
    for epoch in range(1, num_epochs+1):
        running_loss = 0.0
        for index, (features_, labels_) in enumerate(trainloader):
            mw.begin() # call this before each step, enables gradient tracking on desired params
            features, labels = features_.to(device), labels_.to(device)

            # forward pass
            pred = mw.forward(features)
            loss = loss_fn(pred, labels)

            # backward pass
            mw.zero_grad()
            loss.backward(create_graph=True) # important! use create_graph=True
            mw.step()
            running_loss += loss.item() * features_.size(0)
        train_loss = running_loss / len(trainloader.dataset)
        print(f"\tEpoch[{epoch}/{num_epochs}]: Training Loss = {train_loss:.5f}", flush=True)

        for key, value in optimizer.parameters.items():
            optimizer_plots_dict[key].append(value.item())
        optimizer_plots_dict['Loss'].append(train_loss)
        Epochs.append(epoch)

    print(f"***Training Complete***\n")
    print(f"Final Optimizer Parameters")
    for key, value in optimizer.parameters.items():
        print(f"\t{key} : {value}")
    
    return mw, optimizer_plots_dict, Epochs

def test(model, testloader, device):
    print(f"\n***Testing Results***")
    print("="*30)
    model.eval()
    with torch.no_grad():
        total, correct = 0, 0
        for index, (features_, labels_) in enumerate(testloader):
            features, labels = features_.to(device), labels_.to(device)
            outputs = model.forward(features)
            
            # top-1 accuracy
            _, prediction = torch.max(outputs, dim=1)
            total += labels.shape[0]
            correct += (prediction == labels).sum().item()

        test_accuracy = (100.0 * correct) / total
    
    print(f"Test Accuracy = {test_accuracy:.3f} %", flush=True)
    print(f"Test Error = {100-test_accuracy:.3f} %", flush=True)
    print("="*30)

def plot(optimizer_plots_dict, Epochs, alpha, kappa, opt, opt_args, hyp, hyp_args, src, path):
    num_plots = len(optimizer_plots_dict)
    num_cols = 2
    num_rows = math.ceil(num_plots / num_cols)

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
    for index, key in enumerate(optimizer_plots_dict):
        ax = axes[index]
        ax.plot(Epochs, optimizer_plots_dict[key])
        ax.scatter(Epochs, optimizer_plots_dict[key])

        if key == 'Loss':
            # Calculate the interquartile range (IQR)
            q1, q3 = np.percentile(optimizer_plots_dict[key], [25, 75])
            iqr = q3 - q1

            # Set the y-axis limits to be 1.5 times the IQR below and above the first and third quartiles
            if iqr > 0:
                lower_limit = q1 - 1.5 * iqr
                upper_limit = q3 + 1.5 * iqr
                ax.set_ylim(max(0, min(round(min(optimizer_plots_dict[key]), 4), lower_limit)), upper_limit)

        ax.set_xlabel("Epochs")
        ax.set_ylabel(f"{key}")
        ax.set_title(f"{key} Optimization Plot")

    plt.suptitle(f"Plots Showing the Optimization of Loss and Hyperparameters against Epochs\nOptimizer [{opt} alpha={alpha} {opt_args}] : Hyperoptimizer [{hyp} kappa={kappa} {hyp_args}]", fontsize=16)
    plt.tight_layout()
    plt.savefig(f'../plots/{src}/{path}.png')
