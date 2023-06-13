import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from packages.utils.plotting_utils import *


def get_RNN_data(model):
    root_dir = f'./MAKE/OUT/{model}'
    Optimizer, Hyperoptimizer, Training_loss, Loss, TestAccuracy, TestError, Final_params_dict = [], [], [], [], [], [], []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.out'):
                with open(os.path.join(subdir, file), 'r') as f:
                    text = f.read()
                    final_params = re.findall(r'Final Optimizer Parameters(.*?)\n\n', text, re.DOTALL)
                    accuracy = re.search(r'Test Accuracy = (.*?) %', text).group(1)
                    error = re.search(r'Test Error = (.*?) %', text).group(1)
                    optimizer = re.search(r'optimizer: (\w+)', text).group(1)
                    hyperoptimizer = re.search(r'hyperoptimizer: (\w+)', text).group(1)

                final_params_dict = {}
                if file != 'baseline.out':
                    final_params_dict['alpha'] = re.search(r"(\d+\.\d+)", file).group(1)
                else:
                    final_params_dict['alpha'] = 0.002
                    final_params_dict['beta1'] = 0.9
                    final_params_dict['beta2'] = 0.99

                train_loss = []
                with open(os.path.join(subdir, file), 'r') as f:
                    for line in f:
                        match = re.search(r'Training Loss = ([\d.]+)', line)
                        if match:
                            training_loss_value = float(match.group(1))
                            train_loss.append(training_loss_value)
                        match1 = re.search(r'Initial Train Loss: ([\d.]+)', line)
                        if match1:
                            training_loss_value = float(match1.group(1))
                            train_loss.append(training_loss_value)

                Optimizer.append(optimizer)
                Hyperoptimizer.append(hyperoptimizer)
                Training_loss.append(train_loss)
                Final_params_dict.append(final_params_dict)
                TestAccuracy.append(accuracy)
                TestError.append(error)

    data = {
        'Optimizer': Optimizer,
        'Hyperoptimizer': Hyperoptimizer,
        'Training_loss': Training_loss,
        'Parameters': Final_params_dict,
        'Test Accuracy': TestAccuracy,
        'Test Error': TestError
    }

    for param_dict in data['Parameters']:
        for key, value in param_dict.items():
            if pd.isna(value):
                param_dict[key] = 'Diverged'

    return data


def get_CNN_data(model):
    root_dir = f'./MAKE/OUT/{model}'
    Optimizer, Hyperoptimizer, Training_loss, Loss, TestAccuracy, TestError, Final_params_dict = [], [], [], [], [], [], []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.out'):
                with open(os.path.join(subdir, file), 'r') as f:
                    text = f.read()
                    final_params = re.findall(r'Final Optimizer Parameters(.*?)\n\n', text, re.DOTALL)
                    accuracy = re.search(r'Test Accuracy = (.*?) %', text).group(1)
                    error = re.search(r'Test Error = (.*?) %', text).group(1)
                    optimizer = re.search(r'optimizer: (\w+)', text).group(1)
                    hyperoptimizer = re.search(r'hyperoptimizer: (\w+)', text).group(1)

                    final_params_dict = {}
                    if file != 'baseline.out':
                        final_params_dict['alpha'] = re.search(r"(\d+\.\d+)", subdir).group(1)
                        final_params_dict['mu'] = re.search(r"(\d+\.\d+)", file).group(1)
                    else:
                        final_params_dict['alpha'] = 0.1
                        final_params_dict['mu'] = 0.9

                train_loss = []
                with open(os.path.join(subdir, file), 'r') as f:
                    for line in f:
                        match = re.search(r'Training Loss = ([\d.]+)', line)
                        if match:
                            training_loss_value = float(match.group(1))
                            train_loss.append(training_loss_value)
                        match1 = re.search(r'Initial Train Loss: ([\d.]+)', line)
                        if match1:
                            training_loss_value = float(match1.group(1))
                            train_loss.append(training_loss_value)

                Optimizer.append(optimizer)
                Hyperoptimizer.append(hyperoptimizer)
                Training_loss.append(train_loss)
                Final_params_dict.append(final_params_dict)
                TestAccuracy.append(accuracy)
                TestError.append(error)

    data = {
        'Optimizer': Optimizer,
        'Hyperoptimizer': Hyperoptimizer,
        'Training_loss': Training_loss,
        'Parameters': Final_params_dict,
        'Test Accuracy': TestAccuracy,
        'Test Error': TestError
    }

    for param_dict in data['Parameters']:
        for key, value in param_dict.items():
            if pd.isna(value):
                param_dict[key] = 'Diverged'

    return data


def get_NN_data(model):
    root_dir = f'./MAKE/OUT/{model}'
    Optimizer, Hyperoptimizer, Training_loss, Loss, TestAccuracy, TestError, Final_params_dict = [], [], [], [], [], [], []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.out'):
                with open(os.path.join(subdir, file), 'r') as f:
                    text = f.read()
                    final_params = re.findall(r'Final Optimizer Parameters(.*?)\n\n', text, re.DOTALL)
                    accuracy = re.search(r'Test Accuracy = (.*?) %', text).group(1)
                    error = re.search(r'Test Error = (.*?) %', text).group(1)
                    optimizer = re.search(r'optimizer: (\w+)', text).group(1)
                    hyperoptimizer = re.search(r'hyperoptimizer: (\w+)', text).group(1)

                    final_params_dict = {}
                    for params in final_params:
                        for line in params.split('\n'):
                            if line.strip():
                                key, value = line.split(':')
                                final_params_dict[key.strip()] = float(value.strip())

                train_loss = []
                with open(os.path.join(subdir, file), 'r') as f:
                    for line in f:
                        match = re.search(r'Training Loss = ([\d.]+)', line)
                        if match:
                            training_loss_value = float(match.group(1))
                            train_loss.append(training_loss_value)
                        match1 = re.search(r'Initial Train Loss: ([\d.]+)', line)
                        if match1:
                            training_loss_value = float(match1.group(1))
                            train_loss.append(training_loss_value)

                Optimizer.append(optimizer)
                Hyperoptimizer.append(hyperoptimizer)
                Training_loss.append(train_loss)
                Final_params_dict.append(final_params_dict)
                TestAccuracy.append(accuracy)
                TestError.append(error)

    data = {
        'Optimizer': Optimizer,
        'Hyperoptimizer': Hyperoptimizer,
        'Training_loss': Training_loss,
        'Parameters': Final_params_dict,
        'Test Accuracy': TestAccuracy,
        'Test Error': TestError
    }

    for param_dict in data['Parameters']:
        for key, value in param_dict.items():
            if pd.isna(value):
                param_dict[key] = 'Diverged'

    return data


def get_stack_data(model):
    root_dir = f'./MAKE/OUT/{model}'
    Optimizer, Hyperoptimizer, Training_loss, Loss, TestAccuracy, TestError, Final_params_dict, Stacks = [], [], [], [], [], [], [], []
    for subdir, _, files in os.walk(root_dir):
        stacks, optimizers, hyperoptimizers, final_params, accuracies, errors, training_losses = [], [], [], [], [], [], []
        for file in files:
            if file.endswith('.out'):
                with open(os.path.join(subdir, file), 'r') as f:
                    text = f.read()
                    final_param = re.findall(r'Final Optimizer Parameters(.*?)\n\n', text, re.DOTALL)
                    accuracy = re.search(r'Test Accuracy = (.*?) %', text).group(1)
                    error = re.search(r'Test Error = (.*?) %', text).group(1)
                    optimizer = re.search(r'optimizer: (\w+)', text).group(1)
                    hyperoptimizer = re.search(r'hyperoptimizer: (\w+)', text).group(1)

                    final_params_dict = {}
                    final_params_dict['alpha'] = re.search(r"(\d+\.\d+)", subdir).group(1)

                train_loss = []
                with open(os.path.join(subdir, file), 'r') as f:
                    for line in f:
                        match = re.search(r'Training Loss = ([\d.]+)', line)
                        if match:
                            training_loss_value = float(match.group(1))
                            train_loss.append(training_loss_value)
                        match1 = re.search(r'Initial Train Loss: ([\d.]+)', line)
                        if match1:
                            training_loss_value = float(match1.group(1))
                            train_loss.append(training_loss_value)


                stacks.append(re.search(r"(\d+)", file).group(1))
                optimizers.append(optimizer)
                hyperoptimizers.append(hyperoptimizer)
                final_params.append(final_params_dict)
                accuracies.append(accuracy)
                errors.append(error)
                training_losses.append(train_loss)

            Optimizer.append(optimizers)
            Hyperoptimizer.append(hyperoptimizers)
            Final_params_dict.append(final_params)
            TestAccuracy.append(accuracies)
            TestError.append(errors)
            Stacks.append(stacks)
            Training_loss.append(training_losses)

    data = {
        'Num_Stacks': Stacks,
        'Optimizer': Optimizer,
        'Hyperoptimizer': Hyperoptimizer,
        'Training_loss': Training_loss,
        'Parameters': Final_params_dict,
        'Test Accuracy': TestAccuracy,
        'Test Error': TestError
    }

    sorted_data = {}
    for key in data:
        sorted_data[key] = []
        for i, lst in enumerate(data[key]):
            if key == 'Num_Stacks':
                indexed_a = list(enumerate(lst))
                indexed_a.sort(key=lambda x: int(x[1]))
                sorted_data[key].append([x[1] for x in indexed_a])
            else:
                if isinstance(lst[0], list):
                    sorted_data[key].append([lst[x[0]] for x in indexed_a])
                else:
                    sorted_data[key].append([lst[x[0]] for x in indexed_a])

    return sorted_data


def plot_RNN_data(data_dict, model):
    # Extract relevant data from the data dictionary
    optimizers = data_dict['Optimizer']
    hyperoptimizers = data_dict['Hyperoptimizer']
    training_losses = data_dict['Training_loss']
    parameters = data_dict['Parameters']
    baseline_index = hyperoptimizers.index('NoOp')
    baseline_loss = training_losses[baseline_index]

    # Determine the number of subplots based on the number of optimizers
    num_optimizers = len(optimizers)

    # Create subplots
    mpl_style(dark=True, minor_ticks=False)
    fig, axes = plt.subplots(num_optimizers - 1, 1, figsize=(10, 5 * (num_optimizers - 1)))

    # Iterate over each optimizer
    for i, optimizer in enumerate(optimizers):
        # Skip plotting the baseline against itself
        if i == baseline_index:
            continue

        # Get the training loss for the current optimizer
        current_loss = training_losses[i]

        # Set up the subplot
        ax = axes[i] if num_optimizers > 1 else axes

        # Plot the training loss for the current optimizer
        ax.plot(range(len(current_loss)), current_loss, label=f'{optimizer} Training Loss')

        # Plot the baseline loss
        ax.plot(range(len(baseline_loss)), baseline_loss, label='Baseline Training Loss')

        # Set labels and title for the subplot
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Training Loss')
        ax.set_title(f"Alpha = {parameters[i]['alpha']}")

        # Add legend to the subplot
        ax.legend()

    title = plt.suptitle(f'Hyperoptimizer vs Baseline (with Alpha=0.002) Training Loss Plots')

    # Adjust spacing between subplots
    plt.tight_layout()
    title.set_position([.5, 0.99])

    # Show the plot
    plt.savefig(f'./comparative_plots/Training_Loss/{model}/lr_comparisons.png')


def plot_CNN_data(data_dict, model):
    # Extract relevant data from the data dictionary
    optimizers = data_dict['Optimizer']
    hyperoptimizers = data_dict['Hyperoptimizer']
    training_losses = data_dict['Training_loss']
    parameters = data_dict['Parameters']
    baseline_index = hyperoptimizers.index('NoOp')
    baseline_loss = training_losses[baseline_index]

    # Determine the number of figures based on the number of alphas
    alphas = list(set([d['alpha'] for d in parameters]))
    num_alphas = len(alphas)

    # Iterate over each alpha
    for alpha_idx, alpha in enumerate(alphas):
        # Skip plotting the baseline hyperoptimizer separately
        if alpha_idx == baseline_index:
            continue

        print(f"Alpha: {alpha}")

        # Create subplots for the current alpha
        mu = [d['mu'] for d in parameters if d['alpha'] == alpha]
        num_mu = len(mu)
        mpl_style(dark=True, minor_ticks=False)
        fig, axes = plt.subplots(num_mu, 1, figsize=(10, 5 * num_mu))

        # Iterate over each mu for the current alpha
        for mu_idx, m in enumerate(mu):
            print(f"\tMu: {m}")
            # Get the training loss for the current alpha and mu
            index = next((j for j, d in enumerate(parameters) if d['alpha'] == alpha and d['mu'] == m), None)
            current_loss = training_losses[index]

            # Set up the subplot
            ax = axes[mu_idx] if num_mu > 1 else axes

            # Plot the training loss for the current alpha and mu
            ax.plot(range(len(current_loss)), current_loss, label=f'{m} Training Loss')

            # Plot the baseline loss
            ax.plot(range(len(baseline_loss)), baseline_loss, label='Baseline Training Loss')

            # Set labels and title for the subplot
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Training Loss')
            ax.set_title(f'Mu = {m}')

            # Add legend to the subplot
            ax.legend()

        title = plt.suptitle(f'Alpha = {alpha} vs Baseline (with Alpha=0.1) Training Loss Plots')

        # Adjust spacing between subplots
        plt.tight_layout()
        title.set_position([.5, 0.99])

        # Save the figure
        plt.savefig(f'./comparative_plots/Training_Loss/{model}/alpha_{alpha}_comparisons.png')


def plot_NN_data(data_dict, model):
    # Extract relevant data from the data dictionary
    optimizers = data_dict['Optimizer']
    hyperoptimizers = data_dict['Hyperoptimizer']
    training_losses = data_dict['Training_loss']
    baseline_index = hyperoptimizers.index('NoOp')
    baseline_loss = training_losses[baseline_index]

    # Determine the number of figures based on the number of optimizers
    optimizer_loop = list(set(optimizers))
    num_optimizers = len(optimizers)

    # Iterate over each optimizer
    for i, optimizer in enumerate(optimizer_loop):
        # Get the indices of hyperoptimizers corresponding to the current optimizer
        optimizer_indices = [j for j, opt in enumerate(optimizers) if opt == optimizer]
        num_hyperoptimizers = len(optimizer_indices)

        # Create subplots for the current optimizer
        mpl_style(dark=True, minor_ticks=False)
        fig, axes = plt.subplots(num_hyperoptimizers-1, 1, figsize=(10, 5 * num_hyperoptimizers))

        # Iterate over each hyperoptimizer for the current optimizer
        j = 0
        for hyperoptimizer_idx in optimizer_indices:
            # Skip plotting the baseline hyperoptimizer separately
            if hyperoptimizers[hyperoptimizer_idx] == 'NoOp':
                continue

            # Get the training loss for the current optimizer and hyperoptimizer
            current_loss = training_losses[hyperoptimizer_idx]

            # Set up the subplot
            ax = axes[j] if num_hyperoptimizers > 1 else axes

            # Plot the training loss for the current optimizer and hyperoptimizer
            ax.plot(range(len(current_loss)), current_loss,
                    label=f'{hyperoptimizers[hyperoptimizer_idx]} Training Loss')

            # Plot the baseline loss
            ax.plot(range(len(baseline_loss)), baseline_loss, label='Baseline Training Loss')

            # Set labels and title for the subplot
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Training Loss')
            ax.set_title(f'Hyperoptimizer : {hyperoptimizers[hyperoptimizer_idx]}')

            # Add legend to the subplot
            ax.legend()
            j+=1

        title = plt.suptitle(f'{optimizer} vs Baseline Training Loss Plots')

        # Adjust spacing between subplots
        plt.tight_layout()
        title.set_position([.5, 0.99])

        # Save the figure
        plt.savefig(f'./comparative_plots/Training_Loss/{model}/{optimizer}_comparisons.png')


def plot_stack_data(data_dict, model):
    # Extract relevant data from the data dictionary
    alphas = list(set([params['alpha'] for sublist in data_dict['Parameters'] for params in sublist]))
    training_losses = data_dict['Training_loss']

    # Iterate over each figure (alpha)
    for i, alpha in enumerate(alphas):
        # Create plots
        mpl_style(dark=True, minor_ticks=False)
        fig, axes = plt.subplots(figsize=(15, 8))

        # Get the training losses for the current alpha
        current_losses = training_losses[i]

        # Iterate over each stack
        for stack_num, stack_losses in enumerate(current_losses):
            # Plot the training loss for the current stack
            linestyle = 'solid' if stack_num != 0 else (0, (1, 10))
            axes.plot(range(len(stack_losses)), stack_losses, linewidth=2, linestyle=linestyle, label=f'Stack {stack_num}')

        # Set labels and title for the subplot
        axes.set_xlabel('Epochs')
        axes.set_ylabel('Training Loss')

        # Add legend to the subplot
        plt.legend(loc='upper right')
        title = plt.suptitle(f'Alpha: {alpha}')

        # Adjust spacing between subplots
        plt.tight_layout()

        # Save the figures
        title.set_position([.5, 0.99])
        plt.savefig(f'./comparative_plots/Training_Loss/{model}/alpha_{alpha}.png')
        plt.close()


if __name__ == '__main__':
    models = ['CNN'] #'CharRNN', 'CNN', 'NN', 'stacked_NN']

    # define Function Dictionary
    FUNCTION_DICT = {'CharRNN': [get_RNN_data, plot_RNN_data], 'CNN': [get_CNN_data, plot_CNN_data],
                     'NN': [get_NN_data, plot_NN_data], 'stacked_NN': [get_stack_data, plot_stack_data]}

    # get data
    for model in models:
        print(f"Doing {model} model")
        data = FUNCTION_DICT[model][0](model)
        print(f"\t{data['Parameters']}")

        # plot data
        FUNCTION_DICT[model][1](data, model)
