import os
import re
import argparse
import pandas as pd
from tabulate import tabulate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='NN')
    args = parser.parse_args()

    root_dir = f'/scratch/bes1g19/DeepLearning/CW/MAKE/OUT/{args.model}'
    Optimizer = []
    Hyperoptimizer = []
    TestAccuracy = []
    TestError = []
    Final_params_dict = []
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

                Optimizer.append(optimizer)
                Hyperoptimizer.append(hyperoptimizer)
                Final_params_dict.append(final_params_dict)
                TestAccuracy.append(accuracy)
                TestError.append(error)

    data = {
        'Optimizer': Optimizer,
        'Hyperoptimizer': Hyperoptimizer,
        'Parameters': Final_params_dict,
        'Test Accuracy': TestAccuracy,
        'Test Error': TestError
    }

    for param_dict in data['Parameters']:
        for key, value in param_dict.items():
            if pd.isna(value):
                param_dict[key] = 'Diverged'

    df = pd.DataFrame(data)
    params_df = pd.DataFrame(df['Parameters'].to_list())
    df = pd.concat([df.drop('Parameters', axis=1), params_df], axis=1)

    table = tabulate(df, headers='keys', tablefmt='grid')
    df.to_csv(f'OUT/{args.model}_summary.csv')
    print(table)
