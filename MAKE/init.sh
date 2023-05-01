#!/bin/bash
#SBATCH --partition=ecsstudents
#SBATCH --account=ecsstudents
#SBATCH --time=12:00:00
#SBATCH --ntasks=6
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bes1g19@soton.ac.uk

model=NN
optimizer=Adam
hyperoptimizer=SGD
loss_fn=CrossEntropyLoss
dataset=MNIST
alpha=0.01
kappa=0.00001

python /scratch/bes1g19/DeepLearning/CW/taskdef.py --model $model --optimizer $optimizer --hyperoptimizer $hyperoptimizer --alpha $alpha --kappa $kappa --loss_fn $loss_fn --dataset $dataset &> /scratch/bes1g19/DeepLearning/CW/MAKE/OUT/run.out
find -type f -name '*slurm*' -delete