#!/bin/bash
#SBATCH --partition=ecsstudents
#SBATCH --account=ecsstudents
#SBATCH --qos=ecsstudentsextra
#SBATCH --time=12:00:00
#SBATCH --ntasks=16
#SBATCH --nodes=1.0.0.0.0.0
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL

model=CNN
optimizer=SGD
optimizer_args="mu=0.9"
hyperoptimizer=NoOp
hyperoptimizer_args=""
loss_fn=CrossEntropyLoss
dataset=CIFAR
num_epochs=50
alpha=0.1
baseline=True

python /scratch/bes1g19/DeepLearning/CW/taskdef.py --model $model --baseline $baseline --optimizer $optimizer --optimizer_args $optimizer_args --hyperoptimizer $hyperoptimizer --hyperoptimizer_args $hyperoptimizer_args  --alpha $alpha --loss_fn $loss_fn --num_epochs $num_epochs --dataset $dataset &> /scratch/bes1g19/DeepLearning/CW/MAKE/OUT/$model/baseline.out
find -type f -name '*slurm*' -delete