#!/bin/bash
#SBATCH --partition=ecsstudents
#SBATCH --account=ecsstudents
#SBATCH --qos=ecsstudentsextra
#SBATCH --time=12:00:00
#SBATCH --ntasks=16
#SBATCH --nodes=1.0.0.0.0.0
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL

model=CharRNN
optimizer=Adam_alpha
optimizer_args="beta1=0.9 beta2=0.99"
hyperoptimizer=NoOp
hyperoptimizer_args=""
loss_fn=CrossEntropyLoss
dataset=WarAndPeace
num_epochs=50
alpha=0.002
batch_size=128
device='cuda'
baseline=True

python /scratch/bes1g19/DeepLearning/CW/taskdef.py --model $model --baseline $baseline --optimizer $optimizer --optimizer_args $optimizer_args --hyperoptimizer $hyperoptimizer --hyperoptimizer_args $hyperoptimizer_args  --alpha $alpha --loss_fn $loss_fn --num_epochs $num_epochs --dataset $dataset --batch_size $batch_size --device $device &> /scratch/bes1g19/DeepLearning/CW/MAKE/OUT/$model/baseline.out
find -type f -name '*slurm*' -delete
