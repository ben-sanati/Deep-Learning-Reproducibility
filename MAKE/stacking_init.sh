#!/bin/bash
#SBATCH --partition=ecsstudents
#SBATCH --account=ecsstudents
#SBATCH --qos=ecsstudentsextra
#SBATCH --time=12:00:00
#SBATCH --ntasks=16
#SBATCH --nodes=1.0.0.0.0.0
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL

model=stacked_NN
optimizer=SGD
optimizer_args=""
hyperoptimizer=RMSProp
hyperoptimizer_args=""
loss_fn=CrossEntropyLoss
dataset=MNIST
num_epochs=10
alpha=(1 0.01 0.0001 0.000001 0.0000001)
batch_size=1024
num_hyperoptimizers=()
device='cuda'

for i in $(seq 0 20); do
  num_hyperoptimizers+=($i)
done

for i in $(seq 0 $((${#alpha[@]} - 1))); do
    al=${alpha[$i]}
    for j in $(seq 0 $((${#num_hyperoptimizers[@]} - 1))); do
        num_layers=${num_hyperoptimizers[$j]}
        python /scratch/bes1g19/DeepLearning/CW/taskdef.py --model $model --optimizer $optimizer --optimizer_args $optimizer_args --hyperoptimizer $hyperoptimizer --hyperoptimizer_args $hyperoptimizer_args --num_hyperoptimizers $num_layers  --alpha $al --loss_fn $loss_fn --num_epochs $num_epochs --dataset $dataset --batch_size $batch_size --device $device &> /scratch/bes1g19/DeepLearning/CW/MAKE/OUT/$model/$al/layers_$num_layers.out
    done
done

find -type f -name '*slurm*' -delete
