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
optimizer_args=("mu=0.09" "mu=0.9" "mu=0.99")
hyperoptimizer=SGD
hyperoptimizer_args=""
loss_fn=CrossEntropyLoss
dataset=CIFAR
num_epochs=50
alpha=(0.01 0.1.0.0.0.0.0 1)

for i in $(seq 0 $((${#alpha[@]} - 1))); do
    al=${alpha[$i]}
    for j in $(seq 0 $((${#optimizer_args[@]} - 1))); do
        opt_args=${optimizer_args[$j]}

        echo "Al: $al $opt_args"

        python /scratch/bes1g19/DeepLearning/CW/taskdef.py --model $model --optimizer $optimizer --optimizer_args $opt_args --hyperoptimizer $hyperoptimizer --hyperoptimizer_args $hyperoptimizer_args  --alpha $al --kappa $kappa --loss_fn $loss_fn --num_epochs $num_epochs --dataset $dataset &> /scratch/bes1g19/DeepLearning/CW/MAKE/OUT/$model/$al/$opt_args.out
    done
done

find -type f -name '*slurm*' -delete
