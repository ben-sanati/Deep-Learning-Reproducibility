#!/bin/bash
#SBATCH --partition=ecsstudents
#SBATCH --account=ecsstudents
#SBATCH --qos=ecsstudentsextra
#SBATCH --time=12:00:00
#SBATCH --ntasks=16
#SBATCH --nodes=1.0.0.0.0.0
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL

model=NN
optimizers=("AdaGrad" "Adam" "RMSProp" "SGD")
optimizer_args=("" "beta1=0.9 beta2=0.99" "gamma=0.99" "")
hyperoptimizer=NoOp
hyperoptimizer_args=""
loss_fn=CrossEntropyLoss
dataset=MNIST
num_epochs=30
alpha=(0.01 0.001 0.01 0.01)
baseline=True

for i in $(seq 0 $((${#alpha[@]} - 1))); do
    al=${alpha[$i]}
    opt=${optimizers[$i]}
    opt_args=${optimizer_args[$i]}
    echo $al
    echo $opt
    echo $opt_args
    python /scratch/bes1g19/DeepLearning/CW/taskdef.py --model $model --baseline $baseline --optimizer $opt --optimizer_args $opt_args --hyperoptimizer $hyperoptimizer --hyperoptimizer_args $hyperoptimizer_args  --alpha $al --loss_fn $loss_fn --num_epochs $num_epochs --dataset $dataset &> /scratch/bes1g19/DeepLearning/CW/MAKE/OUT/$model/$opt.out
done


find -type f -name '*slurm*' -delete