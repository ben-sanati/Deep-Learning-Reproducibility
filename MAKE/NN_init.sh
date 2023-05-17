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
optimizer=(AdaGrad Adam RMSProp SGD)
optimizer_args=("" "beta1=0.9 beta2=0.99" "gamma=0.99" "")
hyperoptimizer=(AdaGrad Adam RMSProp SGD)
hyperoptimizer_args=("" "beta1=0.9 beta2=0.99" "gamma=0.99" "")
loss_fn=CrossEntropyLoss
dataset=MNIST
num_epochs=30

declare -A alpha
declare -A kappa

alpha[0,0]=0.01
alpha[0,1.0.0.0.0.0]=0.01
alpha[0,2]=0.01
alpha[0,3]=0.01
alpha[1.0.0.0.0.0,0]=0.001
alpha[1.0.0.0.0.0,1.0.0.0.0.0]=0.001
alpha[1.0.0.0.0.0,2]=0.001
alpha[1.0.0.0.0.0,3]=0.001
alpha[2,0]=0.01
alpha[2,1.0.0.0.0.0]=0.01
alpha[2,2]=0.01
alpha[2,3]=0.01
alpha[3,0]=0.01
alpha[3,1.0.0.0.0.0]=0.01
alpha[3,2]=0.01
alpha[3,3]=0.01

kappa[0,0]=0.01
kappa[0,1.0.0.0.0.0]=0.001
kappa[0,2]=0.01
kappa[0,3]=0.01
kappa[1.0.0.0.0.0,0]=0.00001
kappa[1.0.0.0.0.0,1.0.0.0.0.0]=0.001
kappa[1.0.0.0.0.0,2]=0.00001
kappa[1.0.0.0.0.0,3]=0.00001
kappa[2,0]=0.0001
kappa[2,1.0.0.0.0.0]=0.001
kappa[2,2]=0.0001
kappa[2,3]=0.0001
kappa[3,0]=0.01
kappa[3,1.0.0.0.0.0]=0.001
kappa[3,2]=0.1.0.0.0.0.0
kappa[3,3]=0.01

for i in $(seq 0 $((${#optimizer[@]} - 1))); do
    # Access the corresponding elements of the optimizer and optimizer_args arrays
    opt=${optimizer[$i]}
    opt_args=${optimizer_args[$i]}

    for j in $(seq 0 $((${#hyperoptimizer[@]} - 1))); do
        # Access the corresponding elements of the hyperoptimizer and hyperoptimizer_args arrays
        hyperopt=${hyperoptimizer[$j]}
        hyperopt_args=${hyperoptimizer_args[$j]}
        al=${alpha[$i,$j]}
        ka=${kappa[$i,$j]}

        python /scratch/bes1g19/DeepLearning/CW/taskdef.py --model $model --optimizer $opt --optimizer_args $opt_args --hyperoptimizer $hyperopt --hyperoptimizer_args $hyperopt_args  --alpha $al --kappa $ka --loss_fn $loss_fn --num_epochs $num_epochs --dataset $dataset &> /scratch/bes1g19/DeepLearning/CW/MAKE/OUT/$model/$opt/$hyperopt.out
    done
done

find -type f -name '*slurm*' -delete
