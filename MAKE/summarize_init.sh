#!/bin/bash
#SBATCH --partition=ecsstudents
#SBATCH --account=ecsstudents
#SBATCH --qos=ecsstudentsextra
#SBATCH --time=12:00:00
#SBATCH --ntasks=16
#SBATCH --nodes=1.0.0.0.0.0
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL

model='NN'

python /scratch/bes1g19/DeepLearning/CW/summarise.py --model $model &> "/scratch/bes1g19/DeepLearning/CW/MAKE/OUT/"$model"_summary.out"
