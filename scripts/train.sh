#!/bin/bash
#SBATCH --account=vision
#SBATCH --partition=svl --qos=normal
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1 

#SBATCH --job-name="train"
#SBATCH --output=logs/train_%A.out
#SBATCH --error=logs/train_%A.err
####SBATCH --mail-user=emilyjin@stanford.edu
####SBATCH --mail-type=ALL

echo "command line arguments"

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

##########################################
# Setting up virtualenv / conda / docker #
##########################################
module load anaconda3
source activate abs
echo "conda env activated"

##############################################################
# Setting up LD_LIBRARY_PATH or other env variable if needed #
##############################################################
export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64:/usr/lib/x86_64-linux-gnu
export WANDB_CACHE_DIR=/vision/u/emilyjin/.cache/wandb
export WANDB_DIR=/vision/u/emilyjin/abstractions/.cache/wandb
export TORCH_HOME=/vision/u/emilyjin/abstractions/tmp
export TMPDIR=/vision/u/emilyjin/abstractions/tmp
export TMPDIR=/vision/u/emilyjin/abstractions/tmp
export TORCH_USE_CUDA_DSA=1

echo "Working with the LD_LIBRARY_PATH: "$LD_LIBRARY_PATH

cd /vision/u/emilyjin/abstractions/src

dataset=$1
similarity=$2
is_hyperbolic=$3
prior_mode=$4
lr=$5
total_epoch=$6
num_hyp_layers=$7
hyp_hidden_dim=$8
hyp_output_dim=$9
num_fc_layers=${10}
fc_hidden_dim=${11}
triplet_selector=${12}
ssl_coeff=${13}
margin=${14}
reg_type=${15}
reg_coeff=${16}
reg_margin=${17}

echo "python train.py --dataset ${dataset} --similarity ${similarity} --is_hyperbolic ${is_hyperbolic} --prior_mode ${prior_mode} --lr ${lr} --total_epoch ${total_epoch} --num_hyp_layers ${num_hyp_layers} --hyp_hidden_dim ${hyp_hidden_dim} --hyp_output_dim ${hyp_output_dim} --num_fc_layers ${num_fc_layers} --fc_hidden_dim ${fc_hidden_dim} --triplet_selector ${triplet_selector} --ssl_coeff ${ssl_coeff} --margin ${margin} --reg_type ${reg_type} --reg_coeff ${reg_coeff} --reg_margin ${reg_margin}"
python train.py --dataset ${dataset} --similarity ${similarity} --is_hyperbolic ${is_hyperbolic} --prior_mode ${prior_mode} --lr ${lr} --total_epoch ${total_epoch} --num_hyp_layers ${num_hyp_layers} --hyp_hidden_dim ${hyp_hidden_dim} --hyp_output_dim ${hyp_output_dim} --num_fc_layers ${num_fc_layers} --fc_hidden_dim ${fc_hidden_dim} --triplet_selector ${triplet_selector} --ssl_coeff ${ssl_coeff} --margin ${margin} --reg_type ${reg_type} --reg_coeff ${reg_coeff} --reg_margin ${reg_margin}
