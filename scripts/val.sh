#!/bin/bash
#SBATCH --account=viscam
#SBATCH --partition=viscam --qos=normal
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --gres=gpu:1 

#SBATCH --job-name="val"
#SBATCH --output=logs/val_%A.out
#SBATCH --error=logs/val_%A.err
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
export WANDB_INIT_TIMEOUT=180
export TMPDIR=/vision/u/emilyjin/abstractions/tmp
export TORCH_HOME=/vision/u/emilyjin/abstractions/tmp
export WANDB_CACHE_DIR=/vision/u/emilyjin/.cache/wandb
export WANDB_DIR=/vision/u/emilyjin/abstractions/wandb

echo "Working with the LD_LIBRARY_PATH: "$LD_LIBRARY_PATH

cd /vision/u/emilyjin/abstractions/src

checkpoint_name=$1
dataset=$2
get_accuracy=$3
visualize_embeddings=$4
visualize_norms=$5
visualize_masks=$6
few_shot=$7
by_pred=$8
num_data=$9

echo "python val.py --checkpoint_name ${checkpoint_name} --dataset ${dataset} ${get_accuracy} ${visualize_embeddings} ${visualize_norms} ${visualize_masks} ${few_shot} ${by_pred} ${num_data}"
python -u val.py --checkpoint_name ${checkpoint_name} --dataset ${dataset} ${get_accuracy} ${visualize_embeddings} ${visualize_norms} ${visualize_masks} ${few_shot} ${by_pred} ${num_data}