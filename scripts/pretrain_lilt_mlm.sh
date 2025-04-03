#!/bin/bash
#SBATCH --account=def-enamul
#SBATCH --ntasks-per-node=16
#SBATCH --time=11:50:00
#SBATCH --mem=128G
#SBATCH --partition=debug 
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/home/masry20/projects/def-enamul/masry20/LiLTReproduce/jobs_output/%x-%j.out

nvidia-smi
module load python/3.11
module load scipy-stack
module load gcc arrow
module load StdEnv/2023

source ~/projects/def-enamul/masry20/lilt_env/bin/activate

export HF_TOKEN=

# cd $SLURM_TMPDIR
# cp /home/masry20/scratch/idl_data/raw_data/selected_data/extracted_data.tar extracted_data.tar
# tar -xf extracted_data.tar 

cd /home/masry20/projects/def-enamul/masry20/LiLTReproduce/
python pretrain_lilt_mlm.py