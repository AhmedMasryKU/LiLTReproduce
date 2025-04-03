#!/bin/bash
#SBATCH --account=def-enamul
#SBATCH --cpus-per-task=8
#SBATCH --time=2:50:00
#SBATCH --mem=256000M
#SBATCH --output=/home/masry20/projects/def-enamul/masry20/LiLTReproduce/jobs_output/%x-%j.out

module load python/3.11
module load gcc arrow
source ~/projects/def-enamul/masry20/chartqa/bin/activate

python3.11 /home/masry20/projects/def-enamul/masry20/LiLTReproduce/data/preprocessing/pretraining/preprocess_data.py