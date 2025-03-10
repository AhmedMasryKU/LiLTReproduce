#!/bin/bash
#SBATCH --account=def-enamul
#SBATCH --cpus-per-task=8
#SBATCH --time=167:0:0
#SBATCH --mem=64000M
#SBATCH --output=/home/masry20/projects/def-enamul/masry20/LiLTReproduce/jobs_output/%x-%j.out

cd /home/masry20/scratch/idl_data/raw_data/
#wget http://datasets.cvc.uab.es/UCSF_IDL/index.txt
# wget -i index.txt
# extract folder f, can be repeated for g,h,j,k
# cat f.*|  tar xvzf -
# cat g.*|  tar xvzf -
# cat h.*|  tar xvzf -
# cat j.*|  tar xvzf -
# cat k.*|  tar xvzf -
wget http://datasets.cvc.uab.es/UCSF_IDL/IMDBs/imdbs_v2.tar.gz
tar -xzf imdbs_v2.tar.gz