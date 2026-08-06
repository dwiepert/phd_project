#!/bin/bash
# Job name:
#SBATCH -J ema_extraction
#
# File where job progress and standard output is written
#SBATCH -o $WORK/job_output/ema.out
#
# File where job errors are written
#SBATCH -e $WORK/job_output/ema.err      
#
# Request only one node:
#SBATCH -n 1
#
# Wall clock limit: HH:MM:SS. Max is 48 hours on most nodes
#SBATCH -t 00:30:00
#
## Command(s) to run
source $WORK/miniforge3/condabin/conda
conda activate acmv1

python $WORK/phd_project/run.py --audio_dir=$WORK/Downsampled/Train --metadata_path=$WORK/Downsampled/Train/metadata.csv --split=speaker+diagnosis --ema_only --debug

