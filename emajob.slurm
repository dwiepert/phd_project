#!/bin/bash
# Job name:
#SBATCH -J ema_extraction
#
# File where job progress and standard output is written
#SBATCH -o job_output/ema.o%j
#
# File where job errors are written
#SBATCH -e job_output/ema.e%j     
#
# Request only one node:
#SBATCH -N 1
#
# Wall clock limit: HH:MM:SS. Max is 48 hours on most nodes
#SBATCH -t 48:00:00
#
#queue name
#SBATCH -p gh
## Command(s) to run
eval "$(conda shell.bash hook)"
conda activate acmv1
python phd_project/run.py --audio_dir=mocha_timit_audio --metadata_path=mocha_timit_audio/metadata.csv --output_dir=ema_mocha-timit/ --split=speaker --ema_only --dataset_name=mocha-timit -tasks=sentence-reading