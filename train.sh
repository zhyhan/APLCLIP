#!/bin/bash
#SBATCH --job-name=test          # Job name
#SBATCH --output=output.txt   # Standard output and error log.%A_%a
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=40G                   # Total RAM to be used
#SBATCH --cpus-per-task=64          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH -p gpu                      # Use the gpu partition
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH --qos=gpu-8                 # To enable the use of up to 8 GPUs

python -m domainbed.scripts.train\
       --data_dir ~/dataset\
       --output_dir ./logs\
       --algorithm APLCLIP\
       --dataset VLCS\
       --hparams "{\"clip_backbone\": \"ViT-B/16\"}"\
       --test_envs 0

# python -m domainbed.scripts.train\
#        --data_dir ~/dataset\
#        --output_dir ./logs\
#        --algorithm APLCLIP\
#        --dataset VLCS\
#        --hparams "{\"clip_backbone\": \"ViT-B/16\"}"\
#        --test_envs 1

# python -m domainbed.scripts.train\
#        --data_dir ~/dataset\
#        --output_dir ./logs\
#        --algorithm APLCLIP\
#        --dataset VLCS\
#        --hparams "{\"clip_backbone\": \"ViT-B/16\"}"\
#        --test_envs 2

# python -m domainbed.scripts.train\
#        --data_dir ~/dataset\
#        --output_dir ./logs\
#        --algorithm APLCLIP\
#        --dataset VLCS\
#        --hparams "{\"clip_backbone\": \"ViT-B/16\"}"\
#        --test_envs 3