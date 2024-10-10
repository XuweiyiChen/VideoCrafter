#!/bin/bash

#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --time=16:00:00  # Adjusted to 96 hours, change as needed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4  # Increased CPU count for parallel processing
#SBATCH --mem=40G  # Adjust as per your job's requirement
#SBATCH --gres=gpu:1  # Adjust based on GPU needs
#SBATCH --job-name="VideoCrafter_unictrl_c=1_seed=123123123"
#SBATCH --output=/scratch/lsa_root/lsa1/tianxia/FastFreeInit/gl-outputs/%x-%j.out
#SBATCH --error=/scratch/lsa_root/lsa1/tianxia/FastFreeInit/gl-outputs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL  # Removed BEGIN and REQUEUE notifications

module load cuda

cd /scratch/lsa_root/lsa1/tianxia/FastFreeInit/examples/AnimateDiff

python submitit_greatlakes.py --prompt_file /home/tianxia/VideoCrafter/prompts/data.txt --motion_ctrl 1 --seed 123123123 --output_folder results/VideoCrafter_unictrl_c=1_seed=123123123