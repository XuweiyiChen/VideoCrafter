#!/bin/bash
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --time=20:00:00  # Adjusted to 96 hours, change as needed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4  # Increased CPU count for parallel processing
#SBATCH --mem=40G  # Adjust as per your job's requirement
#SBATCH --gres=gpu:1  # Adjust based on GPU needs
#SBATCH --job-name="VideoCrafter_unictrl_c=0.4_seed=0"
#SBATCH --output=/scratch/lsa_root/lsa1/tianxia/VideoCrafter/gl-outputs/%x-%j.out
#SBATCH --error=/scratch/lsa_root/lsa1/tianxia/VideoCrafter/gl-outputs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL  # Removed BEGIN and REQUEUE notifications
module load cuda
cd /scratch/lsa_root/lsa1/tianxia/VideoCrafter
python scripts/evaluation/infer.py --motion_ctrl 0.4 --seed 0 --savedir results/VideoCrafter_unictrl_c=0.4_seed=0