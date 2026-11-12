#!/bin/bash
#SBATCH --job-name=tactile_lam_egotouch
#SBATCH --output=logs/tactile_lam_egotouch.%j.out
#SBATCH --error=logs/tactile_lam_egotouch.%j.err

#SBATCH -A iyg@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -t 50:00:00
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --mail-user=clement.prigent@ec-lyon.fr
#SBATCH --mail-type=FAIL

# TactileLAM on EgoTouch (conf/train_tactile_lam.yaml), with the Jean Zay modules only (no conda env).
# The packages missing from the pytorch-gpu module (vector_quantize_pytorch and its deps, rotary_embedding_torch,
# decord, moviepy + proglog for wandb.Video) are installed once with `pip install --user --no-deps` in PYTHONUSERBASE below.
# Extra hydra overrides can be passed: sbatch train_tactile_lam_egotouch.sh trainer.max_steps=1000

echo ${SLURM_NODELIST}

module purge
module load arch/h100              # required on the H100 partition
module load pytorch-gpu/py3/2.8.0

export PYTHONUSERBASE=$WORK/.local_lam_pt2.8
export PYTHONPATH=/lustre/fswork/projects/rech/iyg/uvx44rt/Tactile-Guided-Latent-Action:$PYTHONPATH

export WANDB_MODE=offline
export WANDB_START_METHOD=thread
export HYDRA_FULL_ERROR=1

cd /lustre/fswork/projects/rech/iyg/uvx44rt/Tactile-Guided-Latent-Action
srun python train.py "$@"
