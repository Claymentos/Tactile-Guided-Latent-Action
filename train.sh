#!/bin/sh
#SBATCH --job-name=lam_egotouch
#SBATCH --output=logs/lam_egotouch.%j.out
#SBATCH --error=logs/lam_egotouch.%j.err 

#SBATCH -A uli@h100  # uli remplace par ton nom de projet 
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -t 20:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --mail-user=clement.prigent@ec-lyon.fr
#SBATCH --mail-typ=FAIL

echo ${SLURM_NODELIST}

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
module purge
module load arch/h100 # a charger absolument avec H100 mais à retirer avec v100
# module load pytorch-gpu/py3/2.7.0 # là tu charge le module pytorhc que tu veux
module load pytorch-gpu/py3/2.8.0

export PYTHONUSERBASE=$WORK/.local_lam_pt2.8
export PYTHONPATH=/lustre/fswork/projects/rech/iyg/uvx44rt/Tactile-Guided-Latent-Action:$PYTHONPATH
proj_name=Tactile-guided-Latent-Action
device_id=0
export WANDB_MODE=offline
export WANDB_START_METHOD=thread
export WANDB_API_KEY=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


export HYDRA_FULL_ERROR=1

srun python train.py "$@"