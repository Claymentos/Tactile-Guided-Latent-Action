#!/bin/sh
#SBATCH --job-name=train_tacile_lam
#SBATCH --output=logs/train_tacile_lam.%j.out
#SBATCH --error=logs/train_tacile_lam.%j.err 

#SBATCH -A yfl@h100  # uli remplace par ton nom de projet 
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -t 50:00:00
#SBATCH --qos=qos_gpu_h100-t4
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

srun python train.py