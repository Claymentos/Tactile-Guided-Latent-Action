#!/bin/sh
#SBATCH --job-name=motion_tokenizer
#SBATCH --output=logs/motion_tokenizer.%j.out
#SBATCH --error=logs/motion_tokenizer.%j.err

#SBATCH -A uli@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -t 10:00:00
#SBATCH --qos=qos_gpu_h100-t3

# Étape 1 de JALA version papier : tokenizer GRVQ du mouvement des mains (conf/train_motion_tokenizer_egodex.yaml).
#   sbatch train_motion_tokenizer_egodex.sh
# Le checkpoint (runs/wandb/offline-run-*/checkpoints/*.ckpt) sert ensuite à train_jala_paper_egodex.sh.
# Pas de clé W&B ici : run hors ligne, `wandb sync` depuis un nœud de login (connexion via `wandb login`).

echo ${SLURM_NODELIST}

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0

export PYTHONUSERBASE=$WORK/.local_lam_pt2.8
export PYTHONPATH=/lustre/fswork/projects/rech/iyg/uvx44rt/Tactile-Guided-Latent-Action:$PYTHONPATH

export WANDB_MODE=offline
export WANDB_START_METHOD=thread
export HYDRA_FULL_ERROR=1

# pas de décodage vidéo : seulement les annotations hdf5
srun python train.py --config-name train_motion_tokenizer_egodex data.data_module.num_workers=20 "$@"
