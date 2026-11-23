#!/bin/sh
#SBATCH --job-name=jala_egodex
#SBATCH --output=logs/jala_egodex.%j.out
#SBATCH --error=logs/jala_egodex.%j.err

#SBATCH -A uli@h100
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

# JALA-LAM sur EgoDex (conf/train_jala_egodex.yaml).
#   sbatch train_jala_egodex.sh                                   # config du papier
#   sbatch train_jala_egodex.sh algo.model.z_action_weight=1.0 algo.model.variance_weight=1.0 \
#          trainer.logger.name=jala_egodex_garde_fous              # avec les garde-fous anti-effondrement
#   sbatch train_jala_egodex.sh checkpoint_path=runs/.../checkpoints/xxx.ckpt   # reprise
# Le dossier logs/ doit exister avant le sbatch (mkdir -p logs).
# Pas de clé W&B ici : le run est hors ligne (WANDB_MODE=offline) ; synchroniser ensuite depuis un nœud de login
# avec `wandb sync runs/wandb/offline-run-*` (connexion via `wandb login`, stockée dans ~/.netrc).

echo ${SLURM_NODELIST}

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0

export PYTHONUSERBASE=$WORK/.local_lam_pt2.8
export PYTHONPATH=/lustre/fswork/projects/rech/iyg/uvx44rt/Tactile-Guided-Latent-Action:$PYTHONPATH

export WANDB_MODE=offline
export WANDB_START_METHOD=thread
export WANDB_API_KEY=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

export HYDRA_FULL_ERROR=1

# DINOv3 est chargé depuis dinov3/ et weights/ à la racine du repo (algo.feature_extractor.model.repo_path / weights_path)
for f in dinov3/hubconf.py weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth; do
    [ -e "$f" ] || { echo "manquant : $f (lancer sbatch depuis la racine du repo)" >&2; exit 1; }
done

# 20 workers sur les 24 CPU : le décodage vidéo (5 frames par clip) est le goulot probable
srun python train.py --config-name train_jala_egodex data.data_module.num_workers=20 "$@"
