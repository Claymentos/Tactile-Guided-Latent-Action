#!/bin/sh
#SBATCH --job-name=jala_paper
#SBATCH --output=logs/jala_paper.%j.out
#SBATCH --error=logs/jala_paper.%j.err

#SBATCH -A uli@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -t 20:00:00
#SBATCH --qos=qos_gpu_h100-t3

# JALA version papier (InternVL3-2B + loss MCP) sur EgoDex (conf/train_jala_paper_egodex.yaml).
#   TOKENIZER=runs/wandb/offline-run-XXX/checkpoints/epoch=0-step=50000.ckpt sbatch train_jala_paper_egodex.sh
#   TOKENIZER=... sbatch train_jala_paper_egodex.sh checkpoint_path=\"runs/.../last.ckpt\"     # reprise
# Le chemin passe par une variable : les checkpoints lightning contiennent des '=' que Hydra refuse sans guillemets.
# Les nœuds de calcul n'ont pas internet : télécharger le VLM avant, depuis un nœud de login :
#   huggingface-cli download OpenGVLab/InternVL3-2B-hf --local-dir $WORK/models/InternVL3-2B-hf
# Pas de clé W&B ici : run hors ligne, `wandb sync` depuis un nœud de login (connexion via `wandb login`).

: "${TOKENIZER:?TOKENIZER=<checkpoint du tokenizer de mouvement> requis (train_motion_tokenizer_egodex.sh)}"
VLM_PATH=${VLM_PATH:-$WORK/models/InternVL3-2B-hf}

echo ${SLURM_NODELIST}

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0

export PYTHONUSERBASE=$WORK/.local_lam_pt2.8
export PYTHONPATH=/lustre/fswork/projects/rech/iyg/uvx44rt/Tactile-Guided-Latent-Action:$PYTHONPATH

export WANDB_MODE=offline
export WANDB_START_METHOD=thread
export HF_HUB_OFFLINE=1
export HYDRA_FULL_ERROR=1

for f in dinov3/hubconf.py weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth "$TOKENIZER" "$VLM_PATH/config.json"; do
    [ -e "$f" ] || { echo "manquant : $f (lancer sbatch depuis la racine du repo)" >&2; exit 1; }
done
# InternVL3 au format transformers (InternVLForConditionalGeneration) : transformers >= 4.52
python -c "import transformers as t, sys; v = tuple(int(x) for x in t.__version__.split('.')[:2]); print('transformers', t.__version__); sys.exit(v < (4, 52))" \
    || { echo "transformers >= 4.52 requis : pip install --user 'transformers>=4.52'" >&2; exit 1; }

srun python train.py --config-name train_jala_paper_egodex data.data_module.num_workers=20 \
    "algo.model.motion_tokenizer_path=\"$TOKENIZER\"" "algo.model.vlm_path=\"$VLM_PATH\"" "$@"
