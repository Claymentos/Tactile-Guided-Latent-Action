from lam.lam import ActVAE
from lam.lightning.lam import ActVAETraining
from lam.datasets.egodex.dataset import EgoDexDataset
import torch
from torch.utils.data import DataLoader
from einops import rearrange, repeat
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
import numpy as np



act_vae_checkpoint = '/home/cl-ment-prigent/LAM/checkpoints/ckpt_act_vae/epoch=6-step=180000.ckpt'

act_vae_model = ActVAE(
    action_dim=264,
    model_dim=512,
    latent_dim=512,
    num_latents=16,
    enc_layers=15,
    dec_layers=15,
    nhead=8,
    dropout=0.1
)
ckpt = ActVAETraining.load_from_checkpoint(act_vae_checkpoint, model=act_vae_model)


DATASET_DIR = "/home/cl-ment-prigent/egodex/"

dataset = EgoDexDataset(root_folder=DATASET_DIR, 
                                    max_context_len=16, 
                                    num_context=4, 
                                    patch_size=14, 
                                    img_size=224, 
                                    backbone_name='dinov2', 
                                    train=False, 
                                    evaluate=True, 
                                    keys='all', 
                                    var_time=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = act_vae_model.to(device)
model.eval()
all_codes = []
all_tasks = []
dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
with torch.no_grad():
    for batch_idx, batch in enumerate(dataloader):

        actions = batch[1]
        metadata = batch[5]
        vid_files = metadata['vid_file']
        B,T = actions.shape[:2]
        x = rearrange(actions, 'b t p c -> b t (p c)') 
        z_e = model.encode(x)
        z_e = model.to_latent(z_e)

        _, _, _, indices = model.vq(z_e)
        indices = indices.view(B,T).cpu().numpy()

        for b in range(B):
            task_name = vid_files[b].split('/')[-2]

            for t in range(T):
                all_codes.append(indices[b,t])
                all_tasks.append(task_name)

all_codes = np.array(all_codes)
all_tasks = np.array(all_tasks)

# --- 2. EVALUATION DU ALIGNEMENT QUANTITATIF (NMI) ---
nmi_score = normalized_mutual_info_score(all_tasks, all_codes)
print("\n📊 --- ANALYSE DE COHÉRENCE SÉMANTIQUE ---")
print(f"Score NMI (Normalized Mutual Information) : {nmi_score:.4f}")
print("👉 [0 = Assignation purement aléatoire | 1 = Alignement parfait avec les tâches]")

df = pd.DataFrame({'Code': all_codes, 'Task': all_tasks})
num_latents = 16

print("\n📖 --- DICTIONNAIRE TEXTUEL DU CODEBOOK (Top 3 tâches par code) ---")
for code_idx in range(num_latents):
    df_code = df[df['Code'] == code_idx]
    if len(df_code) == 0:
        print(f"Code #{code_idx:02d} : ❌ Code mort (jamais activé dans ce split)")
        continue
        
    # Calcul des proportions des tâches pour ce code spécifique
    top_tasks = df_code['Task'].value_counts(normalize=True).head(3)
    task_str = ", ".join([f"'{task}' ({pct*100:.1f}%)" for task, pct in top_tasks.items()])
    print(f"Code #{code_idx:02d} -> Spécialisé dans : {task_str}")

# --- 4. VISUALISATION GRAPHIQUE (Heatmap de Contingence) ---
# Si le dataset de test contient trop de tâches uniques, on se focalise sur les 12 plus fréquentes pour garder le graphique lisible
top_12_tasks = df['Task'].value_counts().head(12).index
df_filtered = df[df['Task'].isin(top_12_tasks)]

# Table de contingence normalisée par colonne (pourcentage d'activation du code par tâche)
contingency_matrix = pd.crosstab(df_filtered['Code'], df_filtered['Task'], normalize='columns')

# Assurer que les 16 codes apparaissent sur l'axe Y même s'ils sont peu activés
contingency_matrix = contingency_matrix.reindex(range(num_latents), fillvalue=0)

plt.figure(figsize=(14, 8))
sns.heatmap(contingency_matrix, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Probabilité du Code sachant la Tâche'})
plt.title("Matrice d'Alignement ActVAE / EgoDex Tasks (Top 12 Tâches)")
plt.xlabel("Tâches EgoDex (Dossiers)")
plt.ylabel("Index des Mots du Codebook (0-15)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

            

        


