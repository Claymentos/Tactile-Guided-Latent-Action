import pickle
import numpy as np

# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE

FILENAME = "latent_to_real_actions_1o6l5rlz_fractal20220817_data.pkl"
MODE = "only_max_context" # "only_min_context"  "only_max_context" "all_contexts"
OUTPUT = f"tsne_features_latent_to_real_actions_1o6l5rlz_fractal20220817_data_{MODE}.pkl"


with open(FILENAME, "rb") as file:
    latent_to_real_actions = pickle.load(file)


action_map = latent_to_real_actions[MODE]           # List[Tensors] Shape: N, (C, D)

print(f"Vocabulary size: {len(action_map)}")                # N
print(f"Action dimension: {action_map[0].shape[-1]}")       # D

# Turn into two numpy arrays
# X: Shape (M, D) where M is the sum of all diffent C
# y: Shape (M, ) where M is the sum of all diffent C
X = []
y = []

for i, action in enumerate(action_map):
    X.append(action.cpu().numpy())
    y.extend([i] * action.shape[0])

X = np.concatenate(X, axis=0)
y = np.array(y)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

tsne = TSNE(n_components=2, verbose=2, n_jobs=8)
tsne_results = tsne.fit_transform(X)

tsne_results = {"tsne_features": tsne_results,
                "labels": y}

with open(OUTPUT, "wb") as file:
    pickle.dump(tsne_results, file)