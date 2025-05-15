import pickle
import argparse
import numpy as np

import umap


def main(args):
    with open(args.filename, "rb") as file:
        latent_to_real_actions = pickle.load(file)

    action_map = latent_to_real_actions[args.mode]           # List[Tensors] Shape: N, (C, D)

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

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(X)

    umap_results = {"umap_features": embedding,
                    "labels": y}

    with open(args.output, "wb") as file:
        pickle.dump(umap_results, file)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="UMAP")
    argparser.add_argument("--filename", type=str, default="latent_to_real_actions_1o6l5rlz_fractal20220817_data.pkl", help="Input filename")
    argparser.add_argument("--mode", type=str, default="only_max_context", choices=["only_min_context", "only_max_context", "all_contexts"], help="Mode")
    argparser.add_argument("--output", type=str, default=f"umap_features_latent_to_real_actions_1o6l5rlz_fractal20220817_data_only_max_context.pkl", help="Output filename")

    args = argparser.parse_args()
    main(args)