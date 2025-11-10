import torch
import pickle
import argparse

from sklearn.metrics import pairwise_distances
from skbio.stats.distance import DistanceMatrix, permanova


def main(args):
    with open(args.filename, "rb") as file:
        action_map = pickle.load(file)                          # List[Tensors] Shape: N, (C, D)

    all_vectors = torch.cat(action_map).numpy()               
    grouping_labels = [f"<ACT_{idx}>" for idx, t in enumerate(action_map) for _ in range(t.shape[0])]

    print(f"Total vectors: {len(all_vectors)}")
    print(f"Total labels: {len(grouping_labels)}")
    print("-" * 30)

    dist_matrix_raw = pairwise_distances(all_vectors, metric='cosine')
    distance_matrix = DistanceMatrix(dist_matrix_raw, validate=False)

    results = permanova(distance_matrix, grouping_labels, permutations=999)

    print("PERMANOVA Results:")
    print(results)

    if results['p-value'] < 0.05:
        print(f"\nThe p-value {results['p-value']} is significant (p < 0.05).")
    else:
        print(f"\nThe p-value {results['p-value']} is not significant (p >= 0.05).")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="t-SNE")
    argparser.add_argument("--filename", type=str, default="latent_to_real_proprio_diff_kz827dan_fractal_100.pkl", help="Input filename")

    args = argparser.parse_args()
    main(args)