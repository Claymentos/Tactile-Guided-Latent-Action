import pickle
import argparse
import numpy as np
import torch

from tqdm import tqdm

def cosine_similarity(A, B):
        '''Measure how similar the actions are using cosine similarity'''
        normalized_A = A / (torch.norm(A, dim=1, keepdim=True) + 1e-12)
        normalized_B = B / (torch.norm(B, dim=1, keepdim=True) + 1e-12)

        similarity_matrix = normalized_A @ normalized_B.T

        if A.shape == B.shape and (A == B).all():
            mask = ~torch.eye(A.shape[0], dtype=bool)
            average_similarity = similarity_matrix[mask].mean()
        else:
            average_similarity = similarity_matrix.mean()

        return average_similarity.item()

def main(args):
    with open(args.filename, "rb") as file:
        action_map = pickle.load(file)                          # List[Tensors] Shape: N, (C, D)

    print(f"Vocabulary size: {len(action_map)}")                # N
    print(f"Action dimension: {action_map[0].shape[-1]}")       # D

    within_similarities = [None] * len(action_map)
    between_similarities = [[] for _ in range(len(action_map))]

    for idx, latent_actions in tqdm(enumerate(action_map)):
        if latent_actions.shape[0] < 2:
            continue

        within_similarities[idx] = cosine_similarity(latent_actions, latent_actions)

        for jdx, other_latent_actions in enumerate(action_map):
            if other_latent_actions.shape[0] < 2 or jdx == idx:
                continue

            between_similarities[idx].append(cosine_similarity(latent_actions, other_latent_actions))

        # print(f"Latent action {idx} average within similarity = {within_similarities[idx]} | average between similarity = {np.mean(between_similarities[idx])} | Support: {latent_actions.shape[0]}")

    print(f"Latent action average within similarity = {np.mean([x for x in within_similarities if x is not None])} | average between similarity = {np.mean([x for x in between_similarities if len(x) > 0])}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="t-SNE")
    argparser.add_argument("--filename", type=str, default="latent_to_real_proprio_diff_kz827dan_fractal_100.pkl", help="Input filename") # kz827dan

    args = argparser.parse_args()
    main(args)