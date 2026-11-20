import cv2
import torch
import numpy as np

from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from einops import rearrange


def psnr(image, reconstruction):
    return 10 * torch.log10((1.0**2) / torch.nn.functional.mse_loss(reconstruction, image)).item()

def snr(features, reconstruction):
    return 10 * torch.log10(torch.var(features, dim=0, unbiased=False).mean().item() / torch.nn.functional.mse_loss(reconstruction, features)).item()

def delta_psnr(video, recon_video, video_imagination):
    if video is None or recon_video is None or video_imagination is None:
        return 0

    return psnr(video, recon_video) - psnr(video, video_imagination)

def delta_snr(features, recon_features, feature_imagination):
    if features is None or recon_features is None or feature_imagination is None:
        return 0

    return snr(features, recon_features) - snr(features, feature_imagination)

def mse_ratio(features, recon_features, feature_imagination):
    if features is None or recon_features is None or feature_imagination is None:
        return 0

    return torch.nn.functional.mse_loss(recon_features, features[:, 1:]).item() / torch.nn.functional.mse_loss(feature_imagination, features[:, 1:]).item()

def explained_variance(features, reconstructed_features, eps=1e-12):
    '''Compute the explained variance between features and reconstructed features'''
    error_var = torch.var(features - reconstructed_features, dim=0, unbiased=False)
    features_var = torch.var(features, dim=0, unbiased=False)
    ev = 1 - error_var / (features_var + eps)
    return ev.mean().item()

def mean_feature_cosine_similarity(features, reconstructed_features, eps=1e-12):
    if features is None or reconstructed_features is None:
        return 0
    
    dot_product = (features * reconstructed_features).sum(dim=-1)
    magnitude = features.norm(dim=-1) * reconstructed_features.norm(dim=-1)
    cosine_similarity = dot_product / (magnitude + eps)
    return cosine_similarity.mean().item()

def codebook_usage(indices, codebook_size):
    if indices is None or codebook_size is None:
        return 0
    
    values, counts = torch.unique(indices, return_counts=True)
    total_counts = torch.full((codebook_size,), 0, device=indices.device)
    total_counts[values] = counts

    codebook_usage =  torch.mean((total_counts != 0).float())

    return codebook_usage

def good_turing_coverage(indices):
    '''Crude Good Turing: coverage ≈ 1 - (singletons / items_sampled)'''
    if indices is None:
        return 0
    
    values, counts = torch.unique(indices, return_counts=True)
    singletons = (counts == 1).sum().item()
    total = indices.numel()
    return 1 - (singletons / total)

def distinctiveness_cosine(codes):
    '''Measure how distinct the codes are using cosine similarity'''
    if codes is None or codes.shape[0] > 5000:
        return 0

    normalized_vectors = codes / torch.norm(codes, dim=1, keepdim=True)

    similarity_matrix = normalized_vectors @ normalized_vectors.T

    mask = ~torch.eye(codes.shape[0], dtype=bool)
    average_similarity = similarity_matrix[mask].mean()

    return 1.0 - average_similarity

def compute_optical_flow(first_frame, second_frame):
    print(f"frames shape: {first_frame.shape}, {second_frame.shape}")
    first_frame = (first_frame * 255).transpose(1, 3).transpose(1, 2).to("cpu", dtype=torch.uint8).numpy()
    second_frame = (second_frame * 255).transpose(1, 3).transpose(1, 2).to("cpu", dtype=torch.uint8).numpy()

    flows = []

    for i in range(first_frame.shape[0]):
        first_frame_gray = cv2.cvtColor(first_frame[i], cv2.COLOR_RGB2GRAY)
        second_frame_gray = cv2.cvtColor(second_frame[i], cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(first_frame_gray, second_frame_gray, None, 
                                                pyr_scale=0.5, levels=3, winsize=15, 
                                                iterations=3, poly_n=5, poly_sigma=1.2, 
                                                flags=0)
        
        flows.append(flow)
    
    flows = np.stack(flows, axis=0)

    return flows

def compute_flow_difference(flow1, flow2):
    flow1_u = flow1[:, :, :, 0]
    flow1_v = flow1[:, :, :, 1]

    flow1_avg_u = np.mean(flow1_u, axis=(1, 2))
    flow1_avg_v = np.mean(flow1_v, axis=(1, 2))
    flow1_avg_magnitude = np.sqrt(flow1_avg_u**2 + flow1_avg_v**2)

    flow2_u = flow2[:, :, :, 0]
    flow2_v = flow2[:, :, :, 1]

    flow2_avg_u = np.mean(flow2_u, axis=(1, 2))
    flow2_avg_v = np.mean(flow2_v, axis=(1, 2))
    flow2_avg_magnitude = np.sqrt(flow2_avg_u**2 + flow2_avg_v**2)

    cos_sim = np.mean((flow1_avg_u * flow2_avg_u + flow1_avg_v * flow2_avg_v) / (flow1_avg_magnitude * flow2_avg_magnitude + 1e-12))
    mag_diff = np.mean(np.abs(flow1_avg_magnitude - flow2_avg_magnitude))

    return cos_sim, mag_diff

def average_optical_flow_difference_same_action(batch, same_action_recon):
    flows = compute_optical_flow(batch[:, 0], same_action_recon[:, 0])

    original_flow = flows[:flows.shape[0]//2]
    same_action_flow = flows[flows.shape[0]//2:]

    cos_sim, mag_diff = compute_flow_difference(original_flow, same_action_flow)

    return cos_sim, mag_diff

def delta_average_optical_flow_difference(batch, recon_video):
    if recon_video.dim() == 5:
        recon_video = recon_video[:, 0]  # [B, C, H, W]

    original_flow = compute_optical_flow(batch[:, 0], batch[:, 1])
    reconstruction_flow = compute_optical_flow(batch[:, 0], recon_video)

    cos_sim, mag_diff = compute_flow_difference(original_flow, reconstruction_flow)
    return cos_sim, mag_diff

def _compute_metrics(metrics, batch, recon_video, features, recon_features, indices, feature_imagination, video_imagination, codebook_size, codebook, same_action_recon):
    ret = {}

    # Image metrics
    if metrics["delta_psnr"] and recon_video is not None:
        delta_psnr_value = delta_psnr(batch[:, 1:], recon_video, video_imagination)
        ret["delta_psnr"] = delta_psnr_value

    if metrics["ssim"] and recon_video is not None:
        ssim_score = ssim(rearrange(recon_video, 'b t c h w -> (b t) c h w'), rearrange(batch[:, 1:], 'b t c h w -> (b t) c h w'))
        ret["ssim"] = ssim_score

    if metrics["avg_optical_flow_diff"] and same_action_recon is not None:
        avg_optical_flow_cos_diff_same_action, avg_optical_flow_mag_diff_same_action = average_optical_flow_difference_same_action(batch, same_action_recon)
        ret["avg_optical_flow_cos_diff_same_action"] = avg_optical_flow_cos_diff_same_action
        ret["avg_optical_flow_mag_diff_same_action"] = avg_optical_flow_mag_diff_same_action

    if metrics["avg_optical_flow_diff"] and recon_video is not None:
        avg_optical_flow_cos_diff_recon, avg_optical_flow_mag_diff_recon = delta_average_optical_flow_difference(batch, recon_video)
        ret["avg_optical_flow_cos_diff_recon"] = avg_optical_flow_cos_diff_recon
        ret["avg_optical_flow_mag_diff_recon"] = avg_optical_flow_mag_diff_recon

    if metrics["avg_optical_flow_diff"] and video_imagination is not None:
        avg_optical_flow_cos_diff_imag, avg_optical_flow_mag_diff_imag = delta_average_optical_flow_difference(batch, video_imagination)
        ret["avg_optical_flow_cos_diff_imag"] = avg_optical_flow_cos_diff_imag
        ret["avg_optical_flow_mag_diff_imag"] = avg_optical_flow_mag_diff_imag

    # Feature metrics
    if metrics["feature_l1_loss"]:
        feature_l1_loss = torch.nn.functional.l1_loss(recon_features, features[:, 1:])
        ret["feature_l1_loss"] = feature_l1_loss

    if metrics["explained_variance"]:
        ev = explained_variance(features[:, 1:], recon_features)
        ret["explained_variance"] = ev

    if metrics["feature_cosine_similarity"]:
        feature_cosine_similarity = mean_feature_cosine_similarity(features[:, 1:], recon_features)
        ret["feature_cosine_similarity"] = feature_cosine_similarity

    if metrics["mse_ratio"]:
        mse_ratio_value = mse_ratio(features, recon_features, feature_imagination)
        ret["mse_ratio"] = mse_ratio_value              # The higher the better, should at least be higher than 1 | 1 is bad, below 1 is weird

    if metrics["delta_snr"]:
        delta_snr_value = delta_snr(features[:, 1:], recon_features, feature_imagination)
        ret["delta_snr"] = delta_snr_value

    if metrics["feature_imagination_and_reconstruction_cosine_similarity"]:
        imagination_cosine_similarity = mean_feature_cosine_similarity(recon_features, feature_imagination)
        ret["feature_imagination_and_reconstruction_cosine_similarity"] = imagination_cosine_similarity

    # Codebook metrics
    if metrics["codebook_usage"]:
        codebook_usage_value = codebook_usage(indices, codebook_size)
        ret["codebook_usage"] = codebook_usage_value

    if metrics["good_turing_coverage"]:
        good_turing_coverage_value = good_turing_coverage(indices)
        ret["good_turing_coverage"] = good_turing_coverage_value

    if metrics["distinctiveness_cosine"]:
        distinctiveness_cosine_value = distinctiveness_cosine(codebook)
        ret["distinctiveness_cosine"] = distinctiveness_cosine_value

    return ret

# ----------------------------------------------------------------------------------------------------------------------
# Cycle-consistance des actions latentes (Garrido et al. 2026, "Learning Latent Action World Models In The Wild",
# arXiv 2601.05230, figure 5 b)
#
# On infère les actions de la vidéo A, on les applique à une autre vidéo B (déroulé autorégressif depuis sa première
# frame), on ré-infère les actions sur la vidéo B ainsi prédite, puis on les applique à A. Si les actions se transfèrent,
# l'erreur de prédiction sur A avec les actions « cycliques » reste proche de celle avec les actions d'origine.
# Fonctionne pour LAM et TactileLAM (mêmes encode / vq / decode / feature_extractor).
# ----------------------------------------------------------------------------------------------------------------------

def _is_pixel_extractor(model):
    from lam.feature_extractor.base import Identity
    return isinstance(model.feature_extractor, Identity)


def _encode_features(model, features):
    """Encodeur + VQ sur une suite de features (B, T, P, D), registres jamais masqués (comme à l'évaluation).
    Renvoie les actions quantifiées (B, T-1, D), leurs codes (B, T-1) et les registres de sortie (ou None)."""
    B, T, P, _ = features.shape
    R = model.num_register_tokens
    registers = model.register_tokens.expand(B, R, -1).unsqueeze(1).expand(B, T, R, -1) if R else None
    mask = torch.zeros((B, T, R + P), dtype=torch.bool, device=features.device)
    logits, readout = model.encode(features, registers=registers, attention_mask=mask)
    tokens, indices, _ = model.vq.quantize(readout[:, 1:])
    return tokens, indices, (logits[:, 1:, :R] if R else None)


def _decode(model, features, tokens, registers):
    """Décodeur : features des frames d'entrée (B, t, P, D) + une action par frame -> tokens prédits (B, t, P, D)."""
    B, t, P, _ = features.shape
    R = model.num_register_tokens
    mask = torch.zeros((B, t, R + P), dtype=torch.bool, device=features.device)
    return model.decode(features, actions=tokens, registers=registers, attention_mask=mask)[:, :, R:]


def _prediction_error(model, predicted, images, features):
    """MSE par clip (B,) dans l'espace de la loss d'entraînement : pixels avec Identity, features avec DINO.
    predicted : tokens prédits des frames 1..T-1 ; images, features : la vidéo réelle (frames 0..T-1)."""
    if _is_pixel_extractor(model):
        video = model.feature_extractor._post_process(predicted)
        return ((video.float() - images[:, 1:].float()) ** 2).mean((1, 2, 3, 4))
    return ((predicted.float() - features[:, 1:].float()) ** 2).mean((1, 2, 3))


def rollout(model, first_features, tokens, registers=None):
    """Déroulé autorégressif : depuis les features de la frame 0 (B, P, D), applique les actions (B, T-1, D) une à une,
    chaque frame prédite servant d'entrée à l'étape suivante. Renvoie les features d'entrée de la vidéo prédite, frame 0
    comprise (B, T, P, D) : avec Identity, chaque prédiction est remise en image puis ré-encodée par l'extracteur, pour
    retrouver l'espace d'entrée de l'encodeur."""
    inputs = [first_features]
    for t in range(tokens.shape[1]):
        context = torch.stack(inputs, dim=1)                                           # (B, t+1, P, D)
        reg = registers[:, :t + 1] if registers is not None else None
        predicted = _decode(model, context, tokens[:, :t + 1], reg)[:, -1]             # (B, P, D) la frame t+1
        if _is_pixel_extractor(model):
            image = model.feature_extractor._post_process(predicted[:, None])          # (B, 1, C, H, W)
            predicted = model.feature_extractor(image)[:, 0]
        inputs.append(predicted.to(first_features.dtype))
    return torch.stack(inputs, dim=1)


def _random_tokens(model, tokens):
    """Actions tirées au hasard : codes uniformes du codebook, ou actions du batch permutées sans codebook."""
    if model.vq.get_codebook() is None:
        return tokens[torch.randperm(tokens.shape[0], device=tokens.device)]
    indices = torch.randint(0, model.vq.codebook_size, tokens.shape[:2], device=tokens.device)
    return model.vq.dequantize(indices).to(tokens.dtype)


@torch.no_grad()
def cycle_consistency(model, images_a, images_b):
    """Cycle-consistance A -> B -> A des actions latentes (figure 5 b de arXiv 2601.05230).

    images_a, images_b : (B, T, C, H, W), deux vidéos différentes par ligne du batch.
      1. z_A = actions inférées sur A
      2. B' = déroulé de z_A depuis la première frame de B
      3. z_cycle = actions ré-inférées sur B'
      4. prédiction de A (frames réelles t -> t+1, comme à l'entraînement) avec z_A, avec z_cycle, et avec des actions
         au hasard (référence : aucune information transférée)
    Renvoie, par clip (B,) : les trois erreurs, et l'accord des codes z_A / z_cycle (fraction des transitions où le code
    ré-inféré est identique ; NaN sans codebook).
    """
    features_a, features_b = model.feature_extractor(images_a), model.feature_extractor(images_b)
    tokens_a, indices_a, registers_a = _encode_features(model, features_a)
    _, _, registers_b = _encode_features(model, features_b)        # les registres décrivent la scène B, pas l'action

    transferred = rollout(model, features_b[:, 0], tokens_a, registers_b)                 # B' (B, T, P, D)
    tokens_cycle, indices_cycle, _ = _encode_features(model, transferred)

    context = features_a[:, :-1]
    errors = {name: _prediction_error(model, _decode(model, context, tokens.to(context.dtype), registers_a), images_a, features_a)
              for name, tokens in (('original', tokens_a), ('cycle', tokens_cycle), ('random', _random_tokens(model, tokens_a)))}
    if indices_a is not None and indices_cycle is not None and indices_a.dim() == 2:
        agreement = (indices_a == indices_cycle).float().mean(1)
    else:
        agreement = torch.full_like(errors['original'], float('nan'))
    return dict(error_original=errors['original'], error_cycle=errors['cycle'], error_random=errors['random'],
                code_agreement=agreement, indices_original=indices_a, indices_cycle=indices_cycle)


def cycle_consistency_summary(error_original, error_cycle, error_random, code_agreement, indices_original=None, **_):
    """Agrège les sorties de cycle_consistency (concaténées sur l'évaluation, tableaux numpy).

    - hausse_erreur : erreur(cycle) - erreur(origine), la quantité de la figure 5 b (0 = transfert parfait)
    - hausse_relative : hausse / (erreur(hasard) - erreur(origine)) ; 0 = parfait, 1 = pas mieux que des actions au
      hasard. Rend les modèles comparables malgré des erreurs absolues différentes
    - accord_des_codes, et l'accord attendu par hasard sum_k p_k^2 avec la fréquence p_k des codes
    """
    mean = lambda x: float(np.nanmean(x))
    psnr_of = lambda mse: float(10 * np.log10(1.0 / max(mean(mse), 1e-12)))
    gap, random_gap = mean(error_cycle) - mean(error_original), mean(error_random) - mean(error_original)
    summary = dict(erreur_origine=mean(error_original), erreur_cycle=mean(error_cycle), erreur_hasard=mean(error_random),
                   psnr_origine=psnr_of(error_original), psnr_cycle=psnr_of(error_cycle), psnr_hasard=psnr_of(error_random),
                   hausse_erreur=gap, hausse_relative=gap / random_gap if random_gap > 0 else float('nan'),
                   accord_des_codes=mean(code_agreement))
    if indices_original is not None:
        _, counts = np.unique(indices_original, return_counts=True)
        summary['accord_des_codes_hasard'] = float(((counts / counts.sum()) ** 2).sum())
    return summary
