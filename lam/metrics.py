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