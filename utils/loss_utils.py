#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchvision.transforms import functional as transformsF
from utils.general_utils import build_rotation


def l1_loss(network_output, gt):
    return torch.abs(network_output - gt).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def _fft_band_loss(network_output, gt, lowpass_ratio=0.25):
    """Match low/high frequency amplitudes without unstable phase wrapping."""
    output_gray = (0.299 * network_output[:, 0:1] +
                   0.587 * network_output[:, 1:2] +
                   0.114 * network_output[:, 2:3])
    gt_gray = (0.299 * gt[:, 0:1] +
               0.587 * gt[:, 1:2] +
               0.114 * gt[:, 2:3])
    output_fft = torch.fft.fftshift(torch.fft.fft2(output_gray, norm="ortho"), dim=(-2, -1))
    gt_fft = torch.fft.fftshift(torch.fft.fft2(gt_gray, norm="ortho"), dim=(-2, -1))
    height, width = output_gray.shape[-2:]
    yy = torch.linspace(-1.0, 1.0, height, device=output_gray.device)
    xx = torch.linspace(-1.0, 1.0, width, device=output_gray.device)
    radius = torch.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
    lowpass = torch.exp(-(radius / max(lowpass_ratio, 1e-3)) ** 2)
    highpass = 1.0 - lowpass
    amplitude_error = torch.abs(torch.abs(output_fft) - torch.abs(gt_fft))
    low_loss = (amplitude_error * lowpass).mean()
    high_loss = (amplitude_error * highpass).mean()
    return low_loss, high_loss


def progressive_frequency_loss(network_output, gt, iteration, start_iter,
                                end_iter, weight=0.01, lowpass_ratio=0.25):
    """Coarse-to-fine Fourier supervision inspired by FreGS (CVPR 2024)."""
    if weight <= 0.0 or iteration < start_iter or iteration >= end_iter:
        return network_output.new_zeros(())
    progress = (iteration - start_iter) / max(end_iter - start_iter, 1)
    low_loss, high_loss = _fft_band_loss(
        network_output, gt, lowpass_ratio=lowpass_ratio
    )
    high_weight = min(1.0, max(0.0, progress))
    return weight * (low_loss + high_weight * high_loss)


# ----------------------------------------------------------------------
# MeshGS-style surface alignment losses
# (Choi et al., "MeshGS: Adaptive Mesh-Aligned Gaussian Splatting for
# High-Quality Rendering", ACCV 2024)
# ----------------------------------------------------------------------
# Reference equations from the paper:
#   L_nc   = sum (1 - n_i . n_f)                                    (5)
#   L_sc   = sum (lam_min * |min(s)| + lam_max * |max(s) - rho|)     (6)
#   s_i    = P_i + ((P_i - v_i) . f_j) * f_j                       (7)
#   L_proj = sum ||s_i - P_i||_2                                   (7)
# ----------------------------------------------------------------------

def mesh_gaussian_normal(rotations, scales):
    """
    Compute per-Gaussian world-space normal as the shortest scale axis
    transformed by the rotation matrix.

    Args:
        rotations: (N, 4) quaternions (w, x, y, z).
        scales:    (N, 3) per-Gaussian scaling factors (post-activation).

    Returns:
        normals: (N, 3) unit normals in world space.
    """
    # Shortest scale axis in local frame
    min_axis_idx = torch.argmin(scales, dim=-1)            # (N,)
    # Convert quat -> rotation matrix (B,3,3)
    R = build_rotation(rotations)                          # (N, 3, 3)
    # Build local axes: [e_x, e_y, e_z]  then pick the shortest
    device = R.device
    e_local = torch.eye(3, device=device).unsqueeze(0).expand(R.shape[0], -1, -1)
    # (N, 3, 3): columns are local axis vectors
    local_axes = e_local
    # World axes = R @ local_axes
    world_axes = torch.bmm(R, local_axes)                  # (N, 3, 3)
    # Select column corresponding to shortest scale axis
    min_idx = min_axis_idx.view(-1, 1, 1).expand(-1, 3, 1)
    normals = torch.gather(world_axes, 2, min_idx).squeeze(-1)  # (N, 3)
    normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
    return normals


def meshgs_normal_consistency_loss(gaussians, face_id, face_normals,
                                    weight=0.1, start_iter=0, end_iter=10**9):
    """
    L_nc = sum (1 - n_i . n_f) for tightly-bound Gaussians.

    All Gaussians are treated as tightly-bound in this implementation
    (we don't compute distances to the surface every step; the
    2D-patch sampler already places points on the mesh).

    Args:
        gaussians: GaussianModel with _rotation, get_scaling
        face_id:   (N,) long tensor mapping Gaussian -> face index
        face_normals: (F, 3) tensor of per-face unit normals
        weight, start_iter, end_iter: scheduler
    """
    if weight <= 0.0:
        return gaussians.get_scaling.new_zeros(())
    n_g = face_id.shape[0]
    if n_g == 0:
        return gaussians.get_scaling.new_zeros(())
    normals = mesh_gaussian_normal(gaussians._rotation, gaussians.get_scaling)
    target_normals = face_normals[face_id]                 # (N, 3)
    cos = (normals * target_normals).sum(dim=-1).clamp(-1.0, 1.0)
    return weight * (1.0 - cos).mean()


def meshgs_scale_regularization(gaussians, rho=0.05,
                                 lam_min=0.01, lam_max=0.01,
                                 start_iter=0, end_iter=10**9):
    """
    L_scale = lam_min * |min(s1,s2,s3)| + lam_max * |max(s1,s2,s3) - rho|

    Encourages Gaussians to be flat (small minimum scale) and not
    grow too large along the surface (max scale ~ rho, the average
    face circumradius).
    """
    if lam_min <= 0.0 and lam_max <= 0.0:
        return gaussians.get_scaling.new_zeros(())
    s = gaussians.get_scaling                            # (N, 3)
    s_min = s.min(dim=-1).values                          # (N,)
    s_max = s.max(dim=-1).values                          # (N,)
    loss = lam_min * s_min.abs() + lam_max * (s_max - rho).abs()
    return loss.mean()


def meshgs_projection_loss(gaussians, face_id, face_verts,
                            weight=0.05, max_dist=None,
                            start_iter=0, end_iter=10**9):
    """
    L_proj = ||s_i - P_i||_2
    where  s_i = P_i + ((P_i - v_i) . f_j) * f_j
    is the closest point on the face plane to Gaussian center P_i.

    We use the face centroid v_i as the in-plane reference; this is
    a standard approximation used by MeshGS (eq. 7) for fast projection.

    Args:
        gaussians: GaussianModel with get_xyz
        face_id:   (N,) long tensor, face index for each Gaussian
        face_verts: (F, 3, 3) tensor, three vertex positions per face
        weight:    loss weight
        max_dist:  if set, only apply loss to Gaussians whose current
                   distance to the face plane exceeds this threshold
                   (acts as a soft "tightly-bound" mask).  None = apply
                   to all.
    """
    if weight <= 0.0:
        return gaussians.get_xyz.new_zeros(())
    xyz = gaussians.get_xyz                                # (N, 3)
    fid = face_id
    if fid.shape[0] == 0:
        return xyz.new_zeros(())
    # Face centroid and normal
    v0 = face_verts[fid, 0]                                # (N, 3)
    v1 = face_verts[fid, 1]
    v2 = face_verts[fid, 2]
    centroid = (v0 + v1 + v2) / 3.0                        # (N, 3)
    e1 = v1 - v0
    e2 = v2 - v0
    fn = torch.cross(e1, e2, dim=-1)                       # (N, 3)
    fn = fn / (fn.norm(dim=-1, keepdim=True) + 1e-8)
    # signed distance from xyz to face plane
    diff = xyz - centroid                                  # (N, 3)
    d = (diff * fn).sum(dim=-1)                            # (N,) signed dist
    abs_d = d.abs()
    if max_dist is not None and max_dist > 0:
        mask = (abs_d > max_dist).float()
        if mask.sum() == 0:
            return xyz.new_zeros(())
        loss = (abs_d * mask).sum() / (mask.sum() + 1e-8)
    else:
        loss = abs_d.mean()
    return weight * loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def blur_2d(tensor, kernel_size):
    """Apply Gaussian blur to a [B, C, H, W] tensor."""
    if kernel_size <= 1:
        return tensor
    return transformsF.gaussian_blur(tensor, kernel_size, sigma=kernel_size / 3.0)


def coarse_to_fine_blur_loss(rendered, gt, iteration, start_iter, fade_iter, kernel_init, weight):
    """
    Opti3DGS-style coarse-to-fine frequency control.
    Train on blurred GT → forces network to learn coarse structure first.
    Step-function fade: large blur until fade_iter, then sharp GT.
    """
    if weight <= 0.0 or iteration < start_iter:
        return rendered.new_zeros(())
    if iteration >= fade_iter:
        return rendered.new_zeros(())
    kernel_size = kernel_init
    blurred_gt = blur_2d(gt, kernel_size)
    return weight * torch.abs(rendered - blurred_gt).mean()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)

