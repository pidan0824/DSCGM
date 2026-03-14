#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from utils.dcputil import quat2mat
import open3d


class MLPHead(nn.Module):
    def __init__(self, args):
        super(MLPHead, self).__init__()
        emb_dims = args.emb_dims
        self.emb_dims = emb_dims
        self.nn = nn.Sequential(nn.Linear(emb_dims * 2, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        embedding = torch.cat((src_embedding, tgt_embedding), dim=1)
        embedding = self.nn(embedding.max(dim=-1)[0])
        rotation = self.proj_rot(embedding)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        translation = self.proj_trans(embedding)
        return quat2mat(rotation), translation


def SVD(src, src_corr):
    # (b, 3, n)
    device = src.device
    batch_size = src.shape[0]
    src_centered = src - src.mean(dim=2, keepdim=True)
    src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

    H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous()).cpu()
    R = []
    for i in range(src.size(0)):
        u, s, v = torch.svd(H[i])
        r = torch.matmul(v, u.transpose(1, 0)).contiguous()
        r_det = torch.det(r).item()
        diag = torch.from_numpy(np.array([[1.0, 0, 0],
                                          [0, 1.0, 0],
                                          [0, 0, r_det]]).astype('float32')).to(v.device)
        r = torch.matmul(torch.matmul(v, diag), u.transpose(1, 0)).contiguous()
        R.append(r)

    R = torch.stack(R, dim=0).to(device)
    t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
    return R, t.view(batch_size, 3)


def SVDslover1(src_o, tgt_o, s_perm_mat):
    """Compute rigid transform (R, t) from src_o to tgt_o using weighted SVD.

    Args:
        src_o: (B, M, 3) source points.
        tgt_o: (B, N, 3) target points.
        s_perm_mat: (B, M, N) soft correspondence matrix.

    Returns:
        R: (B, 3, 3), t: (B, 3).
    """
    weights1 = torch.sum(s_perm_mat, dim=2)
    weights_normalized1 = weights1[..., None] / (torch.sum(weights1[..., None], dim=1, keepdim=True) + 1e-5)
    weights2 = torch.sum(s_perm_mat, dim=1)  # [B, N]
    weights_normalized2 = weights2[:, :, None] / torch.sum(weights2, dim=1, keepdim=True)[:, None, :]  # [B, N, 1]
    centroid_src_o = torch.sum(src_o * weights_normalized1, dim=1)
    centroid_tgt_o = torch.sum(tgt_o * weights_normalized2, dim=1)
    src_o_centered = src_o - centroid_src_o[:, None, :]
    tgt_o_centered = tgt_o - centroid_tgt_o[:, None, :]
    cov = src_o_centered.transpose(-2, -1) @ (tgt_o_centered * weights_normalized2)

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    R = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(R) > 0)

    # Compute translation (uncenter centroid)
    t = -R @ centroid_src_o[:, :, None] + centroid_tgt_o[:, :, None]

    return R, t.view(s_perm_mat.shape[0], 3)


def SVDslover(src_o, tgt_o, s_perm_mat, src1, src_corr1, src2, src_corr2):
    """Compute rigid transform using weighted SVD with point and edge correspondences.

    Returns:
        R: (B, 3, 3), t: (B, 3).
    """
    weights = torch.sum(s_perm_mat, dim=2)
    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + 1e-5)
    centroid_src_o = torch.sum(src_o * weights_normalized, dim=1)
    centroid_tgt_o = torch.sum(tgt_o * weights_normalized, dim=1)
    src_o_centered = src_o - centroid_src_o[:, None, :]
    tgt_o_centered = tgt_o - centroid_tgt_o[:, None, :]
    cov_point = src_o_centered.transpose(-2, -1) @ (tgt_o_centered * weights_normalized)

    src_centered1 = src1 - src1.mean(dim=2, keepdim=True)
    src_centered2 = src2 - src2.mean(dim=2, keepdim=True)

    src_corr_centered1 = src_corr1 - src_corr1.mean(dim=2, keepdim=True)
    src_corr_centered2 = src_corr2 - src_corr2.mean(dim=2, keepdim=True)

    H1 = torch.matmul(src_centered1, src_corr_centered1.transpose(2, 1).contiguous())
    H2 = torch.matmul(src_centered2, src_corr_centered2.transpose(2, 1).contiguous())
    cov_edge = H1 + H2

    cov = cov_point + cov_edge

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    R = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(R) > 0)

    # Compute translation (uncenter centroid)
    t = -R @ centroid_src_o[:, :, None] + centroid_tgt_o[:, :, None]
    return R, t.view(s_perm_mat.shape[0], 3)


def RANSACSVDslover(src_o, tgt_o, s_perm_mat):
    """Compute rigid transform with RANSAC-based outlier rejection.

    Returns:
        R: (B, 3, 3), t: (B, 3), s_perm_mat_re: (B, M, N) filtered correspondence.
    """
    weights = torch.sum(s_perm_mat, dim=2)
    weights_inlier = torch.where(weights==1)

    src_o0 = [src_o[i, list((weights_inlier[1][weights_inlier[0]==i]).cpu().numpy())].cpu().numpy()
              for i in range(s_perm_mat.shape[0])]
    tgt_o0 = [tgt_o[i, list((weights_inlier[1][weights_inlier[0]==i]).cpu().numpy())].cpu().numpy()
              for i in range(s_perm_mat.shape[0])]
    R = torch.zeros((s_perm_mat.shape[0], 3, 3)).to(s_perm_mat)
    t = torch.zeros((s_perm_mat.shape[0], 3, 1)).to(s_perm_mat)
    s_perm_mat_re = torch.zeros_like(s_perm_mat).to(s_perm_mat)
    for i in range(len(src_o0)):
        src_o0i = open3d.geometry.PointCloud()
        tgt_o0i = open3d.geometry.PointCloud()
        src_o0i.points = open3d.utility.Vector3dVector(src_o0[i])
        tgt_o0i.points = open3d.utility.Vector3dVector(tgt_o0[i])
        corr = open3d.utility.Vector2iVector(np.arange(src_o0[i].shape[0])[:,None].repeat(2, axis=1))
        reg_result = open3d.pipelines.registration.registration_ransac_based_on_correspondence(src_o0i, tgt_o0i, corr, 0.2)
        transformation_np = np.array(reg_result.transformation)
        R[i] = torch.from_numpy(transformation_np[:3, :3]).to(s_perm_mat)
        t[i] = torch.from_numpy(transformation_np[:3, 3])[:,None].to(s_perm_mat)
        corr_re = np.asarray(reg_result.correspondence_set)
        s_perm_mat_re[i,corr_re[:,0]] = s_perm_mat[i,corr_re[:,0]]

    return R, t.view(s_perm_mat.shape[0], 3), s_perm_mat_re
