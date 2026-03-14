import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")
from models.feature_extract import PointNet, DGCNN


def mask_point(mask_idx, points):
    device = points.device
    batch_size = points.shape[0]
    points = points.permute(0, 2, 1)
    mask_idx = mask_idx.reshape(batch_size, -1, 1)
    new_pcs = points * mask_idx
    new_points = []

    for new_pc in new_pcs:
        temp = new_pc[:, ...] == 0
        temp = temp.cpu()
        idx = np.argwhere(temp.all(axis=1))
        new_point = np.delete(new_pc.cpu().detach().numpy(), idx, axis=0)

        new_points.append(new_point)

    new_points = np.array(new_points)
    new_points = torch.from_numpy(new_points).to(device)
    return new_points.permute(0, 2, 1)


def mask_point0(mask_idx, points):
    batch_size = points.shape[0]
    num_points_per_batch = points.shape[2]
    M=points.shape[1]
    points = points.permute(0, 2, 1)
    mask_idx_expanded = mask_idx.unsqueeze(-1).repeat(1, 1, M)
    masked_points = points * mask_idx_expanded
    non_zero_rows_idx_per_batch = []
    for i in range(batch_size):
        batch_mask = mask_idx[i]
        non_zero_rows_idx_batch = torch.nonzero(batch_mask, as_tuple=False)[:, 0]
        non_zero_rows_idx_per_batch.append(non_zero_rows_idx_batch)
    new_points_list = []
    for i in range(batch_size):
        batch_indices = non_zero_rows_idx_per_batch[i]
        new_points_list.append(masked_points[i, batch_indices])
    min_num_points = min(len(x) for x in new_points_list)
    truncated_points_list = [x[:min_num_points] for x in new_points_list]
    truncated_points = torch.stack(truncated_points_list, dim=0)

    truncated_points = truncated_points.permute(0, 2, 1)

    return truncated_points


def mask_point1(mask_idx, points):
    # masks: [b, n] : Tensor, binary mask with 0 and 1
    # points: [b, n, m] : Tensor
    # return: [b, n2, m] : Tensor, n2 is the number of remaining points per batch

    batch_size = points.shape[0]
    num_points = points.shape[1]
    num_features = points.shape[2]  # m

    # Expand mask to match points shape
    mask_expanded = mask_idx.unsqueeze(-1).expand_as(points)  # [b, n, m]

    # Apply mask
    masked_points = points * mask_expanded

    # Find indices of unmasked points (where mask is 1)
    valid_mask_3d = mask_expanded.bool()

    valid_idx_list = []
    for i in range(batch_size):
        valid_idx_i = valid_mask_3d[i].any(dim=-1).nonzero(as_tuple=False)[:, 0]  # [num_valid_in_batch]
        valid_idx_list.append(valid_idx_i)

    new_points_list = []

    for i in range(batch_size):
        valid_idx = valid_idx_list[i]
        new_points_i = points[i, valid_idx]  # [num_valid_in_batch, m]
        new_points_list.append(new_points_i)

    new_points = torch.nn.utils.rnn.pad_sequence(new_points_list, batch_first=True, padding_value=0)

    return new_points


def mask_point2(mask_idx, points):
    """Mask rows in point cloud and return padded tensor [b, max_valid_rows, n]."""
    batch_size = points.shape[0]
    n = points.shape[1]
    num_valid_rows = mask_idx.sum(dim=1)  # [b,]
    max_num_valid_rows = int(num_valid_rows.max().item())
    masked_points = points.new_zeros(batch_size, max_num_valid_rows, n)
    for batch_idx in range(batch_size):
        mask_this_batch = mask_idx[batch_idx]  # [n,]
        points_this_batch = points[batch_idx]  # [n, n]
        valid_rows_indices = torch.nonzero(mask_this_batch).squeeze(1)  # [num_valid_rows,]
        for row_idx, valid_row in enumerate(valid_rows_indices):
            masked_points[batch_idx, row_idx, :] = points_this_batch[valid_row]

    return masked_points



def mask_point3(mask_idx, points):
    """Mask columns in point cloud [b, M, n] and return [b, M, m] with valid columns."""
    batch_size = points.shape[0]
    num_channels = points.shape[1]
    num_points = points.shape[2]
    if batch_size != mask_idx.shape[0] or num_points != mask_idx.shape[1]:
        raise ValueError("Batch size or number of points mismatch between points and mask_idx")

    if mask_idx.dtype != torch.bool:
        mask_idx = mask_idx.bool()
    mask_expanded = mask_idx.unsqueeze(1).expand(-1, num_channels, -1)  # [b, M, n]
    valid_cols = mask_idx.any(dim=0).nonzero(as_tuple=False).squeeze(1)  # [m,]
    new_points = points[:, :, valid_cols]  # [b, M, m]

    return new_points


def mask_point_cloud(row_mask_idx, col_mask_idx, points):
    """Mask rows and columns in point cloud [b, n, n] and return padded tensor."""
    batch_size = points.shape[0]
    n = points.shape[1]
    num_valid_rows = row_mask_idx.sum(dim=1)  # [b,]
    num_valid_cols = col_mask_idx.sum(dim=0)  # [n,]
    max_num_valid_rows = int(num_valid_rows.max().item())
    valid_cols_indices = torch.nonzero(col_mask_idx.any(dim=0)).squeeze(1)
    num_valid_cols = len(valid_cols_indices)
    masked_points = points.new_zeros(batch_size, max_num_valid_rows, num_valid_cols)
    for batch_idx in range(batch_size):
        row_mask_this_batch = row_mask_idx[batch_idx]  # [n,]
        points_this_batch = points[batch_idx]  # [n, n]
        valid_rows_indices = torch.nonzero(row_mask_this_batch).squeeze(1)  # [num_valid_rows,]
        valid_points_this_batch = points_this_batch[:, valid_cols_indices]  # [n, num_valid_cols]
        for row_idx, valid_row in enumerate(valid_rows_indices):
            masked_points[batch_idx, row_idx, :] = valid_points_this_batch[valid_row]

    return masked_points




def mask_point5(masks, points):
    # masks: [b, n] : Tensor, containing 0s and 1s
    # points: [b, n, 1] : Tensor, note the changed assumption about points's input shape
    # return: [b, n2, 1] : Tensor, where n2 is the number of unmasked points

    batch_size = points.shape[0]
    num_points = points.shape[1]

    # Expand masks to match the shape of points in all dimensions except the last one
    masks_expanded = masks.unsqueeze(-1)  # [b, n, 1]

    # Apply the mask
    masked_points = points * masks_expanded  # [b, n, 1]

    # Find the indices of all masked points (i.e., points that are zero)
    # Since we have only one channel, we can squeeze the last dimension before comparing
    mask_3d = (masked_points.squeeze(-1) == 0)  # [b, n]

    # Get the indices of the remaining points in each batch
    valid_idx = (~mask_3d).nonzero(
        as_tuple=False)  # [num_valid, 2], where the second column is the index of valid points

    # Create a new list to store the processed point clouds
    new_points_list = []

    for i in range(batch_size):
        # For each batch, select the remaining points
        batch_idx = valid_idx[valid_idx[:, 0] == i][:, 1]  # [num_valid_in_batch]
        new_points_i = points[i, batch_idx]  # [num_valid_in_batch, 1]
        new_points_list.append(new_points_i)

        # Convert the list to a tensor and adjust the shape
    new_points = torch.stack(new_points_list)  # [b, num_valid_points, 1]

    return new_points  # [b, num_valid_points, 1]


def mask_point6(mask_idx, points):
    """Mask rows in point cloud [b, n, M] and return padded tensor [b, max_valid_rows, M]."""
    batch_size = points.shape[0]
    n = points.shape[1]
    M = points.shape[2]

    num_valid_rows = mask_idx.sum(dim=1)  # [b,]
    max_num_valid_rows = int(num_valid_rows.max().item())

    masked_points = points.new_zeros(batch_size, max_num_valid_rows, M)

    for batch_idx in range(batch_size):
        mask_this_batch = mask_idx[batch_idx]  # [n,]
        points_this_batch = points[batch_idx]  # [n, M]

        valid_rows_indices = torch.nonzero(mask_this_batch).squeeze(1)

        for row_idx, valid_row in enumerate(valid_rows_indices):
            masked_points[batch_idx, row_idx, :] = points_this_batch[valid_row]

    return masked_points

def mask_point_nodet(mask_idx, points):
    # masks: [b, n] : Tensor, binary mask with 0 and 1
    # points: [b, 3, n] : Tensor
    # return: [b, 3, n] : Tensor, masked points are set to 0
    batch_size = points.shape[0]
    num_points = points.shape[2]

    # Expand mask_idx from [b, n] to [b, 1, n] for broadcasting with points
    mask_idx = mask_idx.unsqueeze(1)

    # Apply mask, setting masked points to 0
    masked_points = points * mask_idx.float()
    return masked_points

def mask_cor(mask_idx, points):
    # masks: [b, n] : Tensor, binary mask with 0 and 1
    # points: [b, n, n] : Tensor
    # return: [b, n2, n2] : Tensor, n2 is the number of unmasked rows and columns
    device = points.device
    batch_size = points.shape[0]
    n = points.shape[1]

    row_mask = mask_idx.unsqueeze(2).expand(batch_size, n, n)
    col_mask = mask_idx.unsqueeze(1).expand(batch_size, n, n)
    combined_mask = row_mask & col_mask
    new_pcs = points * combined_mask.float()
    new_points = []

    for b in range(batch_size):
        # Find indices of unmasked points
        mask = new_pcs[b] != 0
        idx = torch.nonzero(mask, as_tuple=False)

        # Save unmasked points if any exist
        if idx.size(0) > 0:
            new_point = new_pcs[b][idx[:, 0], idx[:, 1]]
            new_points.append(new_point.unsqueeze(0))
        else:
            # Add empty tensor if all points are masked
            new_points.append(torch.tensor([]).float().unsqueeze(0))

    new_points = torch.cat(new_points, dim=0)
    new_points_list = [new_points[i] for i in range(batch_size)]
    new_points_padded = [F.pad(point, (0, n - point.shape[0], 0, n - point.shape[1])) for point in new_points_list]
    new_points = torch.stack(new_points_padded).to(device)

    return new_points


def gather_points(points, inds):
    '''
    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]


def feature_interaction(src_embedding, tar_embedding):
    # embedding: (batch, emb_dims, num_points)
    num_points1 = src_embedding.shape[2]

    simi1 = cos_simi(src_embedding, tar_embedding)  # (num_points1, num_points2)

    src_embedding = src_embedding.permute(0, 2, 1)
    tar_embedding = tar_embedding.permute(0, 2, 1)

    simi_src = nn.Softmax(dim=2)(simi1)
    glob_tar = torch.matmul(simi_src, tar_embedding)
    glob_src = torch.max(src_embedding, dim=1, keepdim=True)[0]
    glob_src = glob_src.repeat(1, num_points1, 1)
    inter_src_feature = torch.cat((src_embedding, glob_tar, glob_src, glob_tar-glob_src), dim=2)
    inter_src_feature = inter_src_feature.permute(0, 2, 1)

    return inter_src_feature


def cos_simi(src_embedding, tgt_embedding):
    # (batch, emb_dims, num_points)
    src_norm = F.normalize(src_embedding, p=2, dim=1)
    tar_norm = F.normalize(tgt_embedding, p=2, dim=1)
    simi = torch.matmul(src_norm.transpose(2, 1).contiguous(), tar_norm)
    return simi


class OverlapNet(nn.Module):
    def __init__(self, n_emb_dims=1024, all_points=1024, src_subsampled_points=716, tgt_subsampled_points=716):
        super(OverlapNet, self).__init__()
        self.emb_dims = n_emb_dims
        self.all_points = all_points
        self.emb_dims1 = int(self.emb_dims / 2)
        self.src_subsampled_points = src_subsampled_points
        self.tgt_subsampled_points = tgt_subsampled_points
        self.emb_nn1 = DGCNN(self.emb_dims1, k=32)
        self.emb_nn2_src = nn.Sequential(
            nn.Conv1d(self.emb_dims1 * 4, self.emb_dims1 * 2, 1), nn.BatchNorm1d(self.emb_dims1 * 2),nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(self.emb_dims1 * 2, self.emb_dims, 1), nn.BatchNorm1d(self.emb_dims),nn.LeakyReLU(negative_slope=0.01),
        )
        self.emb_nn2_tgt = nn.Sequential(
            nn.Conv1d(self.emb_dims1 * 4, self.emb_dims1 * 2, 1), nn.BatchNorm1d(self.emb_dims1 * 2),nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(self.emb_dims1 * 2, self.emb_dims, 1), nn.BatchNorm1d(self.emb_dims),nn.LeakyReLU(negative_slope=0.01),
        )
        self.score_nn_src = nn.Sequential(
            nn.Conv1d(self.emb_dims, 512, 1), nn.BatchNorm1d(512), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(512, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(128, 1, 1), nn.Sigmoid()
        )
        self.score_nn_tgt = nn.Sequential(
            nn.Conv1d(self.emb_dims, 512, 1), nn.BatchNorm1d(512), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(512, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(128, 1, 1), nn.Sigmoid()
        )
        self.mask_src_nn = nn.Sequential(
            nn.Conv1d(716, 1024, 1), nn.BatchNorm1d(1024), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(1024, 512, 1), nn.BatchNorm1d(512), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(512, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(128, 2, 1)
        )
        self.mask_tgt_nn = nn.Sequential(
            nn.Conv1d(716, 1024, 1), nn.BatchNorm1d(1024), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(1024, 512, 1), nn.BatchNorm1d(512), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(512, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(128, 2, 1)
        )

    def forward(self, *input):
        src = input[0]  #
        tgt = input[1]  #
        batch_size = src.shape[0]

        src_embedding = self.emb_nn1(src)
        tgt_embedding = self.emb_nn1(tgt)
        # Feature fusion
        inter_src_feature = feature_interaction(src_embedding, tgt_embedding)
        inter_tar_feature = feature_interaction(tgt_embedding, src_embedding)
        # Further feature extraction
        src_embedding = self.emb_nn2_src(inter_src_feature)
        tgt_embedding = self.emb_nn2_tgt(inter_tar_feature)
        # Compute scores
        src_score = self.score_nn_src(src_embedding).reshape(batch_size, 1, -1)
        tgt_score = self.score_nn_tgt(tgt_embedding).reshape(batch_size, 1, -1)

        src_score = nn.Softmax(dim=2)(src_score)
        tgt_score = nn.Softmax(dim=2)(tgt_score)

        simi1 = cos_simi(src_embedding, tgt_embedding)
        simi2 = cos_simi(tgt_embedding, src_embedding)

        # Score-weighted similarity
        simi_src = simi1 * tgt_score
        simi_tgt = simi2 * src_score
        mask_src = self.mask_src_nn(simi_src.permute(0, 2, 1))
        mask_tgt = self.mask_tgt_nn(simi_tgt.permute(0, 2, 1))

        mask_src_score = torch.softmax(mask_src, dim=1)[:, 1, :].detach()  # (B, N)
        mask_tgt_score = torch.softmax(mask_tgt, dim=1)[:, 1, :].detach()
        # Select top-k points as overlap points
        mask_src_idx = torch.zeros(mask_src_score.shape, device=src.device)
        _, indices = torch.topk(mask_src_score, k=700, dim=1)
        mask_src_idx.scatter_(1, indices, 1)

        mask_tgt_idx = torch.zeros(mask_tgt_score.shape, device=src.device)
        _, indices = torch.topk(mask_tgt_score, k=700, dim=1)
        mask_tgt_idx.scatter_(1, indices, 1)

        return mask_src, mask_tgt, mask_src_idx, mask_tgt_idx

