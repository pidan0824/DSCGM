from typing import Tuple, Union
import torch
from torch import Tensor, LongTensor

from vision3d.utils.misc import load_ext
from models.knn import knn, keops_knn

ext_module = load_ext("vision3d.ext", ["knn_points"])


def knn_point1s(
    q_points: Tensor, s_points: Tensor, k: int, transposed: bool = False, return_distance: bool = False
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Heap sort based kNN search for point cloud.

    Args:
        q_points: (B, N, 3) or (B, 3, N) if transposed.
        s_points: (B, M, 3) or (B, 3, M) if transposed.
        k: number of neighbors.
        transposed: if True, points shape is (B, 3, N).
        return_distance: if True, also return distances.

    Returns:
        knn_distances: (B, N, k), only if return_distance is True.
        knn_indices: (B, N, k).
    """
    if transposed:
        q_points = q_points.transpose(1, 2)  # (B, N, 3) -> (B, 3, N)
        s_points = s_points.transpose(1, 2)  # (B, M, 3) -> (B, 3, M)
    q_points = q_points.contiguous()
    s_points = s_points.contiguous()

    knn_distances = q_points.new_zeros(size=(q_points.shape[0], q_points.shape[1], k))  # (B, N, k)
    knn_indices = torch.zeros(size=(q_points.shape[0], q_points.shape[1], k), dtype=torch.long)  # (B, N, k)
    knn(q_points, s_points, knn_distances, knn_indices, k)

    if return_distance:
        return knn_distances, knn_indices

    return knn_indices


def knn_points(
    q_points: Tensor,
    s_points: Tensor,
    k: int,
    dilation: int = 1,
    distance_limit: float = None,
    return_distance: bool = None,
    remove_nearest: bool = True,
    transposed: bool = False,
    padding_mode: str = "nearest",
    padding_value: float = 1e10,
    squeeze: bool = False,
):
    """kNN search using KeOps acceleration. Supports dilation, distance limit, and self-removal.

    Args:
        q_points: (*, C, N) or (*, N, C), query points.
        s_points: (*, C, M) or (*, M, C), support points.
        k: number of nearest neighbors.
        dilation: dilation factor for dilated knn.
        distance_limit: ignore neighbors beyond this radius.
        return_distance: if True, also return distances.
        remove_nearest: if True, remove the nearest neighbor (itself).
        transposed: if True, points shape is (*, C, N).
        padding_mode: 'nearest' or 'empty' for out-of-range neighbors.
        padding_value: fill value for 'empty' padding mode.
        squeeze: if True, squeeze output when k=1.

    Returns:
        knn_distances: (*, M, k), only if return_distance is True.
        knn_indices: (*, M, k).
    """
    if transposed:
        q_points = q_points.transpose(-1, -2)  # (*, C, N) -> (*, N, C)
        s_points = s_points.transpose(-1, -2)  # (*, C, M) -> (*, M, C)
    q_points = q_points.contiguous()
    s_points = s_points.contiguous()

    num_s_points = s_points.shape[-2]
    dilated_k = (k - 1)* dilation+ 1
    if remove_nearest:
        dilated_k += 1
    final_k = min(dilated_k, num_s_points)
    knn_distances, knn_indices = keops_knn(q_points, s_points, final_k)  # (*, N, k)
    if remove_nearest:
        knn_distances = knn_distances[..., 1:]
        knn_indices = knn_indices[..., 1:]
    if dilation > 1:
        knn_distances = knn_distances[..., ::dilation]
        knn_indices = knn_indices[..., ::dilation]
    knn_distances = knn_distances.contiguous()
    knn_indices = knn_indices.contiguous()
    if distance_limit is not None:
        assert padding_mode in ["nearest", "empty"]
        knn_masks = torch.ge(knn_distances, distance_limit)
        if padding_mode == "nearest":
            knn_distances = torch.where(knn_masks, knn_distances[..., :1], knn_distances)
            knn_indices = torch.where(knn_masks, knn_indices[..., :1], knn_indices)
        else:
            knn_distances[knn_masks] = padding_value
            knn_indices[knn_masks] = num_s_points
    if squeeze and k == 1:
        knn_distances = knn_distances.squeeze(-1)
        knn_indices = knn_indices.squeeze(-1)

    if return_distance:
        return knn_distances, knn_indices
    return knn_indices
