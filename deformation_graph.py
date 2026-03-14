from typing import Tuple

import torch
from torch import Tensor

from models.knn_points import knn_points
from .pairwise_distance import pairwise_distance


def compute_skinning_weights(distances: Tensor, node_coverage: float) -> Tensor:
    """Skinning weight proposed in DynamicFusion.

    w = exp(-d^2 / (2 * r^2))

    Args:
        distances (Tensor): The distance tensor in arbitrary shape.
        node_coverage (float): The node coverage.

    Returns:
        weights (Tensor): The skinning weights in arbitrary shape.
    """
    weights = torch.exp(-(distances ** 2) / (2.0 * node_coverage ** 2))
    return weights


def build_euclidean_deformation_graph(
    points: Tensor,
    nodes: Tensor,
    num_anchors: int,
    node_coverage: float,
    return_point_anchor: bool = True,
    return_node_graph: bool = True,
    return_distance: bool = False,
    return_adjacent_matrix: bool = False,
    eps: float = 1e-6,
) -> Tuple[Tensor, ...]:
    """Build deformation graph using Embedded Deformation method with euclidean distance.

    Each point is assigned to k-nearest nodes; nodes sharing a point are connected by an edge.
    Skinning weights follow DynamicFusion: w = exp(-d^2 / (2 * r^2)).

    Args:
        points: (N, 3), input points.
        nodes: (M, 3), graph nodes.
        num_anchors: number of anchor nodes per point.
        node_coverage: coverage radius for skinning weights.
        return_point_anchor: if True, return anchor indices and weights for points.
        return_node_graph: if True, return node graph edges/adjacency.
        return_distance: if True, include distances in output.
        return_adjacent_matrix: if True, return adjacency matrix instead of edge list.
        eps: safe division epsilon.

    Returns:
        Tuple of requested tensors (anchor_indices, anchor_weights, distances,
        edge_indices/adjacent_mat, edge_weights/weight_mat, edge_distances/distance_mat).
    """
    output_list = []

    anchor_distances, anchor_indices = knn_points(points, nodes, num_anchors, return_distance=True)  # (N, K)
    anchor_weights = compute_skinning_weights(anchor_distances, node_coverage)  # (N, K)
    anchor_weights = anchor_weights / anchor_weights.sum(dim=1, keepdim=True)  # (N, K)

    if return_point_anchor:
        output_list.append(anchor_indices)
        output_list.append(anchor_weights)
        if return_distance:
            output_list.append(anchor_distances)

    if return_node_graph:
        point_indices = torch.arange(points.shape[0], device=points.device).unsqueeze(1).expand_as(anchor_indices)  # (N, K)
        node_to_point = torch.zeros(size=(nodes.shape[0], points.shape[0]), device=points.device)  # (N, M)
        node_to_point[anchor_indices, point_indices] = 1.0
        adjacent_mat = torch.gt(torch.einsum("nk,mk->nm", node_to_point, node_to_point), 0)
        distance_mat = pairwise_distance(nodes, nodes, squared=False)
        weight_mat = compute_skinning_weights(distance_mat, node_coverage)
        weight_mat = weight_mat * adjacent_mat.float()
        weight_mat = weight_mat / weight_mat.sum(dim=-1, keepdim=True).clamp(min=eps)
        if return_adjacent_matrix:
            output_list.append(adjacent_mat)
            output_list.append(weight_mat)
            if return_distance:
                distance_mat = distance_mat * adjacent_mat.float()
                output_list.append(distance_mat)
        else:
            edge_indices = torch.nonzero(adjacent_mat).contiguous()
            edge_weights = weight_mat[adjacent_mat].contiguous()
            output_list.append(edge_indices)
            output_list.append(edge_weights)
            if return_distance:
                edge_distances = distance_mat[adjacent_mat].contiguous()
                output_list.append(edge_distances)

    return tuple(output_list)
