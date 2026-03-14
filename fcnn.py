import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deformation_graph import compute_skinning_weights
from torch import Tensor

from models.knn import knn

from vision3d.layers import ConvBlock, NonRigidICP
from vision3d.ops import (
    apply_deformation,
    build_euclidean_deformation_graph,
    index_select,
    pairwise_distance,
)
# isort: split
from models.graphsc import GraphSCModule
from typing import Any, Dict, Tuple


def create_encoder(cfg: Dict[str, Any]) -> Any:
    """Create encoder instance from config."""
    return GraphSCModule(
        input_dim=cfg["model"]["transformer"]["input_dim"],
        output_dim=cfg["model"]["transformer"]["output_dim"],
        hidden_dim=cfg["model"]["transformer"]["hidden_dim"],
        num_heads=cfg["model"]["transformer"]["num_heads"],
        num_blocks=cfg["model"]["transformer"]["num_blocks"],
        num_layers_per_block=cfg["model"]["transformer"]["num_layers_per_block"],
        sigma_d=cfg["model"]["transformer"]["sigma_d"],
        embedding_k=cfg["model"]["transformer"]["embedding_k"],
        embedding_dim=cfg["model"]["transformer"]["embedding_dim"],
        dropout=cfg["model"]["transformer"]["dropout"],
        act_cfg=cfg["model"]["transformer"]["activation_fn"],
    )

def create_classifier(cfg: Dict[str, Any]) -> Any:
    """Create classifier instance from config."""
    return nn.Sequential(
        ConvBlock(
            in_channels=cfg["model"]["classifier"]["input_dim"],
            out_channels=cfg["model"]["classifier"]["input_dim"] // 2,
            kernel_size=1,
            conv_cfg="Conv1d",
            norm_cfg="GroupNorm",
            act_cfg="LeakyReLU",
            dropout=cfg["model"]["classifier"]["dropout"],
        ),
        ConvBlock(
            in_channels=cfg["model"]["classifier"]["input_dim"] // 2,
            out_channels=cfg["model"]["classifier"]["input_dim"] // 4,
            kernel_size=1,
            conv_cfg="Conv1d",
            norm_cfg="GroupNorm",
            act_cfg="LeakyReLU",
            dropout=cfg["model"]["classifier"]["dropout"],
        ),
        ConvBlock(
            in_channels=cfg["model"]["classifier"]["input_dim"] // 4,
            out_channels=1,
            kernel_size=1,
            conv_cfg="Conv1d",
            norm_cfg="None",
            act_cfg="None",
        ),
    )

config = {
    "model": {
        "transformer": {
            "input_dim": 6,
            "hidden_dim": 256,
            "output_dim": 256,
            "num_heads": 4,
            "num_blocks": 3,
            "num_layers_per_block": 2,
            "sigma_d": 0.08,
            "dropout": None,
            "activation_fn": "ReLU",
            "embedding_k": -1,
            "embedding_dim": 1,
        },
        "classifier": {
            "input_dim": 256,
            "dropout": None,
        },
    },
}

encoder_instance = create_encoder(config)
classifier_instance = create_classifier(config)


def filter_correspondences(data_dict, num_anchors, node_coverage, max_local_correspondences, min_local_correspondences):
    src_points = data_dict.get("src_points")
    tgt_points = data_dict.get("tgt_points")
    if src_points.dim() == 3:

        src_points = src_points[0]
        tgt_points = tgt_points[0]
    src_corr_points = data_dict["src_corr_points"]
    tgt_corr_points = data_dict["tgt_corr_points"]
    if src_corr_points.dim() == 3:

        src_corr_points = src_corr_points[0]

        tgt_corr_points = tgt_corr_points[0]
    node_indices = data_dict["node_indices"]  # (M,)
    src_nodes = src_points[node_indices]  # (M, 3)

    # Build deformation graph (placeholder implementation)
    def build_euclidean_deformation_graph(src_points, nodes, num_anchors, node_coverage):
        anchor_distances, anchor_indices = knn(src_points, nodes, num_anchors, return_distance=True)  # (N, K)
        anchor_weights = compute_skinning_weights(anchor_distances, node_coverage)  # (N, K)
        anchor_weights = anchor_weights / anchor_weights.sum(dim=1, keepdim=True)  # (N, K)
        return anchor_indices, anchor_weights
    corr_anchor_indices, corr_anchor_weights = build_euclidean_deformation_graph(src_corr_points, src_nodes,num_anchors, node_coverage)

    device = src_points.device
    anchor_masks = torch.ne(corr_anchor_indices, -1)  # (C, Ka)
    anchor_corr_indices, anchor_col_indices = torch.nonzero(anchor_masks, as_tuple=True)
    anchor_node_indices = corr_anchor_indices[anchor_corr_indices, anchor_col_indices]
    anchor_weights = corr_anchor_weights[anchor_corr_indices, anchor_col_indices]
    anchor_weights = anchor_weights.to(device)
    node_to_corr_weights = torch.zeros(src_nodes.shape[0], src_corr_points.shape[0], device=device)
    anchor_node_indices = anchor_node_indices.long()
    anchor_corr_indices = anchor_corr_indices.long()
    max_node_index = anchor_node_indices.max().item()
    min_node_index = anchor_node_indices.min().item()
    max_corr_index = anchor_corr_indices.max().item()
    min_corr_index = anchor_corr_indices.min().item()
    assert min_node_index >= 0, "Minimum node index is less than 0"
    assert max_node_index < src_nodes.shape[0], "Maximum node index is out of bounds"
    assert min_corr_index >= 0, "Minimum correspondence index is less than 0"
    assert max_corr_index < src_corr_points.shape[0], "Maximum correspondence index is out of bounds"
    node_to_corr_weights[anchor_node_indices, anchor_corr_indices] = anchor_weights
    # Assign correspondences to nodes
    max_local_correspondences = min(max_local_correspondences, node_to_corr_weights.shape[1])
    local_corr_weights, local_corr_indices = node_to_corr_weights.topk(k=max_local_correspondences, dim=1, largest=True)
    local_corr_masks = torch.gt(local_corr_weights, 0.0)

    # Remove small nodes
    local_corr_counts = local_corr_masks.sum(dim=-1)
    node_masks = torch.gt(local_corr_counts, min_local_correspondences)
    local_corr_indices = local_corr_indices[node_masks]
    local_corr_weights = local_corr_weights[node_masks]
    local_corr_masks = local_corr_masks[node_masks]


    device = torch.device("cpu")
    src_corr_points = src_corr_points.to(device)
    tgt_corr_points = tgt_corr_points.to(device)
    local_corr_indices = local_corr_indices.to(device)
    local_corr_weights = local_corr_weights.to(device)
    local_corr_masks = local_corr_masks.to(device)

    corr_feats, corr_masks = encoder_instance(src_corr_points, tgt_corr_points, local_corr_indices, local_corr_weights, local_corr_masks)
    corr_feats_norm = F.normalize(corr_feats, p=2, dim=1)

    # Classifier (placeholder implementation)
    corr_feats = corr_feats.transpose(0, 1).unsqueeze(0)
    corr_logits = classifier_instance(corr_feats)
    corr_logits = corr_logits.flatten()
    corr_scores = torch.sigmoid(corr_logits)


    # Feature consistency
    local_corr_feats_norm = corr_feats_norm[local_corr_indices]  # (M', k, d)
    local_affinity_mat = pairwise_distance(local_corr_feats_norm, local_corr_feats_norm, normalized=True, squared=False)
    local_fc_mat = torch.relu(1.0 - local_affinity_mat.pow(2))  # Note: Removed `/ sigma_f.pow(2)` since `sigma_f` is no longer an input

    # Pack output dictionary
    output_dict = {
        "local_corr_indices": local_corr_indices,
        "local_corr_weights": local_corr_weights,
        "local_corr_masks": local_corr_masks,
        "corr_feats": corr_feats_norm,
        "corr_logits": corr_logits,
        "corr_scores": corr_scores,
        "corr_masks": corr_masks,
        "feature_consistency": local_fc_mat,
    }



    if node_indices.device != node_masks.device:
        node_masks = node_masks.to(node_indices.device)
    filtered_node_indices = node_indices[node_masks]
    filtered_node_num_nodes = filtered_node_indices.shape[0]
    return output_dict,filtered_node_num_nodes, filtered_node_indices