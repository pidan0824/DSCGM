import torch
import torch.nn as nn
import math
from models.dgcnn import DGCNN
from models.gconv import Siamese_Gconv
from models.affinity_layer import Affinity
from models.transformer import Transformer

from utils.config import cfg
from models.cfg import make_cfg
from models.fcnn import filter_correspondences
from models.edge_1 import compute_edge_vectors
from models.point_sample import native_furthest_point_sample,furthest_point_sample
from models.overlapdect import mask_point,mask_cor,mask_point_nodet,mask_point1,mask_point0
from models.overlapdect import OverlapNet
from models.SCNETmodel import GraphSCNet,create_model
from models.RGNmodel import RegNet
import numpy as np
from utils.build_graphs import build_graphs
from utils.hungarian import hungarian


def sinkhorn_rpm(log_alpha, n_iters: int = 5, slack: bool = True, eps: float = -1) -> torch.Tensor:
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)

    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    # Sinkhorn iterations
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

    return log_alpha


class Net(nn.Module):
    def __init__(self,cfg1):
        super(Net, self).__init__()
        # Overlap
        self.overlap= OverlapNet(all_points=1024, src_subsampled_points=717,tgt_subsampled_points=717)
        self.GraphSCNet=GraphSCNet(cfg1)
        self.rgn=RegNet()
        self.pointfeaturer = DGCNN(cfg.PGM.FEATURES, cfg.PGM.NEIGHBORSNUM, cfg.PGM.FEATURE_EDGE_CHANNEL)
        self.gnn_layer = cfg.PGM.GNN_LAYER
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_Gconv(cfg.PGM.FEATURE_NODE_CHANNEL + cfg.PGM.FEATURE_EDGE_CHANNEL, cfg.PGM.GNN_FEAT)
            else:
                gnn_layer = Siamese_Gconv(cfg.PGM.GNN_FEAT, cfg.PGM.GNN_FEAT)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(i), Affinity(cfg.PGM.GNN_FEAT))
            if cfg.PGM.USEATTEND == 'attentiontransformer':
                self.add_module('gmattend{}'.format(i), Transformer(2 * cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                    if i == 0 else cfg.PGM.GNN_FEAT))
            self.add_module('InstNorm_layer_{}'.format(i), nn.InstanceNorm2d(1, affine=True))
            if i == self.gnn_layer - 2:  # only second last layer will have cross-graph module
                self.add_module('cross_graph_{}'.format(i), nn.Linear(cfg.PGM.GNN_FEAT * 2, cfg.PGM.GNN_FEAT))

    def forward(self, P1_gt_reshaped3, P2_gt_reshaped3,P1_gt_reshaped6,P2_gt_reshaped6,Inlier_src_gt, Inlier_ref_gt,perm_mat):
        mask_src, mask_tgt, mask_src_idx, mask_tgt_idx = self.overlap(P1_gt_reshaped3, P2_gt_reshaped3)
        P1_gt =mask_point(mask_src_idx, P1_gt_reshaped6)
        P2_gt =mask_point(mask_tgt_idx, P2_gt_reshaped6)

        B=P1_gt.shape[0]
        N = P1_gt.shape[2]
        W=P1_gt.shape[1]
        perm_mat1=mask_point1(mask_src_idx,perm_mat)
        perm_mat1 = perm_mat1.permute(0, 2, 1)
        perm_mat1=mask_point1(mask_tgt_idx,perm_mat1)
        perm_mat1 = perm_mat1.permute(0, 2, 1)
        n1_gt = perm_mat1.shape[1]

        n2_gt = perm_mat1.shape[2]
        Inlier_src_gt=mask_point1(mask_src_idx,Inlier_src_gt)

        Inlier_src_gt_edge=Inlier_src_gt
        Inlier_src_gt=Inlier_src_gt.cpu().numpy()
        Inlier_ref_gt = mask_point1(mask_tgt_idx, Inlier_ref_gt)
        Inlier_ref_gt_edge=Inlier_ref_gt
        Inlier_ref_gt = Inlier_ref_gt.cpu().numpy()
        P1_np = P1_gt.cpu().numpy()
        P2_np = P2_gt.cpu().numpy()
        P1_np = np.transpose(P1_np, (0, 2, 1))
        P2_np = np.transpose(P2_np, (0, 2, 1))
        A1_gt, e1_gt = build_graphs(P1_np, Inlier_src_gt, n1_gt, stg=cfg.PAIR.GT_GRAPH_CONSTRUCT)
        if cfg.PAIR.REF_GRAPH_CONSTRUCT == 'same':
            A2_gt = A1_gt.transpose().contiguous()
            e2_gt = e1_gt
        else:
            A2_gt, e2_gt = build_graphs(P2_np, Inlier_ref_gt, n2_gt,stg=cfg.PAIR.REF_GRAPH_CONSTRUCT)

        A_src = A1_gt
        A_tgt = A2_gt
        n1_gt = torch.full((B,), perm_mat1.shape[1])
        n2_gt = torch.full((B,), perm_mat1.shape[2])

        P1_gttrue = P1_gt.transpose(1, 2)
        P2_gttrue = P2_gt.transpose(1, 2)

        src_points_edge = P1_gttrue[:,:, :3]  
        ref_points_edge = P2_gttrue[:,:, :3]  



        Inlier_src_gt_edge = Inlier_src_gt_edge.squeeze(-1)
        src_points_edge_in = src_points_edge.permute(0, 2, 1)
        src_points_edge_in = mask_point0(Inlier_src_gt_edge, src_points_edge_in)
        src_points_edge_in = src_points_edge_in.permute(0, 2, 1)

        Inlier_ref_gt_edge = Inlier_ref_gt_edge.squeeze(-1)

        ref_points_edge_in = ref_points_edge.permute(0, 2, 1)
        ref_points_edge_in = mask_point0(Inlier_ref_gt_edge, ref_points_edge_in)
        ref_points_edge_in = ref_points_edge_in.permute(0, 2, 1)



        points_np = src_points_edge.cpu().numpy().astype(np.float32)
        sampled_indices_list = []
        for i in range(points_np.shape[0]):
            current_points = points_np[i]
            sampled_indices = furthest_point_sample(current_points, min_distance=0.02)
            sampled_indices_list.append(sampled_indices)

        min_sampled_points = min(len(indices) for indices in sampled_indices_list)
        truncated_sampled_indices_list = [indices[:min_sampled_points] for indices in sampled_indices_list]
        truncated_sampled_indices_np = np.array(truncated_sampled_indices_list, dtype=int)
        src_nodes = points_np[np.arange(B)[:, None], truncated_sampled_indices_np, :]
        P_src = P1_gttrue
        P_tgt = P2_gttrue


        Node_src, Edge_src = self.pointfeaturer(P_src)
        Node_tgt, Edge_tgt = self.pointfeaturer(P_tgt)

        emb_src, emb_tgt = torch.cat((Node_src, Edge_src), dim=1).transpose(1, 2).contiguous(), \
                           torch.cat((Node_tgt, Edge_tgt), dim=1).transpose(1, 2).contiguous()
        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            if cfg.PGM.USEATTEND == 'attentiontransformer':
                gmattends_layer = getattr(self, 'gmattend{}'.format(i))
                src_embedding, tgt_embedding = gmattends_layer(emb_src, emb_tgt)
                d_k = src_embedding.size(1)
                scores_src = torch.matmul(src_embedding.transpose(2, 1).contiguous(), src_embedding) / math.sqrt(d_k)
                scores_tgt = torch.matmul(tgt_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
                A_src1 = torch.softmax(scores_src, dim=-1)
                A_tgt1 = torch.softmax(scores_tgt, dim=-1)
                emb_src, emb_tgt = gnn_layer([A_src1, emb_src], [A_tgt1, emb_tgt])
            else:
                emb_src, emb_tgt = gnn_layer([A_src, emb_src], [A_tgt, emb_tgt])
            affinity = getattr(self, 'affinity_{}'.format(i))

            s = affinity(emb_src, emb_tgt)
            InstNorm_layer = getattr(self, 'InstNorm_layer_{}'.format(i))
            s = InstNorm_layer(s[:,None,:,:]).squeeze(dim=1)
            log_s = sinkhorn_rpm(s, n_iters=20, slack=cfg.PGM.SKADDCR)
            s = torch.exp(log_s)


            if i == self.gnn_layer - 2:
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                emb1_new = cross_graph(torch.cat((emb_src, torch.bmm(s, emb_tgt)), dim=-1))
                emb2_new = cross_graph(torch.cat((emb_tgt, torch.bmm(s.transpose(1, 2), emb_src)), dim=-1))
                emb_src = emb1_new
                emb_tgt = emb2_new

        if cfg.DATASET.NOISE_TYPE != 'clean':
            srcinlier_s = torch.sum(s, dim=-1, keepdim=True)
            refinlier_s = torch.sum(s, dim=-2)[:, :, None]
        else:
            srcinlier_s = None
            refinlier_s = None

        if cfg.DATASET.NOISE_TYPE != 'clean':
            if cfg.PGM.USEINLIERRATE:
                s = srcinlier_s * s * refinlier_s.transpose(2, 1).contiguous()
        # hard_correspondence
        s_perm_mat = hungarian(s, n1_gt, n2_gt, srcinlier_s, refinlier_s)
        filtered_node_num_nodes = src_nodes.shape[1]


        src_nodes_tensor = torch.from_numpy(src_nodes).to(P1_gt_reshaped3.device)
        truncated_sampled_indices_tensor = torch.from_numpy(truncated_sampled_indices_np).to(P1_gt_reshaped3.device)


        perm_mat_edge = mask_point0(Inlier_ref_gt_edge, s_perm_mat)
        perm_mat_edge = perm_mat_edge.permute(0, 2, 1)
        perm_mat_edge = mask_point0(Inlier_src_gt_edge, perm_mat_edge)
        perm_mat_edge = perm_mat_edge.permute(0, 2, 1)



        src_edges, tgt_edges, src_knn_edges, tgt_knn_edges = compute_edge_vectors(src_points_edge,
                                                                                  ref_points_edge,
                                                                                  src_nodes_tensor,truncated_sampled_indices_tensor,
                                                                                  filtered_node_num_nodes,
                                                                                  s_perm_mat)


        src_edges = src_edges.permute(0, 3, 2, 1)
        tgt_edges = tgt_edges.permute(0, 3, 2, 1)
        src_knn_edges = src_knn_edges.permute(0, 3, 2,1)
        tgt_knn_edges = tgt_knn_edges.permute(0, 3, 2, 1)
        tgt_edges = tgt_edges.reshape(B, 3, -1)
        src_knn_edges = src_knn_edges.reshape(B, 3, -1)
        tgt_knn_edges = tgt_knn_edges.reshape(B, 3, -1)
        src_edges = src_edges.reshape(B, 3, -1)


        R1, t1,src1,src_corr1=self.rgn(src_edges,tgt_edges)
        R2, t2,src2,src_corr2=self.rgn(src_knn_edges,tgt_knn_edges)

        return s, srcinlier_s, refinlier_s,s_perm_mat,src1,src_corr1,src2,src_corr2,R1,R2,perm_mat1,mask_src, mask_tgt,src_points_edge,ref_points_edge,n1_gt,n2_gt,mask_src_idx, mask_tgt_idx,P_src,P_tgt,A_src,A_tgt
