from models.knn_points import knn_points
import torch

def compute_edge_vectors(src_points, tgt_points, src_nodes,node_indices, filtered_node_num_nodes, s_perm_mat):

    batch_size, _, _ = src_points.size()
    num_nodes = filtered_node_num_nodes
    if node_indices.numel() != batch_size * num_nodes:
        raise ValueError("The number of node indices does not match the expected value.")
    k_nearest_neighbors = 3

    tgt_node_indices = []
    for b in range(batch_size):
        batch_tgt_indices = []
        current_batch_node_indices = node_indices[b]
        for row_idx in current_batch_node_indices:
            max_col_idx = torch.argmax(s_perm_mat[b, row_idx, :])
            batch_tgt_indices.append(max_col_idx.item())
        tgt_node_indices.append(batch_tgt_indices)
    tgt_node_indices = torch.tensor(tgt_node_indices)


    all_knn_indices = []

    for b in range(batch_size):
        batch_knn_indices = knn_points(src_nodes[b, :num_nodes], src_points[b], k=k_nearest_neighbors)
        all_knn_indices.append(batch_knn_indices)

    src_node_knn_indices = torch.stack(all_knn_indices, dim=0)  

    neighbors_src = src_points[torch.arange(batch_size).view(-1, 1, 1), src_node_knn_indices]

    src_nodes_expanded = src_nodes.unsqueeze(2).repeat(1, 1, k_nearest_neighbors, 1)
    src_edges_tensor_true = neighbors_src - src_nodes_expanded.float()
    tgt_node_knn_indices = []
    for b in range(batch_size):
        batch_s_perm_mat = s_perm_mat[b]
        batch_src_node_knn_indices = src_node_knn_indices[b]

        batch_tgt_indices = torch.empty((num_nodes, k_nearest_neighbors), dtype=torch.long, device=s_perm_mat.device)


        for m in range(num_nodes):
            node_knn_indices = batch_src_node_knn_indices[m]
            knn_rows = batch_s_perm_mat[node_knn_indices]
            node_tgt_indices = torch.argmax(knn_rows, dim=1)
            batch_tgt_indices[m] = node_tgt_indices
        tgt_node_knn_indices.append(batch_tgt_indices)

    tgt_node_knn_indices = torch.stack(tgt_node_knn_indices)
    node_tgt = tgt_points[torch.arange(batch_size).unsqueeze(1).repeat(1,num_nodes), tgt_node_indices]  #  (batch_size, num_nodes, 3)
    node_tgt = node_tgt.unsqueeze(2)
    neighbors_tgt = tgt_points[torch.arange(batch_size).unsqueeze(1).unsqueeze(2).repeat(1, num_nodes,
                                                                                         k_nearest_neighbors), tgt_node_knn_indices]  #  (batch_size, num_nodes, k_nearest_neighbors, 3)

    edges_tgt = neighbors_tgt - node_tgt
    src_knn = src_points[torch.arange(batch_size)[:, None, None].repeat(1, num_nodes,
                                                                        k_nearest_neighbors), src_node_knn_indices] 
    src_knn_diff = src_knn[:, :, :-1, :] - src_knn[:, :, 1:, :]  
    src_loop_edge = src_knn[:, :, 0, :] - src_knn[:, :, -1, :]  
    src_loop_edge = src_loop_edge[:, :, None, :]  
    src_knn_edges = torch.cat((src_knn_diff, src_loop_edge), dim=2)  
    tgt_knn = tgt_points[torch.arange(batch_size)[:, None, None].repeat(1, num_nodes,
                                                                        k_nearest_neighbors), tgt_node_knn_indices]  

    tgt_knn_diff = tgt_knn[:, :, :-1, :] - tgt_knn[:, :, 1:, :]  
    tgt_loop_edge = tgt_knn[:, :, 0, :] - tgt_knn[:, :, -1, :]  
    tgt_loop_edge = tgt_loop_edge[:, :, None, :]  
    tgt_knn_edges = torch.cat((tgt_knn_diff, tgt_loop_edge), dim=2)  
    return src_edges_tensor_true, edges_tgt, src_knn_edges, tgt_knn_edges
