import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import to_dense_adj, get_laplacian, degree
from torch_geometric.data import Data
import random
import os
import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple, List

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

class Generator(nn.Module):

    def __init__(self, noise_dim, input_dim, output_dim, dropout):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.emb_layer = nn.Embedding(output_dim, output_dim)

        hid_layers = []
        dims = [noise_dim + output_dim, 64, 128, 256]
        for i in range(len(dims) - 1):
            d_in = dims[i]
            d_out = dims[i + 1]
            hid_layers.append(nn.Linear(d_in, d_out))
            hid_layers.append(nn.Tanh())
            hid_layers.append(nn.Dropout(p = dropout, inplace = False))
        self.hid_layers = nn.Sequential(* hid_layers)
        self.nodes_layer = nn.Linear(256, input_dim)
    
    def forward(self, z, c):
        label_emb = self.emb_layer.forward(c)    
        z_c = torch.cat((label_emb, z), dim = -1)
        hid = self.hid_layers(z_c)
        node_logits = self.nodes_layer(hid)
        return node_logits

  


def add_self_loops(edge_index, num_nodes=None):
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1
    self_loops = torch.arange(num_nodes, device=edge_index.device)
    self_loops = self_loops.unsqueeze(0).repeat(2, 1)
    new_edge_index = torch.cat([edge_index, self_loops], dim=1)
    return new_edge_index

def edge_distribution_low(edge_idx, student_out, teacher_out):
    edge_idx = add_self_loops(edge_idx)
    src = edge_idx[0]
    dst = edge_idx[1]
    criterion = nn.KLDivLoss(reduction="batchmean", log_target=True)

    loss = criterion(student_out[src], teacher_out[dst])
    
    print(f"low loss requires_grad:{loss.requires_grad}")
    return loss

def get_khop_feature(graph, edge_idx, khop):
    feature = graph.clone()
    num_nodes = graph.shape[0]
    adj = torch.zeros((num_nodes, num_nodes), device=graph.device)
    adj[edge_idx[0], edge_idx[1]] = 1
    
    for _ in range(khop):
        degree = adj.sum(dim=-1, keepdim=True)
        neighbor_sum = torch.matmul(adj, feature)
        feature = (feature + neighbor_sum) / (1 + degree)
    
    return feature
    
def AA(M_acc, T = None):
    if T is None:
        T = M_acc.size(0)
    ret = 0
    for i in range(0, T):
        ret += M_acc[T - 1, i]
    ret /= T
    return ret

def AF(M_acc, T = None):
    if T is None:
        T = M_acc.size(0)
    if T == 1:
        return -1
    ret = 0
    for i in range(0, T - 1):
        forgetting = M_acc[i, i] - M_acc[T - 1, i]
        ret += forgetting
    ret /= T - 1
    return ret

def compute_led(graph_data):
    nodes_feature = graph_data.x
    edge_index = graph_data.edge_index
    num_nodes = nodes_feature.shape[0]
        
    edge_index_laplacian, edge_weight_laplacian = get_laplacian(
        edge_index, 
        num_nodes=num_nodes, 
        normalization='sym'
    )
        
    L = to_dense_adj(edge_index_laplacian, edge_attr=edge_weight_laplacian, max_num_nodes=num_nodes)[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    U = eigenvectors
    X_hat = torch.matmul(U.T, nodes_feature)
    energy_per_freq = torch.sum(X_hat ** 2, dim=1)
    total_energy = torch.sum(energy_per_freq)
        
    if total_energy > 0:
        energy_distribution = energy_per_freq / total_energy
    else:
        energy_distribution = torch.zeros_like(energy_per_freq)
    
    return energy_distribution


def get_Shigh(synthetic_data, args):
    node_features = synthetic_data.x
    edge_index = synthetic_data.edge_index
    num_nodes = node_features.shape[0]
    feature_dim = node_features.shape[1]
    
    edge_index_laplacian, edge_weight_laplacian = get_laplacian(
        edge_index, 
        num_nodes=num_nodes, 
        normalization=None
    )
    
    L = to_dense_adj(edge_index_laplacian, edge_attr=edge_weight_laplacian, max_num_nodes=num_nodes)[0]
    if args.use_gpu:
        device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    L = L.to(device)
    
    num_selected_features = max(1, int(args.feature_prop * feature_dim))
    print(f"选取的特征数量{num_selected_features}")
    
    selected_feature_indices = torch.randperm(feature_dim)[:num_selected_features]
    
    S_high_x_i = []
    
    for feature_idx in selected_feature_indices:
        x = node_features[:, feature_idx]
        
        xTLx = torch.matmul(torch.matmul(x.unsqueeze(0), L), x.unsqueeze(1)).squeeze()
        xTx = torch.dot(x, x)
        
        if xTx > 1e-8:
            S_high_i = xTLx / xTx
            S_high_x_i.append(S_high_i)
    
    if len(S_high_x_i) > 0:
        S_high_x = torch.stack(S_high_x_i).mean()
    else:
        S_high_x = torch.tensor(0.0, device=node_features.device)
    

    A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].to(device)
    

    L_indices = edge_index_laplacian
    L_values = edge_weight_laplacian
    L_sparse = torch.sparse_coo_tensor(L_indices, L_values, (num_nodes, num_nodes)).to(synthetic_data.x.device)
    
    S_high_A_i = []
    
    for node_idx in range(num_nodes):
        Ai = A[:, node_idx]
        
        L_Ai = torch.sparse.mm(L_sparse, Ai.unsqueeze(1)).squeeze()
        
        AiTLAi = torch.dot(Ai, L_Ai)
        AiTAi = torch.dot(Ai, Ai)
        
        if AiTAi > 1e-8:
            S_high_A_i_val = AiTLAi / AiTAi
            S_high_A_i.append(S_high_A_i_val)
    
    if len(S_high_A_i) > 0:
        S_high_A = torch.stack(S_high_A_i).mean()
    else:
        S_high_A = torch.tensor(0.0, device=node_features.device)
    
    S_high_x_weight = args.S_high_x_weight
    S_high_A_weight = args.S_high_A_weight
    
    print(f"S_high_x: {S_high_x}")
    print(f"S_high_A: {S_high_A}")

    S_high = S_high_x_weight * S_high_x + S_high_A_weight * S_high_A
    
    return S_high

def get_Homophily_Level(graph):
    edge_index = graph.edge_index
    node_labels = graph.y
    
    if edge_index.size(1) == 0:
        return 0.0
    
    if node_labels is None:
        return 0.0
    
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]
    
    src_labels = node_labels[src_nodes]
    dst_labels = node_labels[dst_nodes]
    

    same_label_edges = torch.sum(src_labels == dst_labels).float()
    
    total_edges = edge_index.size(1)
    
    homophily_ratio = same_label_edges / total_edges
    
    return homophily_ratio.item()



def reduce_homophilic_edges(data, edge_index, reduction_ratio=0.1):
    labels = data.y
    
    homophilic_edges = []
    heterophilic_edges = []
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if labels[src] == labels[dst]:
            homophilic_edges.append(i)
        else:
            heterophilic_edges.append(i)
    
    num_to_remove = int(len(homophilic_edges) * reduction_ratio)
    if num_to_remove > 0:
        remove_indices = torch.randperm(len(homophilic_edges))[:num_to_remove]
        edges_to_remove = [homophilic_edges[i] for i in remove_indices]
        
        keep_mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
        keep_mask[edges_to_remove] = False
        new_edge_index = edge_index[:, keep_mask]
    else:
        new_edge_index = edge_index
        
    return new_edge_index

def reduce_heterophilic_edges(data, edge_index, reduction_ratio=0.1):
    labels = data.y
    
    heterophilic_edges = []
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if labels[src] != labels[dst]:
            heterophilic_edges.append(i)
    
    num_to_remove = int(len(heterophilic_edges) * reduction_ratio)
    if num_to_remove > 0:
        remove_indices = torch.randperm(len(heterophilic_edges))[:num_to_remove]
        edges_to_remove = [heterophilic_edges[i] for i in remove_indices]
        
        keep_mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
        keep_mask[edges_to_remove] = False
        new_edge_index = edge_index[:, keep_mask]
    else:
        new_edge_index = edge_index
        
    return new_edge_index
