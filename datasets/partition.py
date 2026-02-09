from community import community_louvain
import networkx as nx
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from torch_geometric.data import Data


def get_subgraph_by_node(dataset, node_list, flag):
    node_id_set = set(node_list)
    global_id_to_local_id = {}
    local_id_to_global_id = []
    local_edge_list = []
    global_edge_list = []
    for local_id, global_id in enumerate(node_list):
        global_id_to_local_id[global_id] = local_id
        local_id_to_global_id.append(global_id)

    for edge_id in range(dataset.edge_index.shape[1]):
        src = dataset.edge_index[0, edge_id].item()
        tgt = dataset.edge_index[1, edge_id].item()
        if src in node_id_set and tgt in node_id_set:
            local_id_src = global_id_to_local_id[src]
            local_id_tgt = global_id_to_local_id[tgt]
            local_edge_list.append((local_id_src, local_id_tgt))
            global_edge_list.append((src, tgt))

    local_edge_index = torch.tensor(local_edge_list).t()
    global_edge_list = torch.tensor(global_edge_list).t()
    if not local_edge_list:
        local_edge_index = torch.empty((2, 0), dtype=torch.int64)
    if flag:
        local_subgraph = Data(x=dataset.x[node_list], edge_index=local_edge_index, y=dataset.y[node_list])
    else:
        local_subgraph = Data(x=dataset.x, edge_index=global_edge_list, y=dataset.y)
    local_subgraph.global_map = local_id_to_global_id

    return local_subgraph

def louvain_partitioner(data, num_clients):
    G = nx.Graph()
    for i in range(data.num_nodes):
        G.add_node(i)
   
   
    edges = data.edge_index.numpy()
    for i in range(edges.shape[1]):
        G.add_edge(edges[0, i], edges[1, i])
    
    for i, feature in enumerate(data.x.numpy()):
        G.nodes[i]['feature'] = feature


    
    le = LabelEncoder()
    labels = le.fit_transform(data.y.numpy())
    for i, label in enumerate(labels):
        G.nodes[i]['label'] = label

    partition = community_louvain.best_partition(G)
   
    community_dict = {}
    for node, community in partition.items():
        if community not in community_dict:
            community_dict[community] = []
        community_dict[community].append(node)

    num_communities = len(community_dict)

    clients_nodes = [[] for _ in range(num_clients)]

    while len(community_dict) < num_clients:
        community_sizes = [len(community_dict[i]) for i in range(len(community_dict))]
        max_len = max(community_sizes)
        min_len = min(community_sizes)
        max_index = np.argmax(community_sizes)
        if max_len < 2 * min_len:
            min_len = max_len // 2
        max_len_nodes = community_dict[max_index]
        new_list_id = len(community_dict)
        community_dict[new_list_id] = max_len_nodes[:min_len]
        community_dict[max_index] = max_len_nodes[min_len:]
    community_sizes = [len(community_dict[i]) for i in range(len(community_dict))]
    community_ids = np.argsort(community_sizes)
    for comid in community_ids:
        clid = np.argmin([len(cs) for cs in clients_nodes])
        clients_nodes[clid].extend(community_dict[comid])

    clients_data = []
    for nodes in clients_nodes:
        node_map = {node: i for i, node in enumerate(nodes)}
        sub_edge_index = []
        for i in range(data.edge_index.size(1)):
            if data.edge_index[0, i].item() in node_map and data.edge_index[1, i].item() in node_map:
                sub_edge_index.append([node_map[data.edge_index[0, i].item()], node_map[data.edge_index[1, i].item()]])
        sub_edge_index = torch.tensor(sub_edge_index, dtype=torch.long).t().contiguous()


        sub_x = data.x[nodes]
        sub_y = data.y[nodes]
        sub_data = Data(x=sub_x, edge_index=sub_edge_index,y =sub_y)
        
        clients_data.append(sub_data)
    return clients_data


