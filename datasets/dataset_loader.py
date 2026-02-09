import os
from torch_geometric.datasets import Planetoid, WebKB, Amazon, WikipediaNetwork, Actor, HeterophilousGraphDataset, Coauthor, Flickr
import gdown

import scipy.io
from sklearn.preprocessing import label_binarize
from torch_geometric.data import Data
import torch_geometric.transforms as T
import scipy.io
import numpy as np
import random
import torch
import scipy.sparse
import csv
import pandas as pd
import json
from ogb.nodeproppred import NodePropPredDataset
from os import path
from torch_sparse import SparseTensor
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import os
from collections import defaultdict

import torch.nn.functional as F
from scipy import sparse as sp
from sklearn.metrics import roc_auc_score, f1_score

from typing import Optional, Callable
import os.path as osp
from torch_geometric.utils import to_undirected
from torch_geometric.data import InMemoryDataset, download_url, Data

from .partition import *


def load_dataset(root_dir, dataset_name, per_task_class_num, agrs, num_masks = 1):
    
    assert dataset_name in ('cora', 'citeseer', 'pubmed', 'ogbn-arxiv',
                            'computers', 'physics', 'roman_empire', 'year', 'actor', 'cs'), 'Invalid dataset'

    if dataset_name in ['cora', 'citeseer']:
        dataset = Planetoid(root = root_dir, name = dataset_name)
        data = dataset[0]
    
    if dataset_name == 'pubmed':
        dataset = Planetoid(root = root_dir, name = dataset_name)
        data = dataset[0]

        
    elif dataset_name in ['ogbn-arxiv']:
        dataset = PygNodePropPredDataset(name=dataset_name, root=root_dir)
        data = dataset[0]
        data.y = data.y.view(-1)

    elif dataset_name in ['computers']:
        dataset = Amazon(root=root_dir, name=dataset_name)
        data = dataset[0]

    elif dataset_name in ['cs', 'physics']:
        dataset = Coauthor(root=root_dir, name=dataset_name, transform=T.NormalizeFeatures())
        data = dataset[0]

    elif dataset_name in ["roman_empire"]:
        dataset = HeterophilousGraphDataset(root=root_dir, name=dataset_name)
        data = dataset[0]
    
    elif dataset_name in ['year']:
        data = load_arxiv_year()

    elif dataset_name in ['actor']:
        dataset = Actor(root=root_dir)
        data = dataset[0]
  
    in_dim = data.x.shape[1]
    out_dim = ((data.y.max().item() + 1) // per_task_class_num) * per_task_class_num

    return data, in_dim, out_dim

def load_arxiv_year(nclass=5):
    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv')
    graph = ogb_dataset.graph
    edge_index = torch.as_tensor(graph['edge_index'])
    x = torch.as_tensor(graph['node_feat'])

    label = even_quantile_labels(graph['node_year'].flatten(), nclass, verbose=False)
    y = torch.as_tensor(label).reshape(-1)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data

def even_quantile_labels(vals, nclasses, verbose=True):
    label = -1 * np.ones(vals.shape[0], dtype=int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label

def class_to_task(data, per_task_class_num, train_prop, valid_prop, test_prop, shuffle_flag = False, classes_order = None):
    nodes_num = data.x.shape[0]
    classes_num = data.y.max().item() + 1
    
    if classes_order is not None:
        class_mapping = {orig_class: new_class for new_class, orig_class in enumerate(classes_order)}
        mapped_y = torch.zeros_like(data.y)
        for orig_class, new_class in class_mapping.items():
            mapped_y[data.y == orig_class] = new_class
        data.y = mapped_y
        classes_num = len(classes_order)
    else:
        class_mapping = {i: i for i in range(classes_num)}

    train_mask = torch.zeros(nodes_num, dtype = torch.bool)
    valid_mask = torch.zeros(nodes_num, dtype = torch.bool)
    test_mask = torch.zeros(nodes_num, dtype = torch.bool)
    
    classes_nodes = []
    for class_i in range(classes_num):
        class_i_node_mask = data.y == class_i
        class_i_node_num = class_i_node_mask.sum().item()

        class_i_node_list = torch.where(class_i_node_mask)[0].numpy()
        classes_nodes.append(class_i_node_list)
        np.random.shuffle(class_i_node_list)
        
        train_num = int(class_i_node_num * train_prop)
        valid_num = int(class_i_node_num * valid_prop)
        test_num = int(class_i_node_num * test_prop)

        train_idx = class_i_node_list[: train_num]
        valid_idx = class_i_node_list[train_num : train_num + valid_num]
        test_idx = class_i_node_list[train_num + valid_num : train_num + valid_num + test_num]

        train_mask[train_idx] = True
        valid_mask[valid_idx] = True
        test_mask[test_idx] = True

    tasks_num = (classes_num + per_task_class_num - 1) // per_task_class_num

    task_classes = [[] for _ in range(tasks_num)]
    label_task = {}
    drop_flag = False

    classes_ind_list = list(range(classes_num))

    for task_i in range(tasks_num):
        l = task_i * per_task_class_num
        r = min((task_i + 1) * per_task_class_num, classes_num)

        if r < (task_i + 1) * per_task_class_num:
            drop_flag = True

        for i in range(l, r):
            label_task[classes_ind_list[i]] = task_i
            task_classes[task_i].append(i)

    if drop_flag:
        tasks_num = tasks_num - 1

    tasks = [{"train_mask": torch.zeros_like(train_mask).bool(),
              "valid_mask": torch.zeros_like(valid_mask).bool(),
              "test_mask": torch.zeros_like(test_mask).bool()} for _ in range(tasks_num)]

    for i in range(classes_num):
        class_i_train = train_mask & (data.y == i)
        class_i_valid = valid_mask & (data.y == i)
        class_i_test = test_mask & (data.y == i)
        task_i = label_task[i]

        if task_i == tasks_num:
            continue
        
        tasks[task_i]["train_mask"] = tasks[task_i]["train_mask"] | class_i_train
        tasks[task_i]["valid_mask"] = tasks[task_i]["valid_mask"] | class_i_valid
        tasks[task_i]["test_mask"] = tasks[task_i]["test_mask"] | class_i_test

    for task_i in range(tasks_num):
        nodes_list = []
        for class_idx in task_classes[task_i]:
            nodes_list.extend(classes_nodes[class_idx])
        sub_graph = get_subgraph_by_node(data, nodes_list, False)

        tasks[task_i]["local_data"] = sub_graph

    return tasks
    
    
def get_client_task(data, clients_num, per_task_class_num, train_prop, valid_prop, test_prop, partition_method = "louvain", shuffle_flag = False):
    clients_data = louvain_partitioner(data, clients_num)

    clients_tasks = {client_id: {"data" : None,
                                 "task" : None} for client_id in range(clients_num)}

    known_class_list = []

    for client_i in range(clients_num):
        client_data = clients_data[client_i]
        clients_tasks[client_i]["data"] = client_data

        client_tasks = class_to_task(client_data, per_task_class_num, train_prop, valid_prop, test_prop, shuffle_flag)
        clients_tasks[client_i]["task"] = client_tasks

        for task_i in client_tasks:
            client_i_task_i_mask = task_i["train_mask"] | task_i["valid_mask"] | task_i["test_mask"]
            client_i_task_i_known_classes = torch.unique(client_data.y[client_i_task_i_mask])
            known_class_list.append(client_i_task_i_known_classes)

        print(f"client {client_i} has {len(clients_tasks[client_i]['task'])} tasks.")

    known_class = torch.unique(torch.hstack(known_class_list))
    classes_used_num = known_class.shape[0]

    in_dim = data.x.shape[1]
    out_dim = classes_used_num

    if classes_used_num != data.y.max().item() + 1:
        print(f"DROPS {data.y.max().item() + 1 - classes_used_num} CLASS(ES).")

    return clients_tasks, in_dim, out_dim