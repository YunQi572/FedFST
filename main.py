import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
import networkx as nx
from community import community_louvain
import concurrent.futures

from args import parser
from models.model import load_model
from algorithm.Ours import load_server_clients, plot_all_losses
from datasets.partition import get_subgraph_by_node,louvain_partitioner
from datasets.dataset_loader import load_dataset, class_to_task
from utils import set_seed, AA, AF

def main():
    args = parser.parse_args()
    print(f"Seed: {args.seed}, Dataset: {args.dataset_name}, Clients: {args.clients_num}, Model: {args.model}")
    print(f"gen_lr={args.gen_lr}, kd_lr={args.kd_lr}, kd_khop={args.kd_khop}, gen_epochs={args.gen_epochs}, kd_epochs={args.kd_epochs}, lr={args.lr}")
    print(f"gen_rounds={args.gen_rounds}, tolerance={args.tolerance}, max_iterations={args.max_iterations}, gen_num_nodes={args.gen_num_nodes}, per_task_class_num={args.per_task_class_num}")
    print(f"kd_ce_weight={args.kd_ce_weight}, kd_low_weight={args.kd_low_weight}, temperature={args.kd_temperature}")
    print(f"gen_ce_weight={args.gen_ce_weight}, gen_kl_weight={args.gen_kl_weight}")
    print(f"S_high_x_weight={args.S_high_x_weight}, S_high_D_weight={args.S_high_D_weight}")
    
    set_seed(args.seed)
    
    if args.use_gpu:
        device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    print(f"Loading dataset: {args.dataset_name}")
    data, input_dim, out_dim = load_dataset(args.dataset_dir, args.dataset_name, args.per_task_class_num, args)
    
    args.input_dim = input_dim
    args.output_dim = out_dim

    print(f"Dataset: {args.dataset_name}")
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"Features: {data.x.shape}, Classes: {data.y.max().item() + 1}")
    
    print("Partitioning graph for federated learning...")
    clients_data = louvain_partitioner(data, args.clients_num)

    args.input_dim = input_dim
    args.out_dim = out_dim
    classes_num = data.y.max().item() + 1
    server, clients, message_pool = load_server_clients(args, clients_data, device, classes_num)
    
    tasks_num = (data.y.max().item() + 1) // args.per_task_class_num
    print(f"Total number of tasks: {tasks_num}")
    
    ACC_matrix = torch.zeros(size = (tasks_num, tasks_num)).to(device)

    for task_id in range(tasks_num):
        print(f"\n========== Task {task_id}/{tasks_num} ==========")

        for round in range(args.rounds):
            print(f"GNN Round {round}")
            print(f"Training clients for task {task_id}...")
            client_losses = []
            for client_id, client in enumerate(clients):
                loss = client.train(task_id)
                client_losses.append(loss)
                print(f"Client {client_id} loss: {loss:.4f}")
            
            avg_client_loss = sum(client_losses) / len(client_losses)
            print(f"Average client loss: {avg_client_loss:.4f}")

            for client_id, client in enumerate(clients):
                client.send_message(task_id)
            
            server.clients_nodes_num = [server.message_pool[f"client_{client_id}"]["nodes_num"] 
                                    for client_id in range(args.clients_num)]
            server.clients_learned_nodes_num = [server.message_pool[f"client_{client_id}"]["learned_nodes_num"] 
                                    for client_id in range(args.clients_num)]

            print("Server aggregating client models...")
            server.aggregate()
            server.send_message()

        if task_id != 0:
            server.feature_gen_init(task_id = task_id)
            for client_id, client in enumerate(clients):
                client.feature_gen_init(task_id = task_id)
                
            for round in range(args.gen_rounds):
                print(f"Generator Round {round}")
                def train_feature_gen_client(client_id, client, task_id, is_not_first_round):
                    client.feature_gen_train(task_id, is_not_first_round)
                    return client_id
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(train_feature_gen_client, client_id, client, task_id, round != 0)
                        for client_id, client in enumerate(clients)
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        client_id = future.result()
                
                for client_id, client in enumerate(clients):
                    client.send_feature_gen(task_id)
                
                server.feature_gen_aggregate()
                server.send_feature_gen()

            server.KD_train(task_id)

        server.update_last_global_model()
        for client_id, client in enumerate(clients):
            client.last_global_model.load_state_dict(server.last_global_model.state_dict())
        
        for client_id, client in enumerate(clients):
            client.save_validation_acc(task_id)

        for eval_task_id in range(0, task_id + 1):
            total_nodes_num = 0
            for client_id in range(args.clients_num):
                evaluation = clients[client_id].evaluate(task_id = eval_task_id, global_flag=True)
                client_acc = evaluation["acc"]
                nodes_num = clients[client_id].tasks[eval_task_id]["test_mask"].sum()
                ACC_matrix[task_id, eval_task_id] += client_acc * nodes_num
                total_nodes_num += nodes_num

            ACC_matrix[task_id, eval_task_id] /= total_nodes_num
            print(f"Task {task_id} trained, Task {eval_task_id} global acc: {ACC_matrix[task_id, eval_task_id]:.2f}")

        print(ACC_matrix)

        aa = AA(ACC_matrix, T = task_id + 1)
        af = AF(ACC_matrix, T = task_id + 1)
        print(f"Task {task_id} - AA: {aa:.4f}, AF: {af:.4f}")
    
    print("\n========== Plotting Loss Curves ==========")
    loss_plots_dir = plot_all_losses(server, clients, save_dir=args.save_dir)
    print(f"Loss plots saved to: {loss_plots_dir}")
    
    print("\n========== Training Complete ==========")
    print(f"Final Accuracy Matrix:")
    print(ACC_matrix)
    print(f"Final AA: {AA(ACC_matrix, T = tasks_num):.4f}")
    print(f"Final AF: {AF(ACC_matrix, T = tasks_num):.4f}")


if __name__ == "__main__":
    main()