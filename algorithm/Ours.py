from algorithm.Base import BaseServer, BaseClient
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, add_self_loops, dense_to_sparse, coalesce, get_laplacian
from models.model import *
from datasets.dataset_loader import class_to_task
from utils import Generator, edge_distribution_low, get_khop_feature, edge_distribution_low1, edge_distribution_low2, get_Shigh, reduce_homophilic_edges, reduce_heterophilic_edges, get_Homophily_Level
from datasets.partition import get_subgraph_by_node
import copy
import matplotlib.pyplot as plt
import os
from datetime import datetime
import random


class OursServer(BaseServer):
    def __init__(self, args, message_pool, device):
        super(OursServer, self).__init__(args, message_pool)
        self.args = args
        self.clients_num = self.args.clients_num
        self.device = device
        self.global_model = load_model(name = args.model, input_dim = args.input_dim, hidden_dim = args.hidden_dim, output_dim = args.output_dim, num_layers = args.num_layers, dropout = args.dropout).to(self.device)
        self.client_model = [load_model(name = args.model, input_dim = args.input_dim, hidden_dim = args.hidden_dim, output_dim = args.output_dim, num_layers = args.num_layers, dropout = args.dropout).to(self.device) for _ in range(self.clients_num)]
        self.last_global_model = load_model(name = args.model, input_dim = args.input_dim, hidden_dim = args.hidden_dim, output_dim = args.output_dim, num_layers = args.num_layers, dropout = args.dropout).to(self.device)
        self.per_task_class_num = self.args.per_task_class_num
        self.noise_dim = getattr(args, 'noise_dim', 128)
        self.clients_nodes_num = []
        self.clients_learned_nodes_num = []
        self.clients_graph_energy = []
        self.clients_mean_var = []
        self.generator_losses = {}
        self.generator_loss_details = {}
        self.kd_losses = {}
        self.kd_loss_details = {}
        self.clients = None

    def set_clients(self, clients):
        self.clients = clients

    def aggregate(self):
        client_weights = []
        for client_idx in range(self.clients_num):
            client_key = f"client_{client_idx}"
            if client_key in self.message_pool and "weight" in self.message_pool[client_key]:
                client_weights.append(self.message_pool[client_key]["weight"])
        
        if not client_weights:
            print("No client weights found in message_pool")
            return
            
        totoal_nodes_num = sum([self.message_pool[f"client_{client_id}"]["nodes_num"] for client_id in range(self.clients_num)])

        for i, client_id in enumerate(range(self.clients_num)):
            weight = self.message_pool[f"client_{client_id}"]["nodes_num"] / totoal_nodes_num
            for (client_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"], self.global_model.parameters()):
                if i == 0:
                    global_param.data.copy_(weight * client_param)
                else:
                    global_param.data += weight * client_param


    def feature_gen_init(self, task_id):
        classes_num = task_id * self.per_task_class_num
        self.generator = Generator(noise_dim = self.noise_dim, input_dim = self.args.input_dim, output_dim = classes_num, dropout = self.args.dropout).to(self.device)

    def feature_gen_aggregate(self):
        client_generators = []
        client_weights = []
        
        for client_id in range(self.clients_num):
            client_gen_key = f"client_{client_id}_generator"
            if client_gen_key in self.message_pool:
                client_gen_info = self.message_pool[client_gen_key]
                
                if (client_gen_info["generator_weight"] is not None and 
                    client_gen_info["learned_nodes_num"] > 0):
                    client_generators.append(client_gen_info["generator_weight"])
                    client_weights.append(client_gen_info["learned_nodes_num"])
        
        
        total_nodes = sum(client_weights)
        
        normalized_weights = [weight / total_nodes for weight in client_weights]
        
        with torch.no_grad():
            for param_idx, server_param in enumerate(self.generator.parameters()):
                aggregated_param = torch.zeros_like(server_param)
                
                for client_idx, client_gen_params in enumerate(client_generators):
                    client_param = client_gen_params[param_idx]
                    weight = normalized_weights[client_idx]
                    aggregated_param += weight * client_param
                
                server_param.data.copy_(aggregated_param)

    def synthesis_data(self, task_id, num_samples_per_class=10):
        client_s_highs = []
        client_nodes_nums = []
        
        for client_id in range(self.clients_num):
            client_gen_key = f"client_{client_id}_generator"
            if client_gen_key in self.message_pool:
                client_info = self.message_pool[client_gen_key]
                if client_info["learned_nodes_num"] > 0: 
                    client_s_highs.append(client_info["S_high"])
                    client_nodes_nums.append(client_info["learned_nodes_num"])
        
        total_nodes = sum(client_nodes_nums)
        weighted_s_high = sum(s_high * nodes_num for s_high, nodes_num in zip(client_s_highs, client_nodes_nums))
        global_s_high = weighted_s_high / total_nodes if total_nodes > 0 else 0.5
    
        print(f"Global weighted S_high: {global_s_high}")
        
        self.generator.eval()
        classes_num = task_id * self.per_task_class_num
        
        if classes_num == 0:
            classes_num = self.per_task_class_num
            
        with torch.no_grad():
            all_features = []
            all_labels = []
            
            for class_id in range(classes_num):
                noise = torch.randn(num_samples_per_class, self.noise_dim).to(self.device)
                labels = torch.full((num_samples_per_class,), class_id, dtype=torch.long).to(self.device)
                
                generated_logits = self.generator(noise, labels)
                features = F.normalize(generated_logits, p=2, dim=1)
                all_features.append(features)
                all_labels.append(labels)
            
            synthetic_features = torch.cat(all_features, dim=0)
            synthetic_labels = torch.cat(all_labels, dim=0)
            
        num_nodes = synthetic_features.shape[0]
        num_edges = int(num_nodes * self.args.gen_num_nodes)

        row = torch.randint(0, num_nodes, (num_edges,), device=synthetic_features.device)
        col = torch.randint(0, num_nodes, (num_edges,), device=synthetic_features.device)
        edge_index = torch.stack([row, col], dim=0)


        synthetic_data = Data(x=synthetic_features, edge_index=edge_index, y=synthetic_labels).to(self.device)

        current_s_high = get_Shigh(synthetic_data, self.args)
        print(f"Initial generated graph S_high: {current_s_high}, Target global S_high: {global_s_high}")
        
        tolerance = self.args.tolerance
        max_iterations = self.args.max_iterations
        
        for iteration in range(max_iterations):
            if abs(current_s_high - global_s_high) <= tolerance:
                print(f"S_high adjustment completed, iterations: {iteration}")
                break
                
            if current_s_high < global_s_high:
                edge_index = reduce_homophilic_edges(synthetic_data, edge_index, reduction_ratio=self.args.gen_reduction_ratio)
            else:
                edge_index = reduce_heterophilic_edges(synthetic_data, edge_index, reduction_ratio=self.args.gen_reduction_ratio)
            
            synthetic_data.edge_index = edge_index
            current_s_high = get_Shigh(synthetic_data, self.args)
        
        num_nodes = synthetic_features.shape[0]
        train_mask = torch.ones(num_nodes, dtype=torch.bool)
        

        self.synthesis_task = {
            "local_data": synthetic_data,
            "train_mask": train_mask,
            "valid_mask": torch.zeros(num_nodes, dtype=torch.bool),
            "test_mask": torch.zeros(num_nodes, dtype=torch.bool)
        }
        
        print(f"Synthetic data generated: {num_nodes} nodes, {edge_index.shape[1]} edges")
        print(f"Final S_high: {current_s_high}, Target S_high: {global_s_high}")
        


    def KD_train(self, task_id):
        if task_id == 0:
            print("First task, no knowledge distillation needed.")
            if task_id not in self.kd_losses:
                self.kd_losses[task_id] = []
                self.kd_loss_details[task_id] = []
            return
            
        self.synthesis_data(task_id, num_samples_per_class=self.args.num_samples_per_class)

        synthetic_task = self.synthesis_task
        synthetic_data = synthetic_task["local_data"]
        print(f"KD train SYS Data.y:{synthetic_data.y}")

        
        self.global_model.train()
        self.last_global_model.eval()
        
        kd_optimizer = torch.optim.Adam(self.global_model.parameters(), 
                                       lr=getattr(self.args, 'kd_lr', 0.001))
        
        temperature = getattr(self.args, 'kd_temperature', 4.0)
        num_epochs = getattr(self.args, 'kd_epochs', 50)
        kd_eval_start_epoch = getattr(self.args, 'kd_eval_start_epoch', 20)
        
        synthetic_data = synthetic_data.to(self.device)
        
        print(f"Starting knowledge distillation for task {task_id}...")
        
        if task_id not in self.kd_losses:
            self.kd_losses[task_id] = []
            self.kd_loss_details[task_id] = []
        
        best_aa_minus_af = -float('inf')
        best_epoch = -1
        best_model_state = None
        epoch_records = []
        
        early_stop = False
        last_two_records = []
        
        print(f"========== KD loss for task {task_id}: ==========\n")
        for epoch in range(num_epochs):
            if early_stop:
                print(f"Early stopping at epoch {epoch}, learned tasks AA > current task AA")
                break
                
            kd_optimizer.zero_grad()
            
            _, student_logits = self.global_model(synthetic_data)
            
            with torch.no_grad():
                _, teacher_logits = self.last_global_model(synthetic_data)
            
            loss_ce = nn.CrossEntropyLoss()(student_logits, synthetic_data.y)

            student_log_probs = F.softmax(student_logits / temperature, dim=1)
            teacher_probs = F.softmax(teacher_logits/ temperature, dim=1)
            
            student_log_probs_khop = get_khop_feature(student_log_probs, synthetic_data.edge_index, self.args.kd_khop)
            teacher_log_probs_khop = get_khop_feature(teacher_probs, synthetic_data.edge_index, self.args.kd_khop)

            loss_kd_low = edge_distribution_low(
                synthetic_data.edge_index,
                student_log_probs_khop,
                teacher_log_probs_khop
            )

            ce_weight = getattr(self.args, 'kd_ce_weight', 0.5)
            low_weight = getattr(self.args, 'kd_low_weight', 0.2)
            
            total_loss = ce_weight * loss_ce + low_weight * loss_kd_low

            current_epoch_in_task = len(self.kd_losses[task_id])
            self.kd_losses[task_id].append(total_loss.item())
            self.kd_loss_details[task_id].append({
                'epoch': current_epoch_in_task,
                'total_loss': total_loss.item(),
                'ce_loss': loss_ce.item(),
                'low_freq_loss': loss_kd_low.item()
            })
            
            total_loss.backward()
            kd_optimizer.step()
            
            if epoch % 10 == 0:
                print(f"KD Epoch {epoch}/{num_epochs}, Total Loss: {total_loss.item():.4f}, "
                      f"CE Loss: {loss_ce.item():.4f}, Low Freq Loss: {loss_kd_low.item():.8f}")
            
            if epoch > kd_eval_start_epoch and self.clients is not None:
                eval_result = self.evaluate_on_clients(task_id)
                weighted_aa = eval_result['weighted_aa']
                weighted_af = eval_result['weighted_af']
                weighted_current_task_acc = eval_result['weighted_current_task_acc']
                weighted_prev_tasks_acc = eval_result['weighted_prev_tasks_acc']
                
                aa_minus_af = weighted_aa - weighted_af
                
                current_record = {
                    'epoch': epoch, 
                    'aa': weighted_aa, 
                    'af': weighted_af, 
                    'aa_minus_af': aa_minus_af,
                    'current_task_acc': weighted_current_task_acc,
                    'prev_tasks_acc': weighted_prev_tasks_acc,
                    'model_state': copy.deepcopy(self.global_model.state_dict())
                }
                epoch_records.append(current_record)
                
                last_two_records.append(current_record)
                if len(last_two_records) > 2:
                    last_two_records.pop(0)
                
                print(f"KD Epoch {epoch}: AA = {weighted_aa:.4f}, AF = {weighted_af:.4f}, AA-AF = {aa_minus_af:.4f}")
                print(f"    Current task ACC = {weighted_current_task_acc:.4f}, Learned tasks ACC = {weighted_prev_tasks_acc:.4f}")
                weighted_task_accs = eval_result.get('weighted_task_accs', {})
                task_acc_str = ", ".join([f"Task{t}: {acc:.2f}%" for t, acc in sorted(weighted_task_accs.items())])
                print(f"    Task ACC: {task_acc_str}")
                
                if aa_minus_af >= best_aa_minus_af:
                    best_aa_minus_af = aa_minus_af
                    best_epoch = epoch
                    best_model_state = copy.deepcopy(self.global_model.state_dict())
                    print(f"  -> New best model at epoch {epoch} with AA-AF = {best_aa_minus_af:.4f}")
                
                if weighted_prev_tasks_acc > weighted_current_task_acc:
                    print(f"  -> Early stop: Learned tasks ACC({weighted_prev_tasks_acc:.4f}) > Current task ACC({weighted_current_task_acc:.4f})")
                    early_stop = True
        
        if early_stop and len(last_two_records) >= 2:
            if last_two_records[-1]['aa_minus_af'] >= last_two_records[-2]['aa_minus_af']:
                final_record = last_two_records[-1]
            else:
                final_record = last_two_records[-2]
            
            self.global_model.load_state_dict(final_record['model_state'])
            print(f"\nEarly stop: Selected model from epoch {final_record['epoch']}")
            print(f"AA-AF = {final_record['aa_minus_af']:.4f} (AA={final_record['aa']:.4f}, AF={final_record['af']:.4f})")
        elif early_stop and len(last_two_records) == 1:
            final_record = last_two_records[0]
            self.global_model.load_state_dict(final_record['model_state'])
            print(f"\nEarly stop: Using model from epoch {final_record['epoch']}")
            print(f"AA-AF = {final_record['aa_minus_af']:.4f} (AA={final_record['aa']:.4f}, AF={final_record['af']:.4f})")
        elif best_model_state is not None:
            self.global_model.load_state_dict(best_model_state)
            print(f"\nLoaded best model from epoch {best_epoch} with AA-AF = {best_aa_minus_af:.4f}")
        else:
            print(f"\nNo evaluation performed (epochs < {kd_eval_start_epoch}), using final model")
        
        if epoch_records:
            print(f"\nAA-AF records during KD training:")
            for record in epoch_records:
                print(f"  Epoch {record['epoch']}: AA = {record['aa']:.4f}, AF = {record['af']:.4f}, AA-AF = {record['aa_minus_af']:.4f}")

        print("Knowledge distillation completed!")
    
    def evaluate_on_clients(self, task_id):
        if self.clients is None:
            print("Warning: No clients available for evaluation")
            return {'weighted_aa': 0.0, 'weighted_af': 0.0, 'weighted_current_task_acc': 0.0, 'weighted_prev_tasks_acc': 0.0}
        
        self.message_pool["server"] = {
            "weight": list(self.global_model.parameters())
        }
        
        client_results = []
        client_nodes_nums = []
        
        for client in self.clients:
            result = client.evaluate_for_kd(task_id)
            nodes_num = client.get_learned_nodes_num(task_id)
            
            client_results.append(result)
            client_nodes_nums.append(nodes_num)
        
        total_nodes = sum(client_nodes_nums)
        if total_nodes == 0:
            return {'weighted_aa': 0.0, 'weighted_af': 0.0, 'weighted_current_task_acc': 0.0, 'weighted_prev_tasks_acc': 0.0, 'weighted_task_accs': {}}
        
        weighted_aa = sum(r['aa'] * nodes / total_nodes for r, nodes in zip(client_results, client_nodes_nums))
        weighted_af = sum(r['af'] * nodes / total_nodes for r, nodes in zip(client_results, client_nodes_nums))
        weighted_current_task_acc = sum(r['current_task_acc'] * nodes / total_nodes for r, nodes in zip(client_results, client_nodes_nums))
        weighted_prev_tasks_acc = sum(r['prev_tasks_acc'] * nodes / total_nodes for r, nodes in zip(client_results, client_nodes_nums))
        
        weighted_task_accs = {}
        for task_idx in range(task_id + 1):
            weighted_acc = 0.0
            for r, nodes in zip(client_results, client_nodes_nums):
                if 'task_accs' in r and task_idx in r['task_accs']:
                    weighted_acc += r['task_accs'][task_idx] * nodes / total_nodes
            weighted_task_accs[task_idx] = weighted_acc
        
        return {
            'weighted_aa': weighted_aa,
            'weighted_af': weighted_af,
            'weighted_current_task_acc': weighted_current_task_acc,
            'weighted_prev_tasks_acc': weighted_prev_tasks_acc,
            'weighted_task_accs': weighted_task_accs
        }

    def update_last_global_model(self):
        with torch.no_grad():
            for last_param, global_param in zip(self.last_global_model.parameters(), self.global_model.parameters()):
                last_param.data.copy_(global_param.data)
   
    def plot_loss_curves(self, save_dir="./loss_plots"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.generator_losses:
            for task_id, task_losses in self.generator_losses.items():
                if task_losses:
                    plt.figure(figsize=(12, 8))
                    epochs = range(len(task_losses))
                    plt.plot(epochs, task_losses, label=f'Task {task_id + 1}', marker='o', markersize=3)
                    
                    plt.title(f'Generator Total Loss - Task {task_id + 1} (All Communication Rounds)')
                    plt.xlabel('Continuous Epoch (All Rounds)')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'generator_total_loss_task_{task_id + 1}_{timestamp}.png'), dpi=300, bbox_inches='tight')
                    plt.close()
            
        if self.generator_loss_details:
            for task_id, task_details in self.generator_loss_details.items():
                if not task_details:
                    continue
                    
                plt.figure(figsize=(15, 10))
                
                epochs = [detail['epoch'] for detail in task_details]
                ce_losses = [detail['ce_loss'] for detail in task_details]
                kl_losses = [detail['kl_loss'] for detail in task_details]
                shigh_losses = [detail['shigh_loss'] for detail in task_details]
                total_losses = [detail['total_loss'] for detail in task_details]
                
                plt.subplot(2, 3, 1)
                plt.plot(epochs, ce_losses, 'b-', marker='o', markersize=2)
                plt.title('Cross Entropy Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 3, 2)
                plt.plot(epochs, kl_losses, 'g-', marker='o', markersize=2)
                plt.title('KL Divergence Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 3, 3)
                plt.plot(epochs, shigh_losses, 'orange', marker='o', markersize=2)
                plt.title('Shigh Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 3, 4)
                plt.plot(epochs, total_losses, 'k-', marker='o', markersize=2)
                plt.title('Total Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 3, 5)
                plt.plot(epochs, ce_losses, 'b-', label='CE Loss', linewidth=2)
                plt.plot(epochs, kl_losses, 'g-', label='KL Loss', linewidth=2)
                plt.plot(epochs, shigh_losses, 'orange', label='Shigh Loss', linewidth=2)
                plt.plot(epochs, total_losses, 'k-', label='Total Loss', linewidth=2)
                plt.title('All Generator Losses Combined')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.suptitle(f'Generator Loss Details - Task {task_id + 1} (All Communication Rounds)', fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'generator_detailed_loss_task_{task_id + 1}_{timestamp}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        if self.kd_losses:
            for task_id, task_losses in self.kd_losses.items():
                if task_losses:
                    plt.figure(figsize=(12, 8))
                    epochs = range(len(task_losses))
                    plt.plot(epochs, task_losses, label=f'Task {task_id + 1}', marker='o', markersize=3)
                    
                    plt.title(f'Knowledge Distillation Total Loss - Task {task_id + 1} (All Communication Rounds)')
                    plt.xlabel('Continuous Epoch (All Rounds)')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'kd_total_loss_task_{task_id + 1}_{timestamp}.png'), dpi=300, bbox_inches='tight')
                    plt.close()
            
        if self.kd_loss_details:
            for task_id, task_details in self.kd_loss_details.items():
                if not task_details:
                    continue
                    
                plt.figure(figsize=(12, 8))
                
                epochs = [detail['epoch'] for detail in task_details]
                ce_losses = [detail['ce_loss'] for detail in task_details]
                low_freq_losses = [detail['low_freq_loss'] for detail in task_details]
                total_losses = [detail['total_loss'] for detail in task_details]
                
                plt.subplot(2, 2, 1)
                plt.plot(epochs, ce_losses, 'b-', marker='o', markersize=2)
                plt.title('Cross Entropy Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 2, 2)
                plt.plot(epochs, low_freq_losses, 'g-', marker='o', markersize=2)
                plt.title('Low Frequency Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 2, 3)
                plt.plot(epochs, total_losses, 'k-', marker='o', markersize=2)
                plt.title('Total Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 2, 4)
                plt.plot(epochs, ce_losses, 'b-', label='CE Loss', linewidth=2)
                plt.plot(epochs, low_freq_losses, 'g-', label='Low Freq Loss', linewidth=2)
                plt.plot(epochs, total_losses, 'k-', label='Total Loss', linewidth=2)
                plt.title('All KD Losses Combined')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.suptitle(f'Knowledge Distillation Loss Details - Task {task_id + 1} (All Communication Rounds)', fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'kd_detailed_loss_task_{task_id + 1}_{timestamp}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"Server loss plots saved to: {save_dir}")

    def send_message(self):
        self.message_pool["server"] = {
            "weight" : list(self.global_model.parameters())   
        }
    
    def send_feature_gen(self):
        generator_weight = list(self.generator.parameters())
        
        self.message_pool["server_generator"] = {
            "weight": generator_weight
        }


class OursClient(BaseClient):   
    def __init__(self, args, client_id, data, message_pool, device, classes_order):
        super(OursClient, self).__init__(args, client_id, data)
        self.args = args
        self.device = device
        self.global_model = load_model(name = self.args.model, input_dim = args.input_dim, hidden_dim = args.hidden_dim, output_dim = args.output_dim, num_layers = args.num_layers, dropout = args.dropout).to(self.device)
        self.client_model = load_model(name = self.args.model, input_dim = args.input_dim, hidden_dim = args.hidden_dim, output_dim = args.output_dim, num_layers = args.num_layers, dropout = args.dropout).to(self.device)
        self.last_global_model = load_model(name = self.args.model, input_dim = args.input_dim, hidden_dim = args.hidden_dim, output_dim = args.output_dim, num_layers = args.num_layers, dropout = args.dropout).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.message_pool = message_pool
        self.data = data
        
        self.per_task_class_num = getattr(args, 'per_task_class_num', 2)
        self.train_prop = getattr(args, 'train_prop', 0.6)
        self.valid_prop = getattr(args, 'valid_prop', 0.2) 
        self.test_prop = getattr(args, 'test_prop', 0.2)
        self.shuffle_flag = getattr(args, 'shuffle_flag', False)
        
        self.tasks = class_to_task(data = data, per_task_class_num = self.per_task_class_num, train_prop = self.train_prop, valid_prop = self.valid_prop, test_prop = self.test_prop, shuffle_flag = self.shuffle_flag, classes_order = classes_order)
        
        print(f"Client {client_id} task set:")
        for i, task in enumerate(self.tasks):
            print(f"Client {self.client_id} - Task {i}:")
            print(f"  Training nodes: {task['train_mask'].sum().item()}")
            print(f"  Validation nodes: {task['valid_mask'].sum().item()}")
            print(f"  Test nodes: {task['test_mask'].sum().item()}")

            print("  Training classes:", torch.unique(task['local_data'].y[task['train_mask']]))
            print("  Validation classes:", torch.unique(task['local_data'].y[task['valid_mask']]))
            print("  Test classes:", torch.unique(task['local_data'].y[task['test_mask']]))
        self.local_epochs = args.local_epochs
        
        self.client_losses = {}
        self.generator_losses = {}
        self.generator_loss_details = {}

        self.noise_dim = getattr(args, 'noise_dim', 128)
        
        self.task_validation_acc = {}

    def train(self, task_id):
        if "server" in self.message_pool and "weight" in self.message_pool["server"]:
            with torch.no_grad():
                for (local_param_old, agg_global_param) in zip(self.client_model.parameters(), self.message_pool["server"]["weight"]):
                    local_param_old.data.copy_(agg_global_param)
            with torch.no_grad():
                for (local_param_old, agg_global_param) in zip(self.global_model.parameters(), self.message_pool["server"]["weight"]):
                    local_param_old.data.copy_(agg_global_param)
        
        task = self.tasks[task_id]
        global_model = self.global_model

        self.client_model.train()
        global_model.eval()
        
        local_data = task["local_data"]
        local_train_mask = task["train_mask"]
        whole_data = self.data.to(self.device)
        local_data = local_data.to(self.device)
        
        optimizer = torch.optim.Adam(self.client_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        
        print(f"========== Client {self.client_id} training loss for task {task_id}: ==========\n")
        
        if task_id not in self.client_losses:
            self.client_losses[task_id] = []
        
        for epoch in range(self.local_epochs):
            optimizer.zero_grad()
            
            _, local_student_out = self.client_model(local_data)
            loss = self.loss_fn(local_student_out[local_train_mask], whole_data.y[local_train_mask])
            loss.backward()
            optimizer.step()
            
            self.client_losses[task_id].append(loss.item())
            
            print(f"Epoch {epoch} loss: {loss}\n")

        return loss.item()
    
    def feature_gen_init(self, task_id):
        classes_num = task_id * self.per_task_class_num
        self.generator = Generator(noise_dim = self.noise_dim, input_dim = self.args.input_dim, output_dim = classes_num, dropout = self.args.dropout).to(self.device)

    def get_learned_graph(self, task_id):
        if task_id == 0:
            return
        
        all_nodes = []
        learned_classes = []
        
        for prev_task_id in range(task_id):
            task_data = self.tasks[prev_task_id]
            task_nodes_mask = task_data["train_mask"] | task_data["valid_mask"] | task_data["test_mask"]
            task_nodes_indices = torch.where(task_nodes_mask)[0].tolist()
            all_nodes.extend(task_nodes_indices)
            
            task_classes = torch.unique(self.data.y[task_nodes_indices]).tolist()
            learned_classes.extend(task_classes)
        
        all_nodes = sorted(list(set(all_nodes)))
        learned_classes = sorted(list(set(learned_classes)))
        
        subgraph = get_subgraph_by_node(self.data, all_nodes, True)

        return subgraph

    def get_learned_Homophily_Level(self, task_id):
        total_edge_nums = 0
        same_label_edges = 0
        
        for task_i in range(task_id):
            task = self.tasks[task_i]
            graph = task["local_data"]
            train_mask = task["train_mask"]
            
            edge_index = graph.edge_index
            node_labels = graph.y
            
            for edge_idx in range(edge_index.size(1)):
                src_node = edge_index[0, edge_idx].item()
                dst_node = edge_index[1, edge_idx].item()
                
                if train_mask[src_node] and train_mask[dst_node]:
                    total_edge_nums += 1
                    
                    if node_labels[src_node] == node_labels[dst_node]:
                        same_label_edges += 1
        
        if total_edge_nums > 0:
            homophily_ratio = same_label_edges / total_edge_nums
        else:
            homophily_ratio = 0.0
            
        return homophily_ratio

    
    def feature_gen_train(self, task_id, global_flag = True):
        if global_flag and "server_generator" in self.message_pool:
            with torch.no_grad():
                for local_param, server_param in zip(self.generator.parameters(), self.message_pool["server_generator"]["weight"]):
                    local_param.data.copy_(server_param)

        subgraph = self.get_learned_graph(task_id)
        subgraph_homo = self.get_learned_Homophily_Level(task_id)
        
        
        if subgraph is None:
            print(f"Client {self.client_id} Task {task_id}: No learned subgraph found, skipping generator training")
            return
            
        learned_classes = torch.unique(subgraph.y).tolist()
        
        self.generator.train()
        self.last_global_model.eval()
        
        gen_optimizer = torch.optim.Adam(self.generator.parameters(), 
                                       lr=getattr(self.args, 'gen_lr', 0.001))
        
        gen_epochs = getattr(self.args, 'gen_epochs', 50)
        num_samples_per_class = getattr(self.args, 'gen_num_samples_per_class', 200)
        
        if not learned_classes:
            print(f"Client {self.client_id} Task {task_id}: No learned classes found, skipping generator training")
            return
            
        subgraph = subgraph.to(self.device)
        
        print(f"Client {self.client_id} Task {task_id}: Starting generator training...")
        print(f"Learned classes: {learned_classes}")
        
        if task_id not in self.generator_losses:
            self.generator_losses[task_id] = []
            self.generator_loss_details[task_id] = []
        
        for epoch in range(gen_epochs):
            gen_optimizer.zero_grad()
            
            generated_features_list = []
            generated_labels_list = []
            
            for class_idx in learned_classes:
                noise = torch.randn(num_samples_per_class, self.noise_dim).to(self.device)
                class_labels = torch.full((num_samples_per_class,), class_idx, dtype=torch.long).to(self.device)
                
                generated_logits = self.generator(noise, class_labels)
                generated_features = F.normalize(generated_logits, p=2, dim=1)

                generated_features_list.append(generated_features)
                generated_labels_list.append(class_labels)
            
            all_generated_features = torch.cat(generated_features_list, dim=0)
            all_generated_labels = torch.cat(generated_labels_list, dim=0)
            
            num_nodes = all_generated_features.shape[0]
            num_edges = int(num_nodes * self.args.gen_num_nodes)

            row = torch.randint(0, num_nodes, (num_edges,), device=all_generated_features.device)
            col = torch.randint(0, num_nodes, (num_edges,), device=all_generated_features.device)
            edge_index = torch.stack([row, col], dim=0)


            synthetic_data = Data(x=all_generated_features, edge_index=edge_index, y=all_generated_labels).to(self.device)

            current_homo = get_Homophily_Level(synthetic_data)
            
            tolerance = self.args.tolerance
            max_iterations = self.args.max_iterations
            
            for iteration in range(max_iterations):
                if abs(current_homo - subgraph_homo) <= tolerance:
                    break
                    
                if current_homo > subgraph_homo:
                    edge_index = reduce_homophilic_edges(synthetic_data, edge_index, reduction_ratio=self.args.gen_reduction_ratio)
                else:
                    edge_index = reduce_heterophilic_edges(synthetic_data, edge_index, reduction_ratio=self.args.gen_reduction_ratio)
                
                synthetic_data.edge_index = edge_index
                current_homo = get_Homophily_Level(synthetic_data)

            _, global_predictions = self.last_global_model(synthetic_data)
            
            cross_entropy_loss = nn.CrossEntropyLoss()
            loss_ce = cross_entropy_loss(global_predictions, all_generated_labels)
            
            kl_losses = []
            for i, gen_label in enumerate(all_generated_labels):
                same_class_mask = (subgraph.y == gen_label)
                if same_class_mask.sum() > 0:
                    same_class_indices = torch.where(same_class_mask)[0]
                    random_idx = torch.randint(0, len(same_class_indices), (1,)).item()
                    real_feature = subgraph.x[same_class_indices[random_idx]]
                    
                    kl_loss = F.kl_div(
                        F.log_softmax(all_generated_features[i], dim = 0),
                        F.softmax(real_feature, dim = 0),
                        reduction = 'sum'
                    )

                    kl_losses.append(kl_loss)
            
            if kl_losses:
                loss_kl = torch.stack(kl_losses).mean()
            else:
                loss_kl = torch.tensor(0.0, device=self.device)
            
            ce_weight = getattr(self.args, 'gen_ce_weight', 1.0)
            kl_weight = getattr(self.args, 'gen_kl_weight', 0.5)
            
            total_loss = ce_weight * loss_ce + kl_weight * loss_kl 
            
            current_epoch_in_task = len(self.generator_losses[task_id])
            self.generator_losses[task_id].append(total_loss.item())
            self.generator_loss_details[task_id].append({
                'epoch': current_epoch_in_task,
                'total_loss': total_loss.item(),
                'ce_loss': loss_ce.item(),
                'kl_loss': loss_kl.item() if isinstance(loss_kl, torch.Tensor) else loss_kl,
                'shigh_loss': 0.0
            })
            
            total_loss.backward()
            gen_optimizer.step()
            
            if epoch % 50 == 0:
                print(f"SYS Epoch {epoch}/{gen_epochs}, "
                      f"Total loss: {total_loss.item():.4f}, "
                      f"CE loss: {loss_ce.item():.4f}, "
                      f"KL loss: {loss_kl.item():.4f}")
        
        print(f"Client {self.client_id} Task {task_id}: Generator training completed!")

    def plot_generator_loss_curves(self, save_dir="./loss_plots"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.generator_losses:
            for task_id, task_losses in self.generator_losses.items():
                if task_losses:
                    plt.figure(figsize=(12, 8))
                    epochs = range(len(task_losses))
                    plt.plot(epochs, task_losses, label=f'Task {task_id + 1}', marker='o', markersize=3)
                    
                    plt.title(f'Client {self.client_id} - Generator Total Loss - Task {task_id + 1}')
                    plt.xlabel('Continuous Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'client_{self.client_id}_generator_loss_task_{task_id + 1}_{timestamp}.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
        
        if self.generator_loss_details:
            for task_id, task_details in self.generator_loss_details.items():
                if not task_details:
                    continue
                    
                plt.figure(figsize=(15, 10))
                
                epochs = [detail['epoch'] for detail in task_details]
                ce_losses = [detail['ce_loss'] for detail in task_details]
                kl_losses = [detail['kl_loss'] for detail in task_details]
                shigh_losses = [detail['shigh_loss'] for detail in task_details]
                total_losses = [detail['total_loss'] for detail in task_details]
                
                plt.subplot(2, 3, 1)
                plt.plot(epochs, ce_losses, 'b-', marker='o', markersize=2)
                plt.title('Cross Entropy Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 3, 2)
                plt.plot(epochs, kl_losses, 'g-', marker='o', markersize=2)
                plt.title('KL Divergence Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 3, 3)
                plt.plot(epochs, shigh_losses, 'orange', marker='o', markersize=2)
                plt.title('Shigh Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 3, 4)
                plt.plot(epochs, total_losses, 'k-', marker='o', markersize=2)
                plt.title('Total Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 3, 5)
                plt.plot(epochs, ce_losses, 'b-', label='CE Loss', linewidth=2)
                plt.plot(epochs, kl_losses, 'g-', label='KL Loss', linewidth=2)
                plt.plot(epochs, shigh_losses, 'orange', label='Shigh Loss', linewidth=2)
                plt.plot(epochs, total_losses, 'k-', label='Total Loss', linewidth=2)
                plt.title('All Generator Losses Combined')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.suptitle(f'Client {self.client_id} - Generator Loss Details - Task {task_id + 1}', fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'client_{self.client_id}_generator_detailed_loss_task_{task_id + 1}_{timestamp}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"Client {self.client_id} generator loss plots saved to: {save_dir}")

    def get_subgraph(self, task_id):
        if task_id == 0:
            return 0.0
            
        nodes_list = []
        
        original_data = self.data 
        
        for i in range(task_id):
            task = self.tasks[i]
            nodes_mask = task["train_mask"] | task["valid_mask"] | task["test_mask"]
            
            nodes_indices = torch.where(nodes_mask)[0].tolist()
            nodes_list.extend(nodes_indices)
        
        subgraph = get_subgraph_by_node(original_data, nodes_list, True)

        return subgraph    

    def evaluate(self, task_id, global_flag = True, mask = "test_mask"):
        task = self.tasks[task_id]
        
        if global_flag:
            print(f"Evaluating with global model for task {task_id}")
            client_param_copy = copy.deepcopy(list(self.client_model.parameters()))
            with torch.no_grad():
                for(client_param, global_param) in zip(self.client_model.parameters(), self.message_pool["server"]["weight"]):
                    client_param.data.copy_(global_param)

        self.client_model.eval()
        
        data = task["local_data"]
        
        data = data.to(self.device)
        
        with torch.no_grad():
            _, out = self.client_model(data)
            
            loss = self.loss_fn(out[task[mask]], data.y[task[mask]])
            
            _, pred = out.max(dim=1)
            correct = pred[task[mask]].eq(data.y[task[mask]]).sum().item()
            acc = (correct / task[mask].sum().item()) * 100
        
        print("True labels:", data.y[task[mask]])
        print("Predicted labels:", pred[task[mask]])
        print("Class distribution:", torch.unique(data.y[task[mask]]), torch.unique(pred[task[mask]]))

        if global_flag:
            with torch.no_grad():
                for(global_param, client_param) in zip(self.client_model.parameters(), client_param_copy):
                    global_param.data.copy_(client_param)
        
        return {"loss": loss.item(), "acc": acc}
    
    def test(self, task_id):
        task = self.tasks[task_id]
        
        self.client_model.eval()
        
        data = task["local_data"]
        test_mask = task["test_mask"]
        
        data = data.to(self.device)
        
        with torch.no_grad():
            _, out = self.client_model(data)
            
            _, pred = out.max(dim=1)
            correct = pred[test_mask].eq(data.y[test_mask]).sum().item()
            acc = correct / test_mask.sum().item()
            
        return {"acc": acc}
    
    def evaluate_for_kd(self, current_task_id):
        if current_task_id == 0:
            return {'aa': 0.0, 'af': 0.0, 'current_task_acc': 0.0, 'prev_tasks_acc': 0.0}
        
        client_param_copy = copy.deepcopy(list(self.client_model.parameters()))
        
        if "server" in self.message_pool and "weight" in self.message_pool["server"]:
            with torch.no_grad():
                for client_param, global_param in zip(self.client_model.parameters(), self.message_pool["server"]["weight"]):
                    client_param.data.copy_(global_param)
        
        self.client_model.eval()
        
        task_accs = {}
        
        for prev_task_id in range(current_task_id + 1):
            task = self.tasks[prev_task_id]
            data = task["local_data"].to(self.device)
            valid_mask = task["valid_mask"]
            
            if valid_mask.sum() == 0:
                task_accs[prev_task_id] = 0.0
                continue
            
            with torch.no_grad():
                _, out = self.client_model(data)
                _, pred = out.max(dim=1)
                correct = pred[valid_mask].eq(data.y[valid_mask]).sum().item()
                acc = (correct / valid_mask.sum().item()) * 100
                task_accs[prev_task_id] = acc
        
        with torch.no_grad():
            for client_param, saved_param in zip(self.client_model.parameters(), client_param_copy):
                client_param.data.copy_(saved_param)
        
        aa = sum(task_accs.values()) / len(task_accs) if task_accs else 0.0
        
        af = 0.0
        if current_task_id > 0:
            forgetting_sum = 0.0
            for prev_task_id in range(current_task_id):
                if prev_task_id in self.task_validation_acc:
                    original_acc = self.task_validation_acc[prev_task_id]
                    current_acc = task_accs.get(prev_task_id, 0.0)
                    forgetting_sum += (original_acc - current_acc)
            af = forgetting_sum / current_task_id
        
        current_task_acc = task_accs.get(current_task_id, 0.0)
        
        prev_tasks_acc = 0.0
        if current_task_id > 0:
            prev_accs = [task_accs.get(i, 0.0) for i in range(current_task_id)]
            prev_tasks_acc = sum(prev_accs) / len(prev_accs) if prev_accs else 0.0
        
        return {'aa': aa, 'af': af, 'current_task_acc': current_task_acc, 'prev_tasks_acc': prev_tasks_acc, 'task_accs': task_accs}
    
    def save_validation_acc(self, task_id):
        task = self.tasks[task_id]
        data = task["local_data"].to(self.device)
        valid_mask = task["valid_mask"]
        
        if valid_mask.sum() == 0:
            self.task_validation_acc[task_id] = 0.0
            return
        
        self.client_model.eval()
        with torch.no_grad():
            _, out = self.client_model(data)
            _, pred = out.max(dim=1)
            correct = pred[valid_mask].eq(data.y[valid_mask]).sum().item()
            acc = (correct / valid_mask.sum().item()) * 100
        
        self.task_validation_acc[task_id] = acc
        print(f"Client {self.client_id} Task {task_id} validation accuracy saved: {acc:.2f}%")
    
    def get_learned_nodes_num(self, current_task_id):
        if current_task_id == 0:
            return 0
        
        total_nodes = 0
        for prev_task_id in range(current_task_id + 1):
            task = self.tasks[prev_task_id]
            nodes_mask = task["train_mask"] | task["valid_mask"] | task["test_mask"]
            total_nodes += nodes_mask.sum().item()
        
        return total_nodes
    
    def update_global_model(self):
        self.global_model.load_state_dict(self.message_pool["global_model_params"])
    
    def update_client_model(self):
        self.client_model.load_state_dict(self.global_model.state_dict())


    def get_task_nodes_num(self, task_id):
        task = self.tasks[task_id]
        nodes_mask = task["train_mask"] | task["valid_mask"] | task["test_mask"]
        nodes_num = nodes_mask.sum()
        return nodes_num 

    def get_mean_var(self, task_id):
        task = self.tasks[task_id]
        local_data = task["local_data"] 
        
        node_features = local_data.x
        
        mean = torch.mean(node_features, dim=0)
        var = torch.var(node_features, dim=0, unbiased=False)
        
        return {
            "mean": mean,
            "var": var
        }


    def send_message(self, task_id):
        if task_id == 0:
            learned_nodes_num = 0
        else:
            subgraph = self.get_subgraph(task_id = task_id)
            learned_nodes_num = subgraph.x.shape[0]

        self.message_pool[f"client_{self.client_id}"] = {
            "nodes_num" : self.get_task_nodes_num(task_id),
            "learned_nodes_num" : learned_nodes_num,
            "mean_var" : self.get_mean_var(task_id),
            "weight" : list(self.client_model.parameters())
        }

    def send_feature_gen(self, task_id):
        subgraph = self.get_learned_graph(task_id)
        
        S_high = get_Shigh(subgraph, self.args)
        
        learned_nodes_num = subgraph.x.shape[0]
        
        generator_weight = list(self.generator.parameters())
        
        self.message_pool[f"client_{self.client_id}_generator"] = {
            "generator_weight": generator_weight,
            "learned_nodes_num": learned_nodes_num,
            "S_high": S_high
        }


def plot_all_losses(server, clients, save_dir="./loss_plots"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    server.plot_loss_curves(save_dir)
    
    for client in clients:
        if hasattr(client, 'generator_losses') and client.generator_losses:
            client.plot_generator_loss_curves(save_dir)
    
    if clients and clients[0].client_losses:
        all_task_ids = set()
        for client in clients:
            all_task_ids.update(client.client_losses.keys())
        
        for task_id in sorted(all_task_ids):
            plt.figure(figsize=(12, 8))
            
            for client in clients:
                if task_id in client.client_losses and client.client_losses[task_id]:
                    epochs = range(len(client.client_losses[task_id]))
                    plt.plot(epochs, client.client_losses[task_id], 
                            label=f'Client {client.client_id}', 
                            marker='o', markersize=3)
            
            plt.title(f'Client Local Training Loss - Task {task_id + 1} (All Communication Rounds)')
            plt.xlabel('Continuous Epoch (All Rounds)')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'clients_local_loss_task_{task_id + 1}_{timestamp}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        for client in clients:
            if client.client_losses:
                plt.figure(figsize=(12, 8))
                
                for task_id, task_losses in client.client_losses.items():
                    if task_losses:
                        epochs = range(len(task_losses))
                        plt.plot(epochs, task_losses, 
                                label=f'Task {task_id + 1}', 
                                marker='o', markersize=3)
                
                plt.title(f'Client {client.client_id} - Local Training Loss Over Tasks (Each Task All Rounds)')
                plt.xlabel('Continuous Epoch (All Rounds per Task)')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'client_{client.client_id}_loss_over_tasks_{timestamp}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        plt.figure(figsize=(15, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(clients)))
        
        for client_id, client in enumerate(clients):
            if client.client_losses:
                for task_id, task_losses in client.client_losses.items():
                    if task_losses:
                        global_epochs = [epoch + task_id * len(task_losses) * 1.2 for epoch in range(len(task_losses))]
                        plt.plot(global_epochs, task_losses, 
                                color=colors[client_id], 
                                linestyle='-' if task_id == 0 else '--' if task_id == 1 else ':',
                                label=f'Client {client_id} Task {task_id + 1}', 
                                marker='o', markersize=2)
        
        plt.title('All Clients Local Training Loss - All Tasks (Each Task All Rounds)')
        plt.xlabel('Global Training Progress (Continuous Epochs)')
        plt.ylabel('Loss')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'all_clients_all_tasks_loss_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"All loss plots saved to: {save_dir}")
    return save_dir

def load_server_clients(args, data, device, classes_num):
    message_pool = {}
    clients_num = args.clients_num
    server = OursServer(args, message_pool, device)

    classes_order = list(range(classes_num))
    if args.shuffle_flag:
        random.seed(args.seed)
        random.shuffle(classes_order)

    clients = [OursClient(args, client_id, data[client_id], message_pool, device, classes_order) for client_id in range(clients_num)]
    
    server.set_clients(clients)

    return server, clients, message_pool