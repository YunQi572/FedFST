import argparse
import os
import torch

parser = argparse.ArgumentParser()

current_path = os.path.abspath(__file__)
dataset_path = os.path.join(os.path.dirname(current_path), 'datasets')
root_dir = os.path.join(dataset_path, 'raw_data')
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

parser.add_argument("--dataset_dir", type=str, default=root_dir)

parser.add_argument('--train_prop', type=float, default=0.6)
parser.add_argument('--valid_prop', type=float, default=0.2)
parser.add_argument('--test_prop', type=float, default=0.2)
parser.add_argument('--shuffle_flag', type=bool, default=True)

parser.add_argument('--input_dim', type=int, default=1433)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--output_dim', type=int, default=7)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.5)

parser.add_argument('--noise_dim', type=int, default=128)
parser.add_argument('--temperature', type=float, default=1.0)

parser.add_argument('--threshold', type=float, default=0.5)

parser.add_argument('--weight_decay', type=float, default=5e-4)


parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--device_id', type=int, default=2)

parser.add_argument('--rounds', type=int, default=10)
parser.add_argument('--gen_rounds', type=int, default=2)
parser.add_argument('--local_epochs', type=int, default=3)
parser.add_argument('--kd_epochs', type=int, default=200)
parser.add_argument('--gen_epochs', type=int, default=200)
parser.add_argument('--kd_eval_start_epoch', type=int, default=1)

parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--gen_lr', type=float, default=0.005)
parser.add_argument('--kd_lr', type=float, default=0.002)
parser.add_argument('--kd_khop', type=int, default=2)

parser.add_argument('--kd_temperature', type=float, default=1.0)
parser.add_argument('--kd_low_weight', type=float, default=1)

parser.add_argument('--gen_kl_weight', type=float, default=100)

parser.add_argument('--num_samples', type=int, default=200)

parser.add_argument('--feature_prop', type=float, default=0.2)
parser.add_argument('--S_high_x_weight', type=float, default=0.5)
parser.add_argument('--S_high_A_weight', type=float, default=0.5)

parser.add_argument('--save_dir', type=str, default="./Seed26/Cora/Plot/Top")

parser.add_argument('--num_samples_per_class', type=int, default=300)

parser.add_argument("--dataset_name", type=str, default="cora")
parser.add_argument('--seed', type=int, default=24)
parser.add_argument('--model', type=str, default='GAT')

parser.add_argument('--gen_num_nodes', type=int, default=2)
parser.add_argument('--gen_reduction_ratio', type=float, default=0.05)
parser.add_argument('--tolerance', type=float, default=0.05)
parser.add_argument('--max_iterations', type=int, default=60)
parser.add_argument('--per_task_class_num', type=int, default=2)

parser.add_argument('--clients_num', type=int, default=3)
