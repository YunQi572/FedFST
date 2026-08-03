# FedFST: Mitigating Spectral Catastrophic Forgetting in Federated Graph Continual Learning

FedFST is a spectral-aware framework for Federated Graph Continual Learning (FGCL). It studies catastrophic forgetting when graph neural networks are trained over distributed graph data under a continual task stream.

From the spectral perspective, forgetting in FGCL has two coupled forms:

- **High-frequency inconsistency forgetting**: newly introduced neighborhood inconsistency disrupts message passing and weakens discriminative historical knowledge.
- **Low-frequency consistency forgetting**: over-adaptation to new tasks dilutes stable global semantic consistency.

FedFST addresses these issues with:

- **HHKR**: Historical High-Frequency Knowledge Restoration.
- **HLST**: Historical Low-Frequency Semantic Transfer.

## Motivation

The figure below is redrawn from `motivations.pdf` as a self-contained GitHub-rendered diagram.

```mermaid
flowchart TB
    subgraph Top["Federated Graph Continual Learning"]
        Server["Server / Global Model"]
        Comm["Model Communication"]
        ClientModel["Client Model"]
        TaskT["Task t"]
        TaskNext["Task t + 1"]
        Client["Client i"]
        Server <--> Comm
        Comm <--> ClientModel
        TaskT --> TaskNext
        TaskNext --> Client
    end

    subgraph HF["(a) High-Frequency Forgetting"]
        HFAlter["Neighborhood Alteration"]
        HFProp["Inconsistency Propagation"]
        HFKnow["Inconsistent Knowledge"]
        HFForget["Inconsistency Forgetting"]
        HFDisc["Discriminated"]
        HFConf["Confused"]
        HFAlter --> HFProp
        HFProp --> HFKnow
        HFProp --> HFForget
        HFKnow --> HFDisc
        HFForget --> HFConf
    end

    subgraph LF["(b) Low-Frequency Forgetting"]
        LFAlter["Neighborhood Alteration"]
        LFProp["Consistency Propagation"]
        LFPass["Consistent Knowledge Passing"]
        LFDilute["Knowledge Dilution"]
        LFStable["Historical Semantics"]
        LFWeak["Diluted Semantics"]
        LFAlter --> LFProp
        LFProp --> LFPass
        LFPass --> LFDilute
        LFStable --> LFDilute
        LFDilute --> LFWeak
    end

    Top --> HF
    Top --> LF
    HF --> HHKR["HHKR restores high-frequency historical inconsistency knowledge"]
    LF --> HLST["HLST transfers low-frequency historical semantic knowledge"]
```

## Repository Structure

```text
.
|-- args.py
|-- main.py
|-- utils.py
|-- algorithm/
|   |-- Base.py
|   `-- Ours.py
|-- datasets/
|   |-- dataset_loader.py
|   `-- partition.py
`-- models/
    |-- GAT.py
    |-- GCN.py
    `-- model.py
```

## Environment Setup

Python 3.9-3.11 is recommended. The project depends on PyTorch, PyTorch Geometric, OGB, NetworkX, Louvain graph partitioning, and common scientific Python packages.

Create an environment:

```powershell
conda create -n fedfst python=3.10 -y
conda activate fedfst
python -m pip install --upgrade pip
```

Install PyTorch. Select one command according to your machine:

```powershell
# CPU
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Install PyTorch Geometric and its compiled extensions after PyTorch is installed:

```powershell
python -m pip install torch-geometric
$TORCH_VER = python -c "import torch; print(torch.__version__.split('+')[0])"
$CUDA_TAG = python -c "import torch; print('cpu' if torch.version.cuda is None else 'cu' + torch.version.cuda.replace('.', ''))"
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f "https://data.pyg.org/whl/torch-$TORCH_VER+$CUDA_TAG.html"
```

Install the remaining packages:

```powershell
python -m pip install numpy scipy pandas scikit-learn networkx python-louvain ogb gdown matplotlib tqdm
```

## Datasets

Supported `--dataset_name` values:

```text
cora, citeseer, pubmed, ogbn-arxiv, computers, physics, roman_empire, year, actor, cs
```

By default, datasets are stored under:

```text
datasets/raw_data
```

Use `--dataset_dir <path>` to specify another dataset directory.

## Run

Enter the repository directory:

```powershell
cd D:\CODING\VScode_coding\Research\FedFST\FedFST
```

Current `main.py` prints `kd_ce_weight`, `gen_ce_weight`, and `S_high_D_weight` before training. If these fields are not defined in your local `args.py`, use this no-code-change launcher:

```powershell
python -c "import args as a; a.parser.add_argument('--kd_ce_weight', type=float, default=0.5); a.parser.add_argument('--gen_ce_weight', type=float, default=1.0); a.parser.add_argument('--S_high_D_weight', type=float, default=0.5); import main; main.main()" `
  --dataset_name cora `
  --model GAT `
  --clients_num 3 `
  --rounds 10 `
  --local_epochs 3 `
  --gen_rounds 2 `
  --gen_epochs 200 `
  --kd_epochs 200 `
  --kd_lr 0.002 `
  --gen_lr 0.005 `
  --lr 0.005 `
  --per_task_class_num 2 `
  --device_id 0 `
  --save_dir .\outputs\cora_gat_seed24
```

For a quick smoke test:

```powershell
python -c "import args as a; a.parser.add_argument('--kd_ce_weight', type=float, default=0.5); a.parser.add_argument('--gen_ce_weight', type=float, default=1.0); a.parser.add_argument('--S_high_D_weight', type=float, default=0.5); import main; main.main()" `
  --dataset_name cora `
  --model GCN `
  --clients_num 2 `
  --rounds 1 `
  --local_epochs 1 `
  --gen_rounds 1 `
  --gen_epochs 1 `
  --kd_epochs 1 `
  --num_samples_per_class 5 `
  --device_id 0 `
  --save_dir .\outputs\smoke
```

If your `args.py` already defines those three fields, run the direct entry point:

```powershell
python main.py --dataset_name cora --model GAT --clients_num 3 --rounds 10 --device_id 0 --save_dir .\outputs\cora_gat_seed24
```

## Important Arguments

```text
--dataset_name           Dataset name.
--dataset_dir            Dataset root directory.
--model                  GNN backbone: GAT or GCN.
--clients_num            Number of federated clients.
--rounds                 Federated aggregation rounds per task.
--local_epochs           Local client training epochs per round.
--gen_rounds             Generator communication rounds.
--gen_epochs             Local feature generator epochs.
--kd_epochs              Server-side knowledge distillation epochs.
--kd_khop                K-hop smoothing depth for low-frequency distillation.
--per_task_class_num     Number of classes introduced per continual task.
--S_high_x_weight        Feature-side high-frequency score weight.
--S_high_A_weight        Structure-side high-frequency score weight.
--num_samples_per_class  Synthetic samples per historical class.
--device_id              CUDA device index when GPU is available.
--save_dir               Directory for generated loss plots.
```

## Outputs

Training prints task-wise global accuracy, the final accuracy matrix, AA, and AF. Loss curves are saved under `--save_dir`.

## Citation

```text
Hanyao Guo, Zihan Tan, Wenke Huang, Bin Yang, and Mang Ye. 2026.
FedFST: Mitigating Spectral Catastrophic Forgetting in Federated Graph
Continual Learning. In Proceedings of the 32nd ACM SIGKDD Conference on
Knowledge Discovery and Data Mining V.2 (KDD ’26), August 09–13, 2026,
Jeju Island, Republic of Korea. ACM, New York, NY, USA, 11 pages.
https://doi.org/10.1145/3770855.3817730
```
