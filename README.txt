FedFST: Mitigating Spectral Catastrophic Forgetting in Federated Graph Continual Learning
=====================================================================================

FedFST is a spectral-aware framework for Federated Graph Continual Learning (FGCL). It targets catastrophic forgetting in privacy-preserving graph neural network training under dynamic task streams.

The core idea is to treat forgetting as a dual spectral problem:

- High-frequency inconsistency forgetting: new neighborhood inconsistency disrupts message passing and erases discriminative historical knowledge.
- Low-frequency consistency forgetting: over-adaptation to new tasks dilutes stable global semantic consistency.

FedFST mitigates these problems with:

- HHKR: Historical High-Frequency Knowledge Restoration for preserving high-frequency inconsistency knowledge.
- HLST: Historical Low-Frequency Semantic Transfer for stabilizing low-frequency semantic consistency through spectral distillation.


Motivation
----------

The motivation figure below is redrawn from `motivations.pdf` as an embedded text figure so the repository can keep the change limited to README documentation.

```text
                              Federated graph continual learning

        Task t: global model <-------- communication --------> client model
                                      |
                                      v
                         Client i receives Task t + 1 graph data

+--------------------------------------------+--------------------------------------------+
| (a) High-Frequency Forgetting              | (b) Low-Frequency Forgetting               |
+--------------------------------------------+--------------------------------------------+
| Neighborhood alteration introduces          | Neighborhood alteration changes             |
| inconsistent / heterophilic signals.        | consistent / homophilic propagation.         |
|                                            |                                            |
| Inconsistency propagation                   | Consistency propagation                     |
|        |                                   |        |                                   |
|        v                                   |        v                                   |
| Historical inconsistent knowledge           | Consistent knowledge passing becomes         |
| that helped discrimination is forgotten.    | diluted by the new task stream.              |
|                                            |                                            |
| Result: discriminated old classes become    | Result: global low-frequency semantic        |
| confused after learning new classes.        | coherence is weakened.                      |
+--------------------------------------------+--------------------------------------------+

FedFST response:
  HHKR restores high-frequency historical inconsistency knowledge.
  HLST transfers low-frequency historical semantic knowledge.
```


Repository Layout
-----------------

```text
.
|-- args.py                      # command-line configuration
|-- main.py                      # training and evaluation entry point
|-- utils.py                     # spectral utilities, generator, metrics
|-- algorithm/
|   |-- Base.py
|   `-- Ours.py                  # FedFST server/client logic
|-- datasets/
|   |-- dataset_loader.py        # dataset loading and task construction
|   `-- partition.py             # Louvain client partitioning
`-- models/
    |-- GAT.py
    |-- GCN.py
    `-- model.py                 # model factory
```


Environment Setup
-----------------

The project is implemented in Python with PyTorch and PyTorch Geometric. A Python 3.9-3.11 environment is recommended.

Create and activate an environment:

```powershell
conda create -n fedfst python=3.10 -y
conda activate fedfst
python -m pip install --upgrade pip
```

Install PyTorch. Choose the command that matches your machine:

```powershell
# CPU
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Install PyTorch Geometric and the graph dependencies. Run this after PyTorch is installed so the wheel URL matches the installed Torch/CUDA build:

```powershell
python -m pip install torch-geometric
$TORCH_VER = python -c "import torch; print(torch.__version__.split('+')[0])"
$CUDA_TAG = python -c "import torch; print('cpu' if torch.version.cuda is None else 'cu' + torch.version.cuda.replace('.', ''))"
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f "https://data.pyg.org/whl/torch-$TORCH_VER+$CUDA_TAG.html"
```

Install the remaining dependencies:

```powershell
python -m pip install numpy scipy pandas scikit-learn networkx python-louvain ogb gdown matplotlib tqdm
```


Datasets
--------

Supported `--dataset_name` values:

```text
cora, citeseer, pubmed, ogbn-arxiv, computers, physics, roman_empire, year, actor, cs
```

By default, datasets are downloaded or loaded under:

```text
datasets/raw_data
```

Use `--dataset_dir <path>` to override the dataset root.


Run
---

Go to the repository root before running:

```powershell
cd D:\CODING\VScode_coding\Research\FedFST\FedFST
```

The current entry point reads `kd_ce_weight`, `gen_ce_weight`, and `S_high_D_weight` before training, while these options may be absent from `args.py` in this release. If your copy has not added them to `args.py`, use the following no-edit launcher:

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

For a quick smoke test, reduce the training budget:

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

If your local `args.py` already defines the three compatibility options, the direct entry point is:

```powershell
python main.py --dataset_name cora --model GAT --clients_num 3 --rounds 10 --device_id 0 --save_dir .\outputs\cora_gat_seed24
```


Important Arguments
-------------------

```text
--dataset_name           Dataset name.
--dataset_dir            Dataset root directory.
--model                  GNN backbone: GAT or GCN.
--clients_num            Number of federated clients.
--rounds                 Federated aggregation rounds per task.
--local_epochs           Client local training epochs per round.
--gen_rounds             Generator communication rounds.
--gen_epochs             Local feature generator epochs.
--kd_epochs              Server-side knowledge distillation epochs.
--per_task_class_num     Number of classes introduced per continual task.
--kd_khop                K-hop smoothing depth for low-frequency distillation.
--S_high_x_weight        Feature-side high-frequency score weight.
--S_high_A_weight        Structure-side high-frequency score weight.
--num_samples_per_class  Synthetic samples per historical class.
--device_id              CUDA device index when GPU is available.
--save_dir               Directory for loss plots and generated figures.
```


Outputs
-------

During training, the program prints:

- task-wise global accuracy,
- the final accuracy matrix,
- AA (Average Accuracy),
- AF (Average Forgetting).

Loss curves are saved under `--save_dir`.


Citation
--------

```bibtex
@inproceedings{guo2026fedfst,
  title     = {FedFST: Mitigating Spectral Catastrophic Forgetting in Federated Graph Continual Learning},
  author    = {Guo, Hanyao and Tan, Zihan and Huang, Wenke and Yang, Bin and Ye, Mang},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  year      = {2026},
  pages     = {11},
  publisher = {ACM},
  address   = {New York, NY, USA},
  doi       = {10.1145/3770855.3817730}
}
```
