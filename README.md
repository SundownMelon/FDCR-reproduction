# Parameter Disparities Dissection for Backdoor Defense in Heterogeneous Federated LearningAdd commentMore actions

> Parameter Disparities Dissection for Backdoor Defense in Heterogeneous Federated Learning,            
> Wenke Huang, Mang Ye, Zekun Shi, Guancheng Wan, Bo Du
> *NeurIPS, 2024*
> [Link]()

## Abstract

Backdoor attacks pose a serious threat to federated systems, where malicious clients optimize on the triggered distribution to mislead the global model towards a predefined target. Existing backdoor defense methods typically require either homogeneous assumption, validation datasets, or client optimization conflicts. In our work, we observe that benign heterogeneous distributions and malicious triggered distributions exhibit distinct parameter importance degrees. We introduce the Fisher Discrepancy Cluster and Rescale (FDCR) method, which utilizes Fisher Information to calculate the degree of parameter importance for local distributions. This allows us to reweight client parameter updates and identify those with large discrepancies as backdoor attackers. Furthermore, we prioritize rescaling important parameters to expedite adaptation to the target distribution, encouraging significant elements to contribute more while diminishing the influence of trivial ones. This approach enables FDCR to handle backdoor attacks in heterogeneous federated learning environments. Empirical results on various heterogeneous federated scenarios under backdoor attacks demonstrate the effectiveness of our method.

## Reproduction Experiments

This section provides commands to reproduce the FDCR experiments with different attack patterns and Non-IID settings.

### Prerequisites

```bash
# Install dependencies
pip install torch torchvision numpy yacs tqdm setproctitle hypothesis
```

### Quick Start

Run all experiments with default settings:

```bash
python run_experiments.py
```

### Single Experiment Commands

#### Base Backdoor Attack (Centralized Trigger)

```bash
# α=0.9 (near-IID)
python main.py --dataset fl_cifar10 --attack_type backdoor --optim FedFish --server OurRandomControl --seed 0 DATASET.beta=0.9 attack.backdoor.evils=base_backdoor

# α=0.1 (highly heterogeneous)
python main.py --dataset fl_cifar10 --attack_type backdoor --optim FedFish --server OurRandomControl --seed 0 DATASET.beta=0.1 attack.backdoor.evils=base_backdoor
```

#### DBA Attack (Distributed Backdoor Attack)

```bash
# α=0.9 (near-IID)
python main.py --dataset fl_cifar10 --attack_type backdoor --optim FedFish --server OurRandomControl --seed 0 DATASET.beta=0.9 attack.backdoor.evils=dba_backdoor

# α=0.1 (highly heterogeneous)
python main.py --dataset fl_cifar10 --attack_type backdoor --optim FedFish --server OurRandomControl --seed 0 DATASET.beta=0.1 attack.backdoor.evils=dba_backdoor
```

### Batch Experiment Runner

The `run_experiments.py` script automates running multiple experiment configurations:

```bash
# Run all combinations (2 attacks × 2 alphas × 3 seeds = 12 experiments)
python run_experiments.py

# Dry run (preview commands without execution)
python run_experiments.py --dry-run

# Custom attack patterns
python run_experiments.py --attacks base_backdoor dba_backdoor

# Custom alpha values
python run_experiments.py --alphas 0.9 0.5 0.1

# Custom seeds for reproducibility
python run_experiments.py --seeds 0 1 2 3 4

# Specify dataset
python run_experiments.py --dataset fl_cifar100

# Generate summary from existing results
python run_experiments.py --summary-only --results-dir ./data
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset name | `fl_cifar10` |
| `--attack_type` | Attack type (`backdoor`, `byzantine`, `None`) | `backdoor` |
| `--optim` | Federated optimizer | `FedFish` |
| `--server` | Aggregation server | `OurRandomControl` |
| `--seed` | Random seed | `0` |
| `DATASET.beta` | Dirichlet α for Non-IID | `1.0` |
| `attack.backdoor.evils` | Attack pattern (`base_backdoor`, `dba_backdoor`) | `base_backdoor` |
| `attack.bad_client_rate` | Fraction of malicious clients | `0.3` |
| `attack.noise_data_rate` | Fraction of poisoned samples | `0.5` |

### Output Metrics

Each experiment logs the following metrics:

- **ACC**: Global model accuracy on clean test set
- **ASR**: Attack success rate on backdoor test set
- **filtered_ratio**: Proportion of malicious clients correctly identified by FDCR
- **aggregation_weights**: Per-client weights assigned during aggregation

Results are saved to `./data/label_skew/{attack}/{bad_rate}/{dataset}/{alpha}/{server}/{optim}/`

### Expected Results

| Attack Pattern | α (Non-IID) | ACC (%) | ASR (%) | Filtered Ratio |
|----------------|-------------|---------|---------|----------------|
| base_backdoor | 0.9 | ~85 | <10 | >0.8 |
| base_backdoor | 0.1 | ~75 | <15 | >0.7 |
| dba_backdoor | 0.9 | ~85 | <10 | >0.8 |
| dba_backdoor | 0.1 | ~75 | <15 | >0.7 |

*Note: Actual results may vary based on hardware and random seed.*


## Citation
```
@inproceedings{FDCR_NeurIPS24,
    title    = {Parameter Disparities Dissection for Backdoor Defense in Heterogeneous Federated Learning},
    author    = {Huang, Wenke and Ye, Mang and Shi, Zekun and Wan, Guancheng and Du, Bo and Tao, Dacheng},
    booktitle = {NeurIPS},
    year      = {2024}
}
```

## Relevant Projects
[3] Rethinking Federated Learning with Domain Shift: A Prototype View - CVPR 2023 [[Link](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Rethinking_Federated_Learning_With_Domain_Shift_A_Prototype_View_CVPR_2023_paper.pdf)][[Code](https://github.com/WenkeHuang/RethinkFL)]

[2] Federated Graph Semantic and Structural Learning - IJCAI 2023 [[Link](https://marswhu.github.io/publications/files/FGSSL.pdf)][[Code](https://github.com/wgc-research/fgssl)]

[1] Learn from Others and Be Yourself in Heterogeneous Federated Learning - CVPR 2022 [[Link](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Learn_From_Others_and_Be_Yourself_in_Heterogeneous_Federated_Learning_CVPR_2022_paper.pdf)][[Code](https://github.com/WenkeHuang/FCCL)]