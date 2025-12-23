# Requirements Document

## Introduction

本文档定义了复现FDCR（Fisher Discrepancy Cluster and Rescale）论文实验的需求。FDCR是一种用于异构联邦学习中后门防御的方法，通过Fisher信息矩阵计算参数重要性，识别并过滤恶意客户端。

复现实验目标：
- 两种攻击模式：集中触发（Centralized Backdoor）和DBA分布式触发（Distributed Backdoor Attack）
- 两种Non-IID强度：α=0.9（接近IID）和α=0.1（高度异构）
- 输出指标：ACC（准确率）、ASR（攻击成功率）、filtered_ratio（过滤比例）、权重统计

## Glossary

- **FDCR**: Fisher Discrepancy Cluster and Rescale，基于Fisher信息的后门防御方法
- **FIM**: Fisher Information Matrix，Fisher信息矩阵，用于衡量参数重要性
- **FCDC**: Fisher Client Discrepancy Cluster，Fisher客户端差异聚类模块
- **FPRA**: Fisher Parameter Rescale Aggregation，Fisher参数重缩放聚合模块
- **ACC**: Accuracy，全局模型在干净测试集上的准确率
- **ASR**: Attack Success Rate，攻击成功率，后门触发样本被误分类为目标标签的比例
- **Non-IID**: Non-Independent and Identically Distributed，非独立同分布数据
- **α (beta)**: Dirichlet分布的浓度参数，控制数据异构程度，值越小异构程度越高
- **Centralized Backdoor**: 集中触发攻击，所有恶意客户端使用相同的完整触发器
- **DBA**: Distributed Backdoor Attack，分布式后门攻击，不同恶意客户端使用触发器的不同部分
- **filtered_ratio**: 被FDCR识别并过滤的恶意客户端比例

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to implement DBA (Distributed Backdoor Attack) pattern, so that I can evaluate FDCR's defense capability against distributed trigger attacks.

#### Acceptance Criteria

1. WHEN a DBA attack is configured THEN the System SHALL distribute trigger patterns across multiple malicious clients such that each client receives a distinct subset of the full trigger
2. WHEN malicious clients with DBA attack perform local training THEN the System SHALL inject only their assigned trigger subset into poisoned samples
3. WHEN the full trigger pattern is applied during testing THEN the System SHALL evaluate ASR using the complete reconstructed trigger
4. WHEN DBA attack parameters are specified THEN the System SHALL support configurable number of trigger partitions matching the number of malicious clients

### Requirement 2

**User Story:** As a researcher, I want to run experiments with different Non-IID intensities (α=0.9 and α=0.1), so that I can evaluate FDCR's robustness under varying data heterogeneity levels.

#### Acceptance Criteria

1. WHEN α=0.9 is configured THEN the System SHALL generate client data distributions that are close to IID with mild label skew
2. WHEN α=0.1 is configured THEN the System SHALL generate highly heterogeneous client data distributions with severe label skew
3. WHEN data partitioning is performed THEN the System SHALL use Dirichlet distribution Dir(α) to allocate class samples to clients
4. WHEN experiments are run THEN the System SHALL support command-line parameter override for the beta (α) value

### Requirement 3

**User Story:** As a researcher, I want to collect ACC and ASR metrics during training, so that I can analyze the defense effectiveness over communication rounds.

#### Acceptance Criteria

1. WHEN each communication round completes THEN the System SHALL evaluate and log the global model accuracy (ACC) on the clean test set
2. WHEN each communication round completes THEN the System SHALL evaluate and log the attack success rate (ASR) on the backdoor test set
3. WHEN training completes THEN the System SHALL output steady-state ACC and ASR values from the final window of communication rounds
4. WHEN metrics are logged THEN the System SHALL save ACC and ASR time series to CSV files for post-analysis

### Requirement 4

**User Story:** As a researcher, I want to track filtered_ratio and aggregation weight statistics, so that I can understand FDCR's malicious client detection performance.

#### Acceptance Criteria

1. WHEN FDCR performs client clustering THEN the System SHALL log the indices of clients identified as benign and malicious
2. WHEN aggregation weights are computed THEN the System SHALL log the weight assigned to each client per round
3. WHEN each round completes THEN the System SHALL compute and log the filtered_ratio as the proportion of actual malicious clients correctly identified
4. WHEN experiments complete THEN the System SHALL output summary statistics including mean filtered_ratio and detection accuracy

### Requirement 5

**User Story:** As a researcher, I want to run batch experiments with different configurations, so that I can systematically compare results across attack patterns and Non-IID settings.

#### Acceptance Criteria

1. WHEN a batch experiment script is executed THEN the System SHALL run all combinations of attack patterns (Centralized, DBA) and Non-IID settings (α=0.9, α=0.1)
2. WHEN each experiment configuration runs THEN the System SHALL use consistent random seeds for reproducibility
3. WHEN all experiments complete THEN the System SHALL generate a summary table with ACC, ASR, and filtered_ratio for each configuration
4. WHEN experiments are run THEN the System SHALL support specifying the dataset (default: CIFAR-10) via command-line argument
