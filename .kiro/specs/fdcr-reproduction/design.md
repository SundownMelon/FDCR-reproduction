# Design Document: FDCR Reproduction Experiments

## Overview

本设计文档描述了复现FDCR（Fisher Discrepancy Cluster and Rescale）论文实验所需的系统架构和实现细节。主要目标是：

1. 实现DBA（Distributed Backdoor Attack）分布式后门攻击模式
2. 支持不同Non-IID强度（α=0.9和α=0.1）的实验配置
3. 收集ACC、ASR、filtered_ratio等关键指标
4. 提供批量实验脚本和结果汇总功能

## Architecture

系统基于现有FDCR代码库扩展，采用模块化设计：

```
┌─────────────────────────────────────────────────────────────────┐
│                        Experiment Runner                         │
│  (run_experiments.py - 批量实验控制)                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                          main.py                                 │
│  (实验入口，参数解析，组件初始化)                                   │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Attack Module │     │  Optims Module  │     │  Server Module  │
│ (backdoor/)   │     │  (FedFish)      │     │ (OurRandomCtrl) │
│ - base_backdoor│    │  - Fisher计算    │     │ - FCDC聚类      │
│ - dba_backdoor │    │  - 本地训练      │     │ - FPRA聚合      │
└───────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Metrics Logger                             │
│  (utils/logger.py - ACC, ASR, filtered_ratio, weights)          │
└─────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. DBA Attack Module (`Attack/backdoor/utils.py`)

扩展现有后门攻击模块，添加DBA分布式攻击支持。

```python
def dba_backdoor(cfg, img, target, noise_data_rate, client_index, num_malicious_clients):
    """
    DBA分布式后门攻击
    
    Args:
        cfg: 配置对象
        img: 输入图像
        target: 原始标签
        noise_data_rate: 投毒比例
        client_index: 当前恶意客户端在恶意客户端列表中的索引 (0-based)
        num_malicious_clients: 恶意客户端总数
    
    Returns:
        img: 处理后的图像（可能带有部分触发器）
        target: 处理后的标签（可能被修改为目标标签）
    """
    pass

def get_dba_trigger_partition(full_trigger_positions, client_index, num_partitions):
    """
    获取DBA攻击中特定客户端的触发器分区
    
    Args:
        full_trigger_positions: 完整触发器位置列表
        client_index: 客户端索引
        num_partitions: 分区数量
    
    Returns:
        list: 该客户端负责的触发器位置子集
    """
    pass
```

### 2. FedFish Optimizer (`Optims/fedfish.py`)

实现带Fisher信息计算的联邦优化器。

```python
class FedFish(FederatedOptim):
    """
    带Fisher信息计算的联邦优化器
    用于FDCR防御方法
    """
    NAME = 'FedFish'
    
    def compute_fisher_information(self, net, data_loader):
        """
        计算Fisher信息矩阵（对角近似）
        
        Args:
            net: 神经网络模型
            data_loader: 数据加载器
        
        Returns:
            dict: 参数名到Fisher信息值的映射
        """
        pass
    
    def loc_update(self, priloader_list):
        """
        本地更新，包含Fisher信息计算
        """
        pass
```

### 3. Enhanced Server Module (`Server/OurRandomControl.py`)

增强服务器模块，添加filtered_ratio计算和详细日志。

```python
class OurRandomControl(ServerMethod):
    def __init__(self, args, cfg):
        # ... existing code ...
        self.filtered_ratio_history = []
        self.detection_results = []
    
    def compute_filtered_ratio(self, predicted_malicious, actual_malicious):
        """
        计算过滤比例
        
        Args:
            predicted_malicious: 预测的恶意客户端索引列表
            actual_malicious: 实际的恶意客户端索引列表
        
        Returns:
            float: 正确识别的恶意客户端比例
        """
        pass
    
    def server_update(self, **kwargs):
        """
        服务器更新，增加filtered_ratio计算和日志
        """
        pass
```

### 4. Metrics Logger Enhancement (`utils/logger.py`)

扩展日志模块，支持新指标。

```python
class CsvWriter:
    def write_filtered_ratio(self, filtered_ratio, epoch_index):
        """记录filtered_ratio"""
        pass
    
    def write_detection_results(self, benign_idx, evil_idx, epoch_index):
        """记录检测结果"""
        pass
    
    def write_summary(self, acc_list, asr_list, filtered_ratio_list):
        """生成实验汇总"""
        pass
```

### 5. Batch Experiment Runner (`run_experiments.py`)

批量实验脚本。

```python
def run_all_experiments(dataset='fl_cifar10', seeds=[0, 1, 2]):
    """
    运行所有实验配置组合
    
    Configurations:
    - Attack: ['base_backdoor', 'dba_backdoor']
    - Alpha: [0.9, 0.1]
    """
    pass

def generate_summary_table(results_dir):
    """
    生成实验结果汇总表
    
    Output columns:
    - Attack Pattern
    - Alpha (Non-IID)
    - ACC (steady-state)
    - ASR (steady-state)
    - Filtered Ratio (mean)
    """
    pass
```

## Data Models

### Configuration Schema

```yaml
# 扩展的配置结构
attack:
  bad_client_rate: 0.3
  noise_data_rate: 0.5
  backdoor:
    evils: 'base_backdoor'  # 或 'dba_backdoor'
    backdoor_label: 2
    trigger_position: [...]
    trigger_value: [...]
    # DBA特有配置
    dba_enabled: false
    dba_num_partitions: null  # null表示自动等于恶意客户端数

DATASET:
  beta: 0.5  # Dirichlet参数α
```

### Metrics Data Structure

```python
@dataclass
class ExperimentMetrics:
    """实验指标数据结构"""
    epoch: int
    acc: float  # 全局模型准确率
    asr: float  # 攻击成功率
    filtered_ratio: float  # 恶意客户端过滤比例
    benign_indices: List[int]  # 识别为良性的客户端
    malicious_indices: List[int]  # 识别为恶意的客户端
    aggregation_weights: List[float]  # 聚合权重

@dataclass
class ExperimentSummary:
    """实验汇总数据结构"""
    attack_pattern: str
    alpha: float
    steady_state_acc: float  # 最后10轮平均ACC
    steady_state_asr: float  # 最后10轮平均ASR
    mean_filtered_ratio: float  # 平均过滤比例
    detection_accuracy: float  # 检测准确率
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

Based on the prework analysis, the following correctness properties are identified:

### Property 1: DBA Trigger Partition Completeness and Disjointness
*For any* set of malicious clients and a full trigger pattern, the union of all client trigger partitions SHALL equal the full trigger pattern, and the intersection of any two client partitions SHALL be empty.
**Validates: Requirements 1.1, 1.4**

### Property 2: DBA Trigger Application Correctness
*For any* poisoned image created by a malicious client in DBA mode, only the trigger positions assigned to that specific client SHALL be modified from the original image.
**Validates: Requirements 1.2**

### Property 3: Data Heterogeneity Inverse Relationship with Alpha
*For any* two Dirichlet parameters α₁ > α₂, the label distribution variance across clients generated with α₁ SHALL be less than that generated with α₂.
**Validates: Requirements 2.1, 2.2**

### Property 4: Per-Round Metric Logging Completeness
*For any* completed communication round, the system SHALL have logged both ACC and ASR values for that round.
**Validates: Requirements 3.1, 3.2**

### Property 5: FDCR Detection Logging Completeness
*For any* completed communication round where FDCR performs clustering, the system SHALL have logged benign indices, malicious indices, and aggregation weights.
**Validates: Requirements 4.1, 4.2**

### Property 6: Filtered Ratio Computation Correctness
*For any* set of predicted malicious clients and actual malicious clients, filtered_ratio SHALL equal |predicted ∩ actual| / |actual|.
**Validates: Requirements 4.3**

### Property 7: Experiment Reproducibility
*For any* experiment configuration run twice with the same random seed, the ACC and ASR sequences SHALL be identical.
**Validates: Requirements 5.2**

## Error Handling

### Attack Configuration Errors
- 如果DBA攻击配置的分区数大于触发器位置数，抛出`ValueError`
- 如果恶意客户端数为0但启用了后门攻击，记录警告并跳过攻击

### Data Partitioning Errors
- 如果α值无效（≤0），抛出`ValueError`
- 如果客户端数据分配后某客户端数据量为0，记录警告

### Metric Logging Errors
- 如果无法写入CSV文件，记录错误并继续实验
- 如果指标计算出现NaN，记录警告并使用默认值

## Testing Strategy

### Dual Testing Approach

本项目采用单元测试和属性测试相结合的方式：

1. **单元测试**: 验证具体示例和边界情况
2. **属性测试**: 验证跨所有输入的通用属性

### Property-Based Testing Framework

使用 **Hypothesis** 库进行Python属性测试。

配置要求：
- 每个属性测试运行至少100次迭代
- 使用`@settings(max_examples=100)`装饰器

### Test Categories

#### Unit Tests
- DBA触发器分区函数的边界情况测试
- 配置解析的正确性测试
- CSV日志格式验证

#### Property-Based Tests
- Property 1: 触发器分区完整性和不相交性
- Property 2: DBA触发器应用正确性
- Property 3: 数据异构性与α的反向关系
- Property 6: filtered_ratio计算正确性
- Property 7: 实验可重复性

### Test File Structure

```
tests/
├── test_dba_attack.py          # DBA攻击相关测试
├── test_data_partition.py      # 数据分区测试
├── test_metrics.py             # 指标计算测试
└── test_reproducibility.py     # 可重复性测试
```

### Test Annotation Format

每个属性测试必须使用以下格式注释：
```python
# **Feature: fdcr-reproduction, Property {number}: {property_text}**
# **Validates: Requirements X.Y**
```
