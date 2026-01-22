# <span style="color: rgb(21, 101, 192)"><span style="background-color: rgb(225, 245, 254)">ToDoList</span></span>

***

**From:** zotero://select/library/items/8NHZ8A8J

# Step 1：最小侵入式记录点（Instrumentation + Logging）

## 1.1 目标

在“去作弊版”FDCR 管线中，**把每轮每个客户端的关键中间量完整记录下来**，使你后续能回答：

*   **信号是否在 $F$（原始 Fisher）阶段就不存在？**
*   **还是被 min–max 归一化扭曲/抹平？**
*   **还是在 $g\odot I$ 的加权几何里被放大了 benign-benign 差异？**

> Step 1 不做任何新防御，只做“管线显微镜”。

***

## 1.2 记录范围（每轮、每客户端）

对每一轮 $t$，对每个参与客户端 $k$，记录以下对象（按优先级）：

### A. 必须记录（核心四件套）

1.  `F_raw`：未归一化的对角经验 Fisher 向量（或你选定层的子向量）

    *   定义对应论文中  $F_{w_k^t}(D_k)\approx\mathbb{E}[(\nabla\log p)^2]$ 。

    *   你要记录：

        *   原始数值（float32 足够）
        *   每层/每模块的切片范围（用于后续层级分析）

2.  `I_minmax`：论文版 per-client min–max 后的重要性向量 $I_k\in[0,1]$

    *   定义对应论文 min–max 公式。
    *   **同时记录 **`min_F`, `max_F`（很重要，用来诊断极值统治与尺度差异）

3.  `delta_w` 或 `g`：客户端更新（两者至少存一个）

    *   `delta_w = w_k^t - w^t`（最原始，最不依赖学习率细节）

    *   或 `g = (w_k^t - w^t)/eta`（对应论文的  $g_k^t$ ）。

    *   你要记录客户端本地的 **eta、local epochs、batch size、optimizer** 作为元数据（否则尺度不可解释）

4.  `g_weighted_minmax`：加权更新 $g_k^{e,t} = g_k^t \odot I_k$（使用 `I_minmax`）

    *   对应论文 Eq.(8)。
    *   这一步是 FDCR “放大差异/制造可分性”的关键中间量，必须存

***

## 1.3 分层粒度（如何定义“层/模块”）

你需要一个在你的实验框架里可复现的“层级切片协议”，用于 Step 2 的层级分解。建议按以下三档同时支持：

*   **L0（最粗）**：`head/classifier` vs `backbone_rest`
*   **L1（推荐）**：`last_block(s)`（最后 1–3 个模块） + `head`
*   **L2（可选）**：按具体参数张量（如 `layer4.2.conv2.weight`）精细切片

> 验收要求：每个记录的向量对象（`F_raw/I_minmax/delta_w/g/g_weighted`）都必须能被**一键切片到上述任意粒度**。

***

## 1.4 压缩与存储格式（给 IDE 的硬性要求）

为了便于后续相似度计算与 top-k/Jaccard，你需要同时存“全量/稀疏/统计”三类视图（至少两类）。

### A. 建议的最小组合（强烈推荐）

*   **全量向量**：只对 `head + last_block` 存全量（其余层可不存）
*   **top-k 稀疏表示**：对所有记录对象（至少 `I_minmax` 和 `g_weighted_minmax`）都存 top-k
*   **层级统计量**：每层存一组标量 stats（用于快速诊断）

### B. top-k 具体要求

对每个对象/每层，保存：

*   `topk_idx`：全局扁平索引（必须全局一致）
*   `topk_val`：对应数值
*   `k`：固定值或按层比例（建议先固定，比如 256/512）

> 重点：**必须定义“全局扁平索引映射”**（例如按参数张量顺序拼接并固定顺序），否则你后续没法做 top-k Jaccard 与一致性诊断。

### C. 层级 stats（每层至少存这些）

对 `F_raw` 与 `I_minmax`（每层）：

*   `l1`, `l2`，`mean`, `std`
*   `p50`, `p90`, `p99`（用于观察极值）
*   `min`, `max`（尤其对 `F_raw`）\
    对 `g` 与 `g_weighted`（每层）：
*   `l2`、`topk_mass`（top-k 值之和 / 总和）

***

## 1.5 目录结构与元数据（保证可复现实验）

建议每次 run 生成一个 `run_manifest.json`，并以 `run_id/round_x/client_y.*` 组织。

### 必须包含的元数据字段（manifest）

*   数据集、模型结构、随机种子
*   Non-IID 划分参数（如 Dirichlet α 或 shard 数）
*   攻击类型（base/DBA 等）、恶意比例、触发器参数
*   本地训练超参：eta、local epochs、batch size、optimizer、weight decay
*   本轮参与客户端列表（及其 benign/malicious ground-truth 标记，仅用于离线分析）
*   代码版本标识（commit hash/版本号）

> 验收要求：仅凭 manifest + logged 文件，可完全复现 Step 2 的全部图表与表格。

***

## 1.6 Step 1 的验收标准（你跑一轮就能自检）

完成 Step 1 后，你至少能回答这些“机械问题”：

*   任意轮、任意客户端，能取到 `F_raw/I_minmax/delta_w(or g)/g_weighted_minmax` 四件套
*   能按 L0/L1 粒度切片并计算每层 stats
*   能取出 `min_F/max_F` 并观察是否存在“单点极值统治”现象（例如 max 远超 p99）

***

# Step 2：三板斧信号体检图（No Defense Change, Only Analysis）

## 2.1 目标

用 Step 1 的日志，在不改防御算法的前提下，回答三件事：

1.  **信号在哪里？**（全模型 vs 最后层/模块）

2.  **信号是什么形态？**（benign-benign 与 benign-malicious 在  $I_k$  的几何距离上是否可分）

3.  **信号丢在哪一步？**（`F_raw` → `I_minmax` → `g_weighted` 的哪一段造成“去作弊后不可分”）

***

## 2.2 分析输入（从 Step 1 读取）

Step 2 默认使用以下对象做分析（你可以先只用 `I_minmax`，但建议同时对照 `F_raw` 与 `g_weighted`）：

*   `F_raw`（或其稳健归一化版本，Step 2.5 会做）
*   `I_minmax`
*   `g_weighted_minmax`
*   `delta_w/g`（用于一致性对照，可选）

分析维度：

*   round 维度：每轮独立画图 + 可选跨轮聚合
*   layer 维度：L0/L1/L2 三档切片（至少 L0、L1）
*   regime 维度：Non-IID 强弱（例如 α=0.9 vs α=0.1），攻击类型（base/DBA），恶意比例

***

## 2.3 体检一：层级/模块分解可分性（“信号是不是只在高层”）

### 要做什么

对每一轮 $t$，对每个层级切片 $l$（例如 head、last\_block、all），用 **仅 $I$（不看梯度/更新）** 的相似度来做两组分布：

*   $S_{bb}^{(l,t)}$ ：benign–benign 两两相似度集合

*   $S_{bm}^{(l,t)}$ ：benign–malicious 两两相似度集合

### 相似度指标（建议三种并行）

对同一对象（如 `I_minmax`）计算：

1.  **Cosine similarity**（看方向）
2.  **Spearman rank correlation**（看排序一致性，弱化尺度影响）
3.  **Top-k Jaccard**（看稀疏支持集是否一致）

> 这三种对 min–max 的敏感性不同：如果只在 rank/Jaccard 上可分而 cosine 不可分，往往意味着“尺度/极值问题”在作祟。

### 产物

*   每层/每轮一张图：bb vs bm 的分布（直方图/小提琴图/箱线图均可）

*   每层/每轮一张表：

    *   `mean_bb`, `std_bb`, `mean_bm`, `std_bm`
    *   `Δ = mean_bb - mean_bm`
    *   `effect_size = Δ / pooled_std`
    *   `overlap`（可用简单的分布重叠系数或 AUC proxy）

### 验收判据（你要看到的现象）

*   若 `head/last_block` 上 bb 与 bm 分离更明显，而 `all_layers` 混在一起：\
    **强烈提示“信号存在但被全模型噪声稀释”**（更偏向你之前的 (i) 或“信号集中于高层”）。

***

## 2.4 体检二：距离谱（只看 I\_k，判断 (ii) Non-IID 是否吞没信号）

### 要做什么

把 Step 2.3 的分布结果，按 Non-IID 强度分组对比（例如 α=0.9 vs α=0.1）：

*   对每个 α、每层  $l$ 、每轮  $t$  输出：`effect_size(l,t,alpha)`

*   再做跨轮聚合：例如取平均、取中位数、看方差

### 你要重点关注的诊断信号

*   **α=0.9**：若 bm 明显更远（相似度更低）且稳定
*   **α=0.1**：若 bb 自身方差显著增大、bb 与 bm 高度重叠\
    则基本坐实你之前的 (ii)：**强 Non-IID 下良性差异 > 恶意差异**，导致任何基于“跨客户端相似性”的检测都会变难。

### 产物

*   一张总览图：横轴层级（head/last\_block/all），纵轴 effect\_size，按 α 分两条曲线
*   一张总览表：每种 α 的 `bb variance`、`bm separation` 对比

***

## 2.5 体检三：稳健归一化替代 min–max（定位 (i) 的关键）

这一部分是你提出的核心困惑点的“实验化回答”：**min–max 是否是可分性崩溃的罪魁祸首**。

### 要做什么

对同一份 `F_raw`，构造三种替代归一化，得到 $\hat I_k$，并重复 Step 2.3/2.4 的全套距离谱分析，和 `I_minmax` 做对照。

#### 归一化方案 A：per-layer L1 归一化 + 截断

*   per-layer： $\hat I_{k,l} = F_{k,l} / (\|F_{k,l}\|_1 + \epsilon)$

*   再做 top-p% 保留（或分位裁剪），抑制极值统治

#### 归一化方案 B：rank-based scaling（每层按排名映射到 \[0,1]）

*   每层将元素按 rank 转为分位数（0\~1）
*   特点：对极值不敏感，保结构排序

#### 归一化方案 C：log 压缩 + 分位裁剪 + 再尺度化

*   $\tilde F = \log(1+F)$

*   clip 到  $[q_{0.01}, q_{0.99}]$

*   再做简单标准化或 L1 归一化

### 你要输出的关键对照表

对每个归一化方案（含 min–max baseline），每层、每个 α：

*   `effect_size_cosine`, `effect_size_spearman`, `effect_size_jaccard`
*   以及 `bb variance`（良性内部差异）与 `bm separation`（恶意分离度）

### 验收判据（用于定位 (i)）

*   如果 `F_raw → (A/B/C)` 能明显恢复分离，而 `F_raw → min–max` 不行：\
    **(i) 基本坐实：min–max（及其极值敏感）导致跨客户端可比性被破坏/可分性被抹平。**
*   如果所有归一化都不行，但 `F_raw` 本身在 head 层也不呈现 bm 可分：\
    更倾向 (ii) 或 (iii)（后者需要 Step 3 才能最终确认）。

***

## 2.6 Step 2 的最终交付物清单（你可以直接作为实验里程碑）

1.  **Figure-Set A（层级分解）**：

    *   每种 α、每种相似度指标、每个层级：bb vs bm 分布图（至少 head vs all）

2.  **Figure-Set B（Non-IID 对比）**：

    *   effect\_size 随 α 变化的对比图（每层一条）

3.  **Table-Set C（归一化消融）**：

    *   min–max vs A/B/C 的分离度总表（按层、按 α）

4.  **一段结论文字模板（你写报告用）**：

    *   “信号集中在哪些层/模块”
    *   “Non-IID 增强后 bb 方差是否吞没 bm”
    *   “替代归一化是否显著改善可分性”

***

# 你现在应该先做什么（按最省力、最高信息增益排序）

1.  **先完成 Step 1 的日志四件套 + L0/L1 切片协议**（只记录 head + last\_block 全量，其余 top-k + stats 即可）。
2.  **立刻跑 Step 2.3 + 2.4（层级分解 + 距离谱）**，先不做归一化替换。
3.  如果看到“可分性很差/随机性大”，再做 **Step 2.5（替代归一化）** 来定位是不是 min–max 的锅。

只要完成到第 3 步，你基本就能把“去作弊后信号消失”的责任定位到：

*   信号本就不存在（F\_raw 阶段就不可分），或
*   信号被 min–max/极值扭曲抹掉，或
*   Non-IID 让 bb 方差压过 bm 分离度。

如果你愿意，我也可以把上述 Step 1/2 再进一步细化成“IDE 任务卡片格式”（每张卡：输入、输出、依赖、验收、边界情况），方便你直接逐卡生成实现。
