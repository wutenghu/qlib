# Double Ensemble 集成策略详解

Double Ensemble 是一种基于**样本重加权 (Sample Reweighting, SR)** 和**特征选择 (Feature Selection, FS)** 的集成学习框架，专为金融数据场景设计。核心实现在 `qlib/contrib/model/double_ensemble.py` 的 `DEnsembleModel` 类中。

相关论文：*DoubleEnsemble: A New Ensemble Method Based on Sample Reweighting and Feature Selection for Financial Data Analysis* ([arxiv](https://arxiv.org/pdf/2010.01265.pdf))

---

## 核心思想

金融预测面临两大挑战：**低信噪比**和**特征冗余**。Double Ensemble 通过双轮驱动机制同时应对这两个问题：

1. **SR 模块** — 识别关键样本，调整样本权重
2. **FS 模块** — 识别关键特征，筛选特征子集

每训练完一个子模型，就利用其训练轨迹进行样本重加权和特征选择，然后基于新的权重和特征子集训练下一个子模型。

---

## Label 定义：从绝对收益到截面排序

### 原始 Label

定义在 `qlib/contrib/data/handler.py:151-152`：

```python
def get_label_config(self):
    return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]
```

在 Qlib 中，`Ref(X, -n)` 表示**未来第 n 期**的值（负号表示向未来看）：

| 符号 | 含义 |
|------|------|
| `$close` | 当日收盘价 |
| `Ref($close, -1)` | 次日收盘价 |
| `Ref($close, -2)` | 次次日收盘价 |

**原始 Label = T+2日收盘价 / T+1日收盘价 - 1**

即"T+1日收盘买入、T+2日收盘卖出"的收益率。选择 T+2/T+1 而非 T+1/T 是为了避免前视偏差——T日收盘做出预测，最早能在T+1日开盘执行买入，成交价约为T+1日收盘价（配置中 `deal_price: close`），持有收益从T+2日起算。

此时 label 仍是**绝对值**，如 0.02 表示涨2%，-0.01 表示跌1%。

### 截面标准化（关键转换）

Alpha158 默认的 `learn_processors`（`handler.py:37-40`）：

```python
_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]
```

`CSZScoreNorm`（`processor.py:300-323`）对每个交易日独立做 Z-Score 标准化：

```
label_normalized[i] = (label[i] - mean(label_当日所有股票)) / std(label_当日所有股票)
```

转换后含义变化：
- **原始 label**：股票A涨2%，股票B涨1% → A=0.02, B=0.01
- **截面 Z-Score 后**：A 的 z-score > B 的 z-score，表示 A 的收益率在当天全市场中相对偏高

Alpha360 配置中使用的是 `CSRankNorm`（`processor.py:326-359`），处理更直白：

```python
t = df[cols].groupby("datetime", group_keys=False).rank(pct=True)  # 百分位排名 [0,1]
t -= 0.5       # 中心化到 [-0.5, 0.5]
t *= 3.46      # 缩放到近似标准正态分布 std≈1
```

例如某日500只股票，收益最高者 rank=1.0 → 处理后 `(1.0-0.5)*3.46 = 1.73`，最低者 → `-1.73`。

### 两种标准化方式对比

| | CSZScoreNorm (Alpha158默认) | CSRankNorm (Alpha360配置) |
|---|---|---|
| 操作 | `(x - mean) / std` | `rank(pct=True)` → 中心化 → ×3.46 |
| 对异常值 | 敏感（极端收益会被放大） | 鲁棒（只看排名，忽略绝对差距） |
| 信息保留 | 保留收益的相对差距大小 | 只保留排名顺序，丢弃间距信息 |
| 分布近似 | 近似正态 | 严格变换为均匀分布→近似正态 |

### 为什么使用截面排序而非绝对收益

策略（TopkDropoutStrategy）只关心排名（按分数排序选股），label 定义为截面排序与策略目标一致，原因有二：

1. **市场Beta剥离** — 某天大盘涨3%，所有股票绝对收益都偏高，但截面标准化后只关心"谁涨得更多"，自动去掉市场整体涨跌的影响
2. **噪声压制** — 绝对收益率噪声大（受大盘、行业轮动等影响），截面排名更稳定，信噪比更高

### 示例

假设某日3只股票：

| 股票 | 原始 label (T+2/T+1 收益率) | 截面排名 |
|------|-----|------|
| A | +3% | 最高 |
| B | +1% | 中间 |
| C | -2% | 最低 |

- **CSZScoreNorm 后**：A > 0 > C（具体值取决于均值和方差）
- **CSRankNorm 后**：A=1.73, B=0, C=-1.73

模型训练目标就是拟合这个截面标准化后的值。预测时分数高的股票排在前面被买入，分数低的被卖出——**和绝对收益率大小无关，只和相对排序有关**。

---

## 策略执行：基于 ML 预测的交易决策

策略使用 `TopkDropoutStrategy`（`qlib/contrib/strategy/signal_strategy.py:75-295`），核心逻辑在 `generate_trade_decision` 方法中。

模型输出的预测值是一个**评分（score）**，每只股票每天一个分数，分数越高代表预期收益越好。策略根据分数排名来决定持仓。

### 具体步骤（以 topk=50, n_drop=5 为例）

每个交易日：

1. **获取当日预测分数** — 从 signal 中拿到所有股票的 pred_score
2. **确定卖出列表** — 取当前持仓中，在合并排名（旧持仓+新候选）里排在最末尾的 `n_drop=5` 只股票；还需检查是否满足最低持有天数（`hold_thresh=1`）
3. **确定买入列表** — 从非持仓股票中，按 pred_score 降序取前 `n_drop + topk - len(当前持仓)` 只，即卖出几只就买入几只，维持总持仓 topk=50 只
4. **执行卖出** — 对卖出列表中的股票全额卖出，检查可交易性（涨跌停、停牌等），卖出所得加入现金
5. **执行买入** — 将可用现金的 95%（`risk_degree=0.95`）平均分配给买入列表中的股票

### 关键约束

| 约束 | 说明 |
|------|------|
| `limit_threshold: 0.095` | 涨跌停9.5%时不交易 |
| `deal_price: close` | 以收盘价成交 |
| `open_cost: 0.0005` | 买入手续费0.05% |
| `close_cost: 0.0015` | 卖出手续费0.15% |
| `hold_thresh: 1` | 至少持有1天才可卖出 |
| `forbid_all_trade_at_limit: True` | 涨跌停时禁止一切交易 |

### 端到端数据流

```
T日特征 (Alpha158/360因子)
        ↓
    DEnsembleModel
        ↓
    pred_score (每只股票一个分数，代表相对收益预期)
        ↓
    截面排名排序
        ↓
    TopkDropoutStrategy
        ↓
    卖出排名最末的5只，买入排名最高的新5只
        ↓
    维持50只股票的等权组合
        ↓
    次日收盘成交，持有至次次日
```

---

## 整体训练流程 (`fit` 方法，第65-103行)

```
初始化: weights = 全1, features = 全部特征
循环 k = 0 到 num_models-1:
    1. 用当前 weights 和 features 训练第 k 个子模型 (LightGBM)
    2. 如果是最后一个子模型，结束
    3. 提取当前子模型的 loss curve (训练轨迹)
    4. 计算当前集成在训练集上的 loss_values
    5. SR 模块: 基于 loss_curve 和 loss_values 计算新 weights
    6. FS 模块: 基于 loss_values 选择新 features 子集
最终预测: 所有子模型的加权平均
```

---

## SR 模块：样本重加权 (第140-173行)

目标：让后续子模型更关注"难学"的样本。

**步骤：**

1. **归一化** — 对 loss_curve 和 loss_values 做 rank 归一化（百分位排名），消除量纲影响

2. **计算 h 值** — 每个样本的 h 值由两部分组成：
   - **h₁** = 当前集成的 loss 排名（loss 越大，h₁ 越高）
   - **h₂** = `(l_end / l_start)` 的排名，其中：
     - `l_start` = loss_curve 前10%迭代的平均排名
     - `l_end` = loss_curve 后10%迭代的平均排名
     - h₂ 衡量的是**训练过程中 loss 下降的幅度**——比值越大说明该样本"越难学会"
   - **h = α₁·h₁ + α₂·h₂**，由 `alpha1` 和 `alpha2` 控制两部分权重

3. **分箱计算权重** — 将 h 值分为 `bins_sr`（默认10）个区间，每个区间内的样本获得权重：
   ```
   weight = 1 / (decay^k · h_avg_bin + 0.1)
   ```
   - h 值越高的区间（越难学的样本）权重越大
   - `decay` 参数使权重随子模型序号 k 衰减，避免后期过度关注噪声样本

**直觉**：h₁ 捕获"集成现在还做不好的样本"，h₂ 捕获"训练过程中进步缓慢的样本"。两者结合，既关注当前误差也关注学习难度。

---

## FS 模块：特征选择 (第175-219行)

目标：识别对预测真正重要的特征，减少冗余特征的干扰。

**步骤：**

1. **排列重要性评估** — 对每个特征，打乱 (shuffle) 该列的值，用当前集成模型预测，计算 loss 变化量：
   ```
   g_value = mean(loss_feat - loss_values) / (std(loss_feat - loss_values) + ε)
   ```
   - `loss_feat` = shuffle 该特征后的 loss
   - `loss_values` = 未 shuffle 的原始 loss
   - g_value 越大，说明该特征对预测越重要（shuffle 后 loss 增加越多）

2. **分箱采样** — 将 g_value 分为 `bins_fs`（默认5）个区间，重要性从高到低排列，每个区间按 `sample_ratios`（默认 `[0.8, 0.7, 0.6, 0.5, 0.4]`）随机采样特征：
   - 重要性最高的区间，保留 80% 的特征
   - 重要性最低的区间，只保留 40% 的特征

**直觉**：重要的特征保留更多，不重要的特征大幅删减，逐步聚焦于核心特征。

---

## 预测阶段 (`predict` 方法，第247-259行)

最终预测是所有子模型的加权平均：

```
pred = Σ(sub_weight[k] × sub_model[k].predict(X[:, sub_features[k]])) / Σ(sub_weight[k])
```

注意每个子模型使用的特征子集可能不同（`sub_features[k]`）。

---

## 关键超参数总结

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `num_models` | 6 | 子模型数量 |
| `enable_sr` | True | 是否启用样本重加权 |
| `enable_fs` | True | 是否启用特征选择 |
| `alpha1` | 1.0 | h₁（当前 loss）在 SR 中的权重 |
| `alpha2` | 1.0 | h₂（学习轨迹）在 SR 中的权重 |
| `bins_sr` | 10 | SR 分箱数 |
| `bins_fs` | 5 | FS 分箱数 |
| `decay` | None | SR 权重衰减系数 |
| `sample_ratios` | [0.8,0.7,0.6,0.5,0.4] | FS 各区间特征保留比例 |
| `sub_weights` | [1]*6 | 各子模型的预测权重 |

---

## 与传统集成的对比

- **Bagging** — 随机采样样本，平等对待所有特征
- **Boosting** — 基于残差调整样本权重，但特征不变
- **Double Ensemble** — 同时自适应调整样本权重（基于训练轨迹）和特征子集（基于排列重要性），是 Bagging 和 Boosting 思想的结合与增强
