# Batch Normalization 反向传播推导

## 1. 前向传播 (Forward Pass)

假设输入 $x$ 的形状是 $(N, D)$，我们针对某一个特征维度（某一列）进行推导（即 $x$ 是长度为 $N$ 的向量，$x = [x_1, \dots, x_N]^T$）。

计算步骤如下：

1. **均值 (Mean):**
    $$
    \mu = \frac{1}{N} \sum_{j=1}^N x_j
    $$

2. **方差 (Variance):**
    $$
    \sigma^2 = \frac{1}{N} \sum_{j=1}^N (x_j - \mu)^2
    $$

3. **标准化 (Normalize):**
    $$
    \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
    $$

4. **缩放和平移 (Scale and Shift):**
    $$
    y_i = \gamma \hat{x}_i + \beta
    $$

## 2. 反向传播 (Backward Pass)

我们已知损失函数 $L$ 对输出 $y$ 的梯度 $\frac{\partial L}{\partial y}$ (代码中的 `dout`)，目标是求 $\frac{\partial L}{\partial x}$ (代码中的 `dx`)。

### 第一步：求 $\hat{x}$ 的梯度 (`dx_hat`)

根据 $y_i = \gamma \hat{x}_i + \beta$，利用链式法则：
$$
\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma
$$
对应代码：

```python
dx_hat = dout * gamma
```

### 第二步：求方差 $\sigma^2$ 的梯度

$\sigma^2$ 影响了所有的 $\hat{x}_j$。根据 $\hat{x}_j = (x_j - \mu)(\sigma^2 + \epsilon)^{-1/2}$：
$$
\frac{\partial \hat{x}_j}{\partial \sigma^2} = (x_j - \mu) \cdot -\frac{1}{2}(\sigma^2 + \epsilon)^{-3/2}
$$
对所有样本求和：
$$
\frac{\partial L}{\partial \sigma^2} = \sum_{j=1}^N \frac{\partial L}{\partial \hat{x}_j} \frac{\partial \hat{x}_j}{\partial \sigma^2} = -\frac{1}{2}(\sigma^2 + \epsilon)^{-3/2} \sum_{j=1}^N \frac{\partial L}{\partial \hat{x}_j} (x_j - \mu)
$$

### 第三步：求均值 $\mu$ 的梯度

$\mu$ 既直接出现在 $\hat{x}$ 的分子中，也通过 $\sigma^2$ 间接影响 $\hat{x}$。
$$
\frac{\partial L}{\partial \mu} = \sum_{j=1}^N \frac{\partial L}{\partial \hat{x}_j} \frac{\partial \hat{x}_j}{\partial \mu} + \frac{\partial L}{\partial \sigma^2} \frac{\partial \sigma^2}{\partial \mu} 
$$

1. **第一项** (分子中的 $\mu$):
    $$
    \frac{\partial \hat{x}_j}{\partial \mu} = \frac{-1}{\sqrt{\sigma^2 + \epsilon}}
    $$
    
    $$
    \sum_{j=1}^N \frac{\partial L}{\partial \hat{x}_j} \frac{\partial \hat{x}_j}{\partial \mu} = \frac{-1}{\sqrt{\sigma^2 + \epsilon}} \sum_{j=1}^N \frac{\partial L}{\partial \hat{x}_j}
    $$

2. **第二项** (通过 $\sigma^2$):
    $$
    \frac{\partial \sigma^2}{\partial \mu} = \frac{1}{N} \sum_{j=1}^N -2(x_j - \mu) = -2 \cdot \frac{1}{N} \sum_{j=1}^N (x_j - \mu) = 0
    $$
    
    (因为 $\sum_{j=1}^N (x_j - \mu) = 0$ )

所以：
$$
\frac{\partial L}{\partial \mu} = \frac{-1}{\sqrt{\sigma^2 + \epsilon}} \sum_{j=1}^N \frac{\partial L}{\partial \hat{x}_j}
$$

### 第四步：求输入 $x_i$ 的梯度 (`dx`)

$x_i$ 影响 $L$ 的路径有三条：

1. 直接影响 $\hat{x}_i$
2. 通过 $\mu$ 影响所有 $\hat{x}$
3. 通过 $\sigma^2$ 影响所有 $\hat{x}$

$$
\frac{\partial L}{\partial x_i} = \underbrace{\frac{\partial L}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial x_i}}_{\text{Term 1}} + \underbrace{\frac{\partial L}{\partial \sigma^2} \frac{\partial \sigma^2}{\partial x_i}}_{\text{Term 2}} + \underbrace{\frac{\partial L}{\partial \mu} \frac{\partial \mu}{\partial x_i}}_{\text{Term 3}}
$$

我们分别计算这三项：

* **Term 1:**
    $$
    \frac{\partial \hat{x}_i}{\partial x_i} = \frac{1}{\sqrt{\sigma^2 + \epsilon}}
    $$
    
    $$
    \text{Term 1} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma^2 + \epsilon}}
    $$

* **Term 2:**
    $$
    \frac{\partial \sigma^2}{\partial x_i} = \frac{2(x_i - \mu)}{N}
    $$
    
    代入之前求得的 $\frac{\partial L}{\partial \sigma^2}$：
    $$
    \text{Term 2} = \left[ -\frac{1}{2}(\sigma^2 + \epsilon)^{-3/2} \sum_j \frac{\partial L}{\partial \hat{x}_j} (x_j - \mu) \right] \cdot \frac{2(x_i - \mu)}{N}
    $$
    利用 $x_j - \mu = \hat{x}_j \sqrt{\sigma^2 + \epsilon}$ 进行替换简化：
    $$
    \text{Term 2} = -\frac{1}{N\sqrt{\sigma^2 + \epsilon}} \hat{x}_i \sum_j \left( \frac{\partial L}{\partial \hat{x}_j} \hat{x}_j \right)
    $$

* **Term 3:**
    $$
    \frac{\partial \mu}{\partial x_i} = \frac{1}{N}
    $$
    代入之前求得的 $\frac{\partial L}{\partial \mu}$：
    $$
    \text{Term 3} = \frac{1}{N} \left[ \frac{-1}{\sqrt{\sigma^2 + \epsilon}} \sum_j \frac{\partial L}{\partial \hat{x}_j} \right]
    $$

### 第五步：合并与简化

令 $\text{std} = \sqrt{\sigma^2 + \epsilon}$，将三项加起来：

$$
\frac{\partial L}{\partial x_i} = \frac{1}{\text{std}} \frac{\partial L}{\partial \hat{x}_i} - \frac{1}{N \cdot \text{std}} \hat{x}_i \sum_j (\frac{\partial L}{\partial \hat{x}_j} \hat{x}_j) - \frac{1}{N \cdot \text{std}} \sum_j \frac{\partial L}{\partial \hat{x}_j}
$$

提取公因式 $\frac{1}{N \cdot \text{std}}$：

$$
\frac{\partial L}{\partial x_i} = \frac{1}{N \cdot \text{std}} \left[ N \cdot \frac{\partial L}{\partial \hat{x}_i} - \sum_j \frac{\partial L}{\partial \hat{x}_j} - \hat{x}_i \sum_j \left( \frac{\partial L}{\partial \hat{x}_j} \hat{x}_j \right) \right]
$$

## 3. 对应代码

```python
dx = (1. / N) * (1. / std) * (N * dx_hat - dx_hat.sum(dim=0) - x_hat * (dx_hat * x_hat).sum(dim=0))
```

* `1. / std` 对应公式中的 $\frac{1}{\text{std}}$
* `N * dx_hat` 对应 $N \cdot \frac{\partial L}{\partial \hat{x}_i}$
* `dx_hat.sum(dim=0)` 对应 $\sum_j \frac{\partial L}{\partial \hat{x}_j}$
* `x_hat * (dx_hat * x_hat).sum(dim=0)` 对应 $\hat{x}_i \sum_j (\frac{\partial L}{\partial \hat{x}_j} \hat{x}_j)$
