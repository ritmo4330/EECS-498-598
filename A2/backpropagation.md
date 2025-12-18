# two_layer_net 反向传播推导

## 问题描述

实现一个两层神经网络的反向传播算法，计算交叉熵损失 $L$ 对各个参数的梯度。

$$
L = \frac{1}{N}\cdot softmax(scores, y) + \lambda (\|W_1\|^2 + \|W_2\|^2)
$$

其中，$W_1$ 和 $W_2$ 分别是第一层和第二层的权重矩阵，$b_1$ 和 $b_2$ 是偏置向量，$X$ 是输入数据，$scores = ReLU(X\cdot W_1 + b_1)\cdot W_2 + b_2$ 是网络的输出分数，$softmax = -\sum_{i=1}^{N} \ln \left( \frac{e^{s_{y_i}}}{\sum_{j=1}^{C} e^{s_j}} \right)$ 是交叉熵损失函数，$y$ 是标签，$\lambda$ 是正则化系数。

## 要求

推导并实现损失函数 $L$ 对 $W_1, b_1, W_2, b_2$ 的梯度。

## 推导

1. **前向传播**：

    - 第一层输出：$H = ReLU(XW_1 + b_1)$
    - 第二层输出（得分）：$scores =s= HW_2 + b_2$
    - 交叉熵损失：$L_{data} = -\frac{1}{N}\sum_{i=1}^{N} \ln \left( \frac{e^{s_{i,y_i}}}{\sum_{j=1}^{C} e^{s_{ij}}} \right)$
    - 正则化损失：$L_{reg} = \lambda (\|W_1\|^2 + \|W_2\|^2)$
    - 总损失：$L = L_{data} + L_{reg}$

    其中，$N$ 是样本数，$C$ 是类别数。

2. **反向传播计算梯度**：

    以下推导中，仅考虑 $L_{data}$ 项，并简记为 $L$；使用 Einstein 求和约定，对重复索引进行求和，有的地方为了清晰起见，显式地写出了求和符号。
    - 对 $scores$ 的梯度：

      $$
        \begin{aligned}
        \frac{\partial L}{\partial s_{\alpha \beta}} &= \frac{1}{N} \frac{\partial}{\partial s_{\alpha \beta}} \sum_{i=1}^{N} \left( - s_{i,y_i} + \ln \sum_{k=1}^{C} e^{s_{ik}} \right) \\
        &= \frac{1}{N} \left( -\delta_{\beta, y_\alpha} + \frac{e^{s_{\alpha k}} \delta_{\beta k}}{\sum_{k=1}^{C} e^{s_{\alpha k}}} \right) \\
        &= \frac{1}{N} (p_{\alpha \beta} - \delta_{\beta, y_\alpha})
        \end{aligned}
      $$

      其中，$p_{ij} = \frac{e^{s_{ij}}}{\sum_{k=1}^{C} e^{s_{ik}}}$ 是样本 $i$ 属于类别 $j$ 的预测概率，$\delta$ 是 Kronecker delta 函数。

    - 对 $W_2$ 和 $b_2$ 的梯度：

      $$
        s_{ij} = H_{i \alpha} W_{2,\alpha j} + b_{2,j}
      $$

      - 其中，注意 $b_{2,j}$ 是对每个类别的偏置，不依赖于样本索引 $i$，因此在求导时需要特别注意。它实际上是通过广播机制（即将偏置向量加到每个样本的得分上，相当于在维度 $i$ 复制自身形成矩阵 $b_{2, ij}$）应用到每个样本 $i$ 的，则在反向传播时需要对所有样本 $i$ 进行求和。（见下面推导$\frac{\partial L}{\partial b_2}$）

      $$
        \frac{\partial s_{ij}}{\partial W_{2,\rho \nu}} = H_{i \alpha} \delta_{\rho \alpha} \delta_{\nu j} = H_{i \rho} \delta_{\nu j}
      $$

      $$
      \frac{\partial s_{ij}}{\partial b_{2,\nu}} = \delta_{\nu j}
      $$

      $$
      \begin{aligned}
        \frac{\partial L}{\partial W_{2, \rho \nu}} &= \frac{\partial s_{ij}}{\partial W_{2,\rho \nu}} \cdot \frac{\partial L}{\partial s_{ij}} = H_{i \rho} \cdot \frac{1}{N} (p_{i \nu} - \delta_{\nu, y_i}) \\
        &= \left( H^T \cdot \frac{\partial L}{\partial s} \right)_{\rho \nu}
        \end{aligned}
      $$

      $$
      \begin{aligned}
        \frac{\partial L}{\partial b_{2,\nu}} &= \frac{\partial s_{ij}}{\partial b_{2,\nu}} \cdot \frac{\partial L}{\partial s_{ij}}
      = \sum_{i=1}^{N} \sum_{j=1}^{C} \delta_{\nu j} \cdot \frac{\partial L}{\partial s_{ij}}
      = \sum_{i=1}^{N} \frac{\partial L}{\partial s_{i \nu}} \\
      &= \sum_{i=1}^{N} \left( \frac{\partial L}{\partial s} \right)_{i \nu}
      \end{aligned}
      $$

    - 对 $H$ 的梯度：

      $$
      \begin{aligned}
        \frac{\partial L}{\partial H_{\rho \nu}} &= \frac{\partial s_{ij}}{\partial H_{\rho \nu}} \cdot \frac{\partial L}{\partial s_{ij}} = W_{2,\nu j} \cdot \frac{1}{N} (p_{\rho j} - \delta_{j, y_{\rho}}) \\
        &= \left( \frac{\partial L}{\partial s} \cdot W_2^T \right)_{\rho \nu}
        \end{aligned}
      $$

    - 对 $W_1$ 和 $b_1$ 的梯度（约定 $Z_{i \alpha} = X_{i \rho} W_{1,\rho \alpha} + b_{1,\alpha}$ 为第一次线性变换的输出）：

      $$
      H_{i \alpha} = ReLU(Z_{i \alpha}) = ReLU(X_{i \rho} W_{1,\rho \alpha} + b_{1,\alpha})
      $$

      $$
        \frac{\partial H_{i \alpha}}{\partial W_{1,\rho \nu}} = ReLU'(Z_{i \alpha}) \cdot X_{i \rho} \delta_{\nu \alpha} = ReLU'(Z_{i \alpha}) \cdot X_{i \rho} \delta_{\nu \alpha}
      $$

      $$
        \frac{\partial H_{i \alpha}}{\partial b_{1,\nu}} = ReLU'(Z_{i \alpha}) \cdot \delta_{\nu \alpha}
      $$

      $$
      \begin{aligned}
        \frac{\partial L}{\partial W_{1,\rho \nu}} &= \frac{\partial H_{i \alpha}}{\partial W_{1,\rho \nu}} \cdot \frac{\partial L}{\partial H_{i \alpha}}
        = X_{i \rho} \delta_{\nu \alpha} \cdot \frac{\partial L}{\partial H_{i \alpha}} \cdot ReLU'(Z_{i \alpha}) \\
        &= X_{i \rho} \cdot \left( \frac{\partial L}{\partial H} \odot ReLU'(X W_1 + b_1) \right)_{i \nu} \\
        &= \left( X^T \cdot \left( \frac{\partial L}{\partial H} \odot ReLU'(X W_1 + b_1) \right) \right)_{\rho \nu}
        \end{aligned}
      $$

      $$
        \begin{aligned}
            \frac{\partial L}{\partial b_{1,\nu}} &= \frac{\partial H_{i \alpha}}{\partial b_{1,\nu}} \cdot \frac{\partial L}{\partial H_{i \alpha}}
            = \delta_{\nu \alpha} \cdot \frac{\partial L}{\partial H_{i \alpha}} \cdot ReLU'(Z_{i \alpha}) \\
            &= \sum_{i=1}^{N} \left( \frac{\partial L}{\partial H} \odot ReLU'(X W_1 + b_1) \right)_{i \nu}
            \end{aligned}
      $$

    其中，$\odot$ 表示元素级乘法，$ReLU'(z)$ 是 ReLU 函数的导数，实际上为：
    $$
    ReLU'(z) = \begin{cases}
    1, & z > 0 \\
    0, & z \leq 0
    \end{cases}
    $$

3. **总结梯度公式**：

    综上所述，再考虑到 $L_{reg}$ 正则化项，损失函数 $L=L_{data} + L_{reg}$ 对各参数的梯度为：

    - $\frac{\partial L}{\partial s} = \frac{1}{N} (P - Y)$，其中 $P$ 是预测概率矩阵，$Y$ 是标签的 one-hot 编码矩阵，定义为 $Y_{ij} = 1$ 当且仅当样本 $i$ 的标签为类别 $j$，否则为 $0$。

    - $\frac{\partial L}{\partial W_2} = H^T \cdot \frac{\partial L}{\partial s} + 2\lambda W_2$

    - $\frac{\partial L}{\partial b_2} = \sum_{i=1}^{N} \left( \frac{\partial L}{\partial s} \right)_{i,:}$

    - $\frac{\partial L}{\partial H} = \frac{\partial L}{\partial s} \cdot W_2^T$

    - $\frac{\partial L}{\partial W_1} = X^T \cdot \left( \frac{\partial L}{\partial H} \odot ReLU'(X W_1 + b_1) \right) + 2\lambda W_1$

    - $\frac{\partial L}{\partial b_1} = \sum_{i=1}^{N} \left( \frac{\partial L}{\partial H} \odot ReLU'(X W_1 + b_1) \right)_{i,:}$

## 代码实现

在 `two_layer_net.py` 文件中实现反向传播：

```python
def nn_forward_backward(
    params: Dict[str, torch.Tensor],
    X: torch.Tensor,
    y: torch.Tensor,
    reg: float = 0.0
):
    """
    计算一个两层全连接神经网络的损失和梯度。

    输入：
    - params: 一个存储模型权重的 PyTorch 张量字典。它应具有以下键：
        W1：第一层权重；形状为 (D, H)
        b1：第一层偏置；形状为 (H,)
        W2：第二层权重；形状为 (H, C)
        b2：第二层偏置；形状为 (C,)
    - X: 输入数据，形状为 (N, D)。每个 X[i] 是一个训练样本。
    - y: 训练标签的向量。y[i]是X[i]的标签，每个y[i]是[0, C)内的整数。
    - reg: 正则化强度。

    返回值：返回一个元组。
    - loss：该批训练样本的损失（包括数据损失和正则化损失）。
    - grads：一个字典，将参数名称映射到相对于损失函数的参数梯度；具有与 self.params 相同的键。
    """
    # 从 params 字典中解包变量
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    N, D = X.shape

    # 前向传播：计算得分，由 nn_forward_pass 函数实现，返回 输出分数scores 和 第一层隐藏层激活值h1
    scores, h1 = nn_forward_pass(params, X)


    # 计算损失
    loss = None
    stable_scores = scores - scores.max(dim=1, keepdim=True).values # (N, C)
    idx_samples = torch.arange(N)
    exp_sum = torch.exp(stable_scores).sum(dim=1, keepdim=True)
    p = torch.exp(stable_scores) / exp_sum # (N, C)
    loss = - torch.sum(torch.log(p[idx_samples, y]))
    loss = loss/N + reg*torch.sum(W1*W1) + reg*torch.sum(W2*W2)


    # 反向传播：计算梯度，推导见笔记`backpropagation.md`
    grads = {}
    p[idx_samples, y] -= 1 # 即 P-Y
    ds = p/N # dL/ds
    dW2 = h1.t().mm(ds) + 2*reg*W2 # dL/dW2
    db2 = ds.sum(dim=0) # dL/db2
    Z = X.mm(W1) + b1 # 第一层线性变换的输出
    dZ = ds.mm(W2.t()) # dL/dH，此时未考虑ReLU的导数 dH/dZ 的影响
    dZ[Z<0] = 0 # 考虑ReLU的导数影响，得 dL/dZ
    dW1 = X.t().mm(dZ) + 2*reg*W1 # dL/dW1
    db1 = dZ.sum(dim=0) # dL/db1
    grads['W2'] = dW2
    grads['b2'] = db2
    grads['W1'] = dW1
    grads['b1'] = db1

    return loss, grads
```
