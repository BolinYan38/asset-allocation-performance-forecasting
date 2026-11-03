# stock_prior.py：从训练标签学习股票权重并生成三列特征

我们利用训练标签学习各策略的股票权重、岭惩罚强度、历史衰减速度和上涨概率修正。每条记录仍以自己的20日策略收益拟合系数，再结合已知的目标日股票行情估计目标日策略收益。

`LearnedStockPrior` 完成一个模型的训练和预测，`StockPriorFeatures` 按市场训练两个模型，并组织训练行与查询行的输出。

## 1. 入口与执行顺序

```python
features = StockPriorFeatures(checkpoint, device).build(
    data, row_ids, validation=True, refit=False,
)
```

| 参数 | 含义 |
|---|---|
| `checkpoint` | 当前阶段的 `stock_prior.pt` 权重文件路径 |
| `device` | 计算设备，`cpu` 为通用处理器，`mps` 为苹果图形加速 |
| `data` | 已载入原始记录、日期和行情的 `AllocationData` 对象 |
| `row_ids` | 本次需要输出特征的原始行号，顺序决定返回顺序 |
| `validation` | 是否限制训练标签早于验证分界日 |
| `refit` | 是否重新训练并覆盖先验权重 |

`build()` 的主要执行顺序如下：

```text
取全部已定位记录和可用标签标记
按市场整理股票行情、历史收益、策略编号
按匿名日期编号的奇偶性分成两折
逐折创建 LearnedStockPrior
  有缓存则加载参数，否则调用 fit()
  fit() 每步调用 forward()，计算损失并更新参数
  predict() 调用 forward()，输出所需行
训练行取排除自身日期标签的模型，查询行取两模型平均
按请求的原始行号重排输出
```

返回数组为“请求行数 × 3”，三列依次是标准化收益估计、上涨概率、不确定度。

## 2. build()：准备每个市场的记录

### 2.1 先整理原始行与面板行的对应

`dated, original = data.dated()` 返回所有已唯一定位的记录和它们在原始数据中的行位置。`allowed` 是 `data.allowed_labels(validation)` 返回的标签使用标记。`wanted` 表示原始记录是否出现在请求行号 `row_ids` 中。

对一个市场调用 `data.panel(dated, group)`，取出股票行情 `public`、各日期市场下标 `positions`、日期与策略对应的行位置表 `row_map`。`valid = row_map >= 0` 选出实际有记录的日期与策略组合。

```python
rows = original[row_map[valid]]
day_positions = np.broadcast_to(positions[:, None], row_map.shape)[valid]
```

`rows` 是这些组合对应的原始行位置，`day_positions` 为每条记录提供市场交易日下标。例如日期位置为 `[100,105]`，行位置表为 `[[0,1],[2,-1]]`，则广播并筛选后的位置是 `[100,100,105]`。

### 2.2 收益尺度与策略编号

本市场出现的策略重新编号为0至策略数减一，保存为 `local_ids`，用于访问模型中的策略参数。股票收益已由 `panel()` 标准化，历史策略收益除以 `data.scales` 中对应策略的尺度。训练时目标策略收益也除以相同尺度。

`requested` 保存本市场中需要输出的行位置，`labeled` 标记这些请求行的标签是否允许用于当前阶段训练。

## 3. 两折的标签划分

`date_folds = raw.dates[rows] % 2` 按匿名日期编号取余数，得到折号0或1。同一天的所有策略拥有相同折号。

循环中的 `fold` 是该模型留出的折号。模型训练行满足：

```python
training = allowed[rows] & (date_folds != fold)
```

例如匿名日期编号10属于折0，11属于折1。模型0用折1的可用标签训练，模型1用折0的可用标签训练。日期10上的可训练记录，由模型0生成特征；日期11上的可训练记录，由模型1生成特征。验证行或官方测试行由两个模型分别预测，再逐列平均。

这是按日期编号奇偶分折，同折中可以包含分散的真实日期。最终共有4个市场 × 2折的模型参数，文件键为 `1_0` 至 `4_1`。

## 4. LearnedStockPrior.__init__()：参数定义

设 p 为当前市场股票数，S 为当前市场策略数。初始化使用固定随机种子，以下参数在训练中通过损失函数更新。

| 参数名 | 形状 | 实际作用 |
|---|---|---|
| `stock` | p | 市场共有的股票偏好 |
| `style` | S × 4 | 每个策略的4个风格系数 |
| `style_stock` | p × 4 | 每只股票对4种风格的系数 |
| `shared` | p × 8 | 将股票收益组合成8个额外收益序列 |
| `ridge` | S | 每个策略的岭惩罚参数 |
| `decay` | S | 每个策略的历史衰减参数 |
| `bias` | S | 每个策略的上涨概率偏置 |
| `gain` | 一个数 | 将收益估计转换为上涨概率时的共同增益 |

风格和额外收益序列均为训练得到的线性组合。`stock` 和 `bias` 初始为0，风格参数使用小随机数，`ridge` 初始为 −0.087，`decay` 初始为 −2，`gain` 初始为2的自然对数。

## 5. forward()：一次记录计算的实际顺序

### 5.1 提取历史与目标日股票收益

对当前记录的交易日下标 `position`，用19至0的偏移提取20日股票收益 `past`，用 `position+1` 提取目标日股票收益 `future`。

目标日股票收益已经在历史行情中，待估计的量是目标日策略收益。例如知道两只股票当天分别涨2%、跌1%，仍需要从策略历史拟合它对两只股票的系数，才能估计策略当天的收益。

### 5.2 生成每只股票的权重

对于策略编号 a、股票编号 j，定义股票偏好 ℓₐⱼ。`stock`、`style`、`style_stock` 对应的参数分别记为 uⱼ、sₐₖ、vⱼₖ，k 为1至4的风格编号：

$$
\ell_{aj}=u_j+\sum_{k=1}^4s_{ak}v_{jk}.
$$

将偏好限制在 −4 至4，再通过 softmax 转成正权重。softmax 对每个值取指数，再除以所有股票的指数之和。

20日收益标准差不超过百万分之一的股票被标记为不活跃，其权重乘零。其余权重重新归一化，使权重总和为股票数 p，分母下限为百万分之一。代码取权重平方根时使用下限一亿分之一。

这些权重用于调整股票在拟合中的重要性。策略对各股票的正负收益系数由后面的岭回归求得。

### 5.3 组成20行的设计矩阵

设 R 为20 × p 的历史股票收益矩阵，r★ 为 p 维目标日股票收益列向量，w 为当前策略的 p 维股票权重，F 为 `shared` 参数构成的 p × 8 矩阵。上标 T 表示转置，`diag` 将向量放到对角线上。令 X 为历史设计矩阵、x 为目标日设计向量：

$$
X=\frac1{\sqrt p}\left[R\operatorname{diag}(\sqrt{\max(w,10^{-8})}),\ RF\right],
$$

$$
x=\frac1{\sqrt p}
\begin{bmatrix}
\operatorname{diag}(\sqrt{\max(w,10^{-8})})r_\star\\
F^{\mathsf T}r_\star
\end{bmatrix}.
$$

向量上的最大值和平方根均逐元素计算。X 有 p+8 列，前 p 列是加权股票收益，后8列是训练得到的股票收益组合。

### 5.4 创建惩罚矩阵并求解

设 ρₐ、δₐ 分别为策略 a 的 `ridge`、`decay` 原始参数，定义有效岭系数 λₐ 和衰减率 dₐ：

$$
\lambda_a=0.05+\log(1+e^{\rho_a}),\qquad
 d_a=0.1\frac1{1+e^{-\delta_a}}.
$$

历史行 i 从1至20按最早到最近排列，滞后量 hᵢ=20−i。令 qᵢ 在该日策略收益缺失时为1，否则为0。对角惩罚矩阵 D 的元素为：

$$
D_{ii}=\lambda_a e^{d_a h_i}+10000q_i.
$$

较早的收益受到更大惩罚。设 y 为20日标准化策略收益向量，缺失值填零；定义 A 为求解矩阵、k 为历史与目标日的内积向量，α、b 为两列待求系数：

$$
A=XX^{\mathsf T}+D,\qquad k=Xx,\qquad
A\begin{bmatrix}\alpha&b\end{bmatrix}
=\begin{bmatrix}y&k\end{bmatrix}.
$$

代码用 `torch.linalg.solve` 一次直接求解两个右端项。小型线性方程在 CPU 上计算，结果转回指定设备，梯度沿设备拷贝继续传播。

### 5.5 输出收益、概率所需数值与不确定度

设收益估计为 ŷ，`gain` 参数为 g，策略偏置为 cₐ，有效增益 G 限制在0.5至6。设 z 为上涨对数几率、P 为上涨概率、u 为不确定度：

$$
\hat y=k^{\mathsf T}\alpha,\qquad
G=\operatorname{clip}(e^g,0.5,6),\qquad z=G\hat y+c_a,
$$

$$
P=\frac1{1+e^{-z}},\qquad
u=\sqrt{\max(0,x^{\mathsf T}x-k^{\mathsf T}b)}.
$$

`clip` 表示限制上下界。u 衡量当前设计矩阵对目标日股票行情的解释程度，同时包含岭惩罚影响。

`forward()` 返回 ŷ、z、u。`predict()` 再将 z 转成概率 P，形成输出的三列。

## 6. fit()：以真实策略收益训练参数

每步从本折允许使用的训练行中有放回抽取至多384条记录，调用 `forward()`，计算损失，再反向传播。当前配置训练160步。

对一条训练记录，设 t 为按策略尺度标准化的真实目标收益，b 为上涨标签，t>0时 b=1，否则为0。本节的 b 是二分类标签。令 tᶜ 为 t 截到 −5 至5后的值，ŷ 和 z 为模型输出。二分类损失和收益损失为：

$$
L_{\mathrm{cls}}=\log(1+e^z)-bz,\qquad
L_{\mathrm{return}}=h(\hat y-t^c),
$$

$$
h(e)=\begin{cases}\frac12e^2,&|e|<1,\\|e|-\frac12,&|e|\ge1.\end{cases}
$$

e 是收益估计误差，h 是平滑绝对误差函数。对批次取平均后，再加参数正则。设集合 Pᵣ 包含 `stock`、`style`、`style_stock`、`shared`、`bias` 五个参数数组，θ 表示其中一个数组，mean 表示所有元素的平均，则总损失为：

$$
L=\operatorname{mean}(L_{\mathrm{cls}})
+0.15\operatorname{mean}(L_{\mathrm{return}})
+0.003\sum_{\theta\in\mathcal P_r}\operatorname{mean}(\theta^2).
$$

优化器采用 Adam，初始学习率0.015，梯度总范数上限为5。设当前步号为 s，总步数为 S，s 从0至 S−1，则当步学习率为：

$$
0.015\left(0.2+0.8\left(1-\frac{s}{S}\right)\right).
$$

训练更新第4节的参数，后续每条记录的岭回归系数由当前参数和该记录历史重新求出。

## 7. predict()、缓存与最终行顺序

`predict()` 按每批最多512行计算，在推理模式下运行 `forward()`，返回三列浮点数组。请求为空时返回0行3列。

`build()` 对可训练行只写入其留出折模型的结果，对验证或官方测试行将每折结果乘0.5再相加。训练完保存每个市场每折的参数字典，已有缓存且 `refit=False` 时加载参数。

内部结果先按原始行位置保存。最后将原始行号排序，用 `searchsorted()` 找到请求行号，再按 `row_ids` 的给定顺序返回。例如请求 `[800,120]` 时，即使120在原始数据中更早，返回首行仍对应800。

| 最终特征名 | 内容 |
|---|---|
| `prior_return` | 标准化的目标日策略收益估计 |
| `prior_probability` | 模型给出的上涨概率 |
| `prior_uncertainty` | 本文定义的不确定度 u |

这三列进入 [model.py 的分类器](model.md)，与其余127列共同用于最终判断。
