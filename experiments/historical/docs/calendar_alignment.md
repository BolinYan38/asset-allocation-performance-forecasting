# calendar_alignment.py：定位官方测试集匿名日期

我们通过采样周期约束、策略收益窗口关系和公开股票收益，恢复官方测试集匿名日期对应的真实日期。`calendar_alignment.py` 提供两个类：`PeriodicDateGrid` 通过采样间隔假设，推断允许使用的真实日期集合，`MarketFingerprintCalendarResolver` 对候选日期评分并完成联合选择。

官方测试集包含120个匿名日期，日期分布较稀疏，历史窗口之间的重合较少。我们将测试日期建模为按固定交易日间隔采样的一组日期，利用已定位的测试日期推断间隔与采样区间，在这些采样点上重建候选，再以岭回归留一日误差比较行情窗口，联合选择满足日期关系和日期不重复约束的对应方案。

## 1. 输入、接口与调用关系

模块的主要输入如下：

| 输入 | 含义 |
|---|---|
| `raw` | 合并训练与官方测试记录的数据对象，通过 `train_mask` 区分记录来源 |
| `domains` | 传给采样网格类的日期候选字典，保存“匿名日期名称 → 真实日期字符串列表” |
| `previous` | 此前一轮日期定位结果，同样为候选字典，供采样区间选择使用 |
| `query_names` | 官方测试匿名日期名称列表 |
| `sessions` | 按市场组别保存的真实交易日期数组 |
| `market` | 股票行情缓存字典，其中 `returns_1` 至 `returns_4` 保存四组股票收益 |
| `candidates` | 传给行情定位类的候选字典，包含训练和测试匿名日期 |
| `device` | 计算设备，例如 `cpu` |

组别与市场的对应关系已体现在 `sessions` 和 `market` 的组织中。我们在这一对应关系下定位日期。

启用采样网格时，调用方先提供至少5个已唯一定位的官方测试日期，以及此前定位结果 `previous`，再按以下关系组织调用：

```text
PeriodicDateGrid.infer(domains, previous, sessions, query_names)
    ↓ 返回真实日期网格 grid，以及网格统计
PeriodicDateGrid.rebuild_domains(raw, domains, grid, sessions)
    ↓ 返回网格上的候选字典 candidates
MarketFingerprintCalendarResolver(device).resolve(raw, candidates, sessions, market)
    ↓ 返回最终日期映射 resolved，以及选择报告 report
```

其中，`grid` 是有序真实日期字符串数组，`resolved` 是更新后的候选字典，`report` 是统计与逐日期选择详情。`resolve()` 接收候选后独立运行，采样网格由调用方在前面组织；模块将计算结果返回给调用方保存。

合并训练和测试记录发生在数据读取阶段：

```python
raw = common.load_data(common.DEFAULT_DATA, include_test=True)
```

这里 `common` 是本方法的数据模块 `experiments.historical.common`，`DEFAULT_DATA` 是原始数据目录，`include_test=True` 表示一并读取官方 `X_test.csv`。`raw.train_mask` 对训练记录标记为 `True`，对官方测试记录标记为 `False`。

## 2. 日期编号与数组位置

本程序分别使用以下位置编号：

| 位置 | 所在数组及用途 |
|---|---|
| 匿名日期编号 | `raw.date_names` 中的位置，只标识匿名日期，不表示时间先后 |
| 测试评分行号 | 匿名日期在 `query_dates` 中的位置，用于访问 `scores` 的行 |
| 自然日下标 | 真实日期在 `calendar` 中的位置，候选关联组及最终选日使用这个下标 |
| 市场交易日下标 | 真实日期在某市场 `sessions` 数组中的位置，用于提取20日股票历史 |
| 联合交易日下标 | 真实日期在各市场交易日期并集 `union` 中的位置，用于推断采样间隔 |
| 网格下标 | 日期在固定间隔数组 `clock` 中的位置，用于选择连续采样区间 |

收益历史均按最早到最近排列，窗口长度为20个市场交易日。

## 3. infer：推断采样间隔与日期集合

### 3.1 建立联合交易日时间尺

`infer()` 先合并四个市场的交易日期，排序并去重，得到 `union`：

```python
union = np.unique(np.concatenate([
    values.astype("datetime64[D]")
    for values in sessions.values()
]))
```

`values` 是循环中某个市场的交易日期数组。只要至少一个市场当天开市，该日期就进入 `union`。采样间隔按这个并集计数。

### 3.2 从测试日期锚点求最大公约数

`anchors` 保存 `domains` 中已经唯一定位的官方测试日期，称为日期锚点。提取时只遍历 `query_names`，训练日期不参与采样周期推断。

```python
anchors = np.asarray(sorted(
    domains[name][0]
    for name in query_names
    if len(domains[name]) == 1
))
positions = np.searchsorted(union, anchors.astype("datetime64[D]"))
stride = reduce(math.gcd, np.diff(positions).tolist())
```

`name` 是测试匿名日期名称，`positions` 是锚点在 `union` 中的下标；`stride` 是相邻锚点间距的最大公约数。`math.gcd` 计算最大公约数，`reduce` 将它依次作用于所有间距。

例如：

```text
锚点位置：[25, 45, 55, 75, 95]
相邻间距：[20, 10, 20, 20]
最大公约数：10
```

我们由此取采样周期为10个联合交易日。程序要求至少5个锚点，推得的周期在5至65之间。

### 3.3 确定相位，建立固定间隔网格

```python
clock = union[int(positions[0] % stride)::stride].astype(str)
```

本例中 `25 % 10 = 5`，所以 `clock` 对应的联合交易日下标为：

```text
[5, 15, 25, 35, 45, 55, 65, 75, 85, 95, ...]
```

周期确定间隔，相位确定采用哪一列同余位置。随后，`anchor_positions` 将各锚点转换成 `clock` 内的下标，本例为 `[2, 4, 5, 7, 9]`。

### 3.4 枚举包含全部锚点的采样区间

我们将官方测试日期建模为 `clock` 上连续的固定数量采样点。令：

- `query_count`，记为 N，表示测试匿名日期数量，项目中为120。
- M 表示 `clock` 的长度。
- s 表示待选区间在 `clock` 中的起点。

锚点在 `clock` 中的最小、最大下标分别记为：

$$
b_{\min},\quad b_{\max}.
$$

要使连续 N 个点覆盖全部锚点，起点须满足：

$$
\max(0,b_{\max}-N+1)\le s\le\min(b_{\min},M-N).
$$

代码分别将两端保存为 `first` 和 `last`。它们是可尝试的起点范围。

以12个测试日期、且 `clock` 至少有14个点为例，本例得到 `first=0`、`last=2`，可尝试：

| 起点 | 网格下标区间 | 对应联合交易日下标 |
|---:|---|---|
| 0 | 0至11 | 5、15、……、115 |
| 1 | 1至12 | 15、25、……、125 |
| 2 | 2至13 | 25、35、……、135 |

### 3.5 用 previous 选择具体区间

`previous_dates` 保存 `previous` 中已经唯一定位的测试日期。对每个候选区间，统计其中包含多少个 `previous_dates`，得到支持数 `support`：

```python
days = clock[start:start + query_count]
support = int(np.isin(previous_dates, days).sum())
```

这里 `start` 是正在尝试的网格起点，`days` 是该区间的真实日期列表。若此前定位结果对应联合交易日下标 `[15, 25, 45, 95, 125]`，上述三个区间的支持数依次为4、5、4，于是选取起点1。

最终 `result` 为：

```text
联合交易日下标：[15, 25, 35, ..., 125]
```

它包含12个采样点。项目中返回120个采样日期，不包含这些点之间的其他日期。最高支持数并列时，程序报告区间无法区分。

`infer()` 返回日期集合与统计：采样间隔 `stride_union_sessions`、锚点 `anchor_dates`、区间首尾日期 `first_date` 和 `last_date`、日期数量 `date_count`、最佳与次佳支持数 `previous_fingerprint_support` 和 `runner_up_support`。`uses_query_targets=False` 记录本步骤的输入不含测试目标收益。

这一步确定允许使用哪些真实日期；匿名日期与这些日期的具体配对由后续函数完成。

## 4. rebuild_domains：在网格上重建候选

`rebuild_domains()` 创建 `CalendarInputs(raw)`，在 `grid` 上调用已经在 `calendar_recovery.py` 中使用的 `candidate_domains()`：

```python
choices_by_date = inputs.candidate_domains(
    grid, session_positions(grid, sessions), query_dates,
)
```

这里 `query_dates` 是官方测试匿名日期编号，`choices_by_date` 保存每个日期通过筛选的网格下标。筛选规则是：当天出现的所有市场都开市，并满足周一或非周一的成交量报告规则。

接着，逐个匿名日期取出网格上的候选，保存为 `choices`：

- 若原候选已经唯一，且仍在 `choices` 中，保留这一单候选。
- 其余日期采用网格与原始特征重新生成的 `choices`，替换旧候选列表。
- 训练日期候选保持原样。

因此，重建的来源是网格上的原始特征规则；旧的多候选列表可以被重新生成的列表修复。

返回报告中的 `singleton_before_fingerprint` 是进入行情评分前的单候选数量，`maximum_grid_candidates` 是最大候选数量。`repaired_empty_old_domains` 记录旧候选与新候选完全不相交的匿名日期名称。

## 5. resolve：准备候选并合并测试日期关系

`MarketFingerprintCalendarResolver` 初始化时保存计算设备 `device` 和岭惩罚强度 `ridge`，后者默认0.7。`resolve()` 的执行顺序为：

```text
创建自然日搜索范围 calendar，以及字符串数组 calendar_strings
区分训练与测试匿名日期，收集训练集已唯一确定的日期 locked
从测试候选中排除 locked，将全部候选转换成自然日下标
初始化 CalendarInputs 与 FunctionalCalendarForest
匹配测试日期之间的收益窗口，合并相对日期关系
_score_dates() 计算行情匹配评分
_choose_dates() 联合选择日期
_result() 整理日期映射与报告
```

`calendar` 覆盖2000年2月1日至2025年12月31日。`calendar_strings` 保存相同日期的字符串，供候选查找和结果输出使用。

`locked` 只收集训练集中已唯一定位的真实日期。测试候选排除这些日期后，转换成自然日下标，存入内部列表 `domains`。这里的 `domains` 按匿名日期编号索引，与采样接口接收的同名字典区分。

随后创建关联组对象 `forest`，并调用：

```python
edges = inputs.edges(device, threshold=0.9, max_gap=19, query_only=True)
```

`edges` 是生成的日期关系列表。`threshold` 是余弦相似度门槛，`max_gap` 是最大市场交易日间隔。`query_only=True` 将查询和参考均限制为测试日期，本轮新增关系只连接测试日期。

我们将关系按间隔升序、同间隔内置信度降序传入 `forest.add()`。同一关联组的候选按整列保存，每列是一套满足既有相对间隔的日期方案。组内合并过程见 `calendar_recovery.md`。

## 6. _score_dates：计算岭回归留一日评分

### 6.1 提取测试收益与训练历史位置

`panel = inputs.panel[inputs.query_dates]` 取出测试日期策略收益，维度为“测试日期 × 策略 × 20日历史”。

`known_dates` 收集已唯一定位的训练日期字符串；`known_days` 将它们转换为 `calendar` 下标，用于估计股票波动率。`lags = [19, 18, ..., 0]` 保存提取20日历史所需的交易日偏移。

函数初始化三个数组：

| 变量 | 形状与用途 |
|---|---|
| `score_sum` | 测试日期数 × 自然日数，累计各市场评分 |
| `score_count` | 与 `score_sum` 同形，累计每个位置获得评分的市场数 |
| `expected_groups` | 每个测试日期一个计数，记录应参与评分的市场数 |

### 6.2 逐市场选择完整策略并标准化

对每个市场，`selected` 选出属于当前市场且通过 `stable` 筛选的策略，`observed` 取出其测试收益。`valid` 标记每条策略是否具备完整20日收益，`counts` 统计每个日期的完整策略数。

`rows` 保存至少具有4条完整策略的测试评分行号。该市场计入这些日期的 `expected_groups`。

`values` 将这些日期的收益调整为“测试日期 × 20日历史 × 策略”，对每条策略沿20日减均值、除标准差，标准差下限为 `1e-6`。随后将不完整策略整列置零，计算：

```python
covariance = values @ values.transpose(0, 2, 1) / counts[rows, None, None]
```

因此，每个测试日期得到一个20×20矩阵，汇总当前市场各完整策略的历史收益结构。

### 6.3 提取股票窗口并统一尺度

`public_returns` 是当前市场的股票日收益矩阵，行对应市场交易日、列对应股票；`stock_count` 是股票数量；`positions` 是自然日到该市场交易日下标的查询表。

先用 `positions[known_days]` 得到训练日期的市场交易日位置，取出具有足够历史的20日股票窗口，堆叠后逐股票计算标准差，得到 `stock_scale`，下限为0.005。

再枚举真实行情窗口。`days` 保存市场交易日下标至少为19、且位于行情最后一个交易日之前的自然日下标。由这些位置提取20日股票历史，除以各股票的 `stock_scale`，得到 `windows`。

每个窗口至少有8只股票在20日内的标准差大于 `1e-5`，才进入评分。保留后，逐股票减去窗口内均值，再将整个窗口除以其所有元素的均方根，均方根下限为 `1e-5`。

此时股票窗口维度为“候选真实日期 × 20日历史 × 股票”。当前实现对完整自然日搜索范围计算可用行情评分，后续 `_choose_dates()` 再从关联组允许的候选中取分；采样网格限制的是可选择日期集合。

### 6.4 岭回归与留一日公式

固定一个市场、一个匿名日期和一个候选真实日期，在上述预处理后的实数矩阵上定义：

| 符号 | 含义 |
|---|---|
| n | 历史日数，固定为20 |
| p | 股票数量，即 `stock_count` |
| m | 当前匿名日期的完整策略数量 |
| X | 标准化后的真实股票窗口 |
| Y | 标准化后的完整策略收益，每列一条策略 |
| W | 股票对策略收益的拟合系数 |
| λ | 岭惩罚参数 `self.ridge`，默认0.7 |

三个矩阵的维度为：

$$
X\in\mathbb R^{n\times p},\qquad
Y\in\mathbb R^{n\times m},\qquad
W\in\mathbb R^{p\times m}.
$$

上标 T 表示转置，范数采用 Frobenius 范数，其记号为：

$$
\|\cdot\|_F.
$$

单位矩阵的下标表示阶数。无截距岭回归写为：

$$
\widehat W=\arg\min_W\left\{\|Y-XW\|_F^2+p\lambda\|W\|_F^2\right\}.
$$

由于代码对股票乘积矩阵除以股票数量，原始系数目标中的岭惩罚也相应乘以股票数量。岭回归的对偶解为：

$$
\widehat W=\frac1pX^{\mathsf T}\left(\frac{XX^{\mathsf T}}p+\lambda I_n\right)^{-1}Y.
$$

定义股票乘积矩阵 G、正则化矩阵 K，以及逆矩阵 A：

$$
G=\frac{XX^{\mathsf T}}p,\qquad
K=G+\lambda I_n,\qquad
A=K^{-1}.
$$

它们对应：

```python
gram = ... / stock_count
gram[:, np.arange(20), np.arange(20)] += self.ridge
inverse = np.linalg.inv(gram)
```

令 H 为帽子矩阵，则拟合值和全样本残差满足：

$$
H=GA=I_n-\lambda A,\qquad
\widehat Y=HY,\qquad
Y-\widehat Y=\lambda AY.
$$

对历史日 i，取值为1至 n，记删除该日、用其余日期拟合后对该日的预测行为：

$$
\widehat Y_i^{(-i)}.
$$

留一日残差满足：

$$
Y_i-\widehat Y_i^{(-i)}
=\frac{(Y-\widehat Y)_i}{1-H_{ii}}
=\frac{(AY)_i}{A_{ii}}.
$$

这里行下标 i 表示历史日，对角元素下标 ii 表示同一行列。定义 D 为 A 的对角元素组成的对角矩阵，R 为按行归一化后的逆矩阵，E 为全部策略的留一日残差矩阵：

$$
D=\operatorname{diag}(A_{11},\ldots,A_{nn}),\qquad
R=D^{-1}A,\qquad
E=RY.
$$

这对应：

```python
inverse /= np.diagonal(inverse, axis1=1, axis2=2).copy()[:, :, None]
```

执行后，`inverse` 保存 R。我们进一步定义误差权重矩阵 P，以及策略收益矩阵 C：

$$
P=R^{\mathsf T}R,\qquad
C=\frac{YY^{\mathsf T}}m.
$$

它们分别对应 `penalty` 与 `covariance`。跨策略、跨历史日平均的留一日均方误差为：

$$
\operatorname{MSE}_{\mathrm{LOO}}
=\frac1{nm}\|RY\|_F^2
=\frac1n\operatorname{tr}(PC)
=\frac1n\langle C,P\rangle_F.
$$

其中，MSE 表示均方误差，LOO 表示留一日验证；tr 表示矩阵的迹；Frobenius 内积是对应元素乘积之和。最后一个等号使用了两个矩阵的对称性。

因此，匿名策略窗口只需计算一次 C，真实股票窗口只需计算一次 P，两者即可配对评分。将20×20矩阵展开为长度400的向量后，代码批量计算：

```python
1 - covariance @ penalty.T / 20
```

也就是：

$$
\operatorname{score}=1-\operatorname{MSE}_{\mathrm{LOO}}.
$$

评分越高，表示该真实股票窗口对匿名策略收益的留一日解释误差越小。

### 6.5 跨市场合并评分

每个市场将评分写入 `score_sum`，并给对应位置的 `score_count` 加1。四个市场处理完后，要求某个测试日期与真实日期的配对满足：

```python
usable = (score_count == expected_groups[:, None]) & (expected_groups[:, None] > 0)
```

`usable` 标记所有应参与的市场都完成评分的位置。函数在这些位置返回各市场评分的平均值，其余位置记为负无穷 `-np.inf`。

返回的 `scores` 是“官方测试日期 × 自然日”的评分矩阵。一个测试日期若没有任何市场满足完整策略数量要求，其整行都不可用；后续选择会报告无可用候选。

## 7. _choose_dates：从关联组方案中联合选日

### 7.1 为每个关联组计算方案总分

`query_lookup` 将匿名日期编号映射到 `scores` 的行号。对 `forest` 中的每个关联组，`members` 保留其中的测试成员。每个成员的候选数组按列对齐，同一列是一套共同可行的日期分配。

例如，以下下标作简化示意：

```python
members = {
    7: np.array([100, 200]),
    9: np.array([101, 201]),
}
```

它提供两套方案：分别将匿名日期7、9安排到100、101，或者200、201。

本节用 L 表示参与选日的关联组数量，ℓ 表示组编号，取值为1至 L。第 ℓ 组的测试成员集合记为：

$$
\mathcal C_\ell.
$$

对组内匿名日期 a，第 k 套方案给出的自然日下标记为：

$$
d_{\ell,a}^{(k)},\qquad a\in\mathcal C_\ell.
$$

其中 k 为候选列编号。匿名日期 a 与自然日下标 d 的评分定义为：

$$
s(a,d)=\texttt{scores[query\_lookup[a],d]}.
$$

则第 ℓ 组第 k 套方案的总分为：

$$
J_\ell(k)=\sum_{a\in\mathcal C_\ell}s\!\left(a,d_{\ell,a}^{(k)}\right).
$$

代码将各列总分保存为 `joint`，剔除非有限总分，保留的方案列编号保存为 `options`，并按总分降序排列。

`margin` 为最佳方案与次佳方案的总分差，只有一个方案时记为正无穷。关联组先按有效方案数量升序处理，数量相同时按 `margin` 降序处理。

### 7.2 beam 搜索跨组不冲突的组合

第 ℓ 组有效方案列编号的集合记为：

$$
\mathcal O_\ell.
$$

第 k 套方案使用的真实日期集合定义为：

$$
\mathcal T_\ell(k)=\{d_{\ell,a}^{(k)}:a\in\mathcal C_\ell\}.
$$

我们搜索以下联合分配：

$$
\max_{k_\ell\in\mathcal O_\ell}\sum_{\ell=1}^{L}J_\ell(k_\ell),
\qquad
\mathcal T_\ell(k_\ell)\cap\mathcal T_t(k_t)=\varnothing
\quad(\ell\ne t).
$$

其中 t 同样为关联组编号。组内方案已保证相对日期关系，跨组约束保证不同匿名日期不占用同一天。

`beam` 保存当前搜索路径，每个状态是三元组：

```text
(total, assignments, occupied)
```

`total` 为累计方案总分，`assignments` 为已选的“匿名日期编号 → 自然日下标”字典，`occupied` 为已经占用的日期集合。初始化为空分配：

```python
beam = [(0.0, {}, frozenset())]
```

每处理一个关联组，对每条保留路径尝试该组的方案。`selected` 提取当前方案的日期分配，`days` 提取它使用的日期；与 `occupied` 有交集则跳过，否则合并映射、累加总分并加入 `expansions`。

每条旧路径最多扩展8个不冲突方案；当前组全部扩展完成后，只保留累计总分最高的64条路径。最终返回最高分路径中的日期映射。该搜索通过每步截断控制组合数量。

例如，两组的候选为：

| 关联组 | 方案日期分配 | 总分 |
|---|---|---:|
| A | `{7: 200, 9: 201}` | 1.75 |
| A | `{7: 100, 9: 101}` | 1.50 |
| B | `{12: 200}` | 0.90 |
| B | `{12: 400}` | 0.10 |

B的分差更大，先处理B。B选200时，A的最佳方案因占用200而冲突，只能采用次佳方案，总分2.40；B选400时，搭配A最佳方案的总分为1.85。最终保留前者，返回 `{12: 200, 7: 100, 9: 101}`。

## 8. _result：输出日期映射和报告

`_result()` 复制输入候选字典，保存为 `resolved`。随后按 `query_dates` 遍历官方测试匿名日期：

1. 从 `_choose_dates()` 返回的 `selected_dates` 中查出所选自然日下标。
2. 用 `calendar_strings` 转换为真实日期字符串。
3. 将该匿名日期的候选列表替换为包含所选日期的单元素列表。
4. 将匿名名称、所选日期及对应评分加入详情列表 `details`。

训练匿名日期保留输入时的候选。报告字段如下：

| 字段 | 含义 |
|---|---|
| `query_dates` | 官方测试匿名日期数量 |
| `accepted_edges` | 本轮关联组合并接受的关系数量 |
| `rejected_edges` | 本轮关联组合并拒绝的关系数量 |
| `assignments` | 逐测试日期的选择详情 |

`assignments` 的每条记录包含匿名日期名称 `TS`、真实日期字符串 `selected_date` 和该配对评分 `fingerprint_score`。函数返回 `resolved, report`，完成本模块的日期定位输出。
