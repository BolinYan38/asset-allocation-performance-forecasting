# common.py：原始数据读取、验证评分与提交导出

我们在本模块统一读取比赛记录、建立行号和名称索引，并提供验证评分及提交文件导出。日期恢复与历史收益重建使用同一个 `RawDataset` 数据结构。

## 1. 数据目录与公共接口

`ROOT` 指向项目根目录，`METHOD_ROOT` 指向本方法的 `experiments/historical/` 目录。原始数据默认放在用户目录下的 `Dataset/Asset Allocation Performance Forecasting/`，由 `DEFAULT_DATA` 保存。

```python
raw = load_data(data_dir, include_test=True)
```

`data_dir` 为原始数据目录，`include_test` 控制是否追加官方测试记录。函数返回 `RawDataset`，保留输入行顺序。

| 文件 | 使用内容 |
|---|---|
| `X_train.csv` | 原始训练记录的身份、20日收益、20日成交量、换手率 |
| `y_train.csv` | 原始行号及目标日策略收益 |
| `X_test.csv` | 官方测试特征，启用 `include_test` 时读取 |
| `sample_submission.csv` | 导出时采用的行号顺序和列标题 |

## 2. _read_features()：按列名提取一张特征表

函数先读取表头，查找所需列的位置。数值列的输出顺序固定为：

```text
RET_20 至 RET_1                         20列历史策略收益
SIGNED_VOLUME_20 至 SIGNED_VOLUME_1     20列带符号成交量
MEDIAN_DAILY_TURNOVER                   1列日换手率中位数
```

两段历史均从最早到最近排列。数值读取使用32位浮点数，空字符串转为 `NaN`，即缺失数值。输出的 `numeric` 每条记录41列。

随后读取身份列 `ROW_ID`、`TS`、`ALLOCATION`、`GROUP`，分别表示原始行号、匿名日期名称、策略名称、市场组别。它们按字符串列表保存为 `identities`。函数返回 `numeric, identities`。

## 3. load_data()：合并记录并关联标签

### 3.1 合并训练与官方测试

首先读取训练特征并记录行数 `training_rows`。启用官方模式时，将测试数值矩阵追加在训练矩阵下面，身份列表按同一顺序追加。

### 3.2 建立名称与编号的对应

`np.unique(..., return_inverse=True)` 对匿名日期和策略分别生成去重名称表及逐行编号。例如原始日期依次为：

```python
["DATE_B", "DATE_A", "DATE_B"]
```

则 `date_names=["DATE_A","DATE_B"]`，`dates=[1,0,1]`。任意一行都能通过 `date_names[dates[行位置]]` 找回原始匿名名称。编号按名称表位置定义，实际时间排序由后续真实日期映射完成。

原始行号转为64位整数，市场组别转为整数。日期和策略编号在返回前转为32位整数。

### 3.3 按原始行号对齐目标收益

`y_train.csv` 中的标签行可以与特征文件顺序不同。我们先按标签的原始行号排序，再用 `searchsorted()` 查出每个训练特征行号的位置，并核对该位置确实对应所需行号。

例如训练特征行号为 `[20,10]`，标签文件为 `[[10,-0.2],[20,0.3]]`，对齐后目标收益为 `[0.3,-0.2]`。

`targets` 先全部设缺失，再填入前 `training_rows` 行的标签。`train_mask` 在这些训练来源行上为真，官方测试行上为假。

### 3.4 数据一致性检查

读取阶段检查行号唯一、训练标签完整对应、每个策略只属于一个市场且具有训练记录，以及官方训练和测试的匿名日期名称互不重合。检查通过后构造 `RawDataset`。

| 字段 | 每行或每项保存的内容 |
|---|---|
| `row_ids` | 比赛原始行号 |
| `dates`、`date_names` | 逐行匿名日期编号及编号对应名称 |
| `allocations`、`allocation_names` | 逐行策略编号及编号对应名称 |
| `groups` | 逐行市场组别 |
| `returns`、`volumes` | 每行20个历史收益和成交量数值 |
| `turnover` | 每行换手率 |
| `targets` | 每行目标日策略收益 |
| `train_mask` | 当前对象中的训练标记，读取时标识原始训练来源 |

本地验证的数据选择会在 `data.sample()` 中重新设置采样记录的训练标记，见 [data.md](data.md)。

## 4. evaluate()：计算验证准确率

`PredictionRows` 声明评分与导出需要的五个字段，即行号、训练标记、目标收益、组别和日期。原始数据与最终特征数据都提供这些字段。

```python
metrics = evaluate(raw, scores, threshold=0.5)
```

`raw` 为具有这些字段的数据对象，`scores` 为与它逐行对齐的上涨概率，`threshold` 为分类阈值。函数选择 `~raw.train_mask` 指定的查询行，要求这些行拥有真实收益标签。

概率严格大于阈值时预测上涨，收益严格大于零时真实标签为上涨。例如概率 `[0.7,0.4,0.5]`、收益 `[0.1,-0.2,0.1]` 在阈值0.5下正确两条，准确率为三分之二。

返回的 `metrics` 包含准确率 `accuracy`、正确数 `correct`、验证行数 `rows`、匿名日期数 `dates`、分组准确率 `by_group`。外部榜单分数 `public_leaderboard_score` 初值为空，由调用方决定报告字段。

## 5. write_submission()：按样例顺序写出方向

函数提取查询行，检查其概率均为有限数值，再按阈值转换为0或1，建立“原始行号 → 预测方向”字典。

随后读取提交样例，检查表头、行号唯一性和行号集合。输出沿用样例表头及行顺序。例如内部预测顺序为 `[20,10]`，样例顺序为 `[10,20]`，导出结果按10、20排列。

文件写到调用方传入的 `path`，供本地检查和提交使用。

## 6. log() 与 save_json()

`log()` 为进度消息添加从模块加载开始经过的秒数，打印到终端。设置 `LOG_PATH` 时，同时追加到对应日志文件。

`save_json()` 将字典等对象写成带缩进的 JSON，保留中文，并拒绝非有限浮点数，保证报告符合标准 JSON 数值格式。
