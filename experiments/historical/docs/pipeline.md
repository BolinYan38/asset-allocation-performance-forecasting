# pipeline.py：特征准备、分类器训练与预测流程

我们将历史重建流程分为特征准备、分类器训练、预测三个独立步骤，由 `ReconstructionProject` 统一调用。各步骤通过阶段目录中的文件衔接。

## 1. 入口与运行目录

```python
project = ReconstructionProject(data_dir, output_dir)
dataset = project.prepare(mode="validation", device="cpu")
project.fit(mode="validation")
probability = project.predict(mode="validation")
```

`data_dir` 为原始数据目录，`output_dir` 为输出根目录，默认是本方法的 `runs/`。`mode` 选择 `validation`（本地验证）或 `official`（官方测试）。对应结果分别写入输出根目录的同名子目录。

`device` 控制学习股票先验的计算设备，`cpu` 表示通用处理器。`prepare()` 返回特征数据对象 `dataset`，`predict()` 返回查询行的上涨概率数组 `probability`。

## 2. prepare()：准备130列特征

### 2.1 初始化当前阶段

`validation = mode == "validation"` 决定是否使用本地验证分界日。`AllocationData(..., official=not validation)` 在官方模式加载训练与官方测试记录，验证模式加载原始训练记录。

`base_width` 从配置读取基础特征数量，当前为112。`records` 为 `FeatureDataset` 的每个字段创建一个列表，用于收集不同采样偏移的结果。

### 2.2 每个偏移分别计算

按配置中的0、5、10、15四个偏移循环。每轮调用 `data.sample(offset, validation)` 取得该套采样，再依次计算：

```python
base = ReturnReconstructor().features(data, raw)
sparse = SharedSparsePrior().features(data, raw)
features = np.column_stack((base, 三列缺失占位, sparse))
```

这里 `raw` 为本轮采样记录，`base` 有112列，`sparse` 有15列，`features` 拼成130列。中间3列留给稍后统一计算的学习股票先验。具体计算见 [reconstruction.md](reconstruction.md)。

### 2.3 合并时的行选择

偏移0的所有行进入最终数据，其余偏移只保留 `raw.train_mask` 为真的行。这样训练数据汇集四套采样，查询集由偏移0确定。

例如偏移0生成100条训练行、20条查询行，另外三个偏移各生成100条训练行和若干查询行，则最终数据含400条训练行及20条查询行。其他偏移中的查询记录参与其所在面板的特征计算，合并结果时通过 `keep` 排除。

每轮对特征、行号、收益标签、训练标记、组别和日期使用同一个 `keep` 标记，保证所有字段逐行对应。循环结束后沿行方向连接各列表，构造 `FeatureDataset`。

### 2.4 填入学习股票先验

使用当前阶段的 `stock_prior.pt` 初始化 `StockPriorFeatures`，将最终数据的 `row_ids` 作为请求行号。

已有权重且 `refit_prior=False` 时加载权重；权重缺失或指定重新训练时，按市场与日期两折拟合。返回结果写入零起始第112至114列。其余列的具体位置见 [reconstruction.md](reconstruction.md)，两折训练见 [stock_prior.md](stock_prior.md)。

最后保存 `features.npz`，包含130列特征、特征名及六个数据字段，并返回当前数据对象。

## 3. fit()：训练分类器

`fit()` 从所选阶段读取 `features.npz`，创建 `DirectionClassifier`，用数据中的训练标记选择拟合行，以目标策略收益大于零为分类标签。

训练所得分类器保存为同目录下的 `all.cbm`。其参数来自 `assets/config.json`，完整接口见 [model.md](model.md)。

## 4. predict()：生成查询概率与结果文件

### 4.1 读取特征和模型

读取所选阶段的特征数据，令 `query = ~dataset.train_mask` 选择查询行。加载同目录下的 `all.cbm`，对这些行调用分类器的 `predict()`，得到上涨概率。

### 4.2 保存预测并恢复全行对齐

`predictions.npz` 保存查询行的原始行号、概率、目标收益和组别。为复用公共评分与导出函数，另创建与整个数据集等长的 `scores`，在查询位置填入预测概率。

例如训练标记为 `[True,False,True,False]`，查询概率为 `[0.8,0.3]`，则 `scores=[0,0.8,0,0.3]`。公共函数只读取查询位置。

### 4.3 验证与官方输出

验证模式调用 `evaluate(..., threshold=0.5)`，记录准确率、正确数、样本数及分市场结果。流程将榜单字段改名为 `public_score`，并记录 `validation_used_for_model_selection=True`，写入 `metrics.json`。

官方模式调用 `write_submission()`，按原始目录中的 `sample_submission.csv` 排列行号，写出 `submission.csv`。

## 5. 文件衔接

| 文件 | 生成步骤 | 后续使用 |
|---|---|---|
| `stock_prior.pt` | `prepare()` 中首次训练或重新训练先验 | 后续准备特征时加载 |
| `features.npz` | `prepare()` | `fit()`、`predict()` |
| `all.cbm` | `fit()` | `predict()` |
| `predictions.npz` | `predict()` | 逐行检查与比较 |
| `metrics.json` | 验证模式 `predict()` | 查看本地验证表现 |
| `submission.csv` | 官方模式 `predict()` | 查看及提交官方测试方向 |

日期候选恢复和行情定位在前置阶段生成 `calendar.json`。本流程读取该映射和行情文件，执行策略收益重建及分类。
