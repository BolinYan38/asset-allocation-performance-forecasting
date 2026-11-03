# 基于历史特征的 CatBoost 涨跌预测

我们根据每条策略记录提供的过去20日收益、带符号成交量、换手率，以及同日同市场其他策略的统计信息，构造124列特征，用 CatBoost 预测目标日策略收益是否大于零。

一条样本对应一个匿名日期上的一条策略记录。模型输出上涨概率，概率严格大于0.5时记为上涨，否则记为非上涨。

## 1. 目录与程序职责

```text
catboost/
  __main__.py                 命令行入口
  market_forecaster.py        特征构造、模型训练与预测
  common.py                   原始数据读取、日期划分、评分及导出
  tune_catboost.py            参数搜索
  configs/forecasting.json    当前选定参数
  docs/CatBoost常规预测.md     特征与原参考方案说明
  references/
    split_dates.csv          保存的日期划分参考
    forecasting/             原参考模型、特征名、预处理统计与验证结果
  runs/
    catboost_trials.json     已有参数比较记录
    submission-v1/           已有官方预测模型、预测及提交结果
```

`market_forecaster.py` 中，`AllocationFeatureProcessor` 构造特征，`CatBoostDirectionModel` 负责分类器，`ForecastingPipeline` 将两者串联并保存结果。

## 2. 原始比赛数据

**原始比赛数据未上传至 GitHub。** 从原始数据运行实验时，需自行从比赛平台取得以下文件，默认放入用户主目录下的 `Dataset/Asset Allocation Performance Forecasting/`。

```text
~/Dataset/Asset Allocation Performance Forecasting/
  X_train.csv             原始训练特征
  y_train.csv             目标日策略收益标签
  X_test.csv              官方测试特征
  sample_submission.csv   提交格式与原始行号顺序
```

本地验证读取前两个文件，官方预测使用全部四个文件。可通过 `--data-dir` 指定其他目录。已有参考模型的复评也需要原始训练数据，以重建日期划分和输入特征。

## 3. 特征与数据划分

收益尺度和策略换手率中位数从本次训练行估计。收益除以各策略的训练标准差，成交量通过反双曲正弦函数压缩极端值，同时保留正负号。

| 特征部分 | 列数 | 内容 |
|---|---:|---|
| 历史收益与成交量 | 40 | 各20个历史日的收益和成交量 |
| 报告标记与换手率 | 3 | 最近一天成交量是否有值、换手率对数、相对策略中位数的换手率 |
| 多窗口收益统计 | 30 | 最近2、3、5、10、20日，每个窗口6项统计 |
| 多窗口成交量统计 | 10 | 上述各窗口排除最新一天后的均值和标准差 |
| 趋势、波动及交互 | 13 | 线性趋势、波动比、极值、偏度、峰度、收益与成交量相关等 |
| 同日同组比较 | 24 | 6个变量各自的组内均值、偏离、标准化偏离和相对排名 |
| 策略数量 | 2 | 同日同组的策略数、当天全部策略数 |
| 类别身份 | 2 | 策略编号与市场组别 |
| **合计** | **124** | 122列数值特征、2列类别特征 |

同日同组的标准化偏离，是当前策略的变量值减去组均值，再除以组内标准差。多窗口收益统计包括均值、标准差、正收益比例、下行波动、均值与波动的比值，以及最近收益相对窗口均值的偏离。详细定义见[特征说明](docs/CatBoost常规预测.md)。

本地验证按匿名日期约80%／20%划分，同一天全部策略进入同一侧。默认随机种子为42，得到505个验证日期、104,724条验证记录。代码还为只有19条记录的稀少策略预留训练日期和4个验证日期，实际划分写入每次运行的 `split_dates.csv`。

## 4. 环境与运行

以下命令均在仓库根目录执行。输出目录需为新目录或空目录，每次运行使用一个独立目录。

```sh
python -m pip install -r experiments/catboost/requirements.txt
python -m experiments.catboost --help
```

### 4.1 用当前配置训练并验证

```sh
python -m experiments.catboost validate \
  --output-dir experiments/catboost/runs/validation-new
```

程序读取原始训练数据，重新生成日期划分，构造特征，按当前配置训练一次，再对全部验证行预测和评分。

### 4.2 使用保存的原参考模型复评

```sh
python -m experiments.catboost validate \
  --checkpoint experiments/catboost/references/forecasting/model.cbm \
  --output-dir experiments/catboost/runs/reference-check
```

`--checkpoint` 加载已有模型，其使用范围为随机种子42的原验证划分。程序仍会从原始训练特征计算输入和预处理统计，并检查特征名称与保存模型一致。

### 4.3 训练全量模型并预测官方测试集

```sh
python -m experiments.catboost predict \
  --output-dir experiments/catboost/runs/official-new
```

程序使用全部原始训练标签拟合模型，预测 `X_test.csv`，按提交样例的行号顺序生成本地 `submission.csv`。

三个命令均可追加 `--data-dir "/实际数据目录"` 或 `--config "/配置文件路径"`。当前入口采用 CPU 计算，模型封装将训练与预测线程数设为2。

## 5. 参数与已有结果

当前默认配置采用118次提升迭代、树深度7、学习率0.04、叶节点正则强度10，随机种子42。完整配置见 [forecasting.json](configs/forecasting.json)。

| 记录 | 结果 | 对应文件 |
|---|---|---|
| 原参考验证模型，245次迭代、学习率0.035 | 55,544／104,724，准确率53.0385% | [参考指标](references/forecasting/metrics.json) |
| 当前配置对应的已记录最佳搜索结果，118次迭代、学习率0.04 | 55,714／104,724，准确率53.2008% | [搜索记录](runs/catboost_trials.json) |
| 已有 `submission-v1` 官方预测 | 31,870行，记录的公榜分数为0.5140257295 | [提交指标](runs/submission-v1/metrics.json) |

公榜分数来自用户提供的比赛结果。参数搜索使用上述开发验证集选择参数与迭代次数，当前配置记录了这一选择过程。特征说明文档中的245次迭代参数对应原参考模型。

## 6. 参数搜索

```sh
python -m experiments.catboost.tune_catboost \
  --output-dir experiments/catboost/runs/tuning-new \
  --learning-rates 0.04 0.1 \
  --depths 6 7 \
  --max-iterations 300 \
  --patience 40
```

程序先计算配置基线，再遍历学习率与树深度组合。候选模型在开发验证集上选取最佳迭代，连续指定次数没有改善时停止。`--patience` 指定这个次数，`--threads` 设置搜索线程数，默认1，`--skip-baseline` 可跳过基线。

每完成一个候选，将参数、正确数、准确率和耗时写入输出目录的 `trials.json`。采用搜索结果时，将所选参数写入配置，再通过正式入口训练和保存模型。

## 7. 运行输出

| 文件 | 内容 |
|---|---|
| `run_config.json` | 本次配置、数据目录、训练与查询行数 |
| `split_dates.csv` | 验证模式下的匿名日期划分 |
| `features.json` | 124列特征的固定顺序 |
| `preprocessing.json` | 各策略收益尺度和换手率中位数 |
| `model.cbm` | 本次使用的 CatBoost 模型 |
| `query_predictions.csv` | 查询行的原始行号、预测概率等详情 |
| `metrics.json` | 验证指标或官方预测记录数 |
| `submission.csv` | 官方预测模式生成的提交文件 |
| `run.log` | 运行进度与耗时 |
