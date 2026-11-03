# Asset Allocation Performance Forecasting

我们实现了基于历史特征的 CatBoost 分类和基于日期定位的历史收益重建两种方法，统一放在 `experiments/` 下：

```text
experiments/
  catboost/       常规特征与CatBoost预测
  historical/     日期定位与历史收益重建
```

两种方法分别使用目录内的 `common.py`，包含各自的数据读取、评分和输出函数。

## 方法来源

我们的匿名日期定位方法参考了 RyanBalech 的 [QRT Asset Allocation — Historical Reconstruction](https://github.com/RyanBalech/qrt-asset-allocation-reconstruction)。该项目结合历史收益窗口重合、真实交易所日历和股票收益拟合恢复匿名日期，为本项目的历史重建方法提供了基础。

对应的原始实现包括：

- [whiten_matches.py](https://github.com/RyanBalech/qrt-asset-allocation-reconstruction/blob/7db1df7ddab4863c346de6b2ac5e9e441bc5501e/research/whiten_matches.py)：通过错位历史窗口的余弦相似度寻找日期间隔关系。
- [v4_exchange_sessions.py](https://github.com/RyanBalech/qrt-asset-allocation-reconstruction/blob/7db1df7ddab4863c346de6b2ac5e9e441bc5501e/research/v4_exchange_sessions.py)：提供日本、香港、韩国和上海市场的交易日历。
- [v4_stock_clock.py](https://github.com/RyanBalech/qrt-asset-allocation-reconstruction/blob/7db1df7ddab4863c346de6b2ac5e9e441bc5501e/research/v4_stock_clock.py)：结合市场开市情况、最新成交量的周一报告规则，以及股票收益岭回归的留一日误差，对候选真实日期评分。

我们在这些方法的基础上整理并修改了实现，当前日期定位流程见 [calendar_recovery.md](experiments/historical/docs/calendar_recovery.md) 和 [calendar_alignment.md](experiments/historical/docs/calendar_alignment.md)。

## 原始比赛数据

**原始比赛数据未上传至 GitHub，本仓库不包含原始训练集、测试集和标签文件。** 从原始数据开始运行实验时，请自行从比赛平台获取以下文件，并放入默认数据目录 `~/Dataset/Asset Allocation Performance Forecasting/`：

```text
~/Dataset/Asset Allocation Performance Forecasting/
  X_train.csv             原始训练特征
  y_train.csv             原始训练标签
  X_test.csv              官方测试特征
  sample_submission.csv   提交格式与行号顺序
```

仓库中已保存的股票行情、特征缓存、模型权重和预测结果属于实验资料，与上述原始比赛文件分别存放。

## 方法与运行说明

| 方法 | 程序入口 | 说明与资料 |
|---|---|---|
| 常规CatBoost | `python -m experiments.catboost` | [方法与特征说明](experiments/catboost/docs/CatBoost常规预测.md) |
| 历史重建 | `python -m experiments.historical` | [运行说明](experiments/historical/README.md)、[日期恢复](experiments/historical/docs/calendar_recovery.md)、[官方日期定位](experiments/historical/docs/calendar_alignment.md) |

历史重建的行情与配置在 `experiments/historical/assets/`，已有特征、权重与预测在 `experiments/historical/runs/`。我们保存的历史重建验证结果为5,792行中正确4,246行，准确率73.308011%。

常规CatBoost的配置在 `experiments/catboost/configs/`，基线验证模型在 `experiments/catboost/references/forecasting/`，既有运行结果在 `experiments/catboost/runs/`。

在项目根目录运行命令。数据整理见 [data.md](experiments/historical/docs/data.md)，历史重建运行步骤见 [pipeline.md](experiments/historical/docs/pipeline.md)，命令参数见 [__main__.md](experiments/historical/docs/__main__.md)。两种方法的依赖分别列在各自目录的 `requirements.txt`。
