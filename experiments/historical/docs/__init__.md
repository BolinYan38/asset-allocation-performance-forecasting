# __init__.py：历史重建包的公共接口

我们在包入口统一导出数据处理、收益重建、股票先验、分类器和流程对象，使调用程序可以通过 `experiments.historical` 访问主要功能。

## 1. 导入顺序与对象来源

| 导入模块 | 对外对象 | 职责 |
|---|---|---|
| `data.py` | `AllocationData` | 读取日期和行情，生成采样及市场面板 |
| `data.py` | `FeatureDataset` | 保存特征、标签、行号及划分 |
| `model.py` | `DirectionClassifier` | 训练 CatBoost 并预测上涨概率 |
| `pipeline.py` | `ReconstructionProject` | 串联准备、训练和预测 |
| `reconstruction.py` | `ReturnReconstructor` | 生成112列基础特征 |
| `reconstruction.py` | `SharedSparsePrior` | 估计股票重要性，生成15列特征 |
| `stock_prior.py` | `LearnedStockPrior` | 学习一个市场一折的股票先验参数 |
| `stock_prior.py` | `StockPriorFeatures` | 按市场和日期两折生成3列学习先验特征 |

导入语句按表中模块顺序执行，各模块自身的依赖由 Python 加载。包入口将类名绑定到包命名空间，供调用方创建对象。

## 2. __all__ 声明公共名称

`__all__` 列出上述8个对象，确定星号导入时导出的名称，同时明确本包提供的主要调用接口。

例如流程调用可以写为：

```python
from experiments.historical import ReconstructionProject

project = ReconstructionProject()
dataset = project.prepare(mode="validation", device="cpu")
```

`project` 是流程对象，`dataset` 是返回的特征数据。也可以直接导入 `AllocationData` 和 `ReturnReconstructor`，由调用程序组织采样与特征计算。

## 3. 与命令入口的分工

包导入负责提供类，实例化和方法调用由使用者完成。`python -m experiments.historical` 则由 [__main__.py](__main__.md) 解析命令并调用流程。

日历恢复和官方测试日期定位分别通过 `calendar_recovery`、`calendar_alignment` 子模块访问，接口见 [calendar_recovery.md](calendar_recovery.md) 和 [calendar_alignment.md](calendar_alignment.md)。
