# 日期定位与历史收益重建

我们通过匿名日期定位，将策略历史与真实股票行情对齐，再重建策略收益，并用 CatBoost 综合各项特征预测涨跌。`common.py` 提供本方法的数据读取、评分和导出，全部内部导入使用 `experiments.historical`。

```text
historical/
  common.py                本方法的数据工具
  calendar_recovery.py     初始日期候选与窗口关系
  calendar_alignment.py    官方测试日期的行情评分与联合定位
  data.py                  采样、日期对齐和行情面板
  reconstruction.py        收益重建与共享稀疏先验
  stock_prior.py           学习股票先验
  model.py                 方向分类
  pipeline.py              准备、训练、预测流程
  assets/                  market.npz、calendar.json、config.json、sources.json
  runs/                    已有验证及官方阶段产物
  docs/                    按文件组织的程序说明
```

在项目根目录运行：

```sh
python -m experiments.historical --help
python -m experiments.historical validate
```

`validate` 读取保存的验证特征和模型，重新预测并更新本目录 `runs/validation/` 下的验证结果。我们保存的验证结果为准确率73.308011%，5,792行中正确4,246行。

特征准备、模型训练和预测分开执行：

```sh
python -m experiments.historical prepare --mode validation --device cpu
python -m experiments.historical fit --mode validation
python -m experiments.historical predict --mode validation
```

[数据说明](docs/data.md)与[运行流程](docs/pipeline.md)列出完整步骤；日期定位分别见 [calendar_recovery.md](docs/calendar_recovery.md) 与 [calendar_alignment.md](docs/calendar_alignment.md)。依赖列在 `requirements.txt`。
