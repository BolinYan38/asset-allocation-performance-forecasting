# __main__.py：历史重建方法的命令行入口

我们通过 `python -m experiments.historical` 调用本方法。Python 执行包中的 `__main__.py`，进入 `main()`，解析命令后交给 `ReconstructionProject`。

## 1. 参数解析

在项目根目录执行：

```sh
python -m experiments.historical --help
```

解析器接受下列参数：

| 参数 | 取值及默认值 | 用途 |
|---|---|---|
| `command` | 必填，`prepare`、`fit`、`predict`、`validate` | 选择执行步骤 |
| `--mode` | `official` 默认，或 `validation` | 选择官方测试或本地验证阶段 |
| `--device` | `mps` 默认，或 `cpu` | 选择股票先验计算设备 |
| `--refit-prior` | 默认关闭，写出参数即开启 | 准备特征时重新拟合股票先验 |

`mps` 使用苹果图形加速，`cpu` 使用通用处理器。参数解析后的对象为 `args`，各项通过同名属性访问。

## 2. main() 的分派顺序

解析参数后创建默认 `ReconstructionProject()`，采用默认原始数据目录及本方法的 `runs/` 输出目录。

| 命令 | 实际调用 |
|---|---|
| `prepare` | `project.prepare(args.mode, args.device, args.refit_prior)` |
| `fit` | `project.fit(args.mode)` |
| `predict` | `project.predict(args.mode)` |
| `validate` | `project.predict("validation")` |

设备和先验重训参数在 `prepare` 分支使用。`validate` 固定读取验证阶段的已有特征和模型，重新预测与评分。

## 3. 本地验证的完整调用

先准备特征，再训练分类器，最后读取该模型生成验证结果：

```sh
python -m experiments.historical prepare --mode validation --device cpu
python -m experiments.historical fit --mode validation
python -m experiments.historical predict --mode validation
```

需要重新训练股票先验时，在第一条命令末尾加入 `--refit-prior`。先验参数的训练发生在特征准备中，最终 CatBoost 的训练发生在 `fit` 中。

已有验证特征与模型时，使用：

```sh
python -m experiments.historical validate
```

## 4. 官方测试的完整调用

```sh
python -m experiments.historical prepare --mode official --device cpu
python -m experiments.historical fit --mode official
python -m experiments.historical predict --mode official
```

这组命令将特征、模型和预测写到 `runs/official/`，最后生成 `submission.csv`。每个子步骤读取的输入与输出文件见 [pipeline.md](pipeline.md)。

模块底部的 `if __name__ == "__main__"` 在命令行执行时调用 `main()`。其他程序通过导入调用具体类时，由调用方决定运行哪些步骤。
