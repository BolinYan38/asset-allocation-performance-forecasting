# model.py：将130列特征转换为上涨概率

我们用一个 CatBoost 分类器共同处理四个市场的策略记录。输入包括历史收益、成交量、各方法的收益估计、拟合误差和策略身份，目标标签为目标日策略收益是否严格大于零。

## 1. 入口与执行顺序

```python
model = DirectionClassifier()
model.fit(dataset)
model.save(directory)
probability = model.predict(query_features)
```

`dataset` 为含特征、标签和训练标记的 `FeatureDataset`，`directory` 为模型保存目录，`query_features` 为查询行的130列特征，返回的 `probability` 每行一个上涨概率。

已训练模型通过 `load(directory)` 恢复，再调用 `predict()`。流程中的调用方是 [pipeline.py](pipeline.md)。

## 2. __init__()：确定列名和训练参数

初始化读取 `assets/config.json`，按基础特征112列、学习先验3列、股票加权重建15列的顺序组成完整列名，保存为 `self.columns["all"]`。`self.parameters` 保存训练参数，`self.models` 保存已拟合分类器。

当前配置如下：

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `iterations` | 348 | 提升迭代次数 |
| `depth` | 5 | 树深度 |
| `learning_rate` | 0.035 | 每轮更新幅度 |
| `l2_leaf_reg` | 15 | 叶节点值的平方惩罚强度 |
| `random_seed` | 20261027 | 随机种子 |
| `loss_function` | Logloss | 二分类对数损失 |
| `thread_count` | 2 | 计算线程数 |

训练直接使用配置的迭代次数。参数搜索属于单独的实验步骤。

## 3. frame()：数组转为带列名的表格

`frame(features)` 用完整列名构造 Pandas 数据表。特征必须按配置顺序排列，因此第一个数对应 `current_ridge_01`，最后一个数对应 `sparse_largest_weight`。

策略编号 `allocation` 和市场编号 `group` 先转成整数，再转成字符串，并在训练时声明为类别特征。例如编号7与市场2转为字符串 `"7"` 和 `"2"`，让分类器识别记录所属类别。

其他列作为数值特征传入，包括已有的缺失值。

## 4. fit()：选择训练行并拟合

先将全部特征转换为表格，随后用 `dataset.train_mask` 选行。训练目标为：

```python
dataset.targets[training] > 0
```

`training` 是训练标记，`targets` 保存真实目标日策略收益。收益恰好为零时标签为假。

例如三个训练收益为 `[0.02,-0.01,0]`，对应标签为 `[True,False,False]`。函数将训练特征、这些标签和类别列名交给 CatBoost，结果保存到 `self.models["all"]`。

设训练行数为 N，第 i 行上涨标签为 bᵢ，模型上涨概率为 pᵢ，i 从1至 N。配置的对数损失为：

$$
-\frac1N\sum_{i=1}^N\left[b_i\log p_i+(1-b_i)\log(1-p_i)\right].
$$

bᵢ 取0或1。分类器通过特征组合学习不同收益估计与最终涨跌之间的关系。

## 5. predict()：返回上涨类别概率

查询特征先经过同一个 `frame()`，再按训练列顺序传入 `predict_proba()`，取第1号类别，即上涨类别的概率。

例如模型给出两类概率 `[0.28,0.72]`，本函数返回0.72。调用方使用0.5阈值转换为方向，因此0.72对应上涨，0.5本身对应非上涨。

## 6. save() 与 load()

`save()` 创建输出目录，将键名为 `all` 的分类器写入 `all.cbm`。`load()` 按初始化得到的模型名称读取同名文件，重新填入 `self.models`。

特征名称和顺序由当前配置提供，特征矩阵由 `features.npz` 提供，模型文件保存已学习的树和相关参数。这三部分共同组成预测输入。
