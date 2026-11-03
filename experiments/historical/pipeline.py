"""统一调用数据整理、特征构造、分类训练与预测。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from experiments.historical.common import DEFAULT_DATA, evaluate, save_json, write_submission
from experiments.historical.data import RESULTS, AllocationData, FeatureDataset
from experiments.historical.model import DirectionClassifier
from experiments.historical.reconstruction import ReturnReconstructor, SharedSparsePrior
from experiments.historical.stock_prior import StockPriorFeatures

#给一种类型起名为Mode
Mode = Literal["validation", "official"]


class ReconstructionProject:
    """提供准备特征、拟合和预测三个调用步骤。"""

    def __init__(self, data_dir: Path = DEFAULT_DATA, output_dir: Path = RESULTS) -> None:
        self.data_dir = data_dir
        self.output_dir = output_dir

    def prepare(self, mode: Mode = "validation", device: str = "mps", refit_prior: bool = False) -> FeatureDataset:
        """构造130列特征，已有学习先验权重会直接复用。"""
        #判断是不是验证阶段
        validation = mode == "validation"
        directory = self.output_dir / mode
        directory.mkdir(parents=True, exist_ok=True)
        data = AllocationData(self.data_dir, official=not validation)
        base_width = len(data.config["base_features"])
        records = {name: [] for name in FeatureDataset.__dataclass_fields__}
        for offset in data.config["sample_offsets"]:
            raw = data.sample(offset, validation)
            base = ReturnReconstructor().features(data, raw)
            sparse = SharedSparsePrior().features(data, raw)
            features = np.column_stack((base, np.full((len(base), 3), np.nan, np.float32), sparse))
            keep = np.ones(len(raw.row_ids), bool) if offset == 0 else raw.train_mask
            records["features"].append(features[keep])
            for name in records:
                if name != "features":
                    records[name].append(getattr(raw, name)[keep])
            print(f"采样偏移 {offset} 的重建特征完成。", flush=True)
        dataset = FeatureDataset(**{name: np.concatenate(parts) for name, parts in records.items()})
        prior = StockPriorFeatures(directory / "stock_prior.pt", device)
        dataset.features[:, base_width:base_width + 3] = prior.build(data, dataset.row_ids, validation, refit_prior)
        dataset.save(directory / "features.npz", data.feature_names)
        return dataset

    def fit(self, mode: Mode = "official") -> None:
        """使用当前阶段的训练标签拟合分类器。"""
        directory = self.output_dir / mode
        model = DirectionClassifier()
        model.fit(FeatureDataset.load(directory / "features.npz"))
        model.save(directory)

    def predict(self, mode: Mode = "official") -> np.ndarray:
        """验证阶段记录准确率，官方阶段生成本地提交文件。"""
        directory = self.output_dir / mode
        dataset = FeatureDataset.load(directory / "features.npz")
        query = ~dataset.train_mask
        model = DirectionClassifier()
        model.load(directory)
        probability = model.predict(dataset.features[query])
        scores = np.zeros(len(dataset.row_ids))
        scores[query] = probability
        np.savez_compressed(
            directory / "predictions.npz", row_ids=dataset.row_ids[query], probability=probability,
            targets=dataset.targets[query], groups=dataset.groups[query],
        )
        if mode == "validation":
            metrics = evaluate(dataset, scores, 0.5)
            metrics["public_score"] = metrics.pop("public_leaderboard_score")
            metrics["validation_used_for_model_selection"] = True
            save_json(directory / "metrics.json", metrics)
            print(f"本地验证准确率：{metrics['accuracy']:.6%}", flush=True)
        else:
            write_submission(directory / "submission.csv", dataset, scores, 0.5, self.data_dir / "sample_submission.csv")
        return probability
