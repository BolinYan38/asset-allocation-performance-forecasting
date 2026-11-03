"""将重建特征转换为下一日上涨概率。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from experiments.historical.data import ASSETS, FeatureDataset

if TYPE_CHECKING:
    import pandas as pd


class DirectionClassifier:
    """四个组统一使用包含扩展股票池重建结果的130列特征。"""

    def __init__(self) -> None:
        config = json.loads((ASSETS / "config.json").read_text())
        general = config["base_features"] + config["prior_features"]
        self.columns = {"all": general + config["sparse_features"]}
        self.parameters = config["classifier_parameters"]
        self.models = {}

    def frame(self, features: np.ndarray) -> pd.DataFrame:
        """按固定列名构造表格，策略和组别作为类别特征。"""
        import pandas as pd

        frame = pd.DataFrame(features, columns=self.columns["all"])
        for name in ("allocation", "group"):
            frame[name] = frame[name].astype(np.int32).astype(str)
        return frame

    def fit(self, dataset: FeatureDataset) -> None:
        """使用训练标记选择标签，按固定参数拟合统一分类器。"""
        from catboost import CatBoostClassifier

        frame = self.frame(dataset.features)
        training = dataset.train_mask
        for name, columns in self.columns.items():
            model = CatBoostClassifier(**self.parameters)
            model.fit(frame.loc[training, columns], dataset.targets[training] > 0, cat_features=["allocation", "group"])
            self.models[name] = model

    def predict(self, features: np.ndarray) -> np.ndarray:
        """使用统一分类器返回上涨概率。"""
        frame = self.frame(features)
        return self.models["all"].predict_proba(
            frame.loc[:, self.columns["all"]], thread_count=2,
        )[:, 1]

    def save(self, directory: Path) -> None:
        """保存统一分类器。"""
        directory.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            model.save_model(str(directory / f"{name}.cbm"))

    def load(self, directory: Path) -> None:
        """读取已保存的分类器。"""
        from catboost import CatBoostClassifier

        for name in self.columns:
            model = CatBoostClassifier()
            model.load_model(str(directory / f"{name}.cbm"))
            self.models[name] = model
