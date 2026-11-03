"""读取比赛数据、日历和股票收益，统一成日期与策略面板。"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from experiments.historical.common import METHOD_ROOT, DEFAULT_DATA, RawDataset, load_data

ASSETS = METHOD_ROOT / "assets"
RESULTS = METHOD_ROOT / "runs"
LAGS = np.arange(19, -1, -1)
ROW_FIELDS = (
    "row_ids", "dates", "allocations", "groups", "returns", "volumes", "turnover", "targets", "train_mask",
)


def subset(raw: RawDataset, rows: np.ndarray) -> RawDataset:
    """选取行，保留原有日期和策略编号。"""
    return replace(raw, **{name: getattr(raw, name)[rows] for name in ROW_FIELDS})


@dataclass
class FeatureDataset:
    """保存130列特征及其行号、标签和训练标记。"""

    features: np.ndarray
    row_ids: np.ndarray
    targets: np.ndarray
    train_mask: np.ndarray
    groups: np.ndarray
    dates: np.ndarray

    @classmethod
    def load(cls, path: Path) -> FeatureDataset:
        """读取已计算的特征。"""
        with np.load(path) as archive:
            #这里的cls.__dataclass_fields__就是前面六个属性。
            return cls(**{name: archive[name] for name in cls.__dataclass_fields__})

    def save(self, path: Path, feature_names: list[str]) -> None:
        """将特征和训练标记保存在同一个文件中。"""
        np.savez_compressed(path, feature_names=feature_names, **vars(self))


class AllocationData:
    """整理比赛原始数据，提供采样日期和市场面板。"""

    def __init__(self, data_dir: Path = DEFAULT_DATA, official: bool = False) -> None:
        self.raw = load_data(data_dir, include_test=official)
        self.config = json.loads((ASSETS / "config.json").read_text())
        self.calendar = json.loads((ASSETS / "calendar.json").read_text())
        with np.load(ASSETS / "market.npz") as archive:
            self.market = {name: archive[name] for name in archive.files}
        self.sessions = {group: self.market[f"sessions_{group}"].astype(str) for group in range(1, 5)}
        self.real_dates = np.asarray([
            self.calendar[str(name)][0] if len(self.calendar[str(name)]) == 1 else "9999-01-01"
            for name in self.raw.date_names
        ])
        training = self.raw.train_mask
        counts = np.bincount(self.raw.allocations[training], minlength=len(self.raw.allocation_names))
        squares = np.bincount(
            self.raw.allocations[training], weights=np.nanmean(self.raw.returns[training] ** 2, axis=1),
            minlength=len(counts),
        )
        self.scales = np.sqrt(squares / counts.clip(1)).clip(1e-6)

    @property
    def feature_names(self) -> list[str]:
        """按基础重建、学习先验、稀疏先验排列130列名称。"""
        return self.config["base_features"] + self.config["prior_features"] + self.config["sparse_features"]

    def allowed_labels(self, validation: bool) -> np.ndarray:
        """选择当前阶段可用于拟合的标签。"""
        allowed = self.raw.train_mask.copy()
        if validation:
            allowed &= self.real_dates[self.raw.dates] < self.config["validation_cutoff"]
        return allowed

    def dated(self) -> tuple[RawDataset, np.ndarray]:
        """返回日期已唯一定位的数据及其原始行位置。"""
        rows = np.flatnonzero(self.real_dates[self.raw.dates] != "9999-01-01")
        return subset(self.raw, rows), rows

    def sample(self, offset: int = 0, validation: bool = True) -> RawDataset:
        """每21个联合交易日取一个快照，官方阶段保留全部查询日期。"""
        stride = self.config["sample_stride"]
        union = np.unique(np.concatenate(list(self.sessions.values())))
        phase = np.searchsorted(union, self.config["validation_cutoff"]) % stride
        clock = union[(phase + offset) % stride::stride]
        dates = np.flatnonzero(np.isin(self.real_dates, clock))
        if not validation:
            dates = np.union1d(dates, self.raw.dates[~self.raw.train_mask])
        result = subset(self.raw, np.flatnonzero(np.isin(self.raw.dates, dates)))
        if validation:
            result.train_mask = self.real_dates[result.dates] < self.config["validation_cutoff"]
        return result

    def panel(
        self, raw: RawDataset, group: int, expanded: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """返回标准化股票收益、策略历史、交易日位置和行号表。"""
        key = "expanded_returns_3" if expanded and group == 3 else f"returns_{group}"
        public = self.market[key]
        sessions = self.sessions[group]
        rows = np.flatnonzero(raw.groups == group)
        allocations = np.unique(raw.allocations[rows])
        dates = sorted(np.unique(raw.dates[rows]), key=lambda date: self.real_dates[date])
        positions = np.searchsorted(sessions, self.real_dates[dates])
        values = np.full((len(dates), 20, len(allocations)), np.nan)
        row_map = np.full((len(dates), len(allocations)), -1, dtype=np.int64)
        date_index = np.empty(len(self.real_dates), dtype=np.int64)
        date_index[dates] = np.arange(len(dates))
        time_index = date_index[raw.dates[rows]]
        columns = np.searchsorted(allocations, raw.allocations[rows])
        values[time_index, :, columns] = raw.returns[rows] / self.scales[raw.allocations[rows], None]
        row_map[time_index, columns] = rows
        history = np.isin(dates, np.unique(raw.dates[raw.train_mask]))
        scale = public[positions[history, None] - LAGS].reshape(-1, public.shape[1]).std(axis=0).clip(0.005)
        return public / scale, values, positions, row_map
