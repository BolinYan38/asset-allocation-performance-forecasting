#!/usr/bin/env python3
"""Predict allocation next-day direction with a compact CatBoost classifier.

The frozen 124-feature model recorded 53.0385% accuracy on the established
20% date holdout. Inputs are observed history and same-date/group X only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from experiments.catboost import common
from experiments.catboost.common import METHOD_ROOT, DEFAULT_DATA, RawDataset


class AllocationFeatureProcessor:
    """Fit historical scales and construct the frozen 124-feature matrix."""

    def build(self, raw: RawDataset, fit_mask: np.ndarray) -> pd.DataFrame:
        """Fit only historical X scales; all aggregations read X, never targets."""
        returns = raw.returns.astype(np.float32)
        volumes = raw.volumes.astype(np.float32)
        allocation_count = len(raw.allocation_names)
        scales = np.ones(allocation_count, np.float32)
        turnovers = np.ones(allocation_count, np.float32)
        self.statistics = {}
        for allocation in range(allocation_count):
            rows = fit_mask & (raw.allocations == allocation)
            scales[allocation] = max(float(np.nanstd(returns[rows])), 1e-6)
            turnovers[allocation] = max(float(np.nanmedian(raw.turnover[rows])), 1e-8)
            self.statistics[str(raw.allocation_names[allocation])] = {
                "return_scale": float(scales[allocation]),
                "turnover_median": float(turnovers[allocation]),
            }
        normalized = returns / scales[raw.allocations, None]
        flow = np.arcsinh(np.nan_to_num(volumes))
        features = {}

        def add(name: str, values: np.ndarray) -> None:
            features[name] = np.nan_to_num(values, nan=0, posinf=30, neginf=-30).astype(np.float32)

        for position, lag in enumerate(range(20, 0, -1)):
            add(f"return_{lag}", normalized[:, position])
            add(f"flow_{lag}", flow[:, position])
        add("volume_1_observed", np.isfinite(volumes[:, -1]))
        add("log_turnover", np.log10(np.maximum(raw.turnover, 1e-12)))
        add("relative_turnover", raw.turnover / turnovers[raw.allocations])
        for window in (2, 3, 5, 10, 20):
            history = normalized[:, -window:]
            mean = history.mean(axis=1)
            std = history.std(axis=1)
            add(f"mean_{window}", mean)
            add(f"std_{window}", std)
            add(f"positive_{window}", (history > 0).mean(axis=1))
            add(f"downside_{window}", np.square(np.minimum(history, 0)).mean(axis=1) ** 0.5)
            add(f"sharpe_{window}", mean / np.maximum(std, 0.05))
            add(f"reversal_{window}", normalized[:, -1] - mean)
            add(f"flow_mean_{window}", flow[:, -window:-1].mean(axis=1))
            add(f"flow_std_{window}", flow[:, -window:-1].std(axis=1))
        std = normalized.std(axis=1).clip(0.05)
        add("last_return_z", normalized[:, -1] / std)
        add("volatility_ratio_5_20", normalized[:, -5:].std(axis=1) / std)
        add("return_min", normalized.min(axis=1))
        add("return_max", normalized.max(axis=1))
        add("absolute_return_mean", np.abs(normalized).mean(axis=1))
        add("sign_continuation", (np.sign(normalized[:, 1:]) == np.sign(normalized[:, :-1])).mean(axis=1))
        add("momentum_change", normalized[:, -5:].mean(axis=1) - normalized[:, :5].mean(axis=1))
        time_axis = np.arange(20, dtype=np.float32) - 9.5
        add("trend", (normalized * time_axis).sum(axis=1) / np.square(time_axis).sum())
        centered = normalized - normalized.mean(axis=1, keepdims=True)
        add("skewness", (centered ** 3).mean(axis=1) / std ** 3)
        add("kurtosis", (centered ** 4).mean(axis=1) / std ** 4)
        add("last_return_turnover", normalized[:, -1] * np.sqrt(raw.turnover.clip(0.001)))
        add("last_return_low_turnover", normalized[:, -1] / np.sqrt(raw.turnover.clip(0.01)))
        past_returns = normalized[:, :-1]
        past_flow = flow[:, :-1]
        covariance = ((past_returns - past_returns.mean(axis=1, keepdims=True)) *
                      (past_flow - past_flow.mean(axis=1, keepdims=True))).mean(axis=1)
        add("return_flow_correlation", covariance / (past_returns.std(axis=1) * past_flow.std(axis=1)).clip(0.01))

        frame = pd.DataFrame(features, copy=False)
        day_groups = pd.Series(raw.dates.astype(np.int64) * 10 + raw.groups)
        columns = ["return_1", "mean_5", "mean_20", "std_20", "flow_mean_5", "log_turnover"]
        grouped = frame[columns].groupby(day_groups, sort=False)
        means = grouped.transform("mean")
        deviations = grouped.transform("std").fillna(0).clip(lower=0.01)
        ranks = grouped.rank(pct=True)
        for column in columns:
            add(f"group_mean_{column}", means[column].to_numpy())
            add(f"group_deviation_{column}", (frame[column] - means[column]).to_numpy())
            add(f"group_z_{column}", ((frame[column] - means[column]) / deviations[column]).to_numpy())
            add(f"group_rank_{column}", ranks[column].to_numpy())
        add("group_size", pd.Series(raw.dates).groupby(day_groups, sort=False).transform("size").to_numpy())
        add("date_size", pd.Series(raw.dates).groupby(raw.dates, sort=False).transform("size").to_numpy())
        result = pd.DataFrame(features)
        result["allocation"] = pd.Categorical(raw.allocations)
        result["group"] = pd.Categorical(raw.groups)
        self.feature_names = result.columns.tolist()
        return result


class CatBoostDirectionModel:
    """Learn nonlinear path responses and allocation effects with small trees."""

    def __init__(self, parameters: dict[str, object]) -> None:
        """Bind the frozen parameters, using two threads and no implicit files."""
        self.parameters = dict(parameters)
        self.parameters.update({"thread_count": 2, "allow_writing_files": False})
        self.model = CatBoostClassifier(**self.parameters)

    def fit(self, frame: pd.DataFrame, raw: RawDataset) -> None:
        """Use only labeled training rows; the iteration count is already fixed."""
        rows = np.flatnonzero(raw.train_mask)
        self.model.fit(
            frame.iloc[rows], (raw.targets[rows] > 0).astype(np.int8),
            cat_features=["allocation", "group"],
        )

    def load(self, path: Path, feature_names: list[str]) -> None:
        """Load a native model and require its feature schema to match."""
        self.model.load_model(str(path))
        if self.model.feature_names_ != feature_names:
            raise ValueError("Saved model feature names do not match the current processor")

    def predict(self, frame: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
        """Return positive-return probabilities with two prediction threads."""
        return self.model.predict_proba(frame.iloc[rows], thread_count=2)[:, 1]

    def save(self, path: Path) -> None:
        """Save the native CatBoost model, including categorical transforms."""
        self.model.save_model(str(path))


class ForecastingPipeline:
    """Connect historical feature fitting, CatBoost, and query probabilities."""

    def __init__(self, configuration: dict[str, object]) -> None:
        """Use the selected model configuration without searching outer labels."""
        self.configuration = configuration

    def run(self, raw: RawDataset, output: Path, checkpoint: Path | None = None) -> np.ndarray:
        """Fit once or reuse the original-split model, then predict query rows."""
        processor = AllocationFeatureProcessor()
        frame = processor.build(raw, raw.train_mask)
        common.log(f"特征构造完成：{frame.shape[1]} 维。")
        model = CatBoostDirectionModel(self.configuration["model_parameters"])
        if checkpoint is None:
            model.fit(frame, raw)
        else:
            model.load(checkpoint, processor.feature_names)
            common.log("已读取保存的80%训练模型，只执行预测。")
        rows = np.flatnonzero(~raw.train_mask)
        probabilities = np.full(len(raw.row_ids), np.nan, dtype=np.float64)
        probabilities[rows] = model.predict(frame, rows)
        if not np.isfinite(probabilities[rows]).all():
            raise RuntimeError("Some query rows have no finite probability")
        model.save(output / "model.cbm")
        common.save_json(output / "preprocessing.json", processor.statistics)
        common.save_json(output / "features.json", processor.feature_names)
        return probabilities


def main() -> None:
    """Explicitly run fixed-split validation or full-train official prediction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["validate", "predict"])
    parser.add_argument("--config", type=Path, default=METHOD_ROOT / "configs/forecasting.json")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--checkpoint", type=Path, help="Original 80/20 validation model; prediction only")
    args = parser.parse_args()
    configuration = json.loads(args.config.read_text(encoding="utf-8"))
    if args.checkpoint is not None and (args.mode != "validate" or configuration["split_seed"] != 42):
        parser.error("Checkpoint reuse requires the original seed-42 validation split")
    output = args.output_dir.resolve()
    common.start_run(output, args.device)
    raw = common.load_data(args.data_dir, include_test=args.mode == "predict")
    if args.mode == "validate":
        common.split_dates(raw, int(configuration["split_seed"]), output)
    configuration.update({"mode": args.mode, "device": "cpu", "cpu_threads": 2,
                          "data_dir": str(args.data_dir.resolve()),
                          "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                          "training_rows": int(raw.train_mask.sum()), "query_rows": int((~raw.train_mask).sum()),
                          "uses_external_data": False, "uses_successor_features": False})
    common.save_json(output / "run_config.json", configuration)
    probabilities = ForecastingPipeline(configuration).run(raw, output, args.checkpoint)
    common.write_details(output / "query_predictions.csv", raw, probabilities, 0.5)
    if args.mode == "validate":
        report = common.evaluate(raw, probabilities, 0.5)
        common.log(f"完整20%日期验证准确率：{report['accuracy']:.6%}。")
    else:
        common.write_submission(
            output / "submission.csv", raw, probabilities, 0.5, args.data_dir / "sample_submission.csv"
        )
        report = {"rows": int((~raw.train_mask).sum()), "accuracy": None, "public_leaderboard_score": None}
        common.log("官方测试预测已保存在本地。")
    report.update({"method": configuration["method"], "uses_external_data": False,
                   "uses_successor_features": False, "elapsed_seconds": time.monotonic() - common.START})
    common.save_json(output / "metrics.json", report)


if __name__ == "__main__":
    main()
