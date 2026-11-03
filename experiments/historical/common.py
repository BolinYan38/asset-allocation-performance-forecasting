"""Read QRT CSVs, preserve date splits, and write local prediction artifacts.

Returns and volumes are stored oldest first: column 0 is lag 20, column 19
is lag 1. Query targets are never supplied to a fitting operation.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
#define the root path for the project
ROOT = Path(__file__).resolve().parents[2]
# Directory containing this method's code and resources.
METHOD_ROOT = Path(__file__).resolve().parent
#obtain the default data path for the dataset
DEFAULT_DATA = Path.home() / "Dataset/Asset Allocation Performance Forecasting"
#the start time for logging purposes
START = time.monotonic()
LOG_PATH: Path | None = None


class PredictionRows(Protocol):
    """原始数据与特征数据共用的评分、导出字段。"""

    row_ids: np.ndarray
    train_mask: np.ndarray
    targets: np.ndarray
    groups: np.ndarray
    dates: np.ndarray

#保留传进来的进度消息message，记录时间，记到log里
def log(message: str) -> None:
    """Print one progress line, optionally saving it to the current run."""
    line = f"[{time.monotonic() - START:8.1f}s] {message}"
    print(line, flush=True)
    if LOG_PATH is not None:
        with LOG_PATH.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def save_json(path: Path, value: object) -> None:
    """Save readable JSON and reject nonfinite numeric metadata."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")




@dataclass
class RawDataset:
    """Store row arrays and identity dictionaries for historical reconstruction.

    Arrays have shapes [N] except returns/volumes [N, 20]. The train mask
    selects rows whose labels may enter fitting. Official test labels are NaN.
    """

    row_ids: np.ndarray
    dates: np.ndarray
    allocations: np.ndarray
    groups: np.ndarray
    date_names: np.ndarray
    allocation_names: np.ndarray
    returns: np.ndarray
    volumes: np.ndarray
    turnover: np.ndarray
    targets: np.ndarray
    train_mask: np.ndarray


def _read_features(path: Path) -> tuple[np.ndarray, list[list[str]]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)
    columns = [header.index(f"RET_{lag}") for lag in range(20, 0, -1)]
    columns += [header.index(f"SIGNED_VOLUME_{lag}") for lag in range(20, 0, -1)]
    columns.append(header.index("MEDIAN_DAILY_TURNOVER"))
    numeric = np.loadtxt(
        path, delimiter=",", skiprows=1, usecols=columns, dtype=np.float32,
        converters=lambda value: float(value) if value else float("nan"), ndmin=2,
    )
    identity_columns = [header.index(name) for name in ("ROW_ID", "TS", "ALLOCATION", "GROUP")]
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        next(reader)
        identities = [[row[column] for column in identity_columns] for row in reader]
    return numeric, identities


def load_data(data_dir: Path, include_test: bool = False) -> RawDataset:
    """Load training data, optionally appending official X_test without labels.

    Label alignment uses ROW_ID. The CSV order is retained for later exports;
    numerical TS codes are dictionary positions, never chronological positions.
    """
    log("读取训练 CSV。")
    numeric, identities = _read_features(data_dir / "X_train.csv")
    training_rows = len(identities)
    #if there exists a test set, read it and append to the training data
    if include_test:
        test_numeric, test_identities = _read_features(data_dir / "X_test.csv")
        numeric = np.concatenate((numeric, test_numeric))
        identities.extend(test_identities)
    row_ids = np.asarray([int(row[0]) for row in identities], dtype=np.int64)
    date_names, dates = np.unique([row[1] for row in identities], return_inverse=True)
    allocation_names, allocations = np.unique([row[2] for row in identities], return_inverse=True)
    groups = np.asarray([int(row[3]) for row in identities], dtype=np.int16)
    labels = np.loadtxt(data_dir / "y_train.csv", delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    order = np.argsort(labels[:, 0])
    positions = np.searchsorted(labels[order, 0], row_ids[:training_rows])
    #Check that the ROW_IDs in the training set match the labels in y_train.csv 
    if np.any(positions >= len(labels)) or not np.array_equal(labels[order[positions], 0], row_ids[:training_rows]):
        raise ValueError("X_train and y_train ROW_ID alignment failed")
    targets = np.full(len(row_ids), np.nan, dtype=np.float32)
    targets[:training_rows] = labels[order[positions], 1]
    train_mask = np.arange(len(row_ids)) < training_rows
    if len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("ROW_ID must be unique across the loaded dataset")
    for allocation in np.unique(allocations):
        rows = allocations == allocation
        if len(np.unique(groups[rows])) != 1 or not np.any(rows & train_mask):
            raise ValueError("Each allocation must have one GROUP and labeled training examples")
    if include_test and np.intersect1d(dates[:training_rows], dates[training_rows:]).size:
        raise ValueError("Official train and test must have disjoint date identities")
    raw = RawDataset(
        row_ids, dates.astype(np.int32), allocations.astype(np.int32), groups,
        date_names, allocation_names, np.ascontiguousarray(numeric[:, :20]),
        np.ascontiguousarray(numeric[:, 20:40]), numeric[:, 40].copy(), targets, train_mask,
    )
    log(f"读取完成：{len(row_ids):,} 行，{len(date_names)} 日期。")
    return raw




def evaluate(raw: PredictionRows, scores: np.ndarray, threshold: float) -> dict[str, object]:
    """Read held-out labels only after predictions have been generated."""
    rows = np.flatnonzero(~raw.train_mask)
    if not np.isfinite(raw.targets[rows]).all():
        raise ValueError("Evaluation requires local validation labels")
    correct = (scores[rows] > threshold) == (raw.targets[rows] > 0)
    return {
        "accuracy": float(correct.mean()), "correct": int(correct.sum()), "rows": len(rows),
        "dates": len(np.unique(raw.dates[rows])),
        "by_group": {
            str(int(group)): float(correct[raw.groups[rows] == group].mean())
            for group in np.unique(raw.groups[rows])
        },
        "public_leaderboard_score": None,
    }


def write_submission(path: Path, raw: PredictionRows, scores: np.ndarray, threshold: float, sample_path: Path) -> None:
    """Write a local CSV in the sample's exact row order; never upload it."""
    rows = np.flatnonzero(~raw.train_mask)
    if not np.isfinite(scores[rows]).all():
        raise ValueError("Every query row needs a finite prediction")
    predictions = dict(zip(raw.row_ids[rows].tolist(), (scores[rows] > threshold).astype(int).tolist(), strict=True))
    with sample_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        sample_ids = [int(row[0]) for row in reader]
    if (
        len(header) != 2 or header[0] != "ROW_ID"
        or len(set(sample_ids)) != len(sample_ids) or set(sample_ids) != set(predictions)
    ):
        raise ValueError("Sample submission and official query rows do not match")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows((row_id, predictions[row_id]) for row_id in sample_ids)


