"""Compare CatBoost parameters and save all results in one trials.json file."""

from __future__ import annotations

import argparse
import io
import json
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, Pool

from experiments.catboost import common
from experiments.catboost.market_forecaster import AllocationFeatureProcessor


def main() -> None:
    """Run a small single-thread search with one result line per candidate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=common.METHOD_ROOT / "configs/forecasting.json")
    parser.add_argument("--data-dir", type=Path, default=common.DEFAULT_DATA)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.1, 0.15, 0.2, 0.3])
    parser.add_argument("--depths", type=int, nargs="+", default=[7])
    parser.add_argument("--max-iterations", type=int, default=180)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()
    if min(args.max_iterations, args.patience, args.threads) < 1:
        parser.error("Iterations, patience, and threads must be positive")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be empty; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    configuration = json.loads(args.config.read_text(encoding="utf-8"))
    common.LOG_PATH = None
    with redirect_stdout(io.StringIO()):
        raw = common.load_data(args.data_dir, include_test=False)
        common.split_dates(raw, configuration["split_seed"], output)
    frame = AllocationFeatureProcessor().build(raw, raw.train_mask)
    train_rows = np.flatnonzero(raw.train_mask)
    validation_rows = np.flatnonzero(~raw.train_mask)
    labels = (raw.targets > 0).astype(np.int8)
    train_pool = Pool(frame.iloc[train_rows], labels[train_rows], cat_features=["allocation", "group"])
    validation_pool = Pool(frame.iloc[validation_rows], labels[validation_rows], cat_features=["allocation", "group"])
    base = dict(configuration["model_parameters"])
    for key in ("verbose", "silent", "logging_level", "verbose_eval", "early_stopping_rounds", "od_type", "od_wait", "od_pval"):
        base.pop(key, None)
    base.update(thread_count=args.threads, allow_writing_files=False, verbose=False, eval_metric="Accuracy")
    results: list[dict[str, object]] = []
    report = {
        "split_seed": configuration["split_seed"], "training_rows": len(train_rows),
        "validation_rows": len(validation_rows), "prediction_threshold": 0.5,
        "selection_uses_development_validation_labels": True,
        "best_trial": None, "trials": results,
    }
    result_path = output / "trials.json"
    common.save_json(result_path, report)
    seen: set[str] = set()

    def run_trial(overrides: dict[str, object], baseline: bool = False) -> None:
        """Save parameters, accuracy, and comparisons after each completed fit."""
        parameters = {**base, **overrides, "use_best_model": not baseline}
        if not baseline:
            parameters["iterations"] = args.max_iterations
        key = json.dumps(parameters, sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        started = time.monotonic()
        model = CatBoostClassifier(**parameters)
        fit_parameters = {} if baseline else {"early_stopping_rounds": args.patience}
        model.fit(train_pool, eval_set=validation_pool, **fit_parameters)
        probabilities = model.predict_proba(validation_pool, thread_count=args.threads)[:, 1]
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Validation probabilities contain nonfinite values")
        correct = int(np.count_nonzero((probabilities > 0.5) == labels[validation_rows]))
        accuracy = correct / len(validation_rows)
        previous = results[-1]["accuracy"] if results else None
        previous_best = max(item["accuracy"] for item in results) if results else None
        parameters.pop("use_best_model")
        iteration_limit = parameters["iterations"]
        parameters["iterations"] = int(model.tree_count_)
        record = {
            "trial": len(results), "baseline": baseline, "parameters": parameters,
            "iteration_limit": iteration_limit, "early_stopping_rounds": None if baseline else args.patience,
            "trained_iterations": len(model.get_evals_result()["validation"]["Accuracy"]),
            "accuracy": accuracy, "correct": correct, "seconds": round(time.monotonic() - started, 2),
            "vs_previous_percentage_points": None if previous is None else (accuracy - previous) * 100,
            "vs_best_before_percentage_points": None if previous_best is None else (accuracy - previous_best) * 100,
        }
        results.append(record)
        best = max(results, key=lambda item: item["correct"])
        report["best_trial"] = best["trial"]
        common.save_json(result_path, report)
        print(f"Trial {record['trial']:02d}: lr={parameters['learning_rate']}, depth={parameters['depth']}, trees={model.tree_count_}, accuracy={accuracy:.6%}, best={best['accuracy']:.6%}", flush=True)

    if not args.skip_baseline:
        run_trial({}, baseline=True)
    for depth in args.depths:
        for rate in args.learning_rates:
            run_trial({"depth": depth, "learning_rate": rate})
    print(f"Results: {result_path}")


if __name__ == "__main__":
    main()
