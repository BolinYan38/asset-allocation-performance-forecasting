"""用股票收益重建策略：岭回归、共享暴露、快照桥接和稀疏先验。"""

from __future__ import annotations

import numpy as np

from experiments.historical.common import RawDataset
from experiments.historical.data import LAGS, AllocationData

PERSISTENCE = (15.0, 60.0, 240.0)


def ridge_predict(
    kernel: np.ndarray, observed: np.ndarray, cross: np.ndarray, weights: np.ndarray,
    ridge: float = 0.7, temporal: tuple[np.ndarray, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """按缺失模式共同求解；cross可以同时提供多个目标日。"""
    if temporal is not None:
        times, query_time, length = temporal
        kernel = kernel * np.exp(-abs(times[:, None] - times) / length)
        cross = (cross.T * np.exp(-abs(times - query_time) / length)).T
    patterns, codes = np.unique(np.isfinite(observed).T, axis=0, return_inverse=True)
    prediction = np.zeros((*cross.shape[1:], observed.shape[1]))
    error = np.ones(observed.shape[1])
    for code, pattern in enumerate(patterns):
        columns = np.flatnonzero(codes == code)
        rows = np.flatnonzero(pattern & (weights > 1e-6))
        if len(rows) < 4:
            continue
        matrix = kernel[np.ix_(rows, rows)].copy()
        penalty = ridge / weights[rows]
        matrix[np.diag_indices_from(matrix)] += penalty
        coefficients = np.linalg.solve(matrix, observed[np.ix_(rows, columns)])
        prediction[..., columns] = cross[rows].T @ coefficients
        error[columns] = np.sqrt(np.mean((penalty[:, None] * coefficients) ** 2, axis=0))
    return prediction, error


def pooled_history(
    positions: np.ndarray, values: np.ndarray, query: int, selected: np.ndarray,
    radius: int, limit: int, decay: float,
) -> tuple[np.ndarray, np.ndarray]:
    """按真实交易日归并附近快照的重复观测，目标日另交桥接处理。"""
    position = positions[query]
    neighbors = np.flatnonzero(abs(positions - position) <= radius)
    neighbors = neighbors[np.argsort(abs(positions[neighbors] - position))[:limit]]
    donor_days = positions[neighbors, None] - LAGS
    valid = donor_days != position + 1
    days, indices = np.unique(donor_days[valid], return_inverse=True)
    observed = values[neighbors][:, :, selected]
    weights = np.exp(-abs(positions[neighbors] - position) / decay)[:, None, None]
    sums = np.zeros((len(days), len(selected)))
    masses = np.zeros_like(sums)
    np.add.at(sums, indices, (weights * np.nan_to_num(observed))[valid])
    np.add.at(masses, indices, (weights * np.isfinite(observed))[valid])
    return days, np.divide(sums, masses, out=np.full_like(sums, np.nan), where=masses > 0)


def turnover_interpolation(predictions: np.ndarray, turnover: np.ndarray) -> list[float]:
    """用换手率在三档暴露衰减长度之间插值。"""
    lengths = (0.75 / np.nan_to_num(turnover, nan=0.2).clip(0.02) ** 2).clip(15, 240)
    return [np.interp(np.log(length), np.log(PERSISTENCE), row) for length, row in zip(lengths, predictions)]


class ReturnReconstructor:
    """生成34列收益信号、7列重复目标日统计，以及基础市场特征。"""

    def _shared_prior(self, public, values, positions, query, selected):
        """按缺失模式批量拟合共享股票暴露，返回6列信号及先验。"""
        position, stocks = positions[query], public.shape[1]
        days, observed = pooled_history(positions, values, query, selected, 260, 26, 100)
        design, target = public[days] / np.sqrt(stocks), public[position + 1] / np.sqrt(stocks)
        weights = np.exp(-abs(days - position) / 200)
        valid = np.isfinite(observed)
        patterns, codes = np.unique(valid.T, axis=0, return_inverse=True)
        matrices = design.T @ ((patterns * weights)[:, :, None] * design)
        right = design.T @ (weights[:, None] * np.nan_to_num(observed))

        def fit(precision):
            solutions = np.linalg.solve(matrices + 0.7 * precision, right)
            coefficients = solutions[codes, :, np.arange(len(selected))].T
            return np.where(valid.sum(axis=0) >= 4, coefficients, 0)

        identity = np.eye(stocks)
        mean = fit(identity)
        standardized = mean / np.nanstd(observed, axis=0).clip(0.1)
        covariance = standardized @ standardized.T / len(selected)
        covariance *= stocks / max(float(np.trace(covariance)), 1e-12)
        prior = 0.5 * identity + 0.5 * covariance
        full = fit(np.linalg.inv(prior))
        diagonal = fit(np.diag(1 / prior.diagonal()))
        residual = np.where(valid, (design @ full - observed) ** 2, 0).sum(axis=0)
        error = np.sqrt(np.divide(residual, valid.sum(axis=0), out=np.zeros(len(selected)), where=valid.sum(axis=0) >= 4))
        concentration = np.trace(covariance @ covariance) / max(np.trace(covariance) ** 2, 1e-12)
        features = np.column_stack((target @ mean, target @ full, target @ diagonal, error,
                                    np.full(len(selected), len(days)), np.full(len(selected), concentration)))
        return features, prior, mean

    def bridge_features(
        self, public: np.ndarray, values: np.ndarray, positions: np.ndarray, query: int,
        selected: np.ndarray, turnover: np.ndarray, prior: np.ndarray | None = None,
        mean: np.ndarray | None = None, multipliers: np.ndarray | float = 1.0,
    ) -> np.ndarray:
        """拟合快照间的暴露变化；有先验均值时增加第9列。"""
        position = positions[query]
        neighbors = np.flatnonzero(abs(positions - position) <= 84)
        neighbors = neighbors[np.argsort(abs(positions[neighbors] - position))[:5]]
        days = (positions[neighbors, None] - LAGS).ravel()
        times = np.repeat(positions[neighbors], 20)
        observed = values[neighbors][:, :, selected].reshape(-1, len(selected))
        design = public[days] * multipliers / np.sqrt(public.shape[1])
        target = public[position + 1] * multipliers / np.sqrt(public.shape[1])
        weights = np.exp(-abs(times - position) / 84) * np.where(days == position + 1, 5, 1)

        def predict(kernel, cross, length=60, observations=observed, mass=weights):
            return ridge_predict(kernel, observations, cross, mass, temporal=(times, position, length))[0]

        kernel, cross = design @ design.T, design @ target
        simple = np.column_stack([predict(kernel, cross, length) for length in PERSISTENCE])
        with_direct = simple[:, 1]
        if prior is not None:
            kernel, cross = design @ prior @ design.T, design @ prior @ target
            with_direct = predict(kernel, cross)
        without_direct = predict(kernel, cross, mass=weights * (days != position + 1))
        gaps = abs(positions[neighbors[neighbors != query]] - position)
        features = [simple, turnover_interpolation(simple, turnover), with_direct, without_direct,
                    with_direct - without_direct, np.full(len(selected), gaps.min() if len(gaps) else np.nan)]
        if mean is not None:
            features.append(target @ mean + predict(kernel, cross, observations=observed - design @ mean))
        return np.column_stack(features)

    def signals(
        self, public: np.ndarray, values: np.ndarray, positions: np.ndarray,
        query: int, selected: np.ndarray, turnover: np.ndarray,
    ) -> np.ndarray:
        """当前窗口7列、附近历史12列、共享先验6列、快照桥接9列。"""
        position, stocks = positions[query], public.shape[1]
        history, observed, target = public[position - LAGS], values[query][:, selected], public[position + 1]
        kernel = history @ history.T / stocks
        crosses = history @ public[[position, position + 1, position + 2]].T / stocks
        middle, _ = ridge_predict(kernel, observed, crosses, np.ones(20))
        complete = np.isfinite(observed).all(axis=1)
        inverse = np.linalg.inv(kernel[np.ix_(complete, complete)] + 0.7 * np.eye(complete.sum()))
        residual = inverse @ np.nan_to_num(observed[complete]) / inverse.diagonal()[:, None]
        leverage = max(0, target @ target / stocks - crosses[complete, 1] @ inverse @ crosses[complete, 1])
        current = [ridge_predict(kernel, observed, crosses[:, 1], np.ones(20), ridge)[0] for ridge in (0.1, 3.0)]
        current = np.column_stack((current[0], middle[1], current[1], np.sqrt(np.mean(residual ** 2, axis=0)),
                                   np.full(len(selected), leverage), middle[0], middle[2]))
        coefficients = (history[complete] / np.sqrt(stocks)).T @ inverse @ np.nan_to_num(observed[complete])
        days, local_observed = pooled_history(positions, values, query, selected, 84, 9, 48)
        local_history = public[days]
        kernel, cross = local_history @ local_history.T / stocks, local_history @ target / stocks
        weights = np.exp(-abs(days - position) / 84)
        constant = ridge_predict(kernel, local_observed, cross, weights)[0]
        daily = np.column_stack([ridge_predict(kernel, local_observed, cross, weights,
                                 temporal=(days, position + 1, length))[0] for length in PERSISTENCE]).astype(np.float32)
        energy = np.mean(coefficients ** 2, axis=1)
        importance = np.sqrt(energy / max(float(energy.mean()), 1e-12) + 0.1)
        importance = (importance / importance.mean()).clip(0.1, 10)
        adaptive, error = ridge_predict((local_history * importance) @ local_history.T / stocks, local_observed,
                                        (local_history * importance) @ target / stocks, weights,
                                        temporal=(days, position + 1, 60))
        diagnostics = np.tile([target.mean(), target.std(), np.mean(target > 0)], (len(selected), 1))
        local = np.column_stack((constant, daily, turnover_interpolation(daily, turnover), adaptive, error,
                                 np.full(len(selected), len(days)), np.isfinite(local_observed).mean(axis=0),
                                 diagnostics)).astype(np.float32)
        long_features, prior, mean = self._shared_prior(public, values, positions, query, selected)
        bridge = self.bridge_features(public, values, positions, query, selected, turnover, prior, mean)
        return np.column_stack((current, local, long_features, bridge))

    @staticmethod
    def _overlap(values, positions, query, selected):
        """批量统计后续快照中出现的目标日收益。"""
        gaps = positions - positions[query]
        donors = np.flatnonzero((gaps >= 1) & (gaps <= 20))
        result = np.full((len(selected), 7), np.nan, dtype=np.float32)
        result[:, 4] = 0
        if not len(donors):
            return result
        observed = values[donors, 20 - gaps[donors]][:, selected]
        columns = np.flatnonzero(np.isfinite(observed).any(axis=0))
        observed, distances = observed[:, columns], gaps[donors]
        valid, filled = np.isfinite(observed), np.nan_to_num(observed)
        weights = np.exp(-distances[:, None] / 10) * valid
        weights /= weights.sum(axis=0)
        average = (weights * filled).sum(axis=0)
        nearest = np.where(valid, distances[:, None], np.inf).argmin(axis=0)
        result[columns] = np.column_stack((average, filled[nearest, np.arange(len(columns))],
            np.nanmedian(observed, axis=0),
            np.sqrt((weights * (filled - average) ** 2).sum(axis=0)), valid.sum(axis=0), distances[nearest],
            abs((weights * np.sign(filled)).sum(axis=0))))
        return result

    def features(self, data: AllocationData, raw: RawDataset) -> np.ndarray:
        """直接填充固定的112列，保持配置中的特征顺序。"""
        result = np.full((len(raw.row_ids), 112), np.nan, dtype=np.float32)
        for group in range(1, 5):
            public, values, positions, row_map = data.panel(raw, group)
            for query in range(len(positions)):
                selected = np.flatnonzero(row_map[query] >= 0)
                rows = row_map[query, selected]
                result[rows, :34] = self.signals(public, values, positions, query, selected, raw.turnover[rows])
                result[rows, 34:41] = self._overlap(values, positions, query, selected)
        returns, flow = raw.returns / data.scales[raw.allocations, None], np.clip(raw.volumes, -10, 10)
        result[:, 41:81] = np.column_stack((returns, flow))
        statistics = [raw.turnover, np.log(np.maximum(raw.turnover, 1e-5))]
        for window in (5, 10, 20):
            statistics.extend((np.nanmean(returns[:, -window:], axis=1), np.nanstd(returns[:, -window:], axis=1),
                               np.nanmean(flow[:, -window:-1], axis=1)))
        result[:, 81:92] = np.column_stack(statistics)
        keys = raw.dates * 10 + raw.groups
        for key in np.unique(keys):
            rows = np.flatnonzero(keys == key)
            signals = result[rows][:, (1, 7, 9, 12, 19, 20, 21, 29, 33)]
            grouped = np.empty((len(rows), 18), dtype=np.float32)
            grouped[:, 0::2] = np.nanmean(signals, axis=0)
            grouped[:, 1::2] = np.argsort(np.argsort(signals, axis=0), axis=0) / max(1, len(rows) - 1)
            result[rows, 92:110] = grouped
        result[:, 110:] = np.column_stack((raw.allocations, raw.groups))
        result[~np.isfinite(result)] = np.nan
        return result


class SharedSparsePrior:
    """估计同组策略共享的股票重要性，生成15列稀疏重建特征。"""

    def estimate(self, history: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, float]:
        """交替估计股票重要性和观测噪声。"""
        lags = np.isfinite(observed).mean(axis=1) >= 0.8
        columns = np.isfinite(observed[lags]).all(axis=0)
        importance, noise = np.ones(history.shape[1]), 0.7
        if lags.sum() < 8 or columns.sum() < 4:
            return importance, noise
        design = history[lags] / np.sqrt(history.shape[1])
        values = observed[lags][:, columns]
        values = values / np.sqrt(np.mean(values ** 2, axis=0)).clip(0.1)
        covariance, identity = values @ values.T / values.shape[1], np.eye(len(design))
        for _ in range(24):
            inverse = np.linalg.inv((design * importance) @ design.T + noise * identity)
            projected = inverse @ design
            numerator = (projected * (covariance @ projected)).sum(axis=0)
            denominator = (projected * design).sum(axis=0).clip(1e-9)
            update = importance * np.sqrt(numerator.clip(1e-12) / denominator)
            importance = (0.5 * importance + 0.5 * update).clip(1e-4, 20)
            ratio = np.trace(inverse @ covariance @ inverse) / np.trace(inverse)
            noise = np.clip(0.5 * noise * (1 + np.sqrt(max(ratio, 1e-12))), 0.015, 4)
        return importance, float(noise)

    def features(self, data: AllocationData, raw: RawDataset) -> np.ndarray:
        """组合当前窗口、附近历史、快照桥接及先验诊断。"""
        output = np.full((len(raw.row_ids), 15), np.nan, dtype=np.float32)
        bridge = ReturnReconstructor()
        for group in range(1, 5):
            public, values, positions, row_map = data.panel(raw, group, expanded=True)
            stocks = public.shape[1]
            for query, position in enumerate(positions):
                selected = np.flatnonzero(row_map[query] >= 0)
                rows = row_map[query, selected]
                importance, noise = self.estimate(public[position - LAGS], values[query])
                multipliers = np.sqrt(importance)
                current, target = public[position - LAGS] * multipliers, public[position + 1] * multipliers
                kernel, cross = current @ current.T / stocks, current @ target / stocks
                predictions = [ridge_predict(kernel, values[query][:, selected], cross, np.ones(20), ridge)[0]
                               for ridge in (noise, 0.7)]
                days, observed = pooled_history(positions, values, query, selected, 84, 9, 48)
                history, weights = public[days] * multipliers, np.exp(-abs(days - position) / 84)
                predictions.extend(ridge_predict(history @ history.T / stocks, observed, history @ target / stocks,
                                   weights, noise, temporal=(days, position + 1, decay))[0] for decay in (30, 100))
                paths = bridge.bridge_features(public, values, positions, query, selected,
                                               raw.turnover[rows], multipliers=multipliers)
                diagnostic = np.tile([noise, (importance ** 2).sum() / importance.sum() ** 2,
                                      importance.max() / importance.mean()], (len(rows), 1))
                output[rows] = np.column_stack((*predictions, paths, diagnostic))
        return output
