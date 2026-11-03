"""官方稀疏日期定位：采样周期约束与公开行情指纹。

保留原 PeriodicDateGrid 与行情评分、约束搜索逻辑。
原先逐股票读取价格缓存的部分改为读取现有 market.npz 中的相同收益数组。
两个类分别接受候选映射；模块本身不运行定位、不保存或覆盖结果。
"""

from __future__ import annotations

import math
from functools import reduce

import numpy as np
import torch

from experiments.historical.common import RawDataset
from experiments.historical.calendar_recovery import (
    CalendarInputs,
    FunctionalCalendarForest,
    session_positions,
)

#根据已经定位的部分测试日期，推断测试日期的采样间隔，缩小候选范围
class PeriodicDateGrid:
    """根据已定位日期推断官方测试日期的采样周期。"""

    def infer(
        self, domains: dict[str, list[str]], previous: dict[str, list[str]],
        sessions: dict[int, np.ndarray], query_names: list[str],
    ) -> tuple[np.ndarray, dict[str, object]]:
        """确定采样间隔、起点及包含全部测试日期的连续区间。"""
        union = np.unique(np.concatenate([values.astype("datetime64[D]") for values in sessions.values()]))
        anchors = np.asarray(sorted(domains[name][0] for name in query_names if len(domains[name]) == 1))
        if len(anchors) < 5:
            raise ValueError("At least five independently constrained query dates are needed")
        positions = np.searchsorted(union, anchors.astype("datetime64[D]"))
        #取已定为日期的最大公约数作为采样间隔
        stride = reduce(math.gcd, np.diff(positions).tolist())
        #最大间隔不超过65天，并大于等于5天
        if not 5 <= stride <= 65:
            raise ValueError(f"Query anchors do not establish a useful periodic grid: {stride}")
        #确立起始日期
        clock = union[int(positions[0] % stride)::stride].astype(str)
        #把已定位日期转换成 clock 内的位置
        anchor_positions = np.searchsorted(clock, anchors)
        query_count = len(query_names)
        first = max(0, int(anchor_positions.max()) - query_count + 1)
        last = min(int(anchor_positions.min()), len(clock) - query_count)
        previous_dates = [previous[name][0] for name in query_names if len(previous[name]) == 1]
        intervals = []
        for start in range(first, last + 1):
            days = clock[start:start + query_count]
            support = int(np.isin(previous_dates, days).sum())
            intervals.append((support, start))
        intervals.sort(reverse=True)
        if not intervals or (len(intervals) > 1 and intervals[0][0] == intervals[1][0]):
            raise ValueError("The previous fingerprints do not distinguish a query-grid interval")
        support, start = intervals[0]
        result = clock[start:start + query_count]
        #返回采样筛选后的真实日期
        return result, {
            "stride_union_sessions": stride, "anchor_dates": anchors.tolist(),
            "first_date": str(result[0]), "last_date": str(result[-1]),
            "date_count": len(result), "previous_fingerprint_support": support,
            "runner_up_support": intervals[1][0] if len(intervals) > 1 else None,
            "uses_query_targets": False,
        }

    def rebuild_domains(
        self, raw: RawDataset, domains: dict[str, list[str]], grid: np.ndarray,
        sessions: dict[int, np.ndarray],
    ) -> tuple[dict[str, list[str]], dict[str, object]]:
        """在采样网格上重新筛候选，保留兼容的单候选日期。"""
        #在这批采样日期中，按开市情况、周一规则筛出每个匿名日期的候选。
        inputs = CalendarInputs(raw)
        query_dates = inputs.query_dates
        choices_by_date = inputs.candidate_domains(grid, session_positions(grid, sessions), query_dates)
        result = {name: list(values) for name, values in domains.items()}
        changed = []
        sizes = []
        for date, indices in zip(query_dates, choices_by_date):
            name = str(raw.date_names[date])
            choices = grid[indices].tolist()
            if len(domains[name]) == 1 and domains[name][0] in choices:
                choices = list(domains[name])
            if not choices:
                raise ValueError(f"No raw-feature-compatible grid date for {name}")
            if not set(domains[name]).intersection(choices):
                changed.append(name)
            result[name] = choices
            sizes.append(len(choices))
        return result, {
            "repaired_empty_old_domains": changed,
            "singleton_before_fingerprint": sizes.count(1),
            "maximum_grid_candidates": max(sizes),
        }


class MarketFingerprintCalendarResolver:
    """用公开行情评分，再联合选择官方测试日期。"""

    def __init__(self, device: str, ridge: float = 0.7) -> None:
        """device 是计算设备，ridge 是岭回归的系数惩罚强度。"""
        self.device = device
        self.ridge = ridge

    @torch.inference_mode()
    def resolve(
        self, raw: RawDataset, candidates: dict[str, list[str]],
        sessions: dict[int, np.ndarray], market: dict[str, np.ndarray],
    ) -> tuple[dict[str, list[str]], dict[str, object]]:
        """筛候选、合并窗口关系、评分，再选出一组互不冲突的日期。"""
        #准备日历
        calendar = np.arange(np.datetime64("2000-02-01"), np.datetime64("2026-01-01"))
        calendar_strings = calendar.astype(str)
        query_dates = np.unique(raw.dates[~raw.train_mask])
        training_dates = np.unique(raw.dates[raw.train_mask])
        training_names = set(raw.date_names[training_dates])
        query_set = set(query_dates.tolist())
        #挑选出日历中具有唯一候选的匿名日期：`locked` 只收集训练集中已经唯一确定的真实日期：
        locked = {
            values[0] for name, values in candidates.items()
            if len(values) == 1 and name in training_names
        }
    
        positions_by_group = session_positions(calendar, sessions)
        domains = []
        for date, name in enumerate(raw.date_names):
            values = candidates[str(name)]
            #排除测试候选中已被训练日期占用的日子，再把日期字符串转换成自然日下标
            if date in query_set:
                values = [value for value in values if value not in locked]
            #如果所有候选都被排除了，就报错
            if not values:
                raise ValueError(f"No distinct calendar option remains for {name}")
            #positions保存这些候选在自然日搜索数组中的位置
            positions = np.searchsorted(calendar_strings, values)
            #检查这些下标是否越过数组末尾
            if (positions >= len(calendar)).any():
                raise ValueError(f"Calendar option outside supported period: {name}")
            #domains按匿名日期编号保存了训练和测试日期的候选，供后面建立关联组使用。
            domains.append(positions.astype(np.int64))

        #接下来走一遍calendar_recovery.py的流程，把测试期的匿名日期和训练期一起来做前后关联筛查。
        inputs = CalendarInputs(raw)
        forest = FunctionalCalendarForest(domains, positions_by_group)
        edges = inputs.edges(self.device, threshold=0.9, max_gap=19, query_only=True)
        for edge in sorted(edges, key=lambda value: (value.gap, -value.confidence)):
            forest.add(edge)

        scores = self._score_dates(inputs, candidates, market, positions_by_group, calendar)
        selected_dates = self._choose_dates(forest, scores, query_dates)
        return self._result(raw, candidates, calendar_strings, query_dates, scores, forest, selected_dates)

    def _score_dates(
        self, inputs: CalendarInputs, candidates: dict[str, list[str]],
        market: dict[str, np.ndarray],
        positions_by_group: dict[int, np.ndarray], calendar: np.ndarray,
    ) -> np.ndarray:
        """计算测试日期对真实行情窗口的岭回归留一日评分，并按市场平均。"""
        #inputs.panel：完整策略收益数组，维度为“匿名日期 × 策略 × 20个历史日”，历史按最早到最近排列
        panel = inputs.panel[inputs.query_dates]
        #保存已经唯一定位的训练日期字符串
        known_dates = [
            candidates[str(inputs.raw.date_names[date])][0]
            for date in inputs.training_dates
            if len(candidates[str(inputs.raw.date_names[date])]) == 1
        ]
        #将这些真实日期转换成 calendar 中的自然日下标
        known_days = np.searchsorted(calendar.astype(str), known_dates)
        #lags 保存历史交易日的偏移量
        lags = np.arange(19, -1, -1)
        #score_sum 保存各市场评分的累加值。
        score_sum = np.zeros((len(inputs.query_dates), len(calendar)), dtype=np.float32)
        #score_count 保存每个位置累计了几个市场的评分
        score_count = np.zeros_like(score_sum, dtype=np.int8)
        expected_groups = np.zeros(len(inputs.query_dates), dtype=np.int8)

        for group in range(1, 5):
            #public_returns是当前市场的股票收益数组
            public_returns = market[f"returns_{group}"]
            #当前市场股票数量
            stock_count = public_returns.shape[1]
            positions = positions_by_group[group]
            #被筛选合格的策略
            selected = (inputs.allocation_groups == group) & inputs.stable
            #提取这些策略出来观察
            observed = panel[:, selected]
            #统计每个测试日期在当前市场，有多少条策略提供了完整的20日收益，大于等于4条以上的市场，就可以参与后续评分计算了。
            valid = np.isfinite(observed).all(axis=-1)
            counts = valid.sum(axis=1)
            rows = np.flatnonzero(counts >= 4)
            if not len(rows):
                continue
            expected_groups[rows] += 1

            #选出可评分的测试日期、把缺失收益填零、调整数组维度。
            values = np.nan_to_num(observed[rows]).transpose(0, 2, 1)
            values = (values - values.mean(axis=1, keepdims=True)) / values.std(axis=1, keepdims=True).clip(1e-6)
            values *= valid[rows, None, :]
            covariance = values @ values.transpose(0, 2, 1) / counts[rows, None, None]
            #用已定位训练日期的股票历史，计算每只股票的波动率，作为后续收益缩放的尺度
            known_positions = positions[known_days]
            stock_history = public_returns[known_positions[known_positions >= 19, None] - lags]
            stock_scale = stock_history.reshape(-1, stock_count).std(axis=0).clip(0.005)
            #取出候选真实日期的20日股票窗口
            days = np.flatnonzero((positions >= 19) & (positions < len(public_returns) - 1))
            #用波动率正规化了
            windows = public_returns[positions[days, None] - lags] / stock_scale
            active = (windows.std(axis=1) > 1e-5).sum(axis=1) >= 8
            days, windows = days[active], windows[active]
            windows -= windows.mean(axis=1, keepdims=True)
            windows /= np.sqrt(np.square(windows).mean(axis=(1, 2))).clip(1e-5)[:, None, None]
            tensors = torch.as_tensor(windows, dtype=torch.float32, device=self.device)
            #岭回归对偶解算法。
            gram = torch.bmm(tensors, tensors.transpose(1, 2)).cpu().numpy().astype(np.float64) / stock_count
            gram[:, np.arange(20), np.arange(20)] += self.ridge
            inverse = np.linalg.inv(gram)
            inverse /= np.diagonal(inverse, axis1=1, axis2=2).copy()[:, :, None]
            penalty = inverse.transpose(0, 2, 1) @ inverse
            covariance = torch.as_tensor(covariance.reshape(len(rows), -1), dtype=torch.float32, device=self.device)
            penalty = torch.as_tensor(penalty.reshape(len(days), -1), dtype=torch.float32, device=self.device)
            score_sum[np.ix_(rows, days)] += (1 - covariance @ penalty.T / 20).cpu().numpy()
            score_count[np.ix_(rows, days)] += 1
        usable = (score_count == expected_groups[:, None]) & (expected_groups[:, None] > 0)
        return np.divide(score_sum, score_count, out=np.full_like(score_sum, -np.inf), where=usable)

    #将同一关联组中的日期作为整体选择，再处理不同关联组之间的日期冲突
    @staticmethod
    def _choose_dates(
        forest: FunctionalCalendarForest, scores: np.ndarray, query_dates: np.ndarray,
    ) -> dict[int, int]:
        """保留相对日期关系，选择评分高且日期互不冲突的组合。"""
        #先对关联组内部挑选
        query_lookup = {int(date): index for index, date in enumerate(query_dates)}
        components = []
        for nodes in forest.components.values():
            members = {date: dates for date, dates in nodes.items() if date in query_lookup}
            if not members:
                continue
            joint = np.sum([scores[query_lookup[date], dates] for date, dates in members.items()], axis=0, dtype=np.float64)
            options = np.flatnonzero(np.isfinite(joint))
            if not len(options):
                raise ValueError(f"No market-supported candidate for query dates {list(members)}")
            options = options[np.argsort(joint[options])[::-1]]
            #只有一个方案的话就有限考虑，因为选择余地不大。方案数量相同时，最佳与次佳分差大的组优先。
            margin = float(joint[options[0]] - joint[options[1]]) if len(options) > 1 else float("inf")
            components.append((members, joint, options, margin))
        components.sort(key=lambda component: (len(component[2]), -component[3]))

        #beam开始组合不同关联组，并排除它们之间占用同一个真实日期的情况。
        beam = [(0.0, {}, frozenset())]
        for members, joint, options, _ in components:
            expansions = []
            for total, assignments, occupied in beam:
                accepted = 0
                for option in options:
                    selected = {date: int(dates[option]) for date, dates in members.items()}
                    days = frozenset(selected.values())
                    #检查与此前分配是否冲突
                    if not days.isdisjoint(occupied):
                        continue
                    expansions.append((total + float(joint[option]), assignments | selected, occupied | days))
                    accepted += 1
                    #每个旧状态最多扩展出 8个不冲突方案
                    if accepted == 8:
                        break
            if not expansions:
                raise ValueError("Distinct-date search exhausted compatible component options")
            #只保留累计总分最高的64个状态，继续处理下一组。
            beam = sorted(expansions, key=lambda state: state[0], reverse=True)[:64]
        #取最高分状态的日期映射
        return beam[0][1]

    #把选好的日期下标转换成日期字符串，整理成映射和报告返回
    def _result(
        self, raw: RawDataset, candidates: dict[str, list[str]], calendar_strings: np.ndarray,
        query_dates: np.ndarray, scores: np.ndarray,
        forest: FunctionalCalendarForest, selected_dates: dict[int, int],
    ) -> tuple[dict[str, list[str]], dict[str, object]]:
        """整理所选日期、候选数量和已有评分，不再执行匹配。"""
        resolved = {name: list(values) for name, values in candidates.items()}
        details = []
        for index, date in enumerate(query_dates):
            position = selected_dates[int(date)]
            name = str(raw.date_names[date])
            resolved[name] = [str(calendar_strings[position])]
            details.append({"TS": name, "selected_date": resolved[name][0],
                            "fingerprint_score": float(scores[index, position])})
        report = {"query_dates": len(query_dates),
                  "accepted_edges": forest.accepted, "rejected_edges": forest.rejected,
                  "assignments": details}
        return resolved, report
