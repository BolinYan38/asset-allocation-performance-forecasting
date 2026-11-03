"""匿名日期定位：筛候选、匹配重合窗口、合并日期约束。入口为 recover()。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch
from torch.nn import functional

from experiments.historical import common


class SuccessorWindowMatcher:
    """用全部选中策略的余弦相似度，核对错开后的窗口重合。"""

    def __init__(
        self,
        device: str = "cpu",
        threshold: float = 0.9,
        max_gap: int = 19,
        batch_size: int = 128,
    ) -> None:
        """设置计算设备、相似度阈值、最大交易日间隔和批量大小。"""
        self.device = torch.device(device)
        self.threshold = threshold
        self.max_gap = max_gap
        self.batch_size = batch_size

    def match(
        self,
        windows: torch.Tensor,
        query_dates: np.ndarray,
        reference_dates: np.ndarray,
        group: int,
        reverse: bool = False,
    ) -> tuple[list[CalendarEdge], np.ndarray]:
        """返回整体窗口相似度达标的日期关系，以及未匹配的查询日期。"""
        #初始化所有查询日期未匹配，
        matched = np.zeros(len(query_dates), dtype=bool)
        #初始化准备保存日期关系的列表
        edges = []
        #记录传输到指定计算设备
        values = windows.to(self.device)
        references = torch.as_tensor(reference_dates, device=self.device)

        #gap 是正在尝试的市场交易日间隔
        for gap in range(1, self.max_gap + 1):
            pending = np.flatnonzero(~matched)
            if not len(pending):
                break
            reference_vectors = functional.normalize(values[references, gap:].flatten(1), dim=1)
            for start in range(0, len(pending), self.batch_size):
                positions = pending[start : start + self.batch_size]
                dates = torch.as_tensor(query_dates[positions], device=self.device)
                query = functional.normalize(values[dates, : 20 - gap].flatten(1), dim=1)
                similarities = query @ reference_vectors.T
                ranked = similarities.masked_fill(dates[:, None] == references[None, :], -2.0)
                best = ranked.argmax(dim=1, keepdim=True)
                scores = similarities.gather(1, best).squeeze(1).cpu().numpy()
                chosen = best.squeeze(1).cpu().numpy()
                accepted = scores >= self.threshold
                matched[positions[accepted]] = True
                for index in np.flatnonzero(accepted):
                    left = int(query_dates[positions[index]])
                    right = int(reference_dates[chosen[index]])
                    if reverse:
                        left, right = right, left
                    confidence = float(scores[index])
                    edges.append(CalendarEdge(left, right, int(group), gap, confidence))
        #edge返回两个匿名日期的关系，query_date返回未匹配的日期。
        return edges, query_dates[~matched]


@dataclass(frozen=True)
class CalendarEdge:
    """记录前后两个匿名日期、市场组别、交易日间隔及置信度。"""

    left: int
    right: int
    group: int
    gap: int
    confidence: float


class FunctionalCalendarForest:
    """联合保留一组日期的兼容候选组合。"""

    def __init__(self, domains: list[np.ndarray], session_positions: dict[int, np.ndarray]) -> None:
        """每个匿名日期从独立的候选集合开始。"""
        #owners 记录“每个匿名日期属于哪个关联组”，通过下标联系。初始还没有发现日期之间的关系，因此各自独立。
        self.owners = np.arange(len(domains))
        #外层键：关联组编号。内层键：匿名日期编号。内层之：匿名组对应的日期。初始还未找到关联，所以内外层键相同。
        self.components = {node: {node: dates.copy()} for node, dates in enumerate(domains)}
        #positions 字典，保存各市场对每个自然日的交易日查询结果。
        self.session_positions = session_positions
        #空的缓存字典，供后面保存“在某市场移动若干交易日”的查询结果
        self.maps: dict[tuple[int, int], np.ndarray] = {}
        #分别记录后续接受和拒绝了多少条日期关系，初始都是零。
        self.accepted = 0
        self.rejected = 0

    def shift_map(self, group: int, gap: int) -> np.ndarray:
        """查找向前或向后移动指定交易日数后的自然日期下标。"""
        key = group, gap
        if key not in self.maps:
            positions = self.session_positions[group]
            valid_days = np.flatnonzero(positions >= 0)
            lookup = {int(positions[day]): int(day) for day in valid_days}
            mapping = np.full(len(positions), -1, dtype=np.int64)
            for day in valid_days:
                mapping[day] = lookup.get(int(positions[day]) + gap, -1)
            self.maps[key] = mapping
        return self.maps[key]

    #再用match得到日期关系候选后，开始合并日期关系
    def add(self, edge: CalendarEdge) -> bool:
        """合并一条日期关系；没有兼容候选时拒绝这条关系。"""
        #最开始owners都是互相独立的，随着关系逐渐合并。
        left_owner, right_owner = self.owners[edge.left], self.owners[edge.right]
        #left，right是要准备被判断的两个匿名日期的备选日期字典
        left, right = self.components[left_owner], self.components[right_owner]
        #根据left推选right应该是在哪一天里选
        shifted = self.shift_map(edge.group, edge.gap)[left[edge.left]]
        #如果两个匿名日期已经在同一个owners关联组里，就要判断一下了是否产生矛盾，会则拒绝这条合并。
        if left_owner == right_owner:
            keep = shifted == right[edge.right]
            if not keep.any():
                self.rejected += 1
                return False
            self.components[left_owner] = {node: dates[keep] for node, dates in left.items()}
        else:
            #左右匿名日期不在同一关联日期表里，则按照shifted的关系挑出一一对应的，分别放到left_position,right_position。
            inverse = np.full(len(self.session_positions[edge.group]), -1, dtype=np.int64)
            inverse[right[edge.right]] = np.arange(len(right[edge.right]))
            right_positions = inverse[shifted.clip(min=0)]
            keep = (shifted >= 0) & (right_positions >= 0)
            if not keep.any():
                self.rejected += 1
                return False
            #left_positions = [0, 2]   # 左边保留第0、2号候选方案
            #right_positions = [1, 3]  # 分别配对右边第1、2号候选方案
            left_positions = np.flatnonzero(keep)
            right_positions = right_positions[keep]
            #写出左右关联组以及他们每个匿名日期代表的候选日期。
            #left = {
            #           0: np.array([4, 5, 7, 9]),  # 甲
            #           2: np.array([3, 4, 6, 8]),  # 丙
            #        }
            #right = {
            #           1: np.array([0, 5, 8]),  # 乙，沿用前面的候选数组
            #           3: np.array([1, 6, 9]),  # 丁
            #         }
            merged = {node: dates[left_positions] for node, dates in left.items()}
            # 甲：[4, 5, 7, 9] → [4, 7]
            # 丙：[3, 4, 6, 8] → [3, 6]
            merged.update({node: dates[right_positions] for node, dates in right.items()})
            # 乙：[0, 5, 8] → [5, 8]
            # 丁：[1, 6, 9] → [6, 9]
            # Different anonymous dates must remain different real dates.
            dates = np.sort(np.stack(list(merged.values())), axis=0)
            distinct = (np.diff(dates, axis=0) > 0).all(axis=0)
            #有日期重复就拒绝合并
            if not distinct.any():
                self.rejected += 1
                return False
        
            merged = {node: dates[distinct] for node, dates in merged.items()}
            #统一存到左边
            self.components[left_owner] = merged
            #删掉右边
            del self.components[right_owner]
            #把owners的组内所述关系的外层key更新一下，现在甲乙丙丁都在同一组关系里了。
            self.owners[list(right)] = left_owner
        self.accepted += 1
        return True

    def domains(self) -> list[np.ndarray]:
        """返回各匿名日期当前剩余的候选下标。"""
        return [self.components[self.owners[node]][node] for node in range(len(self.owners))]


class CalendarInputs:
    """整理两种日期定位方法共用的策略信息、候选规则和窗口关系。"""

    def __init__(self, raw: common.RawDataset) -> None:
        self.raw = raw
        #查询训练行和非训练行
        self.training_dates = np.unique(raw.dates[raw.train_mask])
        self.query_dates = np.unique(raw.dates[~raw.train_mask])

        #数每条策略有多少条训练记录
        counts = np.bincount(raw.allocations[raw.train_mask], minlength=len(raw.allocation_names))
        #数每个策略有多少条最后一天交易量存在
        reporting = np.bincount(
            raw.allocations[raw.train_mask & np.isfinite(raw.volumes[:, -1])], minlength=len(counts)
        )
        #策略训练记录大于500，且最后一天交易量汇报量小于0.9的，这些策略涉及的记录多，成交量也有一定缺陷，适合观察成交量报告模式的筛选。
        self.stable = (counts > 500) & (reporting < 0.9 * counts)

        #记录每条策略属于哪个组
        self.allocation_groups = np.asarray([
            raw.groups[np.flatnonzero(raw.allocations == allocation)[0]]
            for allocation in range(len(raw.allocation_names))
        ])

    @cached_property
    #构造 匿名日期 × 策略 × 20日历史收益 的数组，并缓存
    def panel(self) -> np.ndarray:
        """首次需要窗口匹配或行情评分时，构造并缓存策略收益面板。"""
        raw = self.raw
        values = np.full((len(raw.date_names), len(raw.allocation_names), 20), np.nan, np.float32)
        values[raw.dates, raw.allocations] = raw.returns
        return values

    def candidate_domains(
        self, days: np.ndarray, positions: dict[int, np.ndarray], date_ids: np.ndarray,
    ) -> list[np.ndarray]:
        """按市场开市和成交量报告模式，筛选每个匿名日期的候选下标。"""
        raw = self.raw
        #开始日期，是一个字典，里面的值是True，False的bool数组
        open_days = {group: indices >= 0 for group, indices in positions.items()}
        #哪些日期是周一：1970 年 1 月 1 日是 NumPy 的 datetime64 类型内置的计数原点该天为周四，+3就是周日。D表示按天计数。
        monday = (days.astype('datetime64[D]').astype(np.int64) + 3) % 7 == 0
        domains = []
        #开始逐个处理匿名日期
        for date in date_ids:
            #取出row中，属于当前匿名日期的行。
            rows = np.flatnonzero(raw.dates == date)
            #初始化匿名日期候选保留标记，初始全为1.
            compatible = np.ones(len(days), dtype=bool)
            #取出要判断的匿名日期里，出现过的所有交易市场Group
            for group in np.unique(raw.groups[rows]):
                #连续并，连续记录是否在每一个出现市场里都是开市。
                compatible &= open_days[int(group)]
            #周一记录：先查当前各行属于哪条策略，再查这些策略是否符合条件，最后保留符合条件的行号，存进 reporters
            reporters = rows[self.stable[raw.allocations[rows]]]
            #如果有任何策略符合周一记录，且前一天有交易量报告的比例大于50%，则断定当天匿名日期是周一。
            is_monday = len(reporters) > 0 and np.isfinite(raw.volumes[reporters, -1]).mean() > 0.5
            #如果当前匿名日期出现交易的市场日期全部开市，且符合周一判断，就把这些日期统统拿出来，作为该匿名日期的真实候选日期。
            choices = np.flatnonzero(compatible & (monday == is_monday))
            if not len(choices):
                raise ValueError(f'No calendar candidate for {raw.date_names[date]}')
            domains.append(choices)
        return domains
    #这一功能开始寻找匿名日期之间的相对关系，例如“乙比甲晚两个日本交易日”：
    def edges(
        self, device: str, threshold: float, max_gap: int, query_only: bool = False,
    ) -> list[CalendarEdge]:
        """每个市场的全部选中策略一起比较，先找后继，再为未匹配日期找前驱。"""
        edges = []
        #创建一个负责比较收益窗口的对象
        matcher = SuccessorWindowMatcher(device, threshold, max_gap)
        #一个市场组一个市场组处理比较
        for group in np.unique(self.raw.groups):
            #选出group中通过stable筛选的策略
            selected = np.flatnonzero((self.allocation_groups == group) & self.stable)
            #打乱顺序
            selected = np.random.default_rng(42 + int(group)).permutation(selected)
            #检查每个匿名日期是否至少有一条选中策略提供了前一天收益
            present = np.isfinite(self.panel[:, selected, -1]).any(axis=1)
            #既是训练日期，在当前市场又有选中策略的最近一天收益记录的日期
            history = self.training_dates[present[self.training_dates]]
            
            queries = self.query_dates[present[self.query_dates]] if query_only else np.flatnonzero(present)
            references = queries if query_only else history

            #取全部选中策略，并将20日历史顺序反转，变成最近日到最早日。
            #调整成“匿名日期 × 历史日 × 策略”的维度顺序。
            #直接使用原始收益，缺失值填零，并转换为用于匹配的张量。
            windows = torch.as_tensor(
                np.nan_to_num(self.panel[:, selected, ::-1].transpose(0, 2, 1)),
                dtype=torch.float32,
            )
            #windows：全部选中策略的收益数组，形状为“匿名日期 × 20个历史日 × 策略”。
            #queries：需要寻找后继的匿名日期下标。
            #references：允许从中寻找后继的匿名日期下标。
            #group：当前市场组别，用来记录找到的关系属于哪个市场。
            forward, pending = matcher.match(windows, queries, references, int(group))
            edges.extend(forward)
            if len(pending):
                backward, _ = matcher.match(
                    windows.flip(1), pending, references, int(group), reverse=True,
                )
                edges.extend(backward)
        return edges


def session_positions(days: np.ndarray, sessions: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """将自然日期换成各市场的交易日下标，休市日记为 -1。"""
    days = days.astype('datetime64[D]')
    result = {}
    for group, dates in sessions.items():
        dates = dates.astype('datetime64[D]')
        #searchsorted把第一个数组按days的日期插入：例如dates有一个7月20日，但在days里面的交易日只有7月19日和7月23日，下标分别为1，2，于是7月20日就给下标1
        #如果dates还有个日期是7月23日，那也给下标1。
        positions = np.searchsorted(dates, days)
        #能查到下标还不能断定dates里的某个日期是交易日，比如上面的例子，7月20日。还必须要position-1的日期恰好对应days里同下标日期，才算找到。比如上面的7月20日
        valid = (positions < len(dates)) & (dates[positions.clip(max=len(dates) - 1)] == days)
        result[group] = np.where(valid, positions, -1)
    #result是每组带初步交易日判断的字典。
    return result


class CalendarRecovery:
    """先筛日期候选，再用窗口关系缩小候选集合。"""

    def __init__(self, device: str, threshold: float = 0.92, max_gap: int = 5) -> None:
        self.device = device
        self.threshold = threshold
        self.max_gap = max_gap

    @torch.inference_mode()
    def recover(
        self, raw: common.RawDataset, sessions: dict[int, np.ndarray],
    ) -> tuple[dict[str, list[str]], dict[str, object]]:
        """输入比赛特征和交易日历，返回候选映射及关系合并摘要。"""
        #创建CalendarInputs对象。
        inputs = CalendarInputs(raw)
        #提供2000年到2026年的所有自然日，后续在里面挑选候选日期
        calendar = np.arange(np.datetime64('2000-02-01'), np.datetime64('2026-01-01'))
        positions = session_positions(calendar, sessions)
        domains = inputs.candidate_domains(calendar, positions, np.arange(len(raw.date_names)))
        #domain就代表：1.该匿名日期实际出现的市场，在候选日期当天都开市。2.候选日期符合成交量报告规则推断出的“周一或非周一”
        forest = FunctionalCalendarForest(domains, positions)
        #返回一堆edge关系的列表
        edges = inputs.edges(self.device, self.threshold, self.max_gap)
        for edge in sorted(edges, key=lambda item: (item.gap, -item.confidence)):
            forest.add(edge)
        resolved = forest.domains()
        sizes = np.asarray([len(values) for values in resolved])
        mapping = {str(name): calendar[values].astype(str).tolist() for name, values in zip(raw.date_names, resolved)}
        report = {
            'dates': len(resolved), 'uniquely_resolved_dates': int((sizes == 1).sum()),
            'query_dates': len(inputs.query_dates), 'unique_query_dates': int((sizes[inputs.query_dates] == 1).sum()),
            'accepted_edges': forest.accepted, 'rejected_edges': forest.rejected,
            'reference_pool': 'labeled_training_dates_X_only', 'uses_targets': False,
        }
        return mapping, report

