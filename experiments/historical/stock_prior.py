"""从训练标签学习股票先验，并按日期两折生成特征。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from experiments.historical.data import AllocationData

if TYPE_CHECKING:
    from torch import Tensor


class LearnedStockPrior:
    """学习股票重要性、策略风格和历史衰减，以20日岭回归估计收益。"""

    def __init__(self, stocks: int, allocations: int, device: str, seed: int) -> None:
        import torch

        self.torch = torch
        self.device = device
        self.lags = torch.arange(19, -1, -1, device=device)
        generator = torch.Generator().manual_seed(seed)

        def parameter(shape, scale=0.0, initial=0.0):
            values = torch.randn(shape, generator=generator) * scale + initial
            return torch.nn.Parameter(values.to(device))

        self.parameters = {
            "stock": parameter((stocks,)),
            "style": parameter((allocations, 4), 0.05),
            "style_stock": parameter((stocks, 4), 0.05),
            "shared": parameter((stocks, 8), 0.03),
            "ridge": parameter((allocations,), initial=-0.087),
            "decay": parameter((allocations,), initial=-2.0),
            "bias": parameter((allocations,)),
            "gain": parameter((), initial=np.log(2.0)),
        }

    def forward(
        self, public: Tensor, positions: Tensor, histories: Tensor,
        allocations: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """统一返回标准化收益、上涨对数几率和反演不确定度。"""
        torch = self.torch
        parameters = self.parameters
        past = public[positions[:, None] - self.lags]
        future = public[positions + 1]
        logits = parameters["stock"] + parameters["style"][allocations] @ parameters["style_stock"].T
        weights = logits.clamp(-4, 4).softmax(dim=-1)
        active = (past.std(dim=1) > 1e-6).to(past.dtype)
        weights = weights * active
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6) * past.shape[-1]
        root = weights.clamp_min(1e-8).sqrt()
        design = torch.cat((past * root[:, None], past @ parameters["shared"]), dim=-1) / np.sqrt(past.shape[-1])
        target = torch.cat((future * root, future @ parameters["shared"]), dim=-1) / np.sqrt(past.shape[-1])
        gram = design @ design.transpose(1, 2)
        cross = (design * target[:, None]).sum(dim=-1)
        ridge = 0.05 + torch.nn.functional.softplus(parameters["ridge"][allocations])
        decay = 0.1 * parameters["decay"][allocations].sigmoid()
        noise = ridge[:, None] * torch.exp(decay[:, None] * self.lags)
        noise = noise + (~histories.isfinite()).to(noise.dtype) * 10000
        matrix = gram + torch.diag_embed(noise)
        observed = histories.nan_to_num().unsqueeze(-1)
        right = torch.cat((observed, cross.unsqueeze(-1)), dim=-1)
        # 小型线性方程在CPU上直接求解；设备拷贝保留梯度，兼容MPS训练。
        coefficients = torch.linalg.solve(matrix.cpu(), right.cpu()).to(self.device)
        prediction = (cross * coefficients[:, :, 0]).sum(dim=-1)
        gain = parameters["gain"].exp().clamp(0.5, 6)
        logit = gain * prediction + parameters["bias"][allocations]
        variance = (target.square().sum(dim=-1) - (cross * coefficients[:, :, 1]).sum(dim=-1)).clamp_min(0)
        return prediction, logit, variance.sqrt()

    def fit(
        self, public: Tensor, positions: Tensor, histories: Tensor,
        allocations: Tensor, targets: Tensor, training: np.ndarray, steps: int, seed: int,
    ) -> None:
        """用固定步数拟合股票先验。"""
        torch = self.torch
        parameters = list(self.parameters.values())
        regularized = [self.parameters[name] for name in ("stock", "style", "style_stock", "shared", "bias")]
        optimizer = torch.optim.Adam(parameters, lr=0.015)
        rng = np.random.default_rng(seed)
        pool = np.flatnonzero(training)
        for step in range(steps):
            batch = torch.as_tensor(rng.choice(pool, min(384, len(pool)), replace=True), device=self.device)
            prediction, logit, _ = self.forward(public, positions[batch], histories[batch], allocations[batch])
            truth = targets[batch]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, (truth > 0).float())
            loss = loss + 0.15 * torch.nn.functional.smooth_l1_loss(prediction, truth.clamp(-5, 5))
            loss = loss + 0.003 * sum(value.square().mean() for value in regularized)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.param_groups[0]["lr"] = 0.015 * (0.2 + 0.8 * (1 - step / steps))
            optimizer.step()

    def predict(
        self, public: Tensor, positions: Tensor, histories: Tensor,
        allocations: Tensor, rows: np.ndarray,
    ) -> np.ndarray:
        """分批推理，保留小型工作内存。"""
        result = np.empty((len(rows), 3), dtype=np.float32)
        with self.torch.inference_mode():
            for start in range(0, len(rows), 512):
                batch = self.torch.as_tensor(rows[start:start + 512], device=self.device)
                prediction, logit, uncertainty = self.forward(public, positions[batch], histories[batch], allocations[batch])
                values = self.torch.stack((prediction, logit.sigmoid(), uncertainty), dim=-1)
                result[start:start + 512] = values.cpu().numpy()
        return result


class StockPriorFeatures:
    """按日期分成两折，生成收益估计、上涨概率和不确定度三列。"""

    def __init__(self, checkpoint: Path, device: str = "mps") -> None:
        self.checkpoint = checkpoint
        self.device = device

    def build(self, data: AllocationData, row_ids: np.ndarray, validation: bool, refit: bool = False) -> np.ndarray:
        """训练行使用未见过自身日期标签的一折，查询行取两折平均。"""
        import torch

        torch.set_num_threads(2)
        raw = data.raw
        dated, original = data.dated()
        allowed = data.allowed_labels(validation)
        wanted = np.isin(raw.row_ids, row_ids)
        features = np.full((len(raw.row_ids), 3), np.nan, dtype=np.float32)
        cached = self.checkpoint.exists() and not refit
        checkpoints = torch.load(self.checkpoint, map_location="cpu", weights_only=True) if cached else {}
        for group in range(1, 5):
            public, _, positions, row_map = data.panel(dated, group)
            valid = row_map >= 0
            rows = original[row_map[valid]]
            day_positions = np.broadcast_to(positions[:, None], row_map.shape)[valid]
            allocation_ids, local_ids = np.unique(raw.allocations[rows], return_inverse=True)
            scale = data.scales[raw.allocations[rows]]
            tensors = (
                torch.as_tensor(public, dtype=torch.float32, device=self.device),
                torch.as_tensor(day_positions, dtype=torch.int64, device=self.device),
                torch.as_tensor(raw.returns[rows] / scale[:, None], dtype=torch.float32, device=self.device),
                torch.as_tensor(local_ids, dtype=torch.int64, device=self.device),
            )
            if not cached:
                targets = torch.as_tensor(raw.targets[rows] / scale, dtype=torch.float32, device=self.device)
            requested = np.flatnonzero(wanted[rows])
            labeled = allowed[rows[requested]]
            date_folds = raw.dates[rows] % 2
            prediction = np.zeros((len(requested), 3), dtype=np.float32)
            for fold in (0, 1):
                model = LearnedStockPrior(public.shape[1], len(allocation_ids), self.device, 42 + group * 10 + fold)
                key = f"{group}_{fold}"
                if cached:
                    model.parameters = {
                        name: torch.nn.Parameter(value.to(self.device)) for name, value in checkpoints[key].items()
                    }
                else:
                    training = allowed[rows] & (date_folds != fold)
                    model.fit(*tensors, targets, training, data.config["prior_steps"], 42 + fold)
                    checkpoints[key] = {name: value.detach().cpu() for name, value in model.parameters.items()}
                needed = ~labeled | (date_folds[requested] == fold)
                part = model.predict(*tensors, requested[needed])
                part[~labeled[needed]] *= 0.5
                prediction[needed] += part
            features[rows[requested]] = prediction
            print(f"GROUP {group} 学习先验完成。", flush=True)
        if not cached:
            self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoints, self.checkpoint)
        order = np.argsort(raw.row_ids)
        return features[order[np.searchsorted(raw.row_ids[order], row_ids)]]

