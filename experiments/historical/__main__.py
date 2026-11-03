"""在项目根目录运行 python -m experiments.historical。"""

from __future__ import annotations

import argparse

from experiments.historical.pipeline import ReconstructionProject


def main() -> None:
    """准备特征、拟合分类器、生成预测，或读取现有模型验证。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "fit", "predict", "validate"])
    parser.add_argument("--mode", choices=["validation", "official"], default="official")
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--refit-prior", action="store_true")
    args = parser.parse_args()
    project = ReconstructionProject()
    if args.command == "prepare":
        project.prepare(args.mode, args.device, args.refit_prior)
    elif args.command == "fit":
        project.fit(args.mode)
    else:
        project.predict("validation" if args.command == "validate" else args.mode)


if __name__ == "__main__":
    main()
