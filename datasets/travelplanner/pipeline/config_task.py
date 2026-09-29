"""TravelPlanner 任务配置：框架 Config 之上叠任务专属参数与产物路径。

自 oak/config.py 迁出（框架/任务分割）：构建循环规模、各环节限额、tp_root、
语料开关、冻结策略、TP 专属 work_dir 派生路径都在这里。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from oak.config import Config, MODEL_ROLES  # noqa: F401  （MODEL_ROLES 重导出兼容旧引用）

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TPConfig(Config):
    # 构建循环规模（对齐论文：每轮抽 train 的 20%，最多 5 轮）
    rounds: int = 5
    train_per_round: int = 9
    test_size: int = 50
    seed: int = 42

    # 各环节限额
    react_max_steps: int = 26
    schema_attempts: int = 4
    func_gen_attempts: int = 3
    plan_repair_attempts: int = 2

    # 任务路径
    tp_root: Path = PROJECT_ROOT / "third_party" / "TravelPlanner"

    # 附加语料：城际地面交通（官方环境数据，非评测答案）
    include_distance_matrix_corpus: bool = True

    # 冻结策略：final=第 5 轮产物；best_round=按 patched Final 选优
    freeze_policy: str = "final"

    # ---- TP 专属派生路径（覆盖基类默认）----
    def __post_init__(self) -> None:
        self.work_dir = PROJECT_ROOT / "datasets" / "travelplanner" / "runs"

    @property
    def data_dir(self) -> Path:
        # queries 落盘与固定抽样索引（已自 runs/data 迁至任务 data/ 目录）
        return PROJECT_ROOT / "datasets" / "travelplanner" / "data"

    @property
    def build_dir(self) -> Path:
        return self.work_dir / "build"

    @property
    def final_dir(self) -> Path:
        return self.work_dir / "final"

    @property
    def inference_dir(self) -> Path:
        return self.work_dir / "inference"


def load_config() -> TPConfig:
    cfg = TPConfig()
    load_dotenv(PROJECT_ROOT / ".env")
    cfg.api_key = os.environ.get("ZHIPU_API_KEY", "")
    if not cfg.api_key:
        raise RuntimeError("ZHIPU_API_KEY 未设置（.env 或环境变量）")
    # fast 档可选切换到第三方网关（省额度）——切外部网关时必须提供该网关自己的 key
    cfg.fast_base_url = os.environ.get("FAST_API_BASE", "") or cfg.api_base_url
    cfg.model_fast = os.environ.get("FAST_MODEL", "") or cfg.model_fast
    if cfg.fast_base_url != cfg.api_base_url:
        cfg.fast_api_key = os.environ.get("FAST_API_KEY", "")
        if not cfg.fast_api_key:
            raise RuntimeError(
                "FAST_API_BASE 指向外部网关时必须在 .env 设置该网关的 FAST_API_KEY"
                "（出于安全不回落使用智谱 key）")
    else:
        cfg.fast_api_key = cfg.api_key
    # JDK（HermiT 依赖）：.env 提供 JAVA_HOME 时注入 PATH，所有子进程继承
    jh = os.environ.get("JAVA_HOME", "")
    if jh and Path(jh).exists():
        os.environ["PATH"] = f"{jh}/bin:" + os.environ.get("PATH", "")
    for d in (cfg.work_dir, cfg.cache_dir, cfg.data_dir, cfg.build_dir,
              cfg.final_dir, cfg.inference_dir):
        d.mkdir(parents=True, exist_ok=True)
    return cfg
