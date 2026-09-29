"""locomo 配置：复用 oak 的 Config/LLMClient，但产物目录与模型路由独立。

模型路由（双档）：
- strong = glm-5.3（智谱直连，本体起草 / 终答 / 判题）
- fast   = deepseek-v1-flash（commandcode 网关，事实抽取 / ReAct 步骤 / 归并审计）
  网关不可用时（probe_models 探测落盘）回退 glm-5.3-flash 同站。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from oak.config import Config

LOCOMO_ROOT = Path(__file__).resolve().parent          # datasets/locomo/pipeline
LOCOMO_TASK_DIR = LOCOMO_ROOT.parent                        # datasets/locomo
PROJECT_ROOT = LOCOMO_ROOT.parents[2]                     # 仓库根

ZHIPU_BASE = "https://open.bigmodel.cn/api/coding/paas/v4"

# namespace 统一 lc* 前缀：避开 LLMClient 对 r*/inf_q* 前缀的预算钩子
NS_PROBE = "lc_probe"
NS_SCHEMA = "lc_schema"


@dataclass
class LocomoConfig:
    cfg: Config                        # oak Config（含 LLMClient 所需一切）
    dataset_path: Path
    anchor_id: str = "conv-26"
    react_max_steps: int = 10
    audit_extract: bool = True         # 建图后二道完整性审计
    fast_available: bool = True        # 探测结果：fast 网关是否可用

    # ---- 派生路径 ----
    @property
    def runs_dir(self) -> Path:
        return LOCOMO_TASK_DIR / "runs"

    def conv_dir(self, sample_id: str) -> Path:
        return self.runs_dir / sample_id

    def probe_path(self) -> Path:
        return self.runs_dir / "model_probe.json"


def _apply_probe_fallback(cfg: Config, probe_path: Path) -> bool:
    """读探测结果：fast 网关不可用则回退 glm-5.3-flash 同站。返回 fast 是否可用。"""
    if not probe_path.exists():
        return True                     # 未探测过：按可用处理（探测脚本会先跑）
    try:
        r = json.loads(probe_path.read_text())
    except Exception:
        return True
    if r.get("fast_ok"):
        return True
    cfg.fast_base_url = cfg.api_base_url
    cfg.fast_api_key = cfg.api_key
    cfg.model_fast = "glm-5.3-flash"
    return False


def load_locomo_config() -> LocomoConfig:
    load_dotenv(PROJECT_ROOT / ".env")
    cfg = Config()
    cfg.api_base_url = ZHIPU_BASE
    cfg.api_key = os.environ.get("ZHIPU_API_KEY", "")
    if not cfg.api_key:
        raise RuntimeError("ZHIPU_API_KEY 未设置（.env 或环境变量）")

    # fast 档：locomo 独立变量 LOCOMO_FAST_*（不与 TravelPlanner 工作流共用 FAST_*，
    # 避免共享 .env 的写冲突）；未配置时回退 FAST_*，再回退智谱 glm-5.3-flash
    cfg.fast_base_url = (os.environ.get("LOCOMO_FAST_API_BASE")
                         or os.environ.get("FAST_API_BASE", "") or ZHIPU_BASE)
    cfg.model_fast = (os.environ.get("LOCOMO_FAST_MODEL")
                      or os.environ.get("FAST_MODEL", "")
                      or "deepseek/deepseek-v4-flash-fast")
    if cfg.fast_base_url != cfg.api_base_url:
        cfg.fast_api_key = (os.environ.get("LOCOMO_FAST_API_KEY")
                            or os.environ.get("FAST_API_KEY", ""))
        if not cfg.fast_api_key:
            raise RuntimeError("fast 档指向外部网关但 LOCOMO_FAST_API_KEY/FAST_API_KEY 未设置")
    else:
        cfg.fast_api_key = cfg.api_key

    # locomo 产物全部隔离在 locomo/runs 下（缓存/台账/重试日志随 work_dir 派生）
    cfg.work_dir = LOCOMO_TASK_DIR / "runs"

    dataset_path = Path(os.environ.get(
        "LOCOMO_DATA",
        str(LOCOMO_TASK_DIR / "data" / "locomo10_zh.json")))
    lc = LocomoConfig(cfg=cfg, dataset_path=dataset_path)
    lc.fast_available = _apply_probe_fallback(cfg, lc.probe_path())

    for d in (cfg.work_dir, cfg.cache_dir, cfg.ledger_path.parent,
              cfg.work_dir / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    return lc


def ns(conv: str, kind: str) -> str:
    """namespace 约定：lc{NN缩写}_{kind}，如 lc26_build / lc26_q3 / lc26_judge。"""
    num = conv.replace("conv-", "")
    return f"lc{num}_{kind}"
