"""全局配置：模型路由、并发、规模、限额、路径。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# role -> tier；tier 再映射到具体模型
MODEL_ROLES: dict[str, str] = {
    "schema": "strong",      # P1 需求分析 / P2 模式草拟
    "func_gen": "strong",    # P4 函数生成（含能力规划）
    "judicator": "strong",   # P6 评判器
    "kg": "fast",            # P3 建图抽取
    "react": "fast",         # P5 ReAct 执行
    "slots": "fast",         # extract_runtime_slots 槽位提取
    "plan_repair": "fast",   # 计划格式修复 / salvage
}


@dataclass
class Config:
    # API（strong 档走智谱；fast 档可切第三方网关省额度）
    api_base_url: str = "https://open.bigmodel.cn/api/coding/paas/v4"
    api_key: str = ""
    model_strong: str = "glm-5.3"
    # fast 档默认同站；.env 可覆盖 FAST_API_BASE/FAST_MODEL/FAST_API_KEY
    fast_base_url: str = ""
    fast_api_key: str = ""
    model_fast: str = "glm-5.3-flash"

    # 并发与重试（账户有限流：8 并发触发 429，降到 4）
    max_concurrency: int = 4
    max_retries: int = 5

    # 构建循环规模（对齐论文：每轮抽 train 的 20%，最多 5 轮）
    rounds: int = 5
    train_per_round: int = 9
    test_size: int = 50
    seed: int = 42

    # 各环节限额
    react_max_steps: int = 20
    schema_attempts: int = 4
    func_gen_attempts: int = 3
    plan_repair_attempts: int = 2

    # 路径
    work_dir: Path = PROJECT_ROOT / "runs"
    tp_root: Path = PROJECT_ROOT / "third_party" / "TravelPlanner"

    # 附加语料：城际地面交通（官方环境数据，非评测答案）
    include_distance_matrix_corpus: bool = True

    # 成本硬顶（按 namespace 计）
    limits: dict = field(default_factory=lambda: {
        "build_round_calls": 600,      # 单轮 LLM 调用上限
        "inference_calls_per_q": 60,   # 单测试题上限
    })

    # 冻结策略：final=第 5 轮产物；best_round=按 patched Final 选优
    freeze_policy: str = "final"

    def model_for(self, role: str) -> str:
        tier = MODEL_ROLES.get(role)
        if tier is None:
            raise ValueError(f"unknown LLM role: {role}")
        return self.model_strong if tier == "strong" else self.model_fast

    # ---- 派生路径 ----
    @property
    def cache_dir(self) -> Path:
        return self.work_dir / "cache" / "llm"

    @property
    def ledger_path(self) -> Path:
        return self.work_dir / "cost_ledger.jsonl"

    @property
    def data_dir(self) -> Path:
        return self.work_dir / "data"

    @property
    def build_dir(self) -> Path:
        return self.work_dir / "build"

    @property
    def final_dir(self) -> Path:
        return self.work_dir / "final"

    @property
    def inference_dir(self) -> Path:
        return self.work_dir / "inference"


def load_config() -> Config:
    load_dotenv(PROJECT_ROOT / ".env")
    cfg = Config()
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
