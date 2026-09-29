"""OaK 框架核心配置：模型路由、并发、限额、缓存路径（任务无关）。

各任务（datasets/travelplanner、datasets/locomo）在此 Config 之上叠自己的
任务参数与产物目录——任务专属字段不在框架层。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# role -> tier；tier 再映射到具体模型（两任务共用；任务可只使用自己的角色前缀）
MODEL_ROLES: dict[str, str] = {
    "schema": "strong",      # P1 需求分析 / P2 模式草拟
    "func_gen": "strong",    # P4 函数生成（含能力规划）
    "judicator": "strong",   # P6 评判器
    "kg": "fast",            # P3 建图抽取
    "react": "fast",         # P5 ReAct 执行
    "slots": "fast",         # extract_runtime_slots 槽位提取
    "plan_repair": "fast",   # 计划格式修复 / salvage
    # ---- locomo 任务（中文 LoCoMo 本体问答）----
    "locomo_schema": "strong",   # P1/P2 本体起草
    "locomo_answer": "strong",   # 终答合成 + 证据自检
    "locomo_judge": "strong",    # 严格判分
    "locomo_extract": "fast",    # 原子事实抽取
    "locomo_util": "fast",       # 实体归并 / 完整性审计 / 格式修复
    "locomo_steps": "fast",      # ReAct 步骤
    # ---- mem0 基线（只-ADD 记忆，中文）----
    "mem0_extract": "fast",      # 加法事实提取
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

    # 框架级路径（任务通常覆盖 work_dir 以隔离产物）
    work_dir: Path = PROJECT_ROOT / "runs"

    # 成本硬顶（按 namespace 计；locomo 用 lc* 前缀天然绕开）
    limits: dict = field(default_factory=lambda: {
        "build_round_calls": 600,      # 单轮 LLM 调用上限
        "inference_calls_per_q": 60,   # 单测试题上限
    })

    def model_for(self, role: str) -> str:
        tier = MODEL_ROLES.get(role)
        if tier is None:
            raise ValueError(f"unknown LLM role: {role}")
        return self.model_strong if tier == "strong" else self.model_fast

    # ---- 框架派生路径 ----
    @property
    def cache_dir(self) -> Path:
        return self.work_dir / "cache" / "llm"

    @property
    def ledger_path(self) -> Path:
        return self.work_dir / "cost_ledger.jsonl"
