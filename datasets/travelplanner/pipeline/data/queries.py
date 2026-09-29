"""query 加载（HF 落盘后离线复用）与抽样。

一个样本 = 一道查询 + 它的参考语料（reference_information），对应论文的 (q, C_q)。
"""
from __future__ import annotations

import ast
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from ..config_task import Config

SPLIT_SIZES = {"train": 45, "validation": 180, "test": 1000}

CS_KEY_COUNT = 8     # 官方 CS 恒 8 项/题（未交付也计失败）


def applicable_hc_keys(level: str, local_constraint: dict | None) -> list[str]:
    """镜像官方 eval.py:98-149 的 HC 适用项规则（唯一真相，adapter 与 Query 共用）。

    - valid_cost 每题恒适用；
    - medium 不计 valid_transportation（官方只对 hard 计 transportation）；
    - 其余按 local_constraint 中该约束存在（非 None/空）计。
    """
    lc = local_constraint or {}
    keys = ["valid_cost"]
    if lc.get("house rule"):
        keys.append("valid_room_rule")
    if lc.get("cuisine"):
        keys.append("valid_cuisine")
    if lc.get("room type"):
        keys.append("valid_room_type")
    if lc.get("transportation") and level == "hard":
        keys.append("valid_transportation")
    return keys


@dataclass
class Query:
    idx: int                    # split 内的行下标（官方评测按下标配对）
    org: str
    dest: str
    days: int
    date: list[str]
    people_number: int
    local_constraint: dict
    budget: int
    query: str
    level: str
    visiting_city_number: int = 1
    reference_information: dict = field(default_factory=dict)  # {"Description": str, "Content": str} 列表转 dict

    @property
    def hc_denominator(self) -> int:
        """该题硬约束适用项数（官方口径，见 applicable_hc_keys）。"""
        return len(applicable_hc_keys(self.level, self.local_constraint))


def _parse_ref_info(raw):
    """reference_information 经 datasets 读出是字符串化的 list[dict]。"""
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, str):
        try:
            v = ast.literal_eval(raw)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        except Exception:
            return []
    return []


def _parse_local_constraint(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            v = ast.literal_eval(raw)
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}
    return {}


def parse_dates(raw) -> list[str]:
    """统一解析 date 字段。

    兼容三种历史形态：
    - 正常 list[str]（如 ["2022-03-01"]）；
    - 字符串化列表（如 "['2022-03-01']"，datasets 落盘常见）；
    - 曾被旧代码 `list(str)` 拆成单字符的 list（如 ['[', "'", '2', ...]）—— 先 join 再解析。
    单个日期字符串转为一项 list；解析失败返回 [] 并保留显式标记（不静默造日期）。
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            v = ast.literal_eval(raw)
            if isinstance(v, list):
                raw = v
            else:
                raw = [raw]
        except Exception:
            raw = [raw]
    if isinstance(raw, list):
        if raw and all(isinstance(x, str) and len(x) <= 2 for x in raw):
            # 字符数组（旧 bug 遗留）：join 后重新走字符串分支
            return parse_dates("".join(raw))
        out = [str(x).strip() for x in raw if str(x).strip()]
        return out
    return []


def load_queries(split: str, cfg: Config | None = None, force_download: bool = False) -> list[Query]:
    """从 HF 拉取并落盘 runs/data/{split}.queries.jsonl，之后离线复用。"""
    from ..config_task import load_config
    cfg = cfg or load_config()
    out: Path = cfg.data_dir / f"{split}.queries.jsonl"
    if out.exists() and not force_download:
        return _from_local(out)

    # 防遮蔽：仓库顶层 datasets/ 包与本 HF 库同名——从 site-packages 精确装载
    import importlib.util
    import sys
    spec = importlib.util.find_spec("datasets")
    ours = str(Path(__file__).resolve().parents[3] / "datasets" / "__init__.py")
    if spec and str(spec.origin) == ours:
        for p in ("", ".", str(Path.cwd())):
            while p in sys.path:
                sys.path.remove(p)
        try:
            from datasets import load_dataset  # noqa: F401  现在解析到 site-packages
        finally:
            for p in ("", "."):
                sys.path.insert(0, p)
    else:
        from datasets import load_dataset  # noqa: F401
    # 官方用法：config 名即 split 名，返回 DatasetDict 后取同名 split
    ds = load_dataset("osunlp/TravelPlanner", split)[split]
    queries: list[Query] = []
    for i, row in enumerate(ds):
        ref = {}
        for item in _parse_ref_info(row.get("reference_information")):
            desc = item.get("Description", "")
            if desc:
                ref[desc] = item.get("Content", "")
        queries.append(Query(
            idx=i,
            org=row.get("org", ""),
            dest=row.get("dest", ""),
            days=int(row.get("days") or 0),
            date=parse_dates(row.get("date")),
            people_number=int(row.get("people_number") or 0),
            local_constraint=_parse_local_constraint(row.get("local_constraint")),
            budget=int(row.get("budget") or 0),
            query=row.get("query", ""),
            level=row.get("level", ""),
            visiting_city_number=int(row.get("visiting_city_number") or 1),
            reference_information=ref,
        ))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for q in queries:
            f.write(json.dumps({
                "idx": q.idx, "org": q.org, "dest": q.dest, "days": q.days,
                "date": q.date, "people_number": q.people_number,
                "local_constraint": q.local_constraint, "budget": q.budget,
                "query": q.query, "level": q.level,
                "visiting_city_number": q.visiting_city_number,
                "reference_information": q.reference_information,
            }, ensure_ascii=False) + "\n")
    return queries


def _from_local(path: Path) -> list[Query]:
    queries = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        queries.append(Query(
            idx=r["idx"], org=r["org"], dest=r["dest"], days=r["days"],
            date=parse_dates(r.get("date")), people_number=r["people_number"],
            local_constraint=r["local_constraint"], budget=r["budget"],
            query=r["query"], level=r["level"],
            visiting_city_number=r.get("visiting_city_number", 1),
            reference_information=r.get("reference_information", {}),
        ))
    return queries


def partition_train(per_round: int = 9, rounds: int = 5, seed: int = 42,
                    cfg: Config | None = None) -> list[list[int]]:
    """train 45 题切成 rounds 组、每组 per_round 题（互斥，尽量按 level 分层）。"""
    from ..config_task import load_config
    cfg = cfg or load_config()
    all_q = load_queries("train", cfg)
    n = len(all_q)
    total = per_round * rounds
    rng = random.Random(seed)
    # 分层：按 level 分桶洗牌后轮流取
    by_level: dict[str, list[int]] = {}
    for q in all_q:
        by_level.setdefault(q.level, []).append(q.idx)
    for ids in by_level.values():
        rng.shuffle(ids)
    pool: list[int] = []
    buckets = list(by_level.values())
    while any(buckets):
        for b in buckets:
            if b:
                pool.append(b.pop())
    pool = pool[:total] if total <= n else pool + rng.sample(
        [i for i in range(n) if i not in set(pool)], total - n)
    groups = [pool[i * per_round:(i + 1) * per_round] for i in range(rounds)]
    out = cfg.data_dir / "sample_indices.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing["rounds"] = groups
    out.write_text(json.dumps(existing, ensure_ascii=False, indent=1))
    return groups


def stratified_test_subset(n: int = 50, seed: int = 42, cfg: Config | None = None) -> list[int]:
    """validation 180 题分层抽 n 题做测试。"""
    from ..config_task import load_config
    cfg = cfg or load_config()
    all_q = load_queries("validation", cfg)
    rng = random.Random(seed)
    by_level: dict[str, list[int]] = {}
    for q in all_q:
        by_level.setdefault(q.level, []).append(q.idx)
    for ids in by_level.values():
        rng.shuffle(ids)
    pool: list[int] = []
    buckets = list(by_level.values())
    while any(buckets):
        for b in buckets:
            if b:
                pool.append(b.pop())
    chosen = sorted(pool[:n])
    out = cfg.data_dir / "sample_indices.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing["test"] = chosen
    out.write_text(json.dumps(existing, ensure_ascii=False, indent=1))
    return chosen


def query_view(q: Query) -> str:
    """ReAct / 需求分析用的紧凑查询视图。"""
    lc = "; ".join(f"{k}={v}" for k, v in q.local_constraint.items() if v not in (None, "", [], {}))
    return (
        f"Natural language request: {q.query}\n"
        f"- origin: {q.org}\n- destination: {q.dest}\n- days: {q.days}\n"
        f"- dates: {', '.join(q.date)}\n- people: {q.people_number}\n"
        f"- budget: {q.budget}\n"
        f"- total distinct cities to visit (including destination): {q.visiting_city_number}\n"
        + (f"- local constraints: {lc}\n" if lc else "")
    )
