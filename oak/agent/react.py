"""步骤③-3b ReAct 执行（≤20 步，三层轨迹记录）。"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config
from ..llm.client import LLMClient
from ..data.queries import Query, query_view
from ..schema.model import Schema
from ..funcs.catalog import FunctionCatalog
from ..prompts import react as P5

log = logging.getLogger("oak.react")

ACTION_RE = re.compile(r"Action:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((\{.*\})\)", re.S)
ACTION_KW_RE = re.compile(r"Action:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)", re.S)
FINAL_RE = re.compile(r"Final Plan:\s*(\[.*\])", re.S)


def _parse_action(out: str):
    """解析 Action 行：优先 JSON 参数；失败降级 kwargs 风格。返回 (fn, args_dict) 或 None。"""
    m = ACTION_RE.search(out)
    if m:
        try:
            args = json.loads(m.group(2))
            if isinstance(args, dict):
                return m.group(1), args
        except Exception:
            pass
    m = ACTION_KW_RE.search(out)
    if not m:
        return None
    fn_name, raw = m.group(1), m.group(2).strip()
    if not raw or raw == "{}":
        return fn_name, {}
    # kwargs 降级：拆 key=value 对（值可能带引号）
    import ast
    args: dict = {}
    ok = True
    for part in _split_kwargs(raw):
        if "=" not in part:
            ok = False
            break
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        try:
            args[k] = ast.literal_eval(v)
        except Exception:
            if v.lower() in ("true", "false"):
                args[k] = v.lower() == "true"
            elif v.replace(".", "").isdigit():
                args[k] = float(v) if "." in v else int(v)
            else:
                args[k] = v.strip('"\'')
    return (fn_name, args) if ok else None


def _split_kwargs(raw: str) -> list[str]:
    """按逗号拆 kwargs，跳过引号内的逗号。"""
    parts, buf, quote = [], [], None
    for ch in raw:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch == ",":
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return [p for p in parts if p.strip()]


@dataclass
class Step:
    n: int
    thought: str = ""
    action: str | None = None
    observation: str = ""


@dataclass
class ReactResult:
    steps: list[Step] = field(default_factory=list)
    raw_plan: str | None = None
    delivered: bool = False
    usage: dict = field(default_factory=dict)


class TrajectoryRecorder:
    def __init__(self, query_idx: int):
        self.query_idx = query_idx
        self.step_layer: list[dict] = []
        self.call_layer: list[dict] = []
        self.plan_layer: dict = {}

    def add_step(self, s: Step) -> None:
        self.step_layer.append({"step": s.n, "thought": s.thought[:500],
                                "action": s.action, "observation": s.observation[:2000]})

    def add_call(self, step: int, fn: str, args: dict, result) -> None:
        n_rows = len(result) if isinstance(result, list) else (1 if result is not None else 0)
        preview = ""
        try:
            preview = json.dumps(result, ensure_ascii=False, default=str)[:800]
        except Exception:
            preview = str(result)[:800]
        self.call_layer.append({"step": step, "fn": fn, "args":
                                {k: v for k, v in list(args.items())[:8]},
                                "n_rows": n_rows, "error": None,
                                "result_preview": preview})

    def add_call_error(self, step: int, fn: str, args: dict, err: str) -> None:
        self.call_layer.append({"step": step, "fn": fn, "args": {}, "error": err[:500]})

    def set_plan(self, plan: list | None, format_errors: list, repairs: int) -> None:
        self.plan_layer = {"delivered": bool(plan), "n_days": len(plan or []),
                           "format_errors": format_errors, "repairs": repairs}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "query_idx": self.query_idx,
            "layers": {"step_layer": self.step_layer,
                       "call_layer": self.call_layer,
                       "plan_layer": self.plan_layer},
            "usage": self.usage if hasattr(self, "usage") else {},
        }, ensure_ascii=False, indent=1))


def _format_observation(result: Any, limit: int = 12000) -> str:
    try:
        s = json.dumps(result, ensure_ascii=False, default=str)
    except Exception:
        s = str(result)
    if isinstance(result, list):
        s = f"[{len(result)} rows] " + s
    if len(s) > limit:
        s = s[:limit] + (f"...(truncated, total {len(s)} chars; use project_properties "
                         f"with tighter filters to narrow)")
    return s


def _envelope(ret: Any) -> dict:
    """函数返回 → 统一观察信封：{status, results, suggestions?, warning?/value?}。

    - list：非空行为候选（过滤全空 dict / 空 name sentinel）；
      带 match_note != 'exact' 或字段级 route/date 不符的行 → suggestions（不可入选）；
    - dict 带 results：解包递归；
    - dict 带 warning/message：status=empty；
    - 标量：value。
    warning/空结果/sentinel 行不再是"一条候选"（旧 recorder 把 warning dict 记 1 行）。
    """
    if isinstance(ret, list):
        rows, sugg = [], []
        for r in ret:
            if not isinstance(r, dict):
                rows.append(r)
                continue
            if not any(v not in (None, "", [], {}) for v in r.values()):
                continue                     # 空占位行
            note = str(r.get("match_note") or "")
            if note and note != "exact":
                sugg.append(r)
            else:
                rows.append(r)
        if rows:
            out = {"status": "ok", "results": rows}
            if sugg:
                out["suggestions"] = sugg[:5]
            return out
        return {"status": "empty", "results": [],
                "warning": "no matching rows (suggestions, if any, are discovery-only)",
                **({"suggestions": sugg[:5]} if sugg else {})}
    if isinstance(ret, dict):
        if isinstance(ret.get("results"), list):
            inner = _envelope(ret["results"])
            merged = dict(inner)
            if "warning" in ret and "warning" not in merged:
                merged["warning"] = str(ret["warning"])[:200]
            return merged
        if any(k in ret for k in ("warning", "Warning", "message", "note")):
            return {"status": "empty", "results": [],
                    "warning": str(ret)[:300]}
        return {"status": "ok", "results": [ret]}
    return {"status": "ok", "value": ret}


async def run_react(client: LLMClient, cfg: Config, q: Query,
                    catalog: FunctionCatalog, schema: Schema,
                    g, module, namespace: str) -> ReactResult:
    """module: catalog.module(g) 的产物（含函数与 G）。"""
    from ..operators import library as ops
    ops.set_graph(g)                     # ContextVar：本题协程上下文内隔离

    recorder = TrajectoryRecorder(q.idx)
    cheat = "\n".join(f"- {e.name}: {', '.join(e.primary_key)}"
                      for e in schema.entities)
    # 权威 covered 城市块（运行时 enrich；模型不再从城市名猜州）
    from ..kg.graph import covered_cities as _cc
    cov_all = _cc(g)
    cov = [c for c in cov_all if c.get("state") == q.dest] or cov_all
    cov_json = json.dumps(cov, ensure_ascii=False)
    system_prompt = P5.build(catalog.render_catalog(), cheat, query_view(q),
                             q.days, q.org, q.dest, q.budget,
                             q.visiting_city_number, q.people_number,
                             covered_cities_json=cov_json)
    messages: list[dict] = [{"role": "system", "content": P5.SYSTEM},
                            {"role": "user", "content": system_prompt}]
    result = ReactResult()
    t0 = time.time()
    prompt_tokens = completion_tokens = 0

    for n in range(1, cfg.react_max_steps + 1):
        left = cfg.react_max_steps - n
        if left == 2:
            messages.append({"role": "user", "content": P5.URGE.format(left=left)})
        if n == cfg.react_max_steps:
            messages.append({"role": "user", "content": P5.FORCE_FINALIZE})

        res = await client.chat(role="react", messages=messages,
                                temperature=0.2, namespace=namespace)
        prompt_tokens += res.usage.get("prompt_tokens", 0)
        completion_tokens += res.usage.get("completion_tokens", 0)
        out = res.content.strip()
        messages.append({"role": "assistant", "content": out})

        # Final Plan?
        m = FINAL_RE.search(out)
        if m:
            step = Step(n, thought=_thought_of(out))
            recorder.add_step(step)
            result.raw_plan = m.group(1)
            result.delivered = True
            break

        # Action?
        parsed = _parse_action(out)
        if not parsed:
            obs = ("Error: could not parse an action. Use exactly "
                   "'Thought: ...\\nAction: fn_name({\"key\": value})' or "
                   "'Thought: ...\\nFinal Plan: <JSON array>'.")
            messages.append({"role": "user", "content": obs})
            recorder.add_step(Step(n, thought=_thought_of(out), observation=obs))
            continue

        fn_name, args = parsed
        step = Step(n, thought=_thought_of(out), action=f"{fn_name}({json.dumps(args, ensure_ascii=False)[:200]})")
        recorder.add_step(step)

        fn = getattr(module, fn_name, None)
        if fn is None or fn_name.startswith("_"):
            avail = [f.name for f in catalog.published()]
            obs = f"Error: unknown function '{fn_name}'. Available: {avail}"
            messages.append({"role": "user", "content": obs})
            recorder.add_call_error(n, fn_name, args, "unknown function")
            continue

        try:
            ret = fn(**args)
            env = _envelope(ret)
            obs = _format_observation(env)
            recorder.add_call(n, fn_name, args, env)
        except Exception as e:
            obs = f"Error while executing {fn_name}: {type(e).__name__}: {e}"
            recorder.add_call_error(n, fn_name, args, obs)
        messages.append({"role": "user", "content": obs})

    # 未交付不再做内联 salvage —— 无计划统一交给 finalize_plan 的结构化组装
    result.usage = {"steps": len(result.steps) or n, "seconds": round(time.time() - t0, 1),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens}
    recorder.usage = result.usage
    result.recorder = recorder
    return result


def _thought_of(out: str) -> str:
    m = re.search(r"Thought:\s*(.+?)(?:\nAction:|\nFinal Plan:|$)", out, re.S)
    return m.group(1).strip()[:400] if m else ""
