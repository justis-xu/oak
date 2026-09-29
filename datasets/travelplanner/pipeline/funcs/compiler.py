"""步骤③-3a 函数编译：能力规划（强）→ 逐函数生成 → AST → 双图试跑。"""
from __future__ import annotations

import logging
import re

import networkx as nx

from ..config_task import Config
from oak.llm.client import LLMClient
from ..data.queries import Query
from oak.schema.model import Schema
from oak.operators import library as ops
from oak.operators.sandbox import exec_function_source, trial_run, SandboxError
from .catalog import CompiledFunction, FunctionCatalog
from ..prompts import func_gen as P4

log = logging.getLogger("oak.funcs")

_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def _extract_code(raw: str) -> str:
    m = _FENCE_RE.search(raw)
    t = m.group(1).strip() if m else raw.strip()
    return t


def query_pattern_digest(round_queries: list[Query]) -> str:
    """local_constraint 取值分布 + 3 个完整 query 例子。"""
    from collections import Counter
    dist: dict[str, Counter] = {}
    for q in round_queries:
        for k, v in q.local_constraint.items():
            if v not in (None, "", [], {}):
                dist.setdefault(k, Counter())[str(v)] += 1
    lines = ["local_constraint value distribution across this round's queries:"]
    for k, c in dist.items():
        top = ", ".join(f"{v}(x{n})" for v, n in c.most_common(5))
        lines.append(f"- {k}: {top}")
    lines.append("")
    for q in round_queries[:3]:
        lc = "; ".join(f"{k}={v}" for k, v in q.local_constraint.items()
                       if v not in (None, "", [], {}))
        lines.append(f"example query: {q.query} (people={q.people_number}, "
                     f"budget={q.budget}, days={q.days}; {lc})")
    return "\n".join(lines)


async def _plan_capabilities(client: LLMClient, schema: Schema,
                             round_queries: list[Query], namespace: str) -> str:
    prompt = P4.build_planning(query_pattern_digest(round_queries),
                               schema.render_brief())
    res = await client.chat(role="func_gen", messages=[
        {"role": "system", "content": "You are a capability planner. Output only the list."},
        {"role": "user", "content": prompt},
    ], temperature=0.2, namespace=namespace)
    return res.content


def _default_sample_args(src: str) -> dict:
    """从签名默认值构造试跑参数（类型感知：list/dict/city 名给真实样例）。"""
    import ast
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    args: dict = {}
    defaults = [None] * (len(fn.args.args) - len(fn.args.defaults)) + fn.args.defaults
    for arg, d in zip(fn.args.args, defaults):
        name = arg.arg
        ann = ""
        if arg.annotation is not None:
            try:
                ann = ast.unparse(arg.annotation)
            except Exception:
                ann = ""
        if d is not None:
            try:
                args[name] = ast.literal_eval(d)
                continue
            except Exception:
                pass
        low = name.lower()
        if "list" in ann or low.endswith(("s", "_list")) and "cit" not in low:
            args[name] = []
        elif "dict" in ann or low.endswith("_dict"):
            args[name] = {}
        elif "bool" in ann or low.startswith(("is_", "allow", "use_")):
            args[name] = True
        elif any(k in low for k in ("city", "origin", "dest", "name", "type", "cuisine")):
            args[name] = _sample_str_value(name)
        else:
            args[name] = 0
    return args


def _sample_str_value(name: str) -> str:
    """给名字/城市类参数一个图内真实首值（试跑更真实）。"""
    g = _g
    if g is not None:
        low = name.lower()
        # 参数名细匹配字段（试跑空结果会误杀 search 函数）
        field_hints = []
        if "origin" in low or low.startswith("org"):
            field_hints = ["OriginCityName", "origin"]
        elif "dest" in low:
            field_hints = ["DestCityName", "destination"]
        elif "date" in low:
            field_hints = ["FlightDate"]
        elif "city" in low or "region" in low or "state" in low:
            # city 参数应取业务实体实际存在的城市值（Restaurant.City 等），
            # 而非 City 节点名（可能来自航班端点，无业务数据）
            for _, nd in g.nodes(data=True):
                if nd.get("etype") in ("Restaurant", "Accommodation", "Attraction"):
                    for f in ("City", "city"):
                        if nd.get(f):
                            return str(nd[f])
            for _, nd in g.nodes(data=True):
                if nd.get("etype") == "City":
                    import json as _json
                    try:
                        v = _json.loads(nd.get("__key__", "{}")).get("name") \
                            or nd.get("name")
                    except Exception:
                        v = nd.get("name")
                    if v:
                        return str(v)
        elif "flight" in low:
            field_hints = ["Flight Number"]
        if field_hints:
            for _, nd in g.nodes(data=True):
                for f in field_hints:
                    v = nd.get(f)
                    if v:
                        return str(v)
        else:
            want = None
            if "flight" in low:
                want = "Flight"
            elif "accom" in low or "hotel" in low or "lodg" in low:
                want = "Accommodation"
            elif "restaur" in low or "meal" in low or "food" in low:
                want = "Restaurant"
            elif "attrac" in low:
                want = "Attraction"
            elif "city" in low:
                want = "City"
            if want:
                for _, nd in g.nodes(data=True):
                    if nd.get("etype") == want:
                        for f in ("name", "NAME", "Name", "Flight Number", "city", "City"):
                            v = nd.get(f)
                            if v:
                                return str(v)
    return ""


async def compile_functions(client: LLMClient, cfg: Config, schema: Schema,
                            g_full: nx.MultiDiGraph,
                            g_mini: nx.MultiDiGraph,
                            round_queries: list[Query],
                            sigma_f: list[dict] | None,
                            prev_catalog: FunctionCatalog | None,
                            namespace: str) -> FunctionCatalog:
    # 无函数反馈且有上轮目录：继承（此前会整套重建丢函数 —— R8）
    if not sigma_f and prev_catalog is not None and prev_catalog.published():
        return prev_catalog

    # 修补模式：只重编译被点名的函数，其余继承
    if sigma_f and prev_catalog is not None:
        return await _apply_patches(client, cfg, schema, g_full, g_mini,
                                    prev_catalog, sigma_f, namespace)

    # (b) 全量编译（首轮或无上轮目录）
    set_trial_graphs(g_full, g_mini)
    plan = await _plan_capabilities(client, schema, round_queries, namespace)
    # 宽松解析：容忍编号/破折号/反引号前缀（此前严格正则会回落到单个 plan）
    tasks = []
    for ln in plan.splitlines():
        s = ln.strip().lstrip("0123456789.-) ").lstrip("`> ")
        if re.match(r"[a-zA-Z_][a-zA-Z0-9_]*\(", s):
            tasks.append(s)
    if not tasks:
        tasks = ["plan(param=default) -> full itinerary skeleton"]

    catalog = FunctionCatalog()
    ops_doc = ops.render_operator_docs()
    schema_brief = schema.render_brief()
    q_patterns = query_pattern_digest(round_queries)

    for i, task in enumerate(tasks[:12]):
        fn = await _gen_one(client, cfg, ops_doc, schema_brief, q_patterns,
                            task, None, None, None, namespace)
        if fn:
            catalog.functions.append(fn)
    log.info("compiled %d/%d functions", len(catalog.published()), len(tasks))
    return catalog


async def _gen_one(client, cfg, ops_doc, schema_brief, q_patterns, task,
                   existing_src, judge_delta, failure_note, namespace
                   ) -> CompiledFunction | None:
    for attempt in range(cfg.func_gen_attempts):
        # 每次尝试重构 prompt（failure_note 变化 → 缓存键变化，避免命中死结果）
        prompt = P4.build(ops_doc, schema_brief, q_patterns, task,
                          existing_fn_source=existing_src, judge_delta=judge_delta,
                          failure_note=failure_note)
        res = await client.chat(role="func_gen", messages=[
            {"role": "system", "content": P4.SYSTEM},
            {"role": "user", "content": prompt},
        ], temperature=0.2, namespace=namespace)
        src = _extract_code(res.content)
        try:
            fn_name, fn = exec_function_source(src)
        except (SandboxError, SyntaxError) as e:
            failure_note = f"```python\n{src}\n```\nERROR: {e}"
            continue
        # 双图试跑
        sample_args = _default_sample_args(src)
        results = trial_run(fn, {"full": _g, "mini": _g_mini_ctx[0]}, sample_args)
        ok = all(r.ok for r in results)
        errs = [f"{r.graph_name}: {r.error}" for r in results if not r.ok]
        # 正向 fixture：search/find/get 类函数在全量图上必须返回非空
        # （空结果不报错的死函数曾大批发布 —— R6）
        if ok and re.match(r"(search|find|get|list)", fn_name):
            for r in results:
                if r.graph_name == "full" and (r.sample_output in ("[]", "{}", "null", "0")):
                    ok = False
                    errs.append("full graph: returned EMPTY for sample args — "
                                "a search function must find something on real data")
        if ok:
            sig = src.split("\n")[0].rstrip(":")
            doc = (fn.__doc__ or task).strip().splitlines()[0]
            return CompiledFunction(name=fn_name, signature=sig, docstring=doc,
                                    source=src, status="published",
                                    trials=[asdict_lite(r) for r in results])
        failure_note = f"```python\n{src}\n```\nTRIAL FAILED:\n" + "\n".join(errs)
    log.warning("function generation failed after retries: %s", task[:80])
    # 记 failed：保留最后源码与错误（可诊断，不进目录渲染）
    return CompiledFunction(
        name=_fn_name_of(src) or "failed", signature=task[:120], docstring="",
        source=src, status="failed",
        trials=[{"graph": "-", "ok": False,
                 "error": (failure_note or "generation failed")[-2000:],
                 "out": None, "ms": 0}])


# 试跑需要图的上下文（避免全局污染，用模块级暂存）
_g: nx.MultiDiGraph | None = None
_g_mini_ctx: list = [None]


def set_trial_graphs(g_full, g_mini) -> None:
    global _g, _g_mini_ctx
    _g = g_full
    _g_mini_ctx[0] = g_mini


def asdict_lite(r):
    return {"graph": r.graph_name, "ok": r.ok, "error": r.error,
            "out": r.sample_output, "ms": r.elapsed_ms}


def _fn_name_of(src: str) -> str | None:
    import ast as _ast
    try:
        tree = _ast.parse(src)
        for n in tree.body:
            if isinstance(n, _ast.FunctionDef):
                return n.name
    except Exception:
        pass
    return None


async def _apply_patches(client, cfg, schema, g_full, g_mini,
                         prev: FunctionCatalog, sigma_f: list[dict],
                         namespace) -> FunctionCatalog:
    """评判器函数补丁：regenerate 按 δ 指令重生成；delete 下架（保 published≥4）；add 新增。"""
    set_trial_graphs(g_full, g_mini)
    ops_doc = ops.render_operator_docs()
    schema_brief = schema.render_brief()
    q_patterns = ""
    new_fns: dict[str, CompiledFunction] = {f.name: f for f in prev.functions}
    n_published = sum(1 for f in prev.functions if f.status == "published")
    for sig_item in sigma_f:
        delta = sig_item.get("delta") or {}
        target = delta.get("function") or sig_item.get("name") or ""
        action = sig_item.get("a", "modify")
        if action == "delete" and target in new_fns:
            if n_published <= 4:
                log.warning("refuse delete of %s: would drop published below 4", target)
                continue
            new_fns[target].status = "failed"
            n_published -= 1
            continue
        existing = new_fns.get(target)
        existing_src = existing.source if existing else None
        task = (f"Modify function `{target}`: {delta.get('instruction', '')}"
                if action == "modify" else
                f"Create new function `{target}`: {delta.get('instruction', '')}")
        fn = await _gen_one(client, cfg, ops_doc, schema_brief, q_patterns,
                            task, existing_src, delta, None, namespace)
        if fn:
            fn.generation = (existing.generation + 1) if existing else 0
            new_fns[fn.name] = fn
    return FunctionCatalog(list(new_fns.values()))
