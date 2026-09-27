"""九件套通用算子库（论文附录 B）。

全部为纯函数；图通过模块级 G 引用（运行时注入）——"字段焊死、值留活口、图是环境"。
extract_runtime_slots 是唯一带 LLM 后端的算子（调用延迟注入，见 set_slot_extractor）。
"""
from __future__ import annotations

import contextvars
import re
from typing import Any

import networkx as nx

# ContextVar：协程级隔离，并发跑题不互相踩图（曾经的错图竞态根源）
_G: contextvars.ContextVar["nx.MultiDiGraph | None"] = \
    contextvars.ContextVar("oak_graph", default=None)
_slot_extractor = None          # async fn(text, slot_names) -> dict（由管线注入）


def set_graph(g: nx.MultiDiGraph) -> None:
    _G.set(g)


def get_graph() -> nx.MultiDiGraph:
    g = _G.get()
    assert g is not None, "graph not injected (call set_graph in this task's context)"
    return g


def set_slot_extractor(fn) -> None:
    global _slot_extractor
    _slot_extractor = fn


def _rows_of(g: nx.MultiDiGraph, etype: str) -> list[dict]:
    from ..kg.graph import node_view
    rows = []
    for nid, nd in g.nodes(data=True):
        if nd.get("etype") == etype:
            row = {"__id__": nid, "__type__": etype}
            # 主键存于 __key__ 元数据 —— 必须物化进行视图，否则按主键字段
            # （Flight Number / NAME / city 等）的检索与过滤全部落空
            row.update(node_view(nd))
            rows.append(row)
    return rows


def _to_num(v) -> float | None:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"-?\d[\d,]*\.?\d*", str(v))
    if not m:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None


# ---------------------------------------------------------------- 九算子
def lookup_entities(etype: str, filters: dict | None = None, limit: int = 50) -> list[dict]:
    """按类型 + 属性条件检索实体（属性值做字符串包含匹配，大小写不敏感）。"""
    g = get_graph()
    rows = _rows_of(g, etype)
    if filters:
        out = []
        for r in rows:
            ok = True
            for k, v in filters.items():
                rv = r.get(k)
                if rv is None or str(v).lower() not in str(rv).lower():
                    ok = False
                    break
            if ok:
                out.append(r)
        rows = out
    return rows[:limit]


def extract_runtime_slots(text: str, slot_names: list[str]) -> dict:
    """自然语言 → 带类型槽位（LLM 后端；未注入时退化为关键词匹配）。"""
    if _slot_extractor is not None:
        return _slot_extractor(text, slot_names)
    # 退化实现：简单关键词兜底
    t = text.lower()
    out = {}
    for s in slot_names:
        for kw in s.lower().replace("_", " ").split():
            if kw in t:
                out[s] = True
                break
    return out


def traverse_relations(start_id: str, relation: str, direction: str = "out",
                       max_hops: int = 1) -> list[dict]:
    """沿指定关系扩展有界邻域。返回到达的节点行。"""
    g = get_graph()
    if start_id not in g:
        return []
    frontier = {start_id}
    visited: set[str] = set()
    for _ in range(max_hops):
        nxt: set[str] = set()
        for nid in frontier:
            if direction in ("out", "both"):
                for _, tgt, ed in g.out_edges(nid, data=True):
                    if ed.get("relation") == relation and tgt not in visited:
                        nxt.add(tgt)
            if direction in ("in", "both"):
                for src, _, ed in g.in_edges(nid, data=True):
                    if ed.get("relation") == relation and src not in visited:
                        nxt.add(src)
        visited |= nxt
        frontier = nxt
        if not frontier:
            break
    rows = []
    for nid in visited:
        nd = g.nodes[nid]
        row = {"__id__": nid, "__type__": nd.get("etype", "?")}
        row.update({k: v for k, v in nd.items() if not k.startswith("__")})
        rows.append(row)
    return rows


def project_properties(rows: list[dict], props: list[str]) -> list[dict]:
    """把实体属性投影成扁平行（保留 __id__/__type__）。"""
    out = []
    for r in rows:
        row = {"__id__": r.get("__id__"), "__type__": r.get("__type__")}
        for p in props:
            row[p] = r.get(p)
        out.append(row)
    return out


def filter_categorical(rows: list[dict], field: str, values: list[str],
                       mode: str = "any") -> list[dict]:
    """类别过滤：mode=any（包含任一）/all（全包含）/exclude（都不含）。大小写不敏感。"""
    vals = [str(v).lower() for v in values]

    def hit(r) -> bool:
        rv = str(r.get(field, "")).lower()
        ins = [v in rv for v in vals]
        if mode == "any":
            return any(ins)
        if mode == "all":
            return all(ins)
        return not any(ins)          # exclude

    return [r for r in rows if hit(r)]


def filter_relation_connected(rows: list[dict], via_relation: str,
                              target_etype: str = "",
                              target_filter: dict | None = None) -> list[dict]:
    """只保留经 via_relation 连到（可选过滤的）目标实体的行。"""
    g = get_graph()
    out = []
    for r in rows:
        nid = r.get("__id__")
        if not nid or nid not in g:
            continue
        for _, tgt, ed in g.out_edges(nid, data=True):
            if ed.get("relation") != via_relation:
                continue
            td = g.nodes[tgt]
            if target_etype and td.get("etype") != target_etype:
                continue
            if target_filter:
                ok = all(str(v).lower() in str(td.get(k, "")).lower()
                         for k, v in target_filter.items())
                if not ok:
                    continue
            out.append(r)
            break
    return out


def filter_numeric(rows: list[dict], field: str, op: str, value: float) -> list[dict]:
    """数值过滤：op ∈ <,<=,>,>=,==；不可解析的值视为不过。"""
    out = []
    for r in rows:
        x = _to_num(r.get(field))
        if x is None:
            continue
        keep = {"<": x < value, "<=": x <= value, ">": x > value,
                ">=": x >= value, "==": x == value}.get(op)
        if keep:
            out.append(r)
    return out


def filter_set_overlap(rows: list[dict], field: str, required: list[str],
                       min_overlap: int = 1) -> list[dict]:
    """多值字段（逗号/分号分隔串，如 Cuisines）与要求集合的重叠过滤。"""
    req = {str(v).strip().lower() for v in required}
    out = []
    for r in rows:
        rv = str(r.get(field, "")).lower()
        items = {p.strip() for p in re.split(r"[;,]", rv) if p.strip()}
        if len(items & req) >= min_overlap:
            out.append(r)
    return out


def aggregate_values(rows: list[dict], field: str, op: str = "sum",
                     group_by: str | None = None) -> Any:
    """count/sum/min/max/average，可按字段分组。"""
    if group_by:
        groups: dict[str, list[float]] = {}
        for r in rows:
            x = _to_num(r.get(field))
            if x is not None:
                groups.setdefault(str(r.get(group_by, "")), []).append(x)
        return {k: _agg(v, op) for k, v in groups.items()}
    if op == "count":
        return len(rows)
    vals = [x for x in (_to_num(r.get(field)) for r in rows) if x is not None]
    return _agg(vals, op)


def _agg(vals: list[float], op: str):
    if op == "count":
        return len(vals)
    if not vals:
        return None
    return {"sum": lambda: sum(vals), "min": lambda: min(vals),
            "max": lambda: max(vals),
            "average": lambda: sum(vals) / len(vals)}.get(op, lambda: None)()


# ---------------------------------------------------------------- 注册表与文档
def _reg(name, sig, params, semantics, example):
    return {"name": name, "signature": sig, "params": params,
            "semantics": semantics, "example": example}


OPERATOR_REGISTRY = {
    "lookup_entities": _reg(
        "lookup_entities", "lookup_entities(etype, filters=None, limit=50) -> list[dict]",
        "etype: entity type name; filters: {field: substring} case-insensitive; limit",
        "Search entities of one type; substring match on attributes.",
        'lookup_entities("Flight", {"OriginCityName": "Columbus"}, limit=20)'),
    "extract_runtime_slots": _reg(
        "extract_runtime_slots", "extract_runtime_slots(text, slot_names) -> dict",
        "text: natural language; slot_names: wanted slots",
        "Parse natural language into typed slots (only explicit constraints).",
        'extract_runtime_slots(query_text, ["budget", "people_number"])'),
    "traverse_relations": _reg(
        "traverse_relations", "traverse_relations(start_id, relation, direction='out', max_hops=1) -> list[dict]",
        "start_id: __id__ value; relation: relation name; direction: out|in|both",
        "Expand bounded neighborhood along a relation.",
        'traverse_relations(flight_id, "arrives_at")'),
    "project_properties": _reg(
        "project_properties", "project_properties(rows, props) -> list[dict]",
        "rows: list from other operators; props: field names",
        "Keep only wanted columns (keeps __id__/__type__).",
        'project_properties(rows, ["NAME", "price"])'),
    "filter_categorical": _reg(
        "filter_categorical", "filter_categorical(rows, field, values, mode='any') -> list[dict]",
        "field: attribute; values: list; mode: any|all|exclude",
        "Filter by categorical containment (case-insensitive).",
        'filter_categorical(rows, "room type", ["Private room"])'),
    "filter_relation_connected": _reg(
        "filter_relation_connected", "filter_relation_connected(rows, via_relation, target_etype='', target_filter=None) -> list[dict]",
        "via_relation: relation name; target_filter: {field: substring}",
        "Keep rows connected via a relation to a (filtered) target.",
        'filter_relation_connected(rows, "serves_cuisine", "Cuisine", {"name": "French"})'),
    "filter_numeric": _reg(
        "filter_numeric", "filter_numeric(rows, field, op, value) -> list[dict]",
        "op: '<'|'<='|'>'|'>='|'=='",
        "Filter by numeric attribute; unparseable values drop out.",
        'filter_numeric(rows, "price", "<=", 200)'),
    "filter_set_overlap": _reg(
        "filter_set_overlap", "filter_set_overlap(rows, field, required, min_overlap=1) -> list[dict]",
        "field: multi-value string field; required: list",
        "Keep rows whose multi-value field overlaps the required set.",
        'filter_set_overlap(rows, "Cuisines", ["American", "French"])'),
    "aggregate_values": _reg(
        "aggregate_values", "aggregate_values(rows, field, op='sum', group_by=None)",
        "op: count|sum|min|max|average",
        "Aggregate a numeric field, optionally grouped.",
        'aggregate_values(rows, "Price", "sum")'),
}


def render_operator_docs() -> str:
    lines = []
    for spec in OPERATOR_REGISTRY.values():
        lines.append(f"### {spec['signature']}")
        lines.append(f"{spec['semantics']}")
        lines.append(f"example: {spec['example']}")
        lines.append("")
    return "\n".join(lines)
