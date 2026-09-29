"""步骤② 抽取：逐 chunk 调 flash，严格 JSON 契约 + 1 次 LLM 修复。"""
from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

from oak.config import Config
from oak.kg.graph import EntityCandidate, RelationCandidate
from oak.llm.client import LLMClient
from oak.schema.model import Schema
if TYPE_CHECKING:  # Chunk 仅需鸭子类型（.source/.text/.idx 等）；定义在各任务的语料模块
    from typing import Protocol

    class Chunk(Protocol):
        source: str
        text: str

from .prompts import kg_extract as P3

log = logging.getLogger("oak.kg")


class ExtractStats:
    chunks_total: int = 0
    parse_ok: int = 0
    repaired: int = 0
    dropped: int = 0
    entities: int = 0
    relations: int = 0
    invalid_refs: int = 0
    ungrounded: int = 0          # 主键值未逐字出现在源 chunk 行 → 丢弃（不做 fuzzy 修正）
    empty_retried: int = 0       # 0 实体但 chunk 有数据行 → 显式重试次数
    errors: list = field(default_factory=list)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_extraction(raw: str, schema: Schema) -> dict:
    """剥栅栏 → json.loads → 按 schema 过滤非法引用。兼容紧凑/冗长两种格式。抛 ValueError 供修复路径。"""
    t = raw.strip()
    m = _FENCE_RE.search(t)
    if m:
        t = m.group(1).strip()
    # 截取首个 { 到最后一个 }（防前后废话）
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("no JSON object found")
    data = json.loads(t[i:j + 1])
    if not isinstance(data, dict):
        raise ValueError("top level is not an object")

    etypes = {e.name for e in schema.entities}
    rnames = {r.name for r in schema.relations}
    ent_out: list[dict] = []
    rel_out: list[dict] = []
    invalid = 0

    if "e" in data:                     # 紧凑格式：{"e": [[type, pk, attrs]], "r": [[rel, i, j]]}
        raw_ents = data.get("e") or []
        parsed_ents = []
        for e in raw_ents:
            try:
                et, key_raw, props_raw = e[0], e[1], (e[2] if len(e) > 2 else {})
                if et not in etypes or not isinstance(key_raw, dict):
                    invalid += 1
                    continue
                ent = schema.entity(et)
                key = {str(k): v for k, v in key_raw.items()
                       if str(k) in set(ent.primary_key)}
                if not key:
                    invalid += 1
                    continue
                props = {}
                for k, v in (props_raw or {}).items():
                    ck = ent.canonical_attr(str(k))
                    if ck is not None:
                        props[ck] = v
                parsed_ents.append((et, key, props))
            except Exception:
                invalid += 1
        for idx, (et, key, props) in enumerate(parsed_ents):
            ent_out.append({"etype": et, "key": key, "properties": props})
        for r in data.get("r") or []:
            try:
                rn, hi, ti = r[0], r[1], r[2]
                if rn not in rnames:
                    invalid += 1
                    continue
                hi, ti = int(hi), int(ti)
                if not (0 <= hi < len(parsed_ents) and 0 <= ti < len(parsed_ents)):
                    invalid += 1
                    continue
                het, hkey, _ = parsed_ents[hi]
                tet, tkey, _ = parsed_ents[ti]
                rel_out.append({"relation": rn,
                                "head": [het, hkey], "tail": [tet, tkey]})
            except Exception:
                invalid += 1
    else:                               # 冗长格式（旧）
        for e in data.get("entities", []):
            try:
                et = str(e["etype"])
                if et not in etypes:
                    invalid += 1
                    continue
                ent = schema.entity(et)
                key = {str(k): v for k, v in e["key"].items()
                       if str(k) in set(ent.primary_key)}
                props = {}
                for k, v in (e.get("properties") or {}).items():
                    ck = ent.canonical_attr(str(k))
                    if ck is not None:
                        props[ck] = v
                if not key:
                    invalid += 1
                    continue
                ent_out.append({"etype": et, "key": key, "properties": props})
            except Exception:
                invalid += 1
        for r in data.get("relations", []):
            try:
                rn = str(r["relation"])
                if rn not in rnames:
                    invalid += 1
                    continue
                head, tail = r["head"], r["tail"]
                if head[0] not in etypes or tail[0] not in etypes:
                    invalid += 1
                    continue
                rel_out.append({"relation": rn,
                                "head": [head[0], {str(k): v for k, v in head[1].items()}],
                                "tail": [tail[0], {str(k): v for k, v in tail[1].items()}]})
            except Exception:
                invalid += 1
    return {"entities": ent_out, "relations": rel_out, "invalid_refs": invalid}


async def _extract_one(client: LLMClient, schema: Schema, chunk: Chunk,
                       namespace: str, stats: ExtractStats) -> dict:
    prompt = P3.build(schema.render_brief(), chunk.render())
    messages = [{"role": "system", "content": P3.SYSTEM},
                {"role": "user", "content": prompt}]
    res = await client.chat(role="kg", messages=messages, temperature=0.0,
                            json_mode=True, namespace=namespace)
    try:
        out = parse_extraction(res.content, schema)
        # 空 chunk 可疑：本 chunk 有数据行却抽出 0 实体（q179 曾整块丢 21 家餐馆）
        # → 带显式反馈重试一次，仍空才接受
        if not out["entities"] and chunk.text.strip():
            out = await _retry_empty(client, schema, chunk, messages, res, out, namespace, stats)
        stats.parse_ok += 1
        return out
    except Exception as e:
        # 1 次修复
        repair = await client.chat(role="kg", messages=[
            *messages,
            {"role": "assistant", "content": res.content},
            {"role": "user", "content":
                f"Your output could not be parsed ({e}). Re-output ONLY the JSON object."},
        ], temperature=0.0, json_mode=True, namespace=namespace)
        try:
            out = parse_extraction(repair.content, schema)
            if not out["entities"] and chunk.text.strip():
                out = await _retry_empty(client, schema, chunk, messages, repair, out, namespace, stats)
            stats.repaired += 1
            return out
        except Exception as e2:
            stats.dropped += 1
            stats.errors.append(f"chunk {chunk.source}/{chunk.seq}: {e2}")
            return {"entities": [], "relations": [], "invalid_refs": 0}


async def _retry_empty(client, schema, chunk, messages, prev_res, prev_out,
                       namespace, stats) -> dict:
    """0 实体但 chunk 有数据行 → 显式反馈重试一次（防整块静默丢失）。"""
    stats.empty_retried += 1
    n_rows = len([ln for ln in chunk.text.splitlines() if ln.strip()])
    retry = await client.chat(role="kg", messages=[
        *messages,
        {"role": "assistant", "content": prev_res.content},
        {"role": "user", "content":
            f"Your output contained 0 entities, but this chunk has {n_rows} data rows "
            f"under the header. Extract one entity per data row (values verbatim). "
            f"Re-output ONLY the JSON object."},
    ], temperature=0.0, json_mode=True, namespace=namespace)
    try:
        out2 = parse_extraction(retry.content, schema)
        return out2 if out2["entities"] else prev_out
    except Exception:
        return prev_out


def _chunk_blob(ch: Chunk) -> str:
    """chunk 的可检索文本（表头 + 数据行，空白折叠）。"""
    import re as _re
    return _re.sub(r"\s+", " ", ch.header + "\n" + ch.text)


def _entity_grounded(key: dict, blob: str) -> bool:
    """精确源 grounding：全部字符串主键值必须逐字出现在源 chunk。

    不做编辑距离/大小写猜测/近邻替换（fuzzy 合并有误改风险）——
    不满足即丢弃（q64 的 Italiize 转录错字应被丢，由其他精确候选替代）。
    数值主键跳过（LLM 可能规范化了千分位/小数格式）。
    """
    for v in key.values():
        if v is None:
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            continue
        s = str(v).strip()
        if s and s not in blob:
            return False
    return True


def _frozen(key: dict) -> tuple:
    return tuple(sorted((str(k), str(v)) for k, v in key.items()))


async def extract_graph_from_chunks(client: LLMClient, cfg: Config, schema: Schema,
                                    chunks: list[Chunk], namespace: str
                                    ) -> tuple[list[EntityCandidate], list[RelationCandidate], ExtractStats]:
    stats = ExtractStats(chunks_total=len(chunks))
    results = await asyncio.gather(*[
        _extract_one(client, schema, ch, namespace, stats) for ch in chunks
    ])
    entities: list[EntityCandidate] = []
    relations: list[RelationCandidate] = []
    for ch, out in zip(chunks, results):
        stats.invalid_refs += out.get("invalid_refs", 0)
        blob = _chunk_blob(ch)
        dropped_keys: set = set()
        for e in out["entities"]:
            if _entity_grounded(e["key"], blob):
                entities.append(EntityCandidate(
                    etype=e["etype"], key=e["key"], properties=e["properties"],
                    chunk_id=f"{ch.source}/{ch.seq}"))
            else:
                dropped_keys.add((e["etype"], _frozen(e["key"])))
                stats.ungrounded += 1
        for r in out["relations"]:
            if (r["head"][0], _frozen(r["head"][1])) in dropped_keys or \
               (r["tail"][0], _frozen(r["tail"][1])) in dropped_keys:
                continue                      # 端点被 grounding 丢弃 → 关系一并丢弃
            relations.append(RelationCandidate(
                relation=r["relation"],
                head=(r["head"][0], r["head"][1]),
                tail=(r["tail"][0], r["tail"][1])))
    stats.entities = len(entities)
    stats.relations = len(relations)
    drop_rate = stats.dropped / max(1, stats.chunks_total)
    if drop_rate > 0.15:
        log.warning("chunk 丢弃率 %.0f%% 超阈值 15%%: %s",
                    drop_rate * 100, stats.errors[:5])
    if stats.ungrounded > stats.entities * 0.1:
        log.warning("ungrounded entities %d / %d (>10%%) — 检查语料格式假设",
                    stats.ungrounded, stats.entities)
    return entities, relations, stats
