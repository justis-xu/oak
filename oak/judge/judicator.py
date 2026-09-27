"""步骤④ 评判器：五输入 → σ 四元组；路由到慢/快回路；补丁应用。"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from ..config import Config
from ..llm.client import LLMClient
from ..schema.model import Schema, Attribute, EntityType, RelationType
from ..funcs.catalog import FunctionCatalog
from ..eval.adapter import SubsetScores
from ..prompts import judicator as P6

log = logging.getLogger("oak.judge")

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


@dataclass
class JudgeInputs:
    schema: Schema
    graph_stats: dict
    graph_samples: dict
    catalog: FunctionCatalog
    trajectories_digest: str
    scores: SubsetScores


def _graph_section(stats: dict, samples: dict) -> str:
    lines = [f"nodes={stats['n_nodes']} edges={stats['n_edges']}",
             "nodes by type: " + json.dumps(stats["by_type"]),
             "edges by relation: " + json.dumps(stats["by_relation"])]
    for t, items in samples.items():
        for it in items[:5]:
            lines.append(f"sample {t}: {json.dumps(it['props'], ensure_ascii=False)[:200]}")
    return "\n".join(lines)


def _functions_section(catalog: FunctionCatalog) -> str:
    lines = []
    for f in catalog.functions:
        status = f.status
        trial = f.trials[-1] if f.trials else {}
        ok = trial.get("ok", "?")
        lines.append(f"- {f.signature} [{status}] last_trial_ok={ok}")
    return "\n".join(lines)


def _scores_section(scores: SubsetScores) -> str:
    lines = [f"delivery={scores.delivery:.2%} micro_cs={scores.micro_cs:.2%} "
             f"macro_cs={scores.macro_cs:.2%} micro_hc={scores.micro_hc:.2%} "
             f"macro_hc={scores.macro_hc:.2%} final={scores.final:.2%}"]
    for it in scores.per_query:
        cs_failed = [k for k, v in it["cs"].items() if v is False]
        hc_failed = [k for k, v in it["hc"].items() if v is False]
        # 官方错误消息是归因的第一证据（哪个天/哪个字段/差什么）
        msgs = []
        for src in ("cs_msg", "hc_msg"):
            for k, m in (it.get(src) or {}).items():
                msgs.append(f"{k}: {m}")
        lines.append(f"query idx={it['idx']} delivered={it['delivered']} "
                     f"cs_all={it['macro_cs']} hc_all={it['macro_hc']}"
                     + (f" CS_FAILED={cs_failed}" if cs_failed else "")
                     + (f" HC_FAILED={hc_failed}" if hc_failed else "")
                     + ((" | official messages: " + " ; ".join(msgs[:6])) if msgs else ""))
    return "\n".join(lines)


async def judge(client: LLMClient, cfg: Config, inputs: JudgeInputs,
                namespace: str) -> list[dict]:
    prompt = P6.build(
        schema_yaml=inputs.schema.to_yaml(),
        graph_section=_graph_section(inputs.graph_stats, inputs.graph_samples),
        functions_section=_functions_section(inputs.catalog),
        trajectories_section=inputs.trajectories_digest,
        scores_section=_scores_section(inputs.scores),
    )
    res = await client.chat(role="judicator", messages=[
        {"role": "system", "content": P6.SYSTEM},
        {"role": "user", "content": prompt},
    ], temperature=0.1, json_mode=True, namespace=namespace)

    t = res.content.strip()
    m = _FENCE_RE.search(t)
    if m:
        t = m.group(1).strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0:
        log.warning("judicator output not JSON: %s", t[:200])
        return []
    try:
        data = json.loads(t[i:j + 1])
    except Exception:
        log.warning("judicator JSON parse failed: %s", t[:200])
        return []
    items = data.get("feedback") or []
    out = []
    for it in items[:6]:
        if not isinstance(it, dict):
            continue
        u, a = it.get("u"), it.get("a")
        if u not in ("entity_type", "relation", "function") or a not in ("add", "delete", "modify"):
            continue
        it.setdefault("name", (it.get("delta") or {}).get("name"))
        out.append({"u": u, "a": a, "delta": it.get("delta") or {},
                    "rho": str(it.get("rho", "")), "name": it.get("name")})
    log.info("judicator issued %d feedback items", len(out))
    return out


def route_feedback(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """(ψ^S 模式反馈→下轮步骤①, σ_F 函数反馈→本轮步骤③)。"""
    psi_s = [it for it in items if it["u"] in ("entity_type", "relation")]
    sigma_f = [it for it in items if it["u"] == "function"]
    return psi_s, sigma_f


def apply_schema_feedback(schema: Schema, psi_s: list[dict]) -> Schema:
    """把 ψ^S 应用到模式（add/delete/modify 实体与关系）。非法引用丢弃并日志。"""
    import dataclasses
    entities = {e.name: e for e in schema.entities}
    relations = {r.name: r for r in schema.relations}

    for it in psi_s:
        a, delta, target = it["a"], it["delta"], it.get("name") or delta.get("name")
        try:
            if it["u"] == "entity_type":
                if a == "add":
                    name = delta.get("name") or target
                    if name and name not in entities:
                        pk = delta.get("primary_key") or []
                        attrs = [Attribute(name=str(x["name"]), dtype=str(x.get("dtype", "string")))
                                 for x in (delta.get("attributes") or []) if isinstance(x, dict)]
                        if pk and attrs:
                            entities[name] = EntityType(
                                name=name, primary_key=[str(k) for k in pk],
                                attributes=attrs,
                                description=str(delta.get("description", "")))
                elif a == "delete" and target in entities:
                    del entities[target]
                elif a == "modify" and target in entities:
                    e = entities[target]
                    for x in (delta.get("add_attributes") or delta.get("attributes") or []):
                        if isinstance(x, dict) and x.get("name"):
                            if not any(a2.name == x["name"] for a2 in e.attributes):
                                e.attributes.append(Attribute(
                                    name=str(x["name"]),
                                    dtype=str(x.get("dtype", "string"))))
                    new_pk = delta.get("primary_key")
                    if new_pk:
                        e.primary_key = [str(k) for k in new_pk]
            elif it["u"] == "relation":
                if a == "add":
                    name = delta.get("name") or target
                    if name and name not in relations and delta.get("range"):
                        dom = delta.get("domain")
                        dom = [dom] if isinstance(dom, str) else list(dom or [])
                        relations[name] = RelationType(
                            name=str(name), domain=[str(d) for d in dom],
                            range=str(delta["range"]),
                            functional=bool(delta.get("functional", False)))
                elif a == "delete" and target in relations:
                    del relations[target]
                elif a == "modify" and target in relations:
                    r = relations[target]
                    if delta.get("range"):
                        r.range = str(delta["range"])
                    if delta.get("functional") is not None:
                        r.functional = bool(delta["functional"])
        except Exception as e:
            log.warning("schema feedback applied with error (dropped): %r -> %s", it, e)

    new_schema = dataclasses.replace(
        schema, entities=list(entities.values()), relations=list(relations.values()))
    errs = new_schema.validate()
    if errs:
        log.warning("schema after feedback has errors (will be re-validated next round): %s", errs)
    return new_schema
