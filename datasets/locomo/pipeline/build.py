"""建图管线：session → LLM 抽取 → 逐字 grounding → 实体归并 → 图组装与派生 → 落盘。

图指纹缓存：sha256(schema+抽取提示词+fast模型+审计开关) 前 8 位——任何一处变了
自动重建，没变直接复用（LLM 请求另有磁盘缓存，重建也几乎零成本）。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from oak.kg.graph import EntityCandidate, RelationCandidate
from oak.kg.graph import build_graph, graph_stats, save_graph
from oak.llm.client import LLMClient
from oak.schema.model import Schema

from .config import LocomoConfig, ns
from .data import Conversation
from .entity_resolve import canonicalize
from .prompts.extract import EXTRACT_SYSTEM, REPAIR_TEMPLATE, render_extract_prompt
from .schema_skeleton import (DEFAULT_TOPICS, EXTRACTABLE_ETYPES,
                              EXTRACTABLE_RELATIONS, FACT_TYPES)

_CHUNK_CHARS = 3000          # 超长 session 切块阈值
_MAX_FACTS_PER_SESSION = 60
# 建图逻辑版本号（进图指纹）：别名消毒/派生规则等代码变化时递增，防旧图缓存复用
BUILD_VER = 6


def _override_dates(facts: list[dict], session) -> None:
    """确定性日期覆写：相对日期由代码按会话锚重算（语义与 gold 对齐：
    上周X=严格早于锚的最近周X；粒度感知：去年→年/上个月→月/上周→周）。
    抽取 LLM 自算的 d 字段仅作兜底。"""
    from .dates import resolve_relative
    if not session.date_iso:
        return
    for f in facts:
        if f.get("o"):
            val, gran = resolve_relative(session.date_iso, f["o"])
            if val:
                f["d"], f["g"] = val, gran


@dataclass
class FactRecord:
    fid: str
    subject: str
    statement: str
    ftype: str
    date_iso: str = ""
    granularity: str = "无"
    date_raw: str = ""
    value: str = ""
    topics: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)   # ev 实体名
    session_no: int = 0

    def row(self) -> dict:
        return {"编号": self.fid, "陈述": self.statement, "主体": self.subject,
                "类型": self.ftype, "日期": self.date_iso, "日期粒度": self.granularity,
                "日期原文": self.date_raw, "数值": self.value,
                "主题": ";".join(self.topics), "出处": ";".join(self.sources)}


@dataclass
class BuildResult:
    graph_dir: Path
    stats: dict
    facts: list[FactRecord]


# ---------------------------------------------------------------- fingerprint
def graph_fingerprint(conv: Conversation, schema: Schema, lc: LocomoConfig,
                      topics: list[str]) -> str:
    blob = json.dumps({
        "build_ver": BUILD_VER,
        "schema": schema.to_yaml(),
        "sys": EXTRACT_SYSTEM,
        "tpl": render_extract_prompt(conv.sessions[0], conv, topics)[:500],
        "model": lc.cfg.model_fast,
        "audit": lc.audit_extract,
        "topics": topics,
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


# ---------------------------------------------------------------- 抽取
def _fold(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _chunk_turns(turns: list) -> list[list]:
    text = "".join(t.text for t in turns)
    if len(text) <= _CHUNK_CHARS * 2:
        return [turns]
    chunks, cur, size = [], [], 0
    for t in turns:
        cur.append(t)
        size += len(t.text)
        if size >= _CHUNK_CHARS:
            chunks.append(cur)
            cur, size = ([cur[-1]], len(cur[-1].text))   # 1 轮重叠
    if cur and (not chunks or len(cur) > 1):
        chunks.append(cur)
    return chunks


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


async def _extract_one(client: LLMClient, conv_id: str, prompt: str) -> dict | None:
    messages = [{"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": prompt}]
    r = await client.chat(role="locomo_extract", messages=messages,
                          temperature=0.0, max_tokens=4096, json_mode=True,
                          namespace=ns(conv_id, "build"))
    obj = _extract_json(r.content)
    if obj is None:
        rep = await client.chat(
            role="locomo_util", temperature=0.0, max_tokens=4096, json_mode=True,
            namespace=ns(conv_id, "build"),
            messages=[{"role": "system", "content": EXTRACT_SYSTEM},
                      {"role": "user", "content": REPAIR_TEMPLATE.format(
                          error="JSON 解析失败") + "\n\n原始任务：\n" + prompt}])
        obj = _extract_json(rep.content)
    return obj


async def _audit_one(client: LLMClient, conv, session, facts: list[dict]) -> dict | None:
    """二道审计：已有事实 + 原文 → 只补遗漏事实。"""
    from .prompts.extract import AUDIT_TEMPLATE, EXTRACT_SYSTEM
    transcript = "\n".join(f"[{t.dia_id}] {t.speaker}: {t.text}" for t in session.turns)
    fact_list = "\n".join(f"- {f['s']}: {f['t']}" for f in facts) or "（无）"
    r = await client.chat(
        role="locomo_util",
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": AUDIT_TEMPLATE.format(
                session_no=session.no,
                session_date_iso=session.date_iso.isoformat() if session.date_iso else "未知",
                speakers="、".join(conv.speakers),
                facts=fact_list, transcript=transcript)},
        ],
        temperature=0.0, max_tokens=4096, json_mode=True,
        namespace=ns(conv.sample_id, "audit"))
    return _extract_json(r.content)


def _norm_fact(o: dict, session_no: int) -> dict | None:
    if not isinstance(o, dict):
        return None
    s = str(o.get("s") or "").strip()
    t = str(o.get("t") or "").strip()
    if not s or not t:
        return None
    y = str(o.get("y") or "其他").strip()
    if y not in FACT_TYPES:
        y = "其他"
    tp = [str(x).strip() for x in (o.get("tp") or []) if str(x).strip()][:3]
    ev = [str(x).strip() for x in (o.get("ev") or []) if str(x).strip()][:8]
    src = [x.strip() for x in re.split(r"[;；,，\s]+", str(o.get("src") or "")) if x.strip()]
    src = [x for x in src if re.fullmatch(r"D\d+:\d+", x)][:4]
    d = str(o.get("d") or "").strip()
    if not re.fullmatch(r"\d{4}(-\d{2})?(-\d{2})?", d):
        d = ""
    g = str(o.get("g") or "无").strip()
    if g not in ("日", "周", "月", "年", "无"):
        g = "无"
    return {"s": s, "t": t, "y": y, "d": d, "g": g,
            "o": str(o.get("o") or "").strip(),
            "n": str(o.get("n") or "").strip(), "tp": tp, "ev": ev,
            "src": src, "session_no": session_no}


def _ground_fact(fact: dict, blob: str, valid_dia: set[str]) -> bool:
    if not fact["src"] or not (set(fact["src"]) & valid_dia):
        return False
    stmt = _fold(fact["t"])
    for num in re.findall(r"\d+(?:\.\d+)?", stmt):
        if num not in blob:
            return False
    if fact["o"] and _fold(fact["o"]) not in blob:
        fact["o"], fact["d"], fact["g"] = "", "", "无"
    if fact["n"] and fact["n"] not in blob:
        fact["n"] = ""
    return True


def _filter_session(obj: dict, session) -> tuple[list[dict], list[list], list[list], int]:
    """解析+grounding 过滤一个 session 的抽取结果。返回 (facts, entities, rels, dropped)。"""
    blob = _fold("".join(f"{t.speaker}{t.text}" for t in session.turns))
    valid_dia = {t.dia_id for t in session.turns}
    facts: list[dict] = []
    dropped = 0
    seen_stmt: set[tuple[str, str]] = set()
    for raw in (obj.get("f") or [])[:_MAX_FACTS_PER_SESSION]:
        f = _norm_fact(raw, session.no)
        if f is None:
            dropped += 1
            continue
        key = (_fold(f["s"]), _fold(f["t"]))
        if key in seen_stmt:
            continue
        if _ground_fact(f, blob, valid_dia):
            seen_stmt.add(key)
            facts.append(f)
        else:
            dropped += 1
    ents: list[list] = []
    for e in obj.get("e") or []:
        if not isinstance(e, (list, tuple)) or len(e) < 2:
            continue
        etype, name = str(e[0]).strip(), str(e[1]).strip()
        alias = str(e[2]).strip() if len(e) > 2 else ""
        if etype in EXTRACTABLE_ETYPES and name and _fold(name) in blob:
            ents.append([etype, name, alias])
    rels: list[list] = []
    ent_names = {e[1] for e in ents}
    for r in obj.get("r") or []:
        if not isinstance(r, (list, tuple)) or len(r) < 3:
            continue
        rel, a, b = str(r[0]).strip(), str(r[1]).strip(), str(r[2]).strip()
        if rel in EXTRACTABLE_RELATIONS and a and b and a != b \
                and a in ent_names and b in ent_names:
            rels.append([rel, a, b])
    return facts, ents, rels, dropped


# ---------------------------------------------------------------- 主流程
async def build_graph_for(conv: Conversation, schema: Schema, lc: LocomoConfig,
                          client: LLMClient, topics: list[str] | None = None) -> BuildResult:
    topics = topics or DEFAULT_TOPICS
    fp = graph_fingerprint(conv, schema, lc, topics)
    gdir = lc.conv_dir(conv.sample_id) / f"graph_{fp}"
    gpath = gdir / "graph.json"
    facts_path = gdir / "facts.jsonl"
    if gpath.exists() and facts_path.exists():
        facts = [FactRecord(**json.loads(line)) for line in facts_path.read_text().splitlines()]
        return BuildResult(graph_dir=gdir, stats=json.loads((gdir / "stats.json").read_text()),
                           facts=facts)

    # 1) 逐 session 抽取（client 内部信号量控并发）
    async def _one(session):
        chunks = _chunk_turns(session.turns)
        facts, ents, rels, dropped = [], [], [], 0
        for ch in chunks:
            sub = type(session)(no=session.no, date_iso=session.date_iso,
                                date_raw=session.date_raw, turns=ch)
            obj = await _extract_one(client, conv.sample_id, render_extract_prompt(sub, conv, topics))
            if obj is None:
                dropped += 1
                continue
            f2, e2, r2, d2 = _filter_session(obj, sub)
            _override_dates(f2, sub)
            facts += f2
            ents += e2
            rels += r2
            dropped += d2
            # 二道审计：对照原文补漏（同契约同 grounding；只补不重写）
            if lc.audit_extract:
                aobj = await _audit_one(client, conv, sub, f2)
                if aobj:
                    af, ae, ar, ad = _filter_session(aobj, sub)
                    _override_dates(af, sub)
                    seen = {_fold(x["t"]) for x in facts}
                    facts += [x for x in af if _fold(x["t"]) not in seen]
                    ents += ae
                    rels += ar
                    dropped += ad
        return facts, ents, rels, dropped

    results = await asyncio.gather(*[_one(s) for s in conv.sessions])
    all_facts = [f for r in results for f in r[0]]
    all_ents = [e for r in results for e in r[1]]
    all_rels = [r for res in results for r in res[2]]
    dropped_total = sum(r[3] for r in results)

    # 第二层语料（确定性，零 LLM，零翻译）：数据集自带已翻译的 observation（逐轮观察句，
    # 带主体与 dia_id）与 event_summary（带日期的事件句）直接转原子事实。语料所有方指示
    # 并入；评测报告必须披露语料含该层。
    for dia_id, spk, stmt in conv.observations:
        m = re.match(r"D(\d+):", dia_id)
        if not m or not stmt.strip():
            continue
        all_facts.append({"s": spk, "t": stmt.strip(), "y": "背景",
                          "d": "", "g": "无", "o": "", "n": "", "tp": [],
                          "ev": [], "src": [dia_id], "session_no": int(m.group(1))})
    for d_iso, spk, ev in conv.events:
        if not ev.strip():
            continue
        all_facts.append({"s": spk, "t": ev.strip(), "y": "事件",
                          "d": d_iso, "g": "日" if d_iso else "无", "o": "", "n": "",
                          "tp": [], "ev": [], "src": [], "session_no": 0})

    # 2) 实体归并
    names = ([e[1] for e in all_ents] + [e[2] for e in all_ents if e[2]]
             + [f["s"] for f in all_facts] + [x for f in all_facts for x in f["ev"]]
             + [x for r in all_rels for x in r[1:]])
    canon = await canonicalize(sorted(set(names)), conv.speakers, client, conv.sample_id)

    def _c(n: str) -> str:
        return canon.get(n, n)

    all_ents = [[e[0], _c(e[1]), _c(e[2]) if e[2] else ""] for e in all_ents]
    for f in all_facts:
        f["s"] = _c(f["s"])
        f["ev"] = [_c(x) for x in f["ev"]]
    all_rels = [[r[0], _c(r[1]), _c(r[2])] for r in all_rels]

    # 别名消毒（iter2 事故：抽取把"梅尔"错标为卡罗琳别名，归并后变"梅拉妮"，
    # 毒化全部按梅拉妮的主体查询）：别名不得等于任何说话人名或其他实体的规范名
    speaker_set = set(conv.speakers)
    ent_canon_names = {e[1] for e in all_ents} | set(canon.values())
    all_ents = [[e[0], e[1], ""] if (e[2] and (e[2] in speaker_set
                                               or (e[2] in ent_canon_names and e[2] != e[1])))
                else e for e in all_ents]

    # 全局去重（主体+陈述 归一化后相同只留一条；层内/跨层都适用）
    dedup_facts: list[dict] = []
    seen_all: set[tuple[str, str]] = set()
    for f in all_facts:
        key = (_fold(f["s"]), _fold(f["t"]))
        if key in seen_all:
            continue
        seen_all.add(key)
        dedup_facts.append(f)
    all_facts = dedup_facts

    # 3) 组装实体候选
    entities: list[EntityCandidate] = []
    relations: list[RelationCandidate] = []
    ent_by_name: dict[str, tuple[str, list]] = {}
    for etype, name, alias in all_ents:
        if name in ent_by_name:
            if alias and alias not in ent_by_name[name][1] + [name]:
                ent_by_name[name][1].append(alias)
            continue
        ent_by_name[name] = (etype, [alias] if alias and alias != name else [])
    for name, (etype, aliases) in ent_by_name.items():
        pk = "姓名" if etype == "人物" else "名称"
        entities.append(EntityCandidate(
            etype=etype, key={pk: name},
            properties={"别名": ";".join(aliases)}, chunk_id=conv.sample_id))
    for s in conv.speakers:                       # 说话人保底入图
        if s not in ent_by_name or ent_by_name[s][0] != "人物":
            entities.append(EntityCandidate(
                etype="人物", key={"姓名": s},
                properties={"身份": f"说话人{'甲' if s == conv.speaker_a else '乙'}"},
                chunk_id=conv.sample_id))
        else:
            for ec in entities:
                if ec.etype == "人物" and ec.key.get("姓名") == s:
                    ec.properties["身份"] = f"说话人{'甲' if s == conv.speaker_a else '乙'}"

    # 4) 事实节点 + 派生边
    person_names = {ec.key["姓名"] for ec in entities if ec.etype == "人物"}
    activity_names = {ec.key["名称"] for ec in entities if ec.etype == "活动"}
    conv_num = conv.sample_id.replace("conv-", "")
    facts: list[FactRecord] = []
    fids_seen: set[str] = set()
    for i, f in enumerate(all_facts, 1):
        fid = f"{conv_num}-{i:04d}"
        while fid in fids_seen:
            i += 1
            fid = f"{conv_num}-{i:04d}"
        fids_seen.add(fid)
        fr = FactRecord(fid=fid, subject=f["s"], statement=f["t"], ftype=f["y"],
                        date_iso=f["d"], granularity=f["g"], date_raw=f["o"],
                        value=f["n"], topics=f["tp"], sources=f["src"],
                        mentions=f["ev"], session_no=f["session_no"])
        facts.append(fr)
        entities.append(EntityCandidate(
            etype="原子事实", key={"编号": fid},
            properties={k: v for k, v in fr.row().items() if k != "编号"},
            chunk_id=conv.sample_id))
        for tp in fr.topics:
            entities.append(EntityCandidate(
                etype="主题", key={"名称": tp}, properties={}, chunk_id=conv.sample_id))
            relations.append(RelationCandidate(
                relation="属于主题", head=("原子事实", {"编号": fid}),
                tail=("主题", {"名称": tp})))
        if fr.subject in person_names:
            relations.append(RelationCandidate(
                relation="归属于", head=("原子事实", {"编号": fid}),
                tail=("人物", {"姓名": fr.subject})))
        m = re.match(r"D(\d+):", fr.sources[0]) if fr.sources else None
        if m:
            relations.append(RelationCandidate(
                relation="记录于", head=("原子事实", {"编号": fid}),
                tail=("会话", {"序号": int(m.group(1))})))
        for ev in fr.mentions:
            if ev in person_names:
                relations.append(RelationCandidate(
                    relation="涉及人物", head=("原子事实", {"编号": fid}),
                    tail=("人物", {"姓名": ev})))
            elif ev in ent_by_name:
                rel = {"地点": "涉及地点", "组织": "涉及组织",
                       "物品": "涉及物品", "活动": "涉及活动"}[ent_by_name[ev][0]]
                relations.append(RelationCandidate(
                    relation=rel, head=("原子事实", {"编号": fid}),
                    tail=(ent_by_name[ev][0], {"名称": ev})))
        # 喜好派生：偏好类事实 + 正向态度 + 涉及活动
        if fr.ftype == "偏好" and fr.subject in person_names and any(
                w in fr.statement for w in ("喜欢", "爱", "享受", "热爱", "着迷", "痴迷")):
            for ev in fr.mentions:
                if ev in activity_names:
                    relations.append(RelationCandidate(
                        relation="喜爱", head=("人物", {"姓名": fr.subject}),
                        tail=("活动", {"名称": ev})))
                    break
    for s in conv.sessions:
        entities.append(EntityCandidate(
            etype="会话", key={"序号": s.no},
            properties={"日期": s.date_iso.isoformat() if s.date_iso else "",
                        "星期": "一二三四五六日"[s.date_iso.weekday()] if s.date_iso else ""},
            chunk_id=conv.sample_id))
    for rel, a, b in all_rels:
        if a not in person_names and a not in ent_by_name:
            continue
        if b not in person_names and b not in ent_by_name:
            continue
        relations.append(RelationCandidate(
            relation=rel,
            head=("人物", {"姓名": a}) if a in person_names
            else (ent_by_name[a][0], {"名称": a}),
            tail=("人物", {"姓名": b}) if b in person_names
            else (ent_by_name[b][0], {"名称": b})))

    g = build_graph(entities, relations, schema)

    # 5) 落盘 + 统计（evidence 覆盖率为事后诊断，不进图）
    gdir.mkdir(parents=True, exist_ok=True)
    save_graph(g, gpath)
    facts_path.write_text("\n".join(json.dumps(f.__dict__, ensure_ascii=False) for f in facts))
    stats = graph_stats(g)
    stats.update({"sessions": len(conv.sessions), "facts": len(facts),
                  "ungrounded_dropped": dropped_total,
                  "entities_canonical": len(ent_by_name),
                  "canon_map_size": len(canon)})
    covered = set()
    for f in facts:
        covered.update(f.sources)
    stats["evidence_coverage"] = round(len(covered & {
        d for qa in conv.qas for d in qa.evidence}) / max(
        len({d for qa in conv.qas for d in qa.evidence}), 1), 3)
    (gdir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    return BuildResult(graph_dir=gdir, stats=stats, facts=facts)
