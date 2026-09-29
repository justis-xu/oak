"""共享执行器：schema 起草（P1/P2）→ 建图 → 作答（断点续跑）→ 判题 → 归因。

run_anchor 与 run_full 共用 eval_conversation()。
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from oak.kg.graph import load_graph
from oak.llm.client import LLMClient
from oak.schema.model import Schema

from .agent import QAOutput, run_qa
from .analyze import attribute_failures, write_failures
from .build import build_graph_for
from .config import LocomoConfig, ns
from .data import Conversation, load_conversation
from .judge import grade_all
from .prompts.schema_draft import render_p1, render_p2
from .schema_skeleton import DEFAULT_TOPICS, SKELETON_YAML, load_skeleton, merge_schema
from .tools import ToolBox

P1_SYSTEM = "你是本体需求分析师，只输出合法 JSON。"
P2_SYSTEM = "你是本体工程师，只输出一个合法 YAML 代码块。"


async def ensure_schema(conv: Conversation, lc: LocomoConfig, client: LLMClient
                        ) -> tuple[Schema, list[str]]:
    """冻结 schema 优先；否则 P1（主题词表）→ P2（扩展）→ 缓存。"""
    frozen = lc.runs_dir / "frozen" / "schema.yaml"
    if frozen.exists():
        schema = Schema.from_yaml(frozen.read_text())
        topics = json.loads((lc.runs_dir / "frozen" / "topics.json").read_text())
        return schema, topics

    cache = lc.runs_dir / "schema_cache.json"
    if cache.exists():
        obj = json.loads(cache.read_text())
        return Schema.from_yaml(obj["yaml"]), obj["topics"]

    # ---- P1：需求分析（只见问题文本+原文抽样，禁 gold）----
    from collections import Counter
    cat_dist = Counter(qa.cat_name for qa in conv.qas)
    sample_q = [qa.question for qa in conv.qas]
    sample_q = sample_q[:80] + sample_q[len(sample_q) // 2:len(sample_q) // 2 + 40]
    transcript = "\n".join(
        f"[{t.dia_id}] {t.speaker}: {t.text}"
        for s in conv.sessions[:2] for t in s.turns[:12])
    p1 = await client.chat(
        role="locomo_schema", temperature=0.3, max_tokens=2048, json_mode=True,
        namespace=ns(conv.sample_id, "schema"),
        messages=[{"role": "system", "content": P1_SYSTEM},
                  {"role": "user", "content": render_p1(sample_q, dict(cat_dist), transcript)}])
    topics: list[str] = list(DEFAULT_TOPICS)
    try:
        m = re.search(r"\{.*\}", p1.content, re.S)
        obj = json.loads(m.group(0)) if m else {}
        p1_topics = [str(t).strip() for t in (obj.get("topics") or []) if str(t).strip()]
        if 10 <= len(p1_topics) <= 40:
            topics = p1_topics
    except Exception:
        pass

    # ---- P2：骨架扩展（可选；校验失败自动回退骨架）----
    schema = load_skeleton()
    try:
        p2 = await client.chat(
            role="locomo_schema", temperature=0.3, max_tokens=4096,
            namespace=ns(conv.sample_id, "schema"),
            messages=[{"role": "system", "content": P2_SYSTEM},
                      {"role": "user", "content": render_p2(SKELETON_YAML, p1.content)}])
        m = re.search(r"```yaml\n(.*?)```", p2.content, re.S) or \
            re.search(r"```\n(.*?)```", p2.content, re.S)
        if m:
            schema = merge_schema(m.group(1))
    except Exception:
        pass

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"yaml": schema.to_yaml(), "topics": topics,
                                 "p1": p1.content[:2000]}, ensure_ascii=False, indent=2))
    return schema, topics


async def answer_all(conv: Conversation, toolbox: ToolBox, client: LLMClient,
                     lc: LocomoConfig, out_dir: Path,
                     idx_filter: set[int] | None = None) -> dict[int, QAOutput]:
    """逐题作答，answers.jsonl + checkpoint.json 断点续跑。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_path = out_dir / "answers.jsonl"
    ckpt_path = out_dir / "checkpoint.json"
    outputs: dict[int, QAOutput] = {}
    done: set[int] = set()
    if ckpt_path.exists() and answers_path.exists():
        done = set(json.loads(ckpt_path.read_text()).get("done", []))
        for line in answers_path.read_text().splitlines():
            try:
                o = json.loads(line)
                outputs[o["idx"]] = o
            except Exception:
                pass

    todo = [qa for qa in conv.qas
            if qa.idx not in done and (idx_filter is None or qa.idx in idx_filter)]
    sem = asyncio.Semaphore(4)

    async def _one(qa):
        async with sem:
            try:
                o = await run_qa(qa.idx, qa.question, conv.header(), toolbox,
                                 client, lc, conv.sample_id)
            except Exception as e:  # noqa: BLE001
                o = QAOutput(idx=qa.idx, question=qa.question,
                             answer=f"对话中未提及该信息", refused=True,
                             trajectory={"error": repr(e)[:300]})
            async with _lock:
                outputs[qa.idx] = o
                done.add(qa.idx)
                with answers_path.open("a") as f:
                    f.write(json.dumps({
                        "idx": o.idx, "question": o.question, "answer": o.answer,
                        "evidence": o.evidence, "refused": o.refused,
                        "n_steps": o.n_steps, "collected": o.collected,
                        "trajectory": o.trajectory,
                    }, ensure_ascii=False) + "\n")
                ckpt_path.write_text(json.dumps({"done": sorted(done)}))
            if len(done) % 20 == 0:
                print(f"  [{conv.sample_id}] {len(done)}/{len(conv.qas)} 题")

    _lock = asyncio.Lock()
    await asyncio.gather(*[_one(qa) for qa in todo])
    out_map: dict[int, QAOutput] = {}
    for i, o in outputs.items():
        if isinstance(o, QAOutput):
            out_map[i] = o
        else:
            out_map[i] = QAOutput(
                idx=i, question=o["question"], answer=o["answer"],
                evidence=o.get("evidence", []), refused=o.get("refused", False),
                n_steps=o.get("n_steps", 0), collected=o.get("collected", []),
                trajectory=o.get("trajectory", {}))
    return out_map


async def eval_conversation(lc: LocomoConfig, client: LLMClient, conv_id: str,
                            tag: str | None = None, idx_filter: set[int] | None = None,
                            ) -> dict:
    """单对话全流程。返回 report（含 attribution）。"""
    conv = load_conversation(lc.dataset_path, conv_id)
    schema, topics = await ensure_schema(conv, lc, client)
    print(f"[{conv_id}] schema 就绪（{len(topics)} 主题）")
    result = await build_graph_for(conv, schema, lc, client, topics)
    print(f"[{conv_id}] 图就绪: {result.stats.get('facts')} 事实 / "
          f"{result.stats.get('n_nodes')} 节点 / 覆盖率 {result.stats.get('evidence_coverage')}")
    g = load_graph(result.graph_dir / "graph.json")
    toolbox = ToolBox(g)

    tags = sorted(p.name for p in lc.conv_dir(conv_id).glob("iter*"))
    tag = tag or f"iter{len(tags) + 1}"
    out_dir = lc.conv_dir(conv_id) / tag
    print(f"[{conv_id}] 作答开始 → {out_dir.name}")
    outputs = await answer_all(conv, toolbox, client, lc, out_dir, idx_filter)
    preds = {i: o.answer for i, o in outputs.items()}
    qas = [qa for qa in conv.qas if qa.idx in preds]
    report = await grade_all(qas, preds, client, conv_id)
    failures = attribute_failures(conv, report, outputs, result.facts)
    write_failures(out_dir, failures, report)
    print(f"[{conv_id}] {tag}: exact {report['exact']}/{report['n']}"
          f" = {report['exact_rate']:.1%} | J口径 {report['j_exact_rate']:.1%}"
          f" | F1 {report['f1_avg']:.3f}")
    for cat, v in report["by_category"].items():
        print(f"    {cat}: {v['exact']}/{v['n']} = {v['exact_rate']:.1%}")
    return report
