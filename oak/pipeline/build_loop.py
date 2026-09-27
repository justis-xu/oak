"""构建循环编排：5 轮 × 四步，state.json 断点续跑，函数修补重测，冻结。"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from ..config import Config, load_config
from ..llm.client import LLMClient
from ..data.queries import Query, load_queries, partition_train
from ..data.corpus import reference_chunks, distance_chunks
from ..schema.model import Schema
from ..schema.builder import build_schema
from ..kg.extract import extract_graph_from_chunks
from ..kg.graph import (build_graph, save_graph, load_graph, graph_stats,
                        graph_samples, derive_relations, enrich_city_nodes)
from ..funcs.catalog import FunctionCatalog
from ..funcs.compiler import compile_functions, set_trial_graphs
from ..agent.react import run_react
from ..agent.planner import finalize_plan, to_plan_record
from ..eval.adapter import TravelPlannerEvaluator
from ..judge.judicator import judge, route_feedback, JudgeInputs

log = logging.getLogger("oak.build")

STAGES = ["schema", "kg", "funcs", "react", "eval", "judge", "patch_rerun"]


def _state_path(round_dir: Path) -> Path:
    return round_dir / "state.json"


def _load_state(round_dir: Path) -> dict:
    p = _state_path(round_dir)
    return json.loads(p.read_text()) if p.exists() else {}


def _mark(round_dir: Path, stage: str) -> None:
    st = _load_state(round_dir)
    st[stage] = "done"
    _state_path(round_dir).write_text(json.dumps(st))


def _done(round_dir: Path, stage: str) -> bool:
    return _load_state(round_dir).get(stage) == "done"


async def _build_kg(client, cfg, schema, queries: list[Query], round_dir: Path, ns: str):
    """本轮全部题的语料 → 一个图；同时保留第一题的 mini 图（函数试跑用）。"""
    all_chunks = []
    first_q_chunks = []
    for qi, q in enumerate(queries):
        chs = reference_chunks(q.reference_information)
        if qi == 0:
            first_q_chunks = list(chs)
        all_chunks.extend(chs)

    # 距离矩阵是结构化源：程序化直建（不走 LLM），City 边由 derive 补
    from ..kg.graph import programmatic_distance_entities
    dist_entities, dist_relations = [], []
    if cfg.include_distance_matrix_corpus:
        dist_entities = programmatic_distance_entities(schema, cfg.tp_root)

    entities, relations, stats = await extract_graph_from_chunks(
        client, cfg, schema, all_chunks, ns)
    entities = entities + dist_entities
    relations = relations + dist_relations
    g = enrich_city_nodes(
        derive_relations(build_graph(entities, relations, schema), schema),
        cfg.tp_root)
    save_graph(g, round_dir / "graph.json")
    (round_dir / "extraction_stats.json").write_text(json.dumps({
        "chunks_total": stats.chunks_total, "parse_ok": stats.parse_ok,
        "repaired": stats.repaired, "dropped": stats.dropped,
        "entities": stats.entities, "relations": stats.relations,
        "invalid_refs": stats.invalid_refs, "errors": stats.errors[:20],
        "dist_entities": len(dist_entities),
    }, ensure_ascii=False, indent=1))
    # mini 图：第一题的 chunks 单独建图（复用全量抽取结果，按 chunk_id 过滤）
    first_ids = {f"{c.source}/{c.seq}" for c in first_q_chunks}
    mini_entities = [e for e in entities if e.chunk_id in first_ids]
    mini_relations = [r for r in relations]
    g_mini = enrich_city_nodes(
        derive_relations(build_graph(mini_entities, mini_relations, schema), schema),
        cfg.tp_root)
    return g, g_mini, stats


async def _run_round_queries(client, cfg, queries, catalog, schema, g, round_dir, ns):
    """9 题 ReAct + 计划规范化修复。返回 (records, recorders)。"""
    module = catalog.module(g)
    sem = asyncio.Semaphore(cfg.max_concurrency)

    async def one(q: Query):
        async with sem:
            res = await run_react(client, cfg, q, catalog, schema, g, module, ns)
        plan, rep, fstats = await finalize_plan(client, cfg, q, res.raw_plan, g, ns)
        res.recorder.set_plan(plan, [i.render() for i in rep.blocking],
                              fstats["repairs"] + fstats["salvage"])
        d = round_dir / "react" / f"q{q.idx}"
        res.recorder.save(d / "trajectory.json")
        return to_plan_record(q.idx, plan), res.recorder, [i.render() for i in rep.blocking]

    outs = await asyncio.gather(*[one(q) for q in queries])
    records = [o[0] for o in outs]
    recorders = [o[1] for o in outs]
    return records, recorders


def _trajectories_digest(queries, recorders, scores) -> str:
    lines = []
    for q, rec, sc in zip(queries, recorders, scores.per_query):
        calls = "; ".join(
            f"{c['fn']}({json.dumps(c['args'], ensure_ascii=False)[:60]})"
            + (f"->{c['n_rows']}rows" if not c.get("error") else "->ERROR")
            for c in rec.call_layer[:10])
        lines.append(
            f"### query idx={q.idx} (level={q.level}, days={q.days}, people={q.people_number})\n"
            f"calls: {calls or '(none)'}\n"
            f"delivered={sc['delivered']} macro_cs={sc['macro_cs']} macro_hc={sc['macro_hc']}\n"
            f"cs_failed={[k for k, v in sc['cs'].items() if v is False]}\n"
            f"hc_failed={[k for k, v in sc['hc'].items() if v is False]}")
    return "\n\n".join(lines)


def _pick_better(records_a, scores_a, records_b, scores_b):
    """逐题取优：delivered > hc 通过数 > cs 通过数。"""
    out_records = []
    for ra, sa, rb, sb in zip(records_a, scores_a.per_query, records_b, scores_b.per_query):
        def key(s):
            hc_ok = sum(1 for v in s["hc"].values() if v is True)
            cs_ok = sum(1 for v in s["cs"].values() if v is True)
            return (s["delivered"], hc_ok, cs_ok)
        out_records.append(rb if key(sb) > key(sa) else ra)
    return out_records


async def run_build(cfg: Config | None = None, rounds: int | None = None) -> dict:
    cfg = cfg or load_config()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    client = LLMClient(cfg)
    evaluator = TravelPlannerEvaluator(cfg)
    n_rounds = rounds or cfg.rounds

    train = load_queries("train", cfg)
    groups = partition_train(cfg.train_per_round, n_rounds, cfg.seed, cfg)
    log.info("round groups: %s", groups)

    prev_schema: Schema | None = None
    prev_catalog: FunctionCatalog | None = None
    psi_s: list[dict] = []
    last_sigma_f: list[dict] = []

    for t in range(1, n_rounds + 1):
        round_dir = cfg.build_dir / f"round_{t}"
        round_dir.mkdir(parents=True, exist_ok=True)
        queries = [train[i] for i in groups[t - 1]]
        ns = f"r{t}"
        log.info("=== ROUND %d: queries %s ===", t, [q.idx for q in queries])

        # ---------- 步骤① 模式 ----------
        if not _done(round_dir, "schema"):
            schema, info = await build_schema(client, cfg, queries, prev_schema,
                                              psi_s, round_dir, ns)
            (round_dir / "schema.yaml").write_text(schema.to_yaml())
            (round_dir / "schema_process.json").write_text(
                json.dumps(info, ensure_ascii=False, indent=1, default=str))
            _mark(round_dir, "schema")
        else:
            schema = Schema.from_yaml((round_dir / "schema.yaml").read_text())

        # ---------- 步骤② KG ----------
        if not _done(round_dir, "kg"):
            g, g_mini, ex_stats = await _build_kg(client, cfg, schema, queries, round_dir, ns)
            log.info("round %d graph: %s", t, graph_stats(g))
            _mark(round_dir, "kg")
        else:
            g = enrich_city_nodes(load_graph(round_dir / "graph.json"), cfg.tp_root)
            g_mini = g    # 断点续跑时 mini 退化为全图（仅试跑用途）

        # ---------- 步骤③ 函数 ----------
        if not _done(round_dir, "funcs"):
            catalog = await compile_functions(
                client, cfg, schema, g, g_mini, queries, last_sigma_f,
                prev_catalog, ns)
            catalog.save(round_dir / "functions")
            log.info("round %d functions: %d published", t, len(catalog.published()))
            _mark(round_dir, "funcs")
        else:
            catalog = FunctionCatalog.load(round_dir / "functions")

        # ---------- 步骤③ 执行 + 步骤④ 评分 ----------
        plans_path = round_dir / "plans.jsonl"
        if not _done(round_dir, "react"):
            records, recorders = await _run_round_queries(
                client, cfg, queries, catalog, schema, g, round_dir, ns)
            plans_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in records))
            _mark(round_dir, "react")
        else:
            records = [json.loads(x) for x in plans_path.read_text().splitlines()]
            recorders = None       # 续跑时轨迹已落盘

        if not _done(round_dir, "eval"):
            plans = [r["plan"] for r in records]
            scores = evaluator.eval_subset(queries, plans)
            (round_dir / "scores.json").write_text(json.dumps(
                {"summary": scores.summary(), "per_query": scores.per_query},
                ensure_ascii=False, indent=1))
            log.info("round %d scores: %s", t, scores.summary())
            _mark(round_dir, "eval")
        else:
            scores = _load_scores(round_dir / "scores.json")

        # ---------- 步骤④ 评判 ----------
        if not _done(round_dir, "judge"):
            if recorders is None:
                recorders = _reload_recorders(round_dir, queries)
            inputs = JudgeInputs(
                schema=schema, graph_stats=graph_stats(g),
                graph_samples=graph_samples(g), catalog=catalog,
                trajectories_digest=_trajectories_digest(queries, recorders, scores),
                scores=scores)
            sigma = await judge(client, cfg, inputs, ns)
            psi_s, sigma_f = route_feedback(sigma)
            (round_dir / "feedback").mkdir(parents=True, exist_ok=True)
            (round_dir / "feedback" / "judgments.json").write_text(
                json.dumps({"all": sigma, "psi_s": psi_s, "sigma_f": sigma_f},
                           ensure_ascii=False, indent=1))
            log.info("round %d judge: psi_s=%d sigma_f=%d", t, len(psi_s), len(sigma_f))
            _mark(round_dir, "judge")
        else:
            fb = json.loads((round_dir / "feedback" / "judgments.json").read_text())
            psi_s, sigma_f = fb["psi_s"], fb["sigma_f"]

        # ---------- 函数修补重测（快回路） ----------
        if sigma_f and not _done(round_dir, "patch_rerun"):
            patched = await compile_functions(
                client, cfg, schema, g, g_mini, queries, sigma_f, catalog, ns)
            patched.save(round_dir / "patches" / "functions")
            records2, _ = await _run_round_queries(
                client, cfg, queries, patched, schema, g, round_dir / "patched_run", ns)
            plans2 = [r["plan"] for r in records2]
            scores2 = evaluator.eval_subset(queries, plans2)
            (round_dir / "scores_patched.json").parent.mkdir(parents=True, exist_ok=True)
            (round_dir / "scores_patched.json").write_text(json.dumps(
                {"summary": scores2.summary(), "per_query": scores2.per_query},
                ensure_ascii=False, indent=1))
            better = _pick_better(records, scores, records2, scores2)
            (round_dir / "round_summary.json").write_text(json.dumps({
                "base": scores.summary(), "patched": scores2.summary(),
            }, indent=1))
            log.info("round %d patched scores: %s", t, scores2.summary())
            # 采纳规则：final>0 时比 final；双方 final=0 时比 blocker 总数
            # （旧规则 final=0 恒真 —— 曾把 9 函数目录劣化成 1 个）
            blockers = sum(sum(1 for v in it["cs"].values() if v is False) +
                           sum(1 for v in it["hc"].values() if v is False)
                           for it in scores.per_query)
            blockers2 = sum(sum(1 for v in it["cs"].values() if v is False) +
                            sum(1 for v in it["hc"].values() if v is False)
                            for it in scores2.per_query)
            # 采纳规则：严格改善才采纳（Final > Macro HC > Macro CS 字典序）；
            # 双方 Final=0 时比 blocker 总数（严格更少）。此前末尾 `>= final` 恒真，
            # 会无条件采纳 tie 候选（曾把 9 函数目录劣化成 1 个）
            adopt = ((scores2.final_cnt, scores2.macro_hc_cnt, scores2.macro_cs_cnt) >
                     (scores.final_cnt, scores.macro_hc_cnt, scores.macro_cs_cnt)) or \
                    (scores2.final_cnt == scores.final_cnt == 0 and blockers2 < blockers)
            if adopt:
                catalog = patched
                records = better
            _mark(round_dir, "patch_rerun")
        else:
            last_sigma_f = []      # 无函数反馈或已跑过

        last_sigma_f = sigma_f if sigma_f else []
        prev_schema = schema
        prev_catalog = catalog

    # ---------- 冻结 ----------
    cfg.final_dir.mkdir(parents=True, exist_ok=True)
    (cfg.final_dir / "schema.yaml").write_text(prev_schema.to_yaml())
    # 冻结保护：published < 4 视为被评判器劣化，回退本轮未打补丁目录
    if len(prev_catalog.published()) < 4:
        log.warning("frozen catalog has only %d published (<4) — "
                    "falling back to pre-patch catalog", len(prev_catalog.published()))
        try:
            prev_catalog = FunctionCatalog.load(cfg.build_dir / f"round_{n_rounds}" / "functions")
        except Exception:
            pass
    prev_catalog.save(cfg.final_dir)
    log.info("frozen kernel -> %s (policy=%s, %d functions)",
             cfg.final_dir, cfg.freeze_policy, len(prev_catalog.published()))
    return {"final_dir": str(cfg.final_dir)}


def _load_scores(path: Path):
    from ..eval.adapter import SubsetScores
    data = json.loads(path.read_text())
    s = SubsetScores(per_query=data["per_query"])
    for k in ("delivery", "micro_cs", "macro_cs", "micro_hc", "macro_hc", "final"):
        setattr(s, k, data["summary"][k])
    return s


def _reload_recorders(round_dir: Path, queries):
    """断点续跑时从落盘轨迹重建摘要（只供 judge digest 用）。"""
    class _R:
        def __init__(self, d):
            self.call_layer = d.get("layers", {}).get("call_layer", [])
    out = []
    for q in queries:
        p = round_dir / "react" / f"q{q.idx}" / "trajectory.json"
        out.append(_R(json.loads(p.read_text()) if p.exists() else {}))
    return out


if __name__ == "__main__":
    import sys
    rounds = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else None
    asyncio.run(run_build(rounds=rounds))
