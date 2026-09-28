"""推理：冻结内核 + 每题现建图 + ReAct → 50 题计划 → subset + 官方 padded 评测。"""
from __future__ import annotations

import asyncio
import json
import logging

from ..config import Config, load_config
from ..llm.client import LLMClient
from ..data.queries import Query, load_queries, stratified_test_subset
from ..data.corpus import reference_chunks, distance_chunks
from ..schema.model import Schema
from ..kg.extract import extract_graph_from_chunks
from ..kg.graph import (build_graph, save_graph, graph_stats, derive_relations,
                        enrich_city_nodes, augment_graph_with_official)
from ..funcs.catalog import FunctionCatalog
from ..agent.react import run_react
from ..agent.planner import finalize_plan, to_plan_record
from ..eval.adapter import TravelPlannerEvaluator

log = logging.getLogger("oak.infer")


async def run_inference(cfg: Config | None = None, limit: int | None = None,
                        anchor: bool = False) -> dict:
    cfg = cfg or load_config()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    client = LLMClient(cfg)
    evaluator = TravelPlannerEvaluator(cfg)

    schema = Schema.from_yaml((cfg.final_dir / "schema.yaml").read_text())
    catalog = FunctionCatalog.load(cfg.final_dir)
    log.info("frozen kernel: %d entities, %d functions",
             len(schema.entities), len(catalog.published()))

    if anchor:
        idx_list = json.loads((cfg.data_dir / "anchor.json").read_text())["anchor_idx"]
        queries_all = load_queries("train", cfg)
        out_dir = cfg.work_dir / "anchor"
    else:
        queries_all = load_queries("validation", cfg)
        idx_list = stratified_test_subset(limit or cfg.test_size, cfg.seed, cfg)
        out_dir = cfg.inference_dir
    queries = [queries_all[i] for i in idx_list]
    log.info("%s queries: %s", "anchor" if anchor else "test", idx_list)

    # 距离矩阵是结构化源：程序化直建（不走 LLM）
    from ..kg.graph import programmatic_distance_entities
    dist_entities = (programmatic_distance_entities(schema, cfg.tp_root)
                     if cfg.include_distance_matrix_corpus else [])
    out_dir.mkdir(parents=True, exist_ok=True)
    plans_by_idx: dict[int, list] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / "checkpoint.json"
    done_idx: set[int] = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    # 恢复已有计划：此前只读 checkpoint 集，汇总评测时 done 题会变回空计划
    plans_path_restore = out_dir / "plans.validation.jsonl"
    if plans_path_restore.exists():
        for line in plans_path_restore.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    plans_by_idx[r["query_idx"]] = r["plan"]
                except Exception:
                    pass

    sem = asyncio.Semaphore(cfg.max_concurrency)

    async def one(q: Query):
        if q.idx in done_idx:
            return
        ns = f"inf_q{q.idx}"
        # 每题现建图（冻结模式 + 该题语料）
        chunks = reference_chunks(q.reference_information)
        entities, relations, stats = await extract_graph_from_chunks(
            client, cfg, schema, chunks, ns)
        g = enrich_city_nodes(
            derive_relations(
                build_graph(entities + dist_entities, relations, schema),
                schema),
            cfg.tp_root)
        # 【第五轮回滚】官方库补图（augment_graph_with_official）经 anchor 三轮验证为
        # 净负改动：7/9 → 6/9 → 3/9。为救 q47/q116 两题把几百实体灌进图，改变了
        # ReAct 全局行为分布（大候选池下 flash 不稳定），第二轮过的题反而丢 11 题。
        # 函数保留在 kg/graph.py 供后续研究，此处停用。
        # n_aug = augment_graph_with_official(g, q, cfg.tp_root)
        # if n_aug:
        #     derive_relations(g, schema)
        #     enrich_city_nodes(g, cfg.tp_root)
        qdir = out_dir / f"q{q.idx}"
        save_graph(g, qdir / "graph.json")
        (qdir / "extraction_stats.json").write_text(json.dumps({
            "chunks_total": stats.chunks_total, "parse_ok": stats.parse_ok,
            "repaired": stats.repaired, "dropped": stats.dropped,
            "entities": stats.entities, "relations": stats.relations,
            "ungrounded": stats.ungrounded, "empty_retried": stats.empty_retried,
            "invalid_refs": stats.invalid_refs,
        }, ensure_ascii=False, indent=1))

        module = catalog.module(g)
        res = await run_react(client, cfg, q, catalog, schema, g, module, ns)
        plan, rep, fstats = await finalize_plan(client, cfg, q, res.raw_plan, g, ns)
        res.recorder.set_plan(plan, [i.render() for i in rep.blocking],
                              fstats["repairs"] + fstats["salvage"])
        res.recorder.save(qdir / "trajectory.json")
        plans_by_idx[q.idx] = plan or []
        log.info("q%d delivered=%s blocking=%d repairs=%d rejected=%d graph_nodes=%d",
                 q.idx, bool(plan), len(rep.blocking), fstats["repairs"],
                 fstats["rejected"], g.number_of_nodes())

    # 分块跑 + 每 5 题落盘 checkpoint
    pending = [q for q in queries if q.idx not in done_idx]
    for i in range(0, len(pending), 5):
        batch = pending[i:i + 5]
        await asyncio.gather(*[_wrap(sem, one, q) for q in batch])
        # 恢复 checkpoint 集
        done_idx |= {q.idx for q in batch if plans_by_idx.get(q.idx) is not None}
        done_path.write_text(json.dumps(sorted(done_idx)))
        # 追加计划文件（幂等：按 idx 重写）
        _write_plans(out_dir, plans_by_idx)

    # 汇总评测
    plans = [plans_by_idx.get(i, []) for i in idx_list]
    test_queries = [queries_all[i] for i in idx_list]
    scores = evaluator.eval_subset(test_queries, plans)
    (out_dir / "scores.json").write_text(json.dumps(
        {"summary": scores.summary(), "per_query": scores.per_query,
         "indices": idx_list}, ensure_ascii=False, indent=1))
    log.info("TEST scores: %s", scores.summary())

    # padded 官方复跑留档
    try:
        padded = out_dir / "plans.validation.official.jsonl"
        evaluator.build_padded_file("validation" if not anchor else "train", idx_list, plans_by_idx, padded)
        official = evaluator.eval_official("train" if anchor else "validation", padded)
        (out_dir / "official_eval_output.json").write_text(json.dumps(official, indent=1))
        log.info("official (padded 180, diluted denominators): %s", official)
    except Exception as e:
        log.warning("official eval rerun failed: %s", e)

    return scores.summary()


async def _wrap(sem, fn, q):
    async with sem:
        await fn(q)


def _write_plans(out_dir, plans_by_idx: dict):
    path = out_dir / "plans.validation.jsonl"
    lines = [json.dumps({"query_idx": i, "plan": p}, ensure_ascii=False)
             for i, p in sorted(plans_by_idx.items())]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    anchor = "--anchor" in sys.argv
    asyncio.run(run_inference(limit=limit, anchor=anchor))
