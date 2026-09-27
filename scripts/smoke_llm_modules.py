"""阶段1 模块冒烟（LLM 部分）：kg 抽取 / react / judge。"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from oak.config import load_config                    # noqa: E402
from scripts.smoke_modules import SEED_YAML           # noqa: E402


async def test_kg() -> None:
    from oak.llm.client import LLMClient
    from oak.data.queries import load_queries
    from oak.data.corpus import reference_chunks
    from oak.schema.model import Schema
    from oak.kg.extract import extract_graph_from_chunks
    from oak.kg.graph import build_graph, graph_stats

    cfg = load_config()
    client = LLMClient(cfg)
    schema = Schema.from_yaml(SEED_YAML)
    train = load_queries("train", cfg)
    q = train[0]
    chunks = reference_chunks(q.reference_information)[:4]     # 只取前 4 块省钱
    entities, relations, stats = await extract_graph_from_chunks(
        client, cfg, schema, chunks, "smoke_kg")
    g = build_graph(entities, relations, schema)
    print(f"[1/3] kg extract OK: chunks={stats.chunks_total} parse_ok={stats.parse_ok} "
          f"repaired={stats.repaired} dropped={stats.dropped} "
          f"entities={stats.entities} relations={stats.relations}")
    print("      graph:", graph_stats(g)["by_type"])
    assert stats.parse_ok + stats.repaired >= stats.chunks_total - 1, \
        f"too many drops: {stats.dropped}"
    assert stats.entities > 0
    # 抽查一个节点的属性保真度（值来自语料）
    sample = [n for n, d in g.nodes(data=True) if d.get("etype") == "Flight"]
    if sample:
        print("      sample flight node:", json.dumps(
            {k: v for k, v in g.nodes[sample[0]].items() if not k.startswith("__")},
            ensure_ascii=False)[:200])


async def test_react() -> None:
    from oak.llm.client import LLMClient
    from oak.data.queries import load_queries
    from oak.data.corpus import reference_chunks
    from oak.schema.model import Schema
    from oak.kg.extract import extract_graph_from_chunks
    from oak.kg.graph import build_graph
    from oak.funcs.catalog import CompiledFunction, FunctionCatalog
    from oak.agent.react import run_react
    from oak.agent.planner import normalize_plan
    from oak.config import Config

    cfg = load_config()
    cfg = Config(**{**cfg.__dict__, "react_max_steps": 10})     # 冒烟限 10 步
    client = LLMClient(cfg)
    schema = Schema.from_yaml(SEED_YAML)
    train = load_queries("train", cfg)
    q = train[0]

    entities, relations, stats = await extract_graph_from_chunks(
        client, cfg, schema, reference_chunks(q.reference_information), "smoke_react_kg")
    g = build_graph(entities, relations, schema)

    # 手写函数目录（冒烟用，两个函数）
    fn1_src = '''\
def search_flights(origin="", dest="", max_price=100000.0):
    """Return flights from origin to dest within max_price.

    Example:
        search_flights(origin="Columbus", dest="Newark")
    """
    rows = lookup_entities("Flight", {"OriginCityName": origin})
    rows = filter_categorical(rows, "DestCityName", [dest])
    rows = filter_numeric(rows, "Price", "<=", max_price)
    return project_properties(rows, ["Flight Number", "OriginCityName", "DestCityName", "Price", "DepTime", "ArrTime"])
'''
    fn2_src = '''\
def get_hotels(city="", max_price=100000.0, people=1):
    """Return hotels in city that fit the party and budget.

    Example:
        get_hotels(city="Newark", max_price=150, people=2)
    """
    rows = lookup_entities("Accommodation", {"city": city})
    rows = filter_numeric(rows, "price", "<=", max_price)
    rows = filter_numeric(rows, "maximum occupancy", ">=", people)
    return project_properties(rows, ["NAME", "city", "price", "room type", "minimum nights", "maximum occupancy"])
'''
    catalog = FunctionCatalog([
        CompiledFunction(name="search_flights",
                         signature="search_flights(origin, dest, max_price=100000.0)",
                         docstring="Return flights from origin to dest within max_price.",
                         source=fn1_src),
        CompiledFunction(name="get_hotels",
                         signature="get_hotels(city, max_price=100000.0, people=1)",
                         docstring="Return hotels in city that fit the party and budget.",
                         source=fn2_src),
    ])
    module = catalog.module(g)
    res = await run_react(client, cfg, q, catalog, schema, g, module, "smoke_react")
    plan, errs = normalize_plan(res.raw_plan, q) if res.raw_plan else (None, ["no plan"])
    print(f"[2/3] react OK: steps={res.usage.get('steps')} delivered={res.delivered} "
          f"calls={len(res.recorder.call_layer)} days={len(plan or [])} fmt_errs={len(errs)}")
    assert res.recorder.call_layer, "no function calls made"
    (PROJECT / "runs" / "smoke_react_trajectory.json").write_text(json.dumps({
        "steps": res.recorder.step_layer, "calls": res.recorder.call_layer,
        "plan": plan, "errors": errs,
    }, ensure_ascii=False, indent=1))


async def test_judge() -> None:
    from oak.llm.client import LLMClient
    from oak.schema.model import Schema
    from oak.funcs.catalog import FunctionCatalog
    from oak.eval.adapter import SubsetScores
    from oak.judge.judicator import judge, route_feedback, JudgeInputs

    cfg = load_config()
    client = LLMClient(cfg)
    schema = Schema.from_yaml(SEED_YAML)
    catalog = FunctionCatalog()
    pair = json.loads((PROJECT / "runs" / "smoke_eval_pair.json").read_text())
    scores = SubsetScores(per_query=pair["subset_per_query"])
    scores.delivery, scores.micro_cs, scores.macro_cs = 0.17, 0.62, 0.0
    scores.micro_hc, scores.macro_hc, scores.final = 0.0, 0.0, 0.0

    inputs = JudgeInputs(
        schema=schema, graph_stats={"n_nodes": 0, "n_edges": 0, "by_type": {},
                                    "by_relation": {}},
        graph_samples={}, catalog=catalog,
        trajectories_digest=(
            "### query idx=0 (level=easy, days=3, people=1)\n"
            "calls: search_flights({\"origin\": \"Fort Lauderdale\"})->12rows\n"
            "delivered=True macro_cs=False macro_hc=False\n"
            "cs_failed=['is_valid_restaurants', 'is_valid_accommodation', 'is_not_absent']\n"
            "hc_failed=[]\n\n"
            "### query idx=1\n(no plan delivered)"),
        scores=scores)
    sigma = await judge(client, cfg, inputs, "smoke_judge")
    psi_s, sigma_f = route_feedback(sigma)
    print(f"[3/3] judge OK: {len(sigma)} items -> psi_s={len(psi_s)} sigma_f={len(sigma_f)}")
    for s in sigma[:3]:
        print(f"      {s['u']}/{s['a']}/{s.get('name')}: {s['rho'][:80]}")
    (PROJECT / "runs" / "smoke_judge.json").write_text(json.dumps(sigma, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    load_config()
    asyncio.run(test_kg())
    asyncio.run(test_react())
    asyncio.run(test_judge())
    print("\nMODULE SMOKE (LLM) PASSED")
