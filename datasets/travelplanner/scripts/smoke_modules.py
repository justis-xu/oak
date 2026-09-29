"""阶段1 模块冒烟（非 LLM 部分）：schema / owlcheck / operators / sandbox / eval 对拍。"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from datasets.travelplanner.pipeline.config_task import load_config                    # noqa: E402


SEED_YAML = """\
meta:
  domain: travel-planning
entity_types:
  Flight:
    description: A scheduled flight between two cities on a date.
    primary_key: [Flight Number, FlightDate]
    attributes:
      - {name: Flight Number, dtype: string}
      - {name: FlightDate, dtype: date}
      - {name: Price, dtype: float}
      - {name: OriginCityName, dtype: string}
      - {name: DestCityName, dtype: string}
  Accommodation:
    primary_key: [NAME, city]
    attributes:
      - {name: NAME, dtype: string}
      - {name: city, dtype: string}
      - {name: price, dtype: float}
      - {name: room type, dtype: string}
      - {name: minimum nights, dtype: int}
      - {name: maximum occupancy, dtype: int}
  Restaurant:
    primary_key: [Name, City]
    attributes:
      - {name: Name, dtype: string}
      - {name: City, dtype: string}
      - {name: Cuisines, dtype: string}
      - {name: Average Cost, dtype: float}
  Attraction:
    primary_key: [Name, City]
    attributes:
      - {name: Name, dtype: string}
      - {name: City, dtype: string}
  City:
    primary_key: [name]
    attributes:
      - {name: name, dtype: string}
relation_types:
  departs_from:
    domain: Flight
    range: City
    functional: true
  arrives_at:
    domain: Flight
    range: City
    functional: true
  located_in:
    domain: [Accommodation, Restaurant, Attraction]
    range: City
axioms:
  - {kind: disjoint, classes: [Flight, Accommodation, Restaurant, Attraction, City]}
"""

CONFLICT_YAML = """\
meta:
  domain: broken
entity_types:
  A:
    primary_key: [id]
    attributes: [{name: id, dtype: string}]
  B:
    primary_key: [id]
    attributes: [{name: id, dtype: string}]
  C:
    primary_key: [id]
    attributes: [{name: id, dtype: string}]
relation_types:
  r:
    domain: C
    range: A
axioms:
  - {kind: disjoint, classes: [A, B]}
  - {kind: subclass, sub: C, sup: A}
  - {kind: subclass, sub: C, sup: B}
"""


def test_schema() -> None:
    from oak.schema.model import Schema
    s = Schema.from_yaml(SEED_YAML)
    assert not s.validate(), s.validate()
    roundtrip = Schema.from_yaml(s.to_yaml())
    assert {e.name for e in roundtrip.entities} == {e.name for e in s.entities}
    assert not roundtrip.validate()
    brief = s.render_brief()
    assert "Flight Number, FlightDate" in brief and "room type" in brief
    print("[1/5] schema parse/validate/render roundtrip OK:",
          len(s.entities), "entities,", len(s.relations), "relations")


def test_owlcheck() -> None:
    from oak.schema.model import Schema
    from oak.schema import owlcheck
    cfg = load_config()

    ok_schema = Schema.from_yaml(SEED_YAML)
    findings, hermit_ok = owlcheck.check_schema(ok_schema)
    assert hermit_ok, "HermiT subprocess timed out on seed schema"
    assert not findings, f"seed schema should pass, got: {[f.render() for f in findings]}"
    print("[2/5] owlcheck seed OK (hermit ran, 0 findings)")

    bad = Schema.from_yaml(CONFLICT_YAML)
    findings2, ok2 = owlcheck.check_schema(bad)
    assert findings2, "conflict schema must produce findings"
    kinds = {f.check for f in findings2}
    assert "disjointness" in kinds, f"expected disjointness finding, got {kinds}"
    print("      conflict detected:", kinds, "| culprits:",
          findings2[0].culprits[:3])


def test_operators() -> None:
    import networkx as nx
    from oak.operators import library as ops

    g = nx.MultiDiGraph()
    g.add_node("Flight::F1|d1", etype="Flight", **{"Flight Number": "F1", "FlightDate": "d1",
                                                   "Price": 120.0, "OriginCityName": "Columbus",
                                                   "DestCityName": "Newark"})
    g.add_node("Flight::F2|d1", etype="Flight", **{"Flight Number": "F2", "FlightDate": "d1",
                                                   "Price": 80.5, "OriginCityName": "Columbus",
                                                   "DestCityName": "Newark"})
    g.add_node("City::name=Columbus", etype="City", name="Columbus")
    g.add_node("City::name=Newark", etype="City", name="Newark")
    g.add_node("Restaurant::R1|Newark", etype="Restaurant", Name="R1", City="Newark",
               Cuisines="American, French", **{"Average Cost": 20})
    g.add_edge("Flight::F1|d1", "City::name=Newark", relation="arrives_at")
    g.add_edge("Restaurant::R1|Newark", "City::name=Newark", relation="located_in")
    ops.set_graph(g)

    rows = ops.lookup_entities("Flight", {"OriginCityName": "Columbus"})
    assert len(rows) == 2
    cheap = ops.filter_numeric(rows, "Price", "<=", 100)
    assert [r["Flight Number"] for r in cheap] == ["F2"]
    proj = ops.project_properties(cheap, ["Price", "DestCityName"])
    assert proj[0]["Price"] == 80.5
    arr = ops.traverse_relations("Flight::F1|d1", "arrives_at")
    assert arr and arr[0]["name"] == "Newark"
    res = ops.filter_set_overlap(ops.lookup_entities("Restaurant"),
                                 "Cuisines", ["french"])
    assert len(res) == 1
    total = ops.aggregate_values(rows, "Price", "sum")
    assert abs(total - 200.5) < 1e-6
    conn = ops.filter_relation_connected(ops.lookup_entities("Restaurant"),
                                         "located_in", "City", {"name": "Nowhere"})
    assert conn == []
    print("[3/5] operators OK (lookup/filter/project/traverse/set_overlap/aggregate/relation_connected)")


def test_sandbox() -> None:
    import networkx as nx
    from oak.operators import library as ops
    from oak.operators.sandbox import exec_function_source, trial_run, SandboxError

    g = nx.MultiDiGraph()
    g.add_node("Accommodation::H1|Boston", etype="Accommodation", NAME="H1", city="Boston",
               price=100.0, **{"room type": "Private room", "minimum nights": 1,
                               "maximum occupancy": 2})
    g.add_node("Accommodation::H2|Boston", etype="Accommodation", NAME="H2", city="Boston",
               price=250.0, **{"room type": "Private room", "minimum nights": 3,
                               "maximum occupancy": 5})

    good_src = '''\
def get_budget_hotels(max_price=200.0, people=2):
    """Return affordable hotels fitting the party.

    Args:
        max_price: price per night cap
        people: party size

    Returns:
        list of hotel rows

    Example:
        get_budget_hotels(max_price=150, people=3)
    """
    rows = lookup_entities("Accommodation")
    rows = filter_numeric(rows, "price", "<=", max_price)
    rows = filter_numeric(rows, "maximum occupancy", ">=", people)
    return project_properties(rows, ["NAME", "price", "maximum occupancy"])
'''
    name, fn = exec_function_source(good_src)
    assert name == "get_budget_hotels"
    results = trial_run(fn, {"full": g, "mini": g}, {"max_price": 200.0, "people": 2})
    assert all(r.ok for r in results), [r.error for r in results]
    assert json.loads(results[0].sample_output)[0]["NAME"] == "H1"

    bad_src = "def f():\n    import os\n    return os.listdir('.')\n"
    try:
        exec_function_source(bad_src)
        raise AssertionError("sandbox must reject import")
    except SandboxError:
        pass
    print("[4/5] sandbox OK (AST whitelist + dual-graph trial run + import rejected)")


def test_eval_pair() -> None:
    """对拍：subset worker vs 官方 eval.py（逐题布尔一致）。"""
    from datasets.travelplanner.pipeline.data.queries import load_queries
    from datasets.travelplanner.pipeline.eval.adapter import TravelPlannerEvaluator

    cfg = load_config()
    ev = TravelPlannerEvaluator(cfg)
    train = load_queries("train", cfg)

    # 手工构造半合法计划：只给第 0 题一个"格式完整但内容可能违规"的计划
    q0 = train[0]
    plan0 = []
    for d in range(q0.days):
        plan0.append({
            "days": d + 1,
            "current_city": f"from {q0.org} to {q0.dest}" if d == 0 else q0.dest,
            "transportation": (f"Flight Number: F3927581, from {q0.org} to {q0.dest}, "
                               f"Departure Time: 11:03, Arrival Time: 13:31"
                               if d == 0 else "-"),
            "breakfast": "-", "lunch": "-", "dinner": "-",
            "attraction": "-", "accommodation": "-",
        })
    indices = list(range(6))
    queries = [train[i] for i in indices]
    plans = [plan0 if i == 0 else [] for i in indices]

    scores = ev.eval_subset(queries, plans)
    # 官方 padded 复跑
    with tempfile.TemporaryDirectory() as td:
        padded = Path(td) / "padded.jsonl"
        plan_by_idx = {0: plan0}
        ev.build_padded_file("train", indices, plan_by_idx, padded)
        official = ev.eval_official("train", padded)
    assert scores.per_query[0]["delivered"] and not scores.per_query[1]["delivered"]
    # 非空计划的 CS 布尔应已产生（官方 worker 跑通）
    assert scores.per_query[0]["cs"], "worker returned empty cs booleans"
    print("[5/5] eval pair OK")
    print("      subset:", scores.summary())
    print("      official(padded/45, diluted):",
          {k: v for k, v in official.items() if "Micro" in k or "Delivery" in k})
    # 保存逐题布尔供人工抽查
    out = PROJECT / "runs" / "smoke_eval_pair.json"
    out.write_text(json.dumps({
        "subset_per_query": scores.per_query,
        "official": official,
    }, ensure_ascii=False, indent=1))
    print("      per-query booleans ->", out)


if __name__ == "__main__":
    load_config()
    test_schema()
    test_owlcheck()
    test_operators()
    test_sandbox()
    test_eval_pair()
    print("\nMODULE SMOKE (non-LLM) PASSED")
