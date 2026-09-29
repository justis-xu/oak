"""旧 50 题离线重放：新终态层（validator + local_repair + budget_downgrade）对旧计划的影响。

不调 LLM；输出 runs/replay/<variant>/，绝不覆盖原结果。
用法: uv run python scripts/replay_plans.py [--plans <path>] [--graphs <dir>] [--out <dir>] [--no-budget]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from datasets.travelplanner.pipeline.config_task import load_config
from datasets.travelplanner.pipeline.data.queries import load_queries
from oak.kg.graph import load_graph, enrich_city_nodes
from datasets.travelplanner.pipeline.agent.graph_index import GraphIndex
from datasets.travelplanner.pipeline.agent.validator import validate_plan_full
from datasets.travelplanner.pipeline.agent.planner import local_repair, budget_downgrade


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", default=str(Path(__file__).resolve().parents[1] / "runs/inference/plans.validation.jsonl"))
    ap.add_argument("--graphs", default=str(Path(__file__).resolve().parents[1] / "runs/inference"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs/replay/v1"))
    ap.add_argument("--no-budget", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    qs = {q.idx: q for q in load_queries("validation")}
    plans: dict[int, list] = {}
    for line in Path(args.plans).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            plans[r["query_idx"]] = r["plan"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    summary = {"n": 0, "delivered_old": 0, "delivered_new": 0,
               "blocking_old": 0, "blocking_new": 0,
               "improved": 0, "worse": 0, "regressed_pass": [],
               "by_code_old": {}, "by_code_new": {}}
    from collections import Counter
    codes_old, codes_new = Counter(), Counter()

    for idx in sorted(plans):
        q = qs[idx]
        gpath = Path(args.graphs) / f"q{idx}" / "graph.json"
        if not gpath.exists():
            continue
        g = enrich_city_nodes(load_graph(gpath), cfg.tp_root)
        gi = GraphIndex(g, cfg.tp_root)
        old_plan = plans[idx] or None
        rep_old = validate_plan_full(old_plan, q, gi)
        new_plan = local_repair(json.loads(json.dumps(plans[idx] or [])) or None, q, gi) \
            if old_plan else None
        if new_plan and not args.no_budget:
            cand = budget_downgrade(new_plan, q, gi)
            new_plan = cand
        rep_new = validate_plan_full(new_plan, q, gi)

        summary["n"] += 1
        if old_plan:
            summary["delivered_old"] += 1
        if new_plan:
            summary["delivered_new"] += 1
        bo, bn = len(rep_old.blocking), len(rep_new.blocking)
        summary["blocking_old"] += bo
        summary["blocking_new"] += bn
        codes_old.update(i.code for i in rep_old.blocking)
        codes_new.update(i.code for i in rep_new.blocking)
        if rep_old.ok and not rep_new.ok:
            summary["regressed_pass"].append(idx)
        if bn < bo:
            summary["improved"] += 1
        if bn > bo:
            summary["worse"] += 1
        recs.append({"query_idx": idx, "plan": new_plan or [],
                     "old_blocking": [i.code for i in rep_old.blocking],
                     "new_blocking": [i.code for i in rep_new.blocking],
                     "old_cost": rep_old.cost, "new_cost": rep_new.cost})

    (out_dir / "plans.replay.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs))
    summary["by_code_old"] = dict(codes_old)
    summary["by_code_new"] = dict(codes_new)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
