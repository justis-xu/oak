"""离线重聚合：对已有 scores.json 的 per_query 按官方固定分母口径复算（不调 LLM）。

用法: uv run python scripts/reaggregate.py [scores.json 路径] [--out 输出路径]
默认读 runs/inference/scores.json，写 runs/inference/scores_fixed_denominator.json。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oak.config import load_config
from oak.data.queries import load_queries, applicable_hc_keys, CS_KEY_COUNT
from oak.eval.adapter import CS_KEYS, HC_KEYS, SubsetScores


def reaggregate(scores_path: Path, split: str = "validation") -> SubsetScores:
    cfg = load_config()
    data = json.loads(scores_path.read_text())
    indices = data["indices"]
    queries_all = load_queries(split, cfg)
    queries = [queries_all[i] for i in indices]
    plans = [None] * len(indices)          # 只用 per_query 的布尔结果，plans 不参与
    per_query = data["per_query"]
    assert len(per_query) == len(queries)

    s = SubsetScores(n=len(queries))
    per: list[dict] = []
    for q, rec in zip(queries, per_query):
        # 旧 per_query: hc 为空 dict 表示 hard 未运行（gated/undelivered/error）
        hc = rec.get("hc") or None
        cs_item = rec.get("cs") or {}
        hc_item = rec.get("hc") or {}
        item = dict(rec)
        s.cs_total += CS_KEY_COUNT
        s.cs_pass += sum(1 for k in CS_KEYS if cs_item.get(k) is True)
        denom = applicable_hc_keys(q.level, q.local_constraint)
        s.hc_total += len(denom)
        s.hc_pass += sum(1 for k in denom if hc_item.get(k) is True)
        if rec.get("delivered"):
            s.delivered += 1
        gated = bool(hc_item)
        cs_all = bool(cs_item) and all(v is not False for v in cs_item.values())
        hc_all = bool(hc_item) and all(v is not False for v in hc_item.values())
        item["macro_cs"] = cs_all and gated
        item["macro_hc"] = hc_all
        item["final"] = cs_all and hc_all
        s.macro_cs_cnt += int(item["macro_cs"])
        s.macro_hc_cnt += int(item["macro_hc"])
        s.final_cnt += int(item["final"])
        per.append(item)

    s.per_query = per
    s.delivery = s.delivered / max(1, s.n)
    s.micro_cs = s.cs_pass / max(1, s.cs_total)
    s.macro_cs = s.macro_cs_cnt / max(1, s.n)
    s.micro_hc = s.hc_pass / max(1, s.hc_total)
    s.macro_hc = s.macro_hc_cnt / max(1, s.n)
    s.final = s.final_cnt / max(1, s.n)
    return s


def main():
    scores_path = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
        else Path("/Users/xu/git/oak/runs/inference/scores.json")
    out_idx = sys.argv.index("--out") + 1 if "--out" in sys.argv else None
    out_path = Path(sys.argv[out_idx]) if out_idx else scores_path.parent / "scores_fixed_denominator.json"
    s = reaggregate(scores_path)
    out_path.write_text(json.dumps({"summary": s.summary(),
                                    "per_query": s.per_query}, ensure_ascii=False, indent=1))
    print(f"source: {scores_path}")
    for k, v in s.summary().items():
        print(f"  {k}: {v}")
    pass_set = [it["idx"] for it in s.per_query if it["final"]]
    print("final-pass idx:", pass_set)
    hard_not_run = {}
    for it in s.per_query:
        r = "error" if it.get("error") else ("empty" if not it.get("delivered")
                                             else ("gated" if not (it.get("hc") or {}) else "ran"))
        hard_not_run[r] = hard_not_run.get(r, 0) + 1
    print("hard constraint run status:", hard_not_run)


if __name__ == "__main__":
    main()
