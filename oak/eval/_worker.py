"""官方评测 worker：在 TravelPlanner/evaluation 目录下以子进程运行。

用法（子进程内）：
    cd <tp_root>/evaluation && python _worker.py <input.json> <output.json>

input.json:  {"queries": [<官方 query 行 dict>...], "plans": [<plan list or []>...]}
output.json: {"per_query": [{"commonsense": {...}|null, "hard": {...}|null}...]}

约束模块 import 时自行 os.chdir 与加载数据库——因此必须在本目录下启动。
"""
import json
import os
import sys


def main() -> None:
    in_path, out_path = sys.argv[1], sys.argv[2]
    # cwd=evaluation（adapter 保证）；把 cwd 与上级加入 sys.path：
    # 前者为了 commonsense_constraint/hard_constraint，后者为了 tools/utils
    sys.path.insert(0, os.getcwd())
    sys.path.insert(1, os.path.abspath(".."))
    from commonsense_constraint import evaluation as commonsense_eval
    from hard_constraint import evaluation as hard_eval

    data = json.loads(open(in_path).read())
    per_query = []
    for q, plan in zip(data["queries"], data["plans"]):
        hard_not_run = None
        try:
            if plan:
                cs = commonsense_eval(q, plan)
            else:
                cs = None
                hard_not_run = "empty"
            if cs and cs["is_not_absent"][0] and cs["is_valid_information_in_sandbox"][0]:
                hc = hard_eval(q, plan)
            else:
                hc = None
                if hard_not_run is None:
                    hard_not_run = "gated"
        except Exception as e:
            cs, hc = None, None
            hard_not_run = "error"
            per_query.append({"commonsense": None, "hard": None, "error": f"{type(e).__name__}: {e}",
                              "hard_not_run_reason": hard_not_run})
            continue
        per_query.append({
            "commonsense": {k: list(v) if isinstance(v, (tuple, list)) else v
                            for k, v in cs.items()} if cs else None,
            "hard": {k: list(v) if isinstance(v, (tuple, list)) else v
                     for k, v in hc.items()} if hc else None,
            "hard_not_run_reason": hard_not_run,
        })
    with open(out_path, "w") as f:
        json.dump({"per_query": per_query}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
