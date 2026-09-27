"""官方评测器对接：子进程调用 + subset 分母重算 + padded 全量复跑。"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from ..data.queries import Query, applicable_hc_keys, CS_KEY_COUNT

WORKER = Path(__file__).parent / "_worker.py"

CS_KEYS = ["is_valid_information_in_current_city", "is_valid_information_in_sandbox",
           "is_reasonable_visiting_city", "is_valid_restaurants",
           "is_valid_transportation", "is_valid_attractions",
           "is_valid_accommodation", "is_not_absent"]
HC_KEYS = ["valid_cost", "valid_room_rule", "valid_cuisine",
           "valid_room_type", "valid_transportation"]


@dataclass
class SubsetScores:
    per_query: list[dict] = field(default_factory=list)
    # 分子/分母（官方固定口径：未交付、门控未运行一律计失败）
    n: int = 0
    delivered: int = 0
    cs_pass: int = 0
    cs_total: int = 0
    hc_pass: int = 0
    hc_total: int = 0
    macro_cs_cnt: int = 0
    macro_hc_cnt: int = 0
    final_cnt: int = 0
    # 兼容旧字段（rate）
    delivery: float = 0.0
    micro_cs: float = 0.0
    macro_cs: float = 0.0
    micro_hc: float = 0.0
    macro_hc: float = 0.0
    final: float = 0.0

    def summary(self) -> dict:
        return {
            "n": self.n,
            "delivery": round(self.delivery, 4), "delivered": f"{self.delivered}/{self.n}",
            "micro_cs": round(self.micro_cs, 4), "cs_pass": f"{self.cs_pass}/{self.cs_total}",
            "macro_cs": round(self.macro_cs, 4), "macro_cs_pass": f"{self.macro_cs_cnt}/{self.n}",
            "micro_hc": round(self.micro_hc, 4), "hc_pass": f"{self.hc_pass}/{self.hc_total}",
            "macro_hc": round(self.macro_hc, 4), "macro_hc_pass": f"{self.macro_hc_cnt}/{self.n}",
            "final": round(self.final, 4), "final_pass": f"{self.final_cnt}/{self.n}",
        }


def _query_official_row(q: Query) -> dict:
    """转成官方评测函数期望的 query 行（local_constraint 为 dict）。"""
    return {
        "org": q.org, "dest": q.dest, "days": q.days, "date": q.date,
        "people_number": q.people_number, "local_constraint": q.local_constraint,
        "budget": q.budget, "query": q.query, "level": q.level,
        "visiting_city_number": q.visiting_city_number,
    }


class TravelPlannerEvaluator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        ev_dir = cfg.tp_root / "evaluation"
        assert (ev_dir / "eval.py").exists(), f"missing {ev_dir}/eval.py"
        assert (cfg.tp_root / "database" / "flights" / "clean_Flights_2022.csv").exists(), \
            "missing flights csv"

    # ---------------- subset 评测（主口径） ----------------
    def eval_subset(self, queries: list[Query], plans: list[list | None]) -> SubsetScores:
        assert len(queries) == len(plans)
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "in.json"
            out = Path(td) / "out.json"
            inp.write_text(json.dumps({
                "queries": [_query_official_row(q) for q in queries],
                "plans": [p or [] for p in plans],
            }, ensure_ascii=False))
            # 子进程：cwd=evaluation（约束模块 import 时 chdir + 相对路径加载库）
            env_path = self.cfg.work_dir.parent / ".venv" / "bin" / "python"
            py = str(env_path) if env_path.exists() else sys.executable
            proc = subprocess.run(
                [py, str(WORKER), str(inp), str(out)],
                cwd=self.cfg.tp_root / "evaluation", capture_output=True,
                text=True, timeout=1800,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"eval worker failed:\n{proc.stderr[-2000:]}")
            per_query = json.loads(out.read_text())["per_query"]

        return self._aggregate(queries, plans, per_query)

    @staticmethod
    def _aggregate(queries, plans, per_query) -> SubsetScores:
        """官方固定分母口径：CS 每题 8 项、HC 每题预定适用项；
        未交付 / 门控未运行 / worker error 一律计失败（进分母不进分子）。
        macro 镜像官方 eval.py:155-176：hard 未运行（gated out）不计 macro_cs/macro_hc。
        """
        n = len(queries)
        s = SubsetScores(n=n)
        per: list[dict] = []

        for q, plan, rec in zip(queries, plans, per_query):
            cs, hc = rec.get("commonsense"), rec.get("hard")
            item: dict = {"idx": q.idx, "delivered": bool(plan),
                          "cs": {}, "hc": {}, "cs_msg": {}, "hc_msg": {},
                          "error": rec.get("error"),
                          "hard_not_run_reason": rec.get("hard_not_run_reason")}
            if cs:
                for k in CS_KEYS:
                    v = cs.get(k)
                    b = None if (v is None or v[0] is None) else bool(v[0])
                    item["cs"][k] = b
                    if b is False and len(v) > 1 and v[1]:
                        item["cs_msg"][k] = str(v[1])[:160]   # 官方错误消息（judge 归因证据）

            # ---- CS 分母固定 8（与官方 count_record 一致，未交付也计失败）----
            s.cs_total += CS_KEY_COUNT
            s.cs_pass += sum(1 for k in CS_KEYS if item["cs"].get(k) is True)

            # ---- HC 适用项（唯一规则 applicable_hc_keys；官方 eval.py:120-149）----
            denom_keys = applicable_hc_keys(q.level, q.local_constraint)
            if hc:
                for k in HC_KEYS:
                    v = hc.get(k)
                    b = None if (v is None or v[0] is None) else bool(v[0])
                    item["hc"][k] = b
                    if b is False and len(v) > 1 and v[1]:
                        item["hc_msg"][k] = str(v[1])[:160]
            s.hc_total += len(denom_keys)
            s.hc_pass += sum(1 for k in denom_keys if item["hc"].get(k) is True)
            item["hc_denominator"] = len(denom_keys)

            if plan:
                s.delivered += 1
            cs_all = bool(cs) and all(v is not False for v in item["cs"].values()) if cs else False
            hc_all = bool(hc) and all(v is not False for v in item["hc"].values()) if hc else False
            gated = hc is not None          # hard 实际运行（not_absent + sandbox 双过）
            item["macro_cs"] = cs_all and gated
            item["macro_hc"] = hc_all
            item["final"] = cs_all and hc_all
            s.macro_cs_cnt += int(item["macro_cs"])
            s.macro_hc_cnt += int(item["macro_hc"])
            s.final_cnt += int(item["final"])
            per.append(item)

        s.per_query = per
        s.delivery = s.delivered / max(1, n)
        s.micro_cs = s.cs_pass / max(1, s.cs_total)
        s.macro_cs = s.macro_cs_cnt / max(1, n)
        s.micro_hc = s.hc_pass / max(1, s.hc_total)
        s.macro_hc = s.macro_hc_cnt / max(1, n)
        s.final = s.final_cnt / max(1, n)
        return s

    # ---------------- padded 全量文件（官方脚本复跑留档） ----------------
    def build_padded_file(self, split: str, indices: list[int],
                          plan_by_idx: dict[int, list], out: Path) -> None:
        from ..data.queries import load_queries
        all_q = load_queries(split, self.cfg)
        lines = []
        for q in all_q:
            plan = plan_by_idx.get(q.idx, [])
            lines.append(json.dumps({"query_idx": q.idx, "plan": plan}, ensure_ascii=False))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines))

    def eval_official(self, split: str, plan_path: Path) -> dict:
        """跑原版 eval.py（仅连通性留档；其分母按全 split 缩水）。"""
        proc = subprocess.run(
            [sys.executable, "eval.py", "--set_type", split,
             "--evaluation_file_path", str(plan_path.resolve())],
            cwd=self.cfg.tp_root / "evaluation", capture_output=True, text=True,
            timeout=3600,
        )
        scores = {}
        for ln in proc.stdout.splitlines():
            if ": " in ln and "%" in ln:
                k, v = ln.rsplit(": ", 1)
                scores[k] = v
        if proc.returncode != 0 and not scores:
            raise RuntimeError(f"official eval failed:\n{proc.stderr[-2000:]}")
        return scores
