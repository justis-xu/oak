"""全量 10 段对话验证入口（冻结 schema 后使用）。

用法：
  uv run python -m locomo.run_full                 # 全部 10 段
  uv run python -m locomo.run_full --only conv-41,conv-44
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

from oak.llm.client import LLMClient

from .config import load_locomo_config
from .data import list_conversations
from .runner import eval_conversation


async def amain(only: list[str] | None) -> None:
    lc = load_locomo_config()
    client = LLMClient(lc.cfg)
    convs = list_conversations(lc.dataset_path)
    if only:
        convs = [c for c in convs if c in only]
    reports = {}
    t0 = time.time()
    for i, conv_id in enumerate(convs, 1):
        print(f"\n===== [{i}/{len(convs)}] {conv_id} =====")
        r = await eval_conversation(lc, client, conv_id, tag="full")
        reports[conv_id] = {k: v for k, v in r.items() if k != "grades"}
    total_n = sum(r["n"] for r in reports.values())
    total_e = sum(r["exact"] for r in reports.values())
    merged = {
        "ts": time.time(), "elapsed_min": round((time.time() - t0) / 60, 1),
        "overall": {"n": total_n, "exact": total_e,
                    "exact_rate": round(total_e / max(total_n, 1), 4)},
        "by_conv": reports,
        "ledger": client.ledger_summary(),
    }
    out = lc.runs_dir / "full" / "report_full.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    print(f"\n===== 全量汇总: {total_e}/{total_n} = "
          f"{total_e / max(total_n, 1):.1%}（{out}）=====")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="逗号分隔 sample_id")
    args = ap.parse_args()
    only = [x.strip() for x in args.only.split(",")] if args.only else None
    asyncio.run(amain(only))


if __name__ == "__main__":
    main()
