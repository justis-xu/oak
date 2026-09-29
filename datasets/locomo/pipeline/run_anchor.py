"""锚点对话迭代入口。

用法：
  uv run python -m locomo.run_anchor                     # conv-26 全量
  uv run python -m locomo.run_anchor --idx 3,5,7        # 冒烟：只跑指定题
  uv run python -m locomo.run_anchor --limit 20          # 前 20 题
  uv run python -m locomo.run_anchor --conv conv-41 --tag gen1
"""
from __future__ import annotations

import argparse
import asyncio

from oak.llm.client import LLMClient

from .config import load_locomo_config
from .runner import eval_conversation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default="conv-26")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--idx", default=None, help="逗号分隔的题号（冒烟用）")
    ap.add_argument("--anchor", action="store_true", help="只跑固定锚点集（快速闭环）")
    args = ap.parse_args()

    lc = load_locomo_config()
    client = LLMClient(lc.cfg)
    idx_filter = None
    if args.anchor:
        import json as _json
        from .config import LOCOMO_TASK_DIR
        obj = _json.loads((LOCOMO_TASK_DIR / "runs" / "anchor_set.json").read_text())
        idx_filter = set(obj["idx"])
    elif args.idx:
        idx_filter = {int(x) for x in args.idx.split(",") if x.strip()}
    elif args.limit:
        idx_filter = set(range(args.limit))

    asyncio.run(eval_conversation(lc, client, args.conv, tag=args.tag,
                                  idx_filter=idx_filter))


if __name__ == "__main__":
    main()
