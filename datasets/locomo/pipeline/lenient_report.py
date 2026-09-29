"""官方宽松口径对照：对严格判分非 exact 的题，按 LoCoMo 论文式二元 LLM 判分复评。

用法：uv run python -m locomo.lenient_report conv-26 iter11
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

from oak.llm.client import LLMClient

from .config import LOCOMO_TASK_DIR, load_locomo_config, ns
from .data import load_conversation

LENIENT_SYSTEM = "你是 LoCoMo 官方口径判分器（binary），只输出合法 JSON。"
LENIENT_TEMPLATE = """判断【待判答案】是否正确回答了【问题】（以【标准答案】为准）。

官方口径：只要答案包含了标准答案的要点即算正确（correct）；完全不含要点或答错才算 incorrect。
注意：比标准答案多给出的正确细节不扣分；措辞不同但事实相同算正确；对抗题（无标准答案）只有明确表示无法回答才算 correct。

【问题】{question}
【标准答案】{gold}
【待判答案】{pred}

输出 JSON：{{"correct": true|false, "reason": "一句话"}}"""


async def amain(conv_id: str, tag: str) -> None:
    lc = load_locomo_config()
    client = LLMClient(lc.cfg)
    conv = load_conversation(lc.dataset_path, conv_id)
    rep = json.loads((LOCOMO_TASK_DIR / "runs" / conv_id / tag / "report.json").read_text())
    answers = {}
    for line in (LOCOMO_TASK_DIR / "runs" / conv_id / tag / "answers.jsonl").read_text().splitlines():
        o = json.loads(line)
        answers[o["idx"]] = o

    sem = asyncio.Semaphore(4)

    async def lenient(qa, pred) -> bool:
        async with sem:
            r = await client.chat(
                role="locomo_judge", temperature=0.0, max_tokens=256, json_mode=True,
                namespace=ns(conv_id, "lenient"),
                messages=[{"role": "system", "content": LENIENT_SYSTEM},
                          {"role": "user", "content": LENIENT_TEMPLATE.format(
                              question=qa.question,
                              gold=qa.gold_text() or "（不可回答——对话中不存在该信息）",
                              pred=pred or "（空）")}])
            m = re.search(r"\{.*\}", r.content, re.S)
            if m:
                try:
                    return bool(json.loads(m.group(0)).get("correct"))
                except Exception:
                    pass
            return "true" in r.content.lower()

    tasks = []
    idxs = []
    for g in rep["grades"]:
        if g["grade"] == "exact":
            continue
        qa = conv.qas[g["idx"]]
        idxs.append(g["idx"])
        tasks.append(lenient(qa, answers.get(g["idx"], {}).get("answer", "")))
    verdicts = await asyncio.gather(*tasks)
    n_exact = rep["exact"]
    n_lenient_extra = sum(verdicts)
    n = rep["n"]
    print(f"[{conv_id}/{tag}] 严格 exact: {n_exact}/{n} = {n_exact / n:.1%}")
    print(f"官方宽松口径:   {n_exact + n_lenient_extra}/{n} = {(n_exact + n_lenient_extra) / n:.1%}"
          f"（非 exact 中 {n_lenient_extra}/{len(verdicts)} 被二元判分判 correct）")
    out = {"strict": n_exact / n, "lenient": (n_exact + n_lenient_extra) / n,
           "lenient_flips": [i for i, v in zip(idxs, verdicts) if v]}
    (LOCOMO_TASK_DIR / "runs" / conv_id / tag / "lenient.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(amain(sys.argv[1] if len(sys.argv) > 1 else "conv-26",
                      sys.argv[2] if len(sys.argv) > 2 else "iter11"))
