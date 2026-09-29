"""失败归因（纯代码，零 LLM）：抽取缺失 / 检索 miss / 误拒答 / 对抗失效 / 推理表述错。

evidence 只在这里（事后诊断）使用，不进入作答路径。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .agent import QAOutput
from .build import FactRecord
from .data import Conversation, QA


def attribute_failures(conv: Conversation, report: dict,
                       answers: dict[int, QAOutput], facts: list[FactRecord]) -> list[dict]:
    by_idx = {qa.idx: qa for qa in conv.qas}
    facts_by_dia: dict[str, list[str]] = {}
    fid_by_subject: dict[str, list[str]] = {}
    for f in facts:
        for d in f.sources:
            facts_by_dia.setdefault(d, []).append(f.fid)
        fid_by_subject.setdefault(f.subject, []).append(f.fid)

    failures: list[dict] = []
    for g in report.get("grades", []):
        if g["grade"] == "exact":
            continue
        qa: QA = by_idx[g["idx"]]
        out = answers.get(g["idx"])
        pred = (out.answer if out else "") or ""
        # 证据覆盖：qa.evidence 有任一 dia 被事实覆盖？
        cover_fids: set[str] = set()
        for d in qa.evidence:
            cover_fids.update(facts_by_dia.get(d, []))
        if qa.answer is None:                                   # 对抗题
            if out and out.refused:
                attr = "拒答带猜测" if g["grade"] == "partial" else "拒答被判错(争议)"
            else:
                attr = "对抗失效(答了不存在的信息)"
        elif not cover_fids:
            attr = "抽取缺失(证据轮未被任何事实覆盖)"
        elif out and not set(out.collected) & cover_fids:
            attr = "检索miss(事实在图但没被收集)"
        elif out and out.refused:
            attr = "误拒答(事实已收集仍拒答)"
        else:
            attr = "推理或表述错(事实已收集)"
        failures.append({
            "idx": g["idx"], "category": qa.cat_name, "attribution": attr,
            "question": qa.question, "gold": qa.gold_text() or "(不可回答)",
            "pred": pred[:300], "grade": g["grade"], "source": g["source"],
            "judge_reason": g.get("reason", ""),
            "evidence": qa.evidence,
            "covered_fids": sorted(cover_fids),
            "missed_fids": sorted(cover_fids - set(out.collected)) if out else [],
            "collected": out.collected if out else [],
        })
    return failures


def attribution_summary(failures: list[dict]) -> dict:
    s: dict[str, dict] = {}
    for f in failures:
        k = f["attribution"].split("(")[0]
        c = s.setdefault(k, {"n": 0, "categories": {}})
        c["n"] += 1
        c["categories"][f["category"]] = c["categories"].get(f["category"], 0) + 1
    return {k: v for k, v in sorted(s.items(), key=lambda x: -x[1]["n"])}


def regression_diff(conv_dir: Path, tag_a: str, tag_b: str) -> dict:
    """两轮 report 的 pass→fail / fail→pass diff（方法论#10 回归保护）。"""
    def grades(tag: str) -> dict[int, str]:
        rep = json.loads((conv_dir / tag / "report.json").read_text())
        return {g["idx"]: g["grade"] for g in rep["grades"]}
    a, b = grades(tag_a), grades(tag_b)
    return {
        "pass_to_fail": sorted(i for i in b if b[i] != "exact" and a.get(i) == "exact"),
        "fail_to_pass": sorted(i for i in b if b[i] == "exact" and a.get(i) != "exact"),
    }


def sample_disputes(report: dict, n: int = 20, seed: int = 7) -> list[int]:
    """分层抽取待人工复核的判分样本（每类别按比例）。"""
    idxs = [g["idx"] for g in report.get("grades", [])]
    random.Random(seed).shuffle(idxs)
    return idxs[:n]


def write_failures(out_dir: Path, failures: list[dict], report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "failures.jsonl").write_text(
        "\n".join(json.dumps(f, ensure_ascii=False) for f in failures))
    report = {**report, "attribution": attribution_summary(failures),
              "dispute_sample": sample_disputes(report)}
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
