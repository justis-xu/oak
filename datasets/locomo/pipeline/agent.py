"""中文 ReAct 作答执行器：步骤=fast 模型，终答+证据自检=strong 模型。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from oak.llm.client import LLMClient

from .config import LocomoConfig, ns
from .prompts.answer import (FINAL_SYSTEM, FINAL_TEMPLATE, REFUSAL, REPAIR_TEMPLATE,
                             STEPS_SYSTEM_TEMPLATE)
from .tools import ToolBox, render_tool_docs

ACTION_RE = re.compile(r"Action[:：]\s*([^\s(（]+)\s*[（(](\{.*\})[)）]", re.S)
ACTION_KW_RE = re.compile(r"Action[:：]\s*([^\s(（]+)\s*[（(](.*)[)）]", re.S)
FINAL_ANSWER_RE = re.compile(r"Final Answer[:：]\s*(.+?)(?:\n|$)", re.S)
EVIDENCE_RE = re.compile(r"证据[:：]\s*\[([^\]]*)\]")
FID_RE = re.compile(r"\d+-\d{3,4}")
COLLECT_RE = re.compile(r"收集完毕")
OBS_TRUNC = 6000


def parse_action_cn(out: str) -> tuple[str, dict] | None:
    m = ACTION_RE.search(out)
    if m:
        try:
            args = json.loads(m.group(2))
            if isinstance(args, dict):
                return m.group(1).strip(), args
        except Exception:
            pass
    m = ACTION_KW_RE.search(out)
    if not m:
        return None
    name, raw = m.group(1).strip(), m.group(2).strip()
    if not raw:
        return name, {}
    args: dict[str, Any] = {}
    ok = True
    buf, quote, parts = [], None, []
    for ch in raw:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'“”":
            quote = ch
            buf.append(ch)
        elif ch in ",，":
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    for part in parts:
        if not part.strip():
            continue
        if "=" not in part:
            ok = False
            break
        k, v = part.split("=", 1)
        k, v = k.strip().strip("\"'“”"), v.strip()
        if v.startswith("[") and v.endswith("]"):
            args[k] = [x.strip().strip("\"'“”") for x in v[1:-1].split(",") if x.strip()]
        else:
            try:
                args[k] = json.loads(v)
            except Exception:
                args[k] = v.strip("\"'“”")
    return (name, args) if ok else None


@dataclass
class QAOutput:
    idx: int
    question: str
    answer: str = ""
    evidence: list[str] = field(default_factory=list)
    refused: bool = False
    n_steps: int = 0
    collected: list[str] = field(default_factory=list)
    trajectory: dict = field(default_factory=dict)


def _steps_system(conv_header: str, toolbox: ToolBox) -> str:
    dates = sorted(str(r.get("日期")) for r in toolbox.facts.values() if r.get("日期"))
    return STEPS_SYSTEM_TEMPLATE.format(
        header=conv_header,
        n_facts=len(toolbox.facts),
        date_lo=dates[0][:7] if dates else "?",
        date_hi=dates[-1][:7] if dates else "?",
        tool_docs=render_tool_docs(),
    )


async def run_qa(idx: int, question: str, conv_header: str, toolbox: ToolBox,
                 client: LLMClient, lc: LocomoConfig, conv_id: str) -> QAOutput:
    out = QAOutput(idx=idx, question=question)
    qns = ns(conv_id, f"q{idx}")
    system = _steps_system(conv_header, toolbox)
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"问题：{question}"},
    ]
    collected: dict[str, str] = {}       # fid -> fact line
    draft = ""
    steps_log: list[dict] = []

    for step in range(1, lc.react_max_steps + 1):
        r = await client.chat(role="locomo_steps", messages=messages,
                              temperature=0.2, max_tokens=1024, namespace=qns)
        reply = r.content.strip()
        messages.append({"role": "assistant", "content": reply})
        act = parse_action_cn(reply)
        if act is None:
            if COLLECT_RE.search(reply):
                m = re.search(r"依据[:：]\s*\[([^\]]*)\]", reply)
                if m:
                    for fid in FID_RE.findall(m.group(1)):
                        if fid in toolbox.facts:
                            collected.setdefault(fid, "")
                dm = re.search(r"结论草稿[:：]\s*(.+)", reply, re.S)
                if dm:
                    draft = dm.group(1).strip()[:300]
                break
            obs = ("格式错误：请输出 Thought+Action 行调用工具，或输出'收集完毕/依据/结论草稿'。"
                   if step < lc.react_max_steps else "已达步数上限。")
            messages.append({"role": "user", "content": obs})
            continue
        name, args = act
        obs, fids = toolbox.execute(name, args)
        for fid in fids:
            if fid in toolbox.facts:
                collected.setdefault(fid, toolbox.fact_line(fid))
        if len(obs) > OBS_TRUNC:
            obs = obs[:OBS_TRUNC] + "…（截断）"
        steps_log.append({"step": step, "action": name, "args": args, "obs": obs[:1500]})
        messages.append({"role": "user", "content": f"观察：\n{obs}"})
        out.n_steps = step

    # ---- 终答合成（strong）----
    # 上下文 = 收集事实（按相关性排序，top 45）+ 全图词法 top15 增补（不依赖 agent 导航，
    # 治"事实在图但没被收集/被截断"）
    scores = toolbox.index.score(question)
    ranked_fids = sorted(collected, key=lambda f: (-scores.get(f, 0.0), f))
    chosen = ranked_fids[:45]
    extra = [fid for fid, _ in toolbox.index.search(question, limit=15)
             if fid not in chosen]
    fact_lines = [toolbox.fact_line(f) for f in chosen]
    if extra:
        fact_lines.append("——以下为全图词法相关的补充事实（可能含未被步骤收集的）——")
        fact_lines += [toolbox.fact_line(f) for f in extra]
    if len(ranked_fids) > 45:
        fact_lines.append(f"（另有 {len(ranked_fids) - 45} 条低相关收集事实未列出）")
    facts_block = "\n".join(fact_lines) or "（未收集到事实）"
    final_msgs = [
        {"role": "system", "content": FINAL_SYSTEM},
        {"role": "user", "content": FINAL_TEMPLATE.format(
            header=conv_header, question=question,
            facts=facts_block, draft=draft or "（无）", refusal=REFUSAL)},
    ]
    # 三采样终答（不同温度→真实多样性；一致采样只是缓存复读）+ 无 gold 共识择优
    candidates: list[tuple[str, list[str]]] = []
    for temp in (0.3, 0.7, 1.0):
        fr = await client.chat(role="locomo_answer", messages=final_msgs,
                               temperature=temp, max_tokens=1024, namespace=qns)
        answer, evidence = _parse_final(fr.content, collected)
        if evidence is None or (not evidence and not answer.startswith(REFUSAL)):
            rr = await client.chat(
                role="locomo_answer", messages=final_msgs + [
                    {"role": "assistant", "content": fr.content},
                    {"role": "user", "content": REPAIR_TEMPLATE.format(
                        refusal=REFUSAL, question=question, facts=facts_block,
                        prev=fr.content[:1500])}],
                temperature=temp, max_tokens=1024, namespace=qns)
            answer, evidence = _parse_final(rr.content, collected)
        if evidence is None or (not evidence and not answer.startswith(REFUSAL)):
            answer, evidence = REFUSAL, []      # 证据强制：无支撑一律拒答
        candidates.append((answer, evidence))

    answer, evidence = await _consensus_pick(question, candidates, client, qns)
    out.answer = answer
    out.evidence = evidence
    out.refused = answer.startswith(REFUSAL)
    out.collected = sorted(collected)
    out.trajectory = {"steps": steps_log, "draft": draft,
                      "final_raw": answer, "evidence": evidence,
                      "candidates": [c[0][:200] for c in candidates]}
    return out


async def _consensus_pick(question: str, candidates: list[tuple[str, list[str]]],
                          client: LLMClient, qns: str) -> tuple[str, list[str]]:
    """三候选择优，不看 gold：归一化多数 → LLM 择同 → 证据最多。"""
    from .dates import normalize_answer_text
    norm = [normalize_answer_text(a) for a, _ in candidates]
    for i in range(len(candidates)):
        if norm[i] and norm.count(norm[i]) >= 2:
            return candidates[i]
    n_ev = [len(ev) for _, ev in candidates]
    try:
        r = await client.chat(
            role="locomo_util", namespace=qns,
            temperature=0.0, max_tokens=256, json_mode=True,
            messages=[
                {"role": "system", "content": "你是答案共识判定器，只输出合法 JSON。"},
                {"role": "user", "content":
                    "同一问题的三个候选回答（互独立采样）。若其中两个**意思一致**，"
                    "返回那个一致答案的序号；若三个各不相同，返回证据数最多的序号。"
                    "拒答也算一种'意思'。\n\n"
                    f"问题：{question}\n"
                    + "\n".join(f"候选{i}（证据数{n_ev[i]}）：{a[:300]}"
                                for i, (a, _) in enumerate(candidates))
                    + '\n\n输出 JSON：{"pick": 0|1|2}'},
            ])
        m = re.search(r"\{.*\}", r.content, re.S)
        pick = int(json.loads(m.group(0))["pick"]) if m else -1
        if 0 <= pick < len(candidates):
            return candidates[pick]
    except Exception:
        pass
    return candidates[n_ev.index(max(n_ev))]


def _parse_final(text: str, collected: dict[str, str]) -> tuple[str, list[str] | None]:
    m = FINAL_ANSWER_RE.search(text)
    answer = m.group(1).strip() if m else text.strip()[:500]
    em = EVIDENCE_RE.search(text)
    evidence: list[str] | None = None
    if em:
        evidence = [f for f in FID_RE.findall(em.group(1)) if f in collected]
        # 证据行存在但都在收集集之外 → 视为无有效证据
        raw_fids = FID_RE.findall(em.group(1))
        if raw_fids and not evidence:
            evidence = []
    return answer, evidence
