"""数据集加载与数据纪律。

数据纪律（评测有效性前提，任何改动不得违反）：
- 建图输入 = session 原文消息 + session 日期。禁止读取 event_summary /
  observation / session_summary（数据集自带摘要，等价于答案线索泄漏）。
- 作答输入 = 问题文本 + 本体图。禁止：gold answer、evidence dia_id、
  题型类别、对话原文全文。
- 判题输入 = 题目 / 类别 / gold / 预测（盲判，不见轨迹）。
- 失败归因（事后诊断）可读 evidence / answer / 轨迹，但结论只能用于修改
  提示词 / schema / 工具，不得把任何题目内容写死进作答路径。

类别映射（对题目实测验证，非网上流传版本）：
  1=多跳  2=时间  3=开放域推理  4=单跳  5=对抗（不可回答，带陷阱答案）
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .dates import parse_cn_date, parse_session_datetime

CATEGORY_MAP = {1: "多跳", 2: "时间", 3: "开放域", 4: "单跳", 5: "对抗"}
TOPIC_CATEGORIES = {1, 2, 3, 4}          # J 口径（可回答题）


@dataclass
class Turn:
    speaker: str
    dia_id: str                          # 如 "D2:3"
    text: str


@dataclass
class Session:
    no: int
    date_iso: date | None
    date_raw: str                        # 原文（英文）日期串
    turns: list[Turn] = field(default_factory=list)


@dataclass
class QA:
    idx: int
    question: str
    category: int
    answer: str | int | None = None      # 对抗题为 None（judge/analyze 专用）
    adversarial_answer: str | None = None    # 陷阱答案（judge/analyze 专用）
    evidence: list[str] = field(default_factory=list)   # dia_id 列表（analyze 专用）

    @property
    def cat_name(self) -> str:
        return CATEGORY_MAP.get(self.category, "?")

    def gold_text(self) -> str:
        return "" if self.answer is None else str(self.answer)


@dataclass
class Conversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[Session] = field(default_factory=list)
    qas: list[QA] = field(default_factory=list)
    # 数据集自带的已翻译标注字段（语料所有方指示并入建图语料，评测报告须披露）
    observations: list[tuple[str, str, str]] = field(default_factory=list)  # (dia_id, 主体, 观察句)
    events: list[tuple[str, str, str]] = field(default_factory=list)       # (日期iso, 主体, 事件句)

    @property
    def speakers(self) -> list[str]:
        return [self.speaker_a, self.speaker_b]

    def session_by_no(self, no: int) -> Session | None:
        return next((s for s in self.sessions if s.no == no), None)

    def turns_by_dia(self, dia_ids: list[str]) -> list[Turn]:
        want = set(dia_ids)
        out = []
        for s in self.sessions:
            for t in s.turns:
                if t.dia_id in want:
                    out.append(t)
        return out

    def header(self) -> str:
        return f"对话双方：{self.speaker_a}（说话人甲）、{self.speaker_b}（说话人乙）"


def list_conversations(path: Path) -> list[str]:
    return [c["sample_id"] for c in json.loads(path.read_text())]


def load_conversation(path: Path, sample_id: str) -> Conversation:
    data = json.loads(path.read_text())
    raw = next((c for c in data if c.get("sample_id") == sample_id), None)
    if raw is None:
        raise KeyError(f"sample_id {sample_id!r} 不存在于 {path}")
    conv_d = raw["conversation"]
    conv = Conversation(
        sample_id=sample_id,
        speaker_a=conv_d.get("speaker_a", ""),
        speaker_b=conv_d.get("speaker_b", ""),
    )
    # session_N 与 session_N_date_time 交替；只收有消息的 session
    ses_nums = sorted(
        int(k.split("_", 1)[1])
        for k in conv_d
        if k.startswith("session_") and not k.endswith("date_time")
        and k.split("_", 1)[1].isdigit() and conv_d[k]
    )
    for no in ses_nums:
        date_raw = conv_d.get(f"session_{no}_date_time", "")
        turns = [
            Turn(speaker=t.get("speaker", ""), dia_id=t.get("dia_id", ""),
                 text=t.get("text", ""))
            for t in conv_d[f"session_{no}"]
            if t.get("text")
        ]
        if not turns:
            continue
        conv.sessions.append(Session(
            no=no, date_iso=parse_session_datetime(date_raw),
            date_raw=date_raw, turns=turns,
        ))
    conv.qas = [
        QA(idx=i, question=q["question"], category=int(q["category"]),
           answer=q.get("answer"), adversarial_answer=q.get("adversarial_answer"),
           evidence=list(q.get("evidence") or []))
        for i, q in enumerate(raw["qa"])
    ]
    # ---- 数据集自带标注（已翻译；语料所有方指示并入建图语料，报告须披露）----
    # observation: {session_N_observation: {说话人: [[观察句, dia_id], ...]}}
    for k, per_speaker in (raw.get("observation") or {}).items():
        for spk, items in (per_speaker or {}).items():
            for it in items or []:
                if isinstance(it, (list, tuple)) and len(it) >= 2 and it[0] and it[1]:
                    conv.observations.append((str(it[1]), str(spk), str(it[0])))
    # event_summary: {events_session_N: {说话人: [事件句], "date": "2023年5月25日" 或英文}}
    for k, per_speaker in (raw.get("event_summary") or {}).items():
        date_raw = str((per_speaker or {}).get("date") or "")
        d = parse_session_datetime(date_raw)
        if d is None:
            ymd = parse_cn_date(date_raw)
            if ymd and ymd[0]:
                try:
                    d = date(*ymd)
                except ValueError:
                    d = None
        for spk, evs in (per_speaker or {}).items():
            if spk == "date":
                continue
            for ev in evs or []:
                if str(ev).strip():
                    conv.events.append((d.isoformat() if d else "", str(spk), str(ev)))
    return conv
