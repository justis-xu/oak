"""mem0 只-ADD 基线的提示词（中文场景）。

来源与对应关系：
- ADDITIVE_EXTRACTION_PROMPT：vendored mem0 2.1.0 `mem0/configs/prompts.py:468`
  （V3 加法提取系统提示：唯一操作是 ADD，无 UPDATE/DELETE）——保真压缩。
- generate_additive_user_prompt：同文件 `generate_additive_extraction_prompt`
  （Summary / Last k Messages / Existing Memories / New Messages / 日期 / 自定义段）。
- CUSTOM_INSTRUCTIONS_ZH：memory-schema-rsi `config/locomo_zh.yaml` 的
  `fact_extraction_prompt`（经 build_mem0_config 的 custom_instructions 插口注入，
  mem0_backend.py:64-73 → prompts.py:1044）——中文强制记录段。
"""
from __future__ import annotations

import json
from datetime import date

# 对应 mem0/configs/prompts.py:468 ADDITIVE_EXTRACTION_PROMPT（只 ADD 契约的骨架）
ADDITIVE_EXTRACTION_PROMPT = """\
# ROLE

You are a Memory Extractor — a precise, evidence-bound processor responsible for \
extracting rich, contextual memories from conversations. Your sole operation is \
ADD: identify every piece of memorable information and produce self-contained, \
contextually rich factual statements.

You extract from BOTH user and assistant messages. User messages reveal personal \
facts, preferences, plans, and experiences. Assistant messages contain \
recommendations, plans, suggestions, and actionable information.

Accuracy and completeness are critical — a missed extraction means lost context. \
When a conversation covers multiple topics, extract each one separately.

# OUTPUT FORMAT

Return a single JSON object: {"memory": [{"text": "<self-contained factual \
statement>", "attributed_to": "<User|Assistant|speaker name, optional>"}]}

- Every statement must be self-contained (readable without the conversation).
- Attribute correctly: "User" for user-stated facts; assistant-generated content \
framed in terms of the user's context ("User was recommended X").
- Do NOT extract: vague characterizations, generic acknowledgments, meta-commentary.
- If nothing memorable, return {"memory": []}.

This is an ADD-ONLY pipeline: never emit update or delete operations.
"""

# locomo_zh.yaml fact_extraction_prompt 的中文强制段（逐字保留）
CUSTOM_INSTRUCTIONS_ZH = (
    "The conversation is in Simplified Chinese. You MUST record ALL facts in "
    "Simplified Chinese — English output is strictly forbidden. Keep proper nouns "
    "such as brand names, product names, and titles exactly as written. "
    "对话是简体中文：所有事实一律用简体中文记录，禁止输出英文；"
    "品牌/产品名等专有名词按原样保留。"
)


def generate_additive_user_prompt(
    existing_memories: list[dict],
    new_messages: list[dict],
    last_k_messages: list[dict] | None = None,
    custom_instructions: str | None = None,
    today: date | None = None,
) -> str:
    """对应 mem0 generate_additive_extraction_prompt 的分段结构。"""
    today = today or date.today()

    def _mems(items: list[dict] | None) -> str:
        items = items or []
        return "\n".join(f"- (id:{m['id']}) {m['text']}" for m in items) or "（无）"

    def _msgs(items: list[dict] | None) -> str:
        items = items or []
        return "\n".join(f"- {m.get('role')}: {m.get('content')}" for m in items) or "（无）"

    sections = [
        f"## Last k Messages\n{_msgs(last_k_messages)}",
        f"## Existing Memories\n{_mems(existing_memories)}",
        f"## New Messages\n{_msgs(new_messages)}",
        f"## Current Date\n{today.isoformat()}",
    ]
    ci = custom_instructions or CUSTOM_INSTRUCTIONS_ZH
    if ci:
        sections.append(f"## Custom Instructions\n{ci}")
    return "\n\n".join(sections)


def parse_extraction(response_text: str) -> list[dict]:
    """解析 {"memory": [...]}；对应 main.py:971-984 的宽容解析。"""
    import re
    text = (response_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
        mems = obj.get("memory", [])
        return [x for x in mems if isinstance(x, dict) and x.get("text")]
    except Exception:
        return []
