"""实体归并：别名→规范名（一次 LLM 调用处理全部实体名；说话人昵称强制归并）。"""
from __future__ import annotations

import json
import re

from oak.llm.client import LLMClient

from .config import ns

CANON_SYSTEM = "你是实体归并器，只输出合法 JSON。"

CANON_TEMPLATE = """下面是一个中文长对话中抽取出的全部实体名与事实主体名。
把"明显指同一实体"的别名归并到一个规范名下。

规则：
- 规范名优先用全名/本名（如"梅拉妮"优于"梅尔"）；说话人已知映射必须遵守：{forced}
- 两位说话人是**不同的两个人**，绝不可归并；一方的昵称绝不归到另一方名下（如"梅尔"只属于梅拉妮）
- 翻译变体、简称、昵称、头衔+姓名 归并（"梅尔"→"梅拉妮"、"史密斯先生"→"约翰·史密斯"）
- 拿不准就不归并（保守：宁可分裂也不要错误合并，如两个不同的人同名时不归并）
- 人物称谓类主体（如"卡罗琳的姐姐"）保持原样，不与具体人名归并，除非文中明确同一

实体名列表：
{names}

输出 JSON：{{"别名1":"规范名","别名2":"规范名"}}（只输出需要归并的，无则输出 {{}}）"""


async def canonicalize(names: list[str], speakers: list[str], client: LLMClient,
                       conv_id: str) -> dict[str, str]:
    """返回 alias→canonical 映射（含说话人强制映射）。identity 映射不包含。"""
    forced = "；".join(f"{s}→{s}" for s in speakers)
    # 说话人本身永远规范；提取昵称映射交给 LLM（forced 提示里已给出全名）
    uniq = sorted({n.strip() for n in names if n and n.strip()})
    mapping: dict[str, str] = {}
    if uniq:
        raw = await client.chat(
            role="locomo_util", namespace=ns(conv_id, "canon"),
            temperature=0.0, max_tokens=2048, json_mode=True,
            messages=[
                {"role": "system", "content": CANON_SYSTEM},
                {"role": "user", "content": CANON_TEMPLATE.format(
                    forced=forced, names="\n".join(f"- {n}" for n in uniq))},
            ],
        )
        try:
            m = re.search(r"\{.*\}", raw.content, re.S)
            obj = json.loads(m.group(0)) if m else {}
            for k, v in obj.items():
                k, v = str(k).strip(), str(v).strip()
                if k and v and k != v and k in set(uniq):
                    mapping[k] = v
        except Exception:
            pass
    return mapping
