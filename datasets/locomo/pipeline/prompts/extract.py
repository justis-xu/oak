"""原子事实抽取提示词（fast 模型，temp 0，json_mode，每 session 一次）。

蓝本：memory-schema-rsi extraction/atomic.py（一事一条/逐字保留/列表逐项拆/
日期锚定），中文化并强化主体归因与紧凑 JSON 契约。
"""
from __future__ import annotations

from ..schema_skeleton import DEFAULT_TOPICS, EXTRACTABLE_ETYPES, EXTRACTABLE_RELATIONS, FACT_TYPES

EXTRACT_SYSTEM = "你是一个精确的事实抽取引擎，只输出合法 JSON。"

EXTRACT_TEMPLATE = """从一个中文对话片段中抽取原子事实、实体和人物关系。宁多勿漏。

【会话序号】{session_no}
【会话日期】{session_date_iso}（{weekday}）原文: {session_date_raw}
【说话人】{speakers}
【对话片段】
{transcript}

抽取规则：
1. 原子：一个条目=关于一个主体的一个事实。"在X公司工作且讨厌这份工作"必须拆成两条。每个对话片段输出 15-45 条，宁多勿漏，但不要编造原文没有的内容。
1b. 覆盖清单（逐项自查，最易漏的打★）：★身份与状态（婚姻/感情、职业、学业、居住）；★感受与反应（对具体事件的情感回应："她很开心也很感恩"）；★象征与含义（"X代表/象征…""让她想起…"）；★具体细节（物品的样子/内容/文字/颜色/材质："牌子上写着…"）；★计划与意图；人际关系及称谓；事件（何时何地做了什么）；偏好与观点；数字与计数。
2. 主体归因（s 字段）：每条事实必须填"这是谁的事实"。说话人用规范名；第三方有名字用名字，无名字用"XX的Y"形式（如"卡罗琳的姐姐"）。转述他人的事，主体填被转述者，并在陈述中注明来源（如"梅拉妮转述"）。
3. 逐字保留：人名、宠物名、书名、品牌、数字、计数、年龄必须逐字（"三个孩子"不许写成"孩子们"）。关键数字同时在 n 字段填阿拉伯数字。**感受/评价/象征类短语沿用原文措辞逐字抄**（原文"对宇宙充满敬畏"就写"梅拉妮对宇宙充满敬畏"，不要改写成"感到震撼"）。
4. 列表逐项拆：提到的每本书/每个孩子/每个地点/每件物品单独成条。
5. 日期三填：o=原文时间表达逐字（如"上周日"、"去年"、"7月2日"；**陈述只要隐含任何时间信息——包括"昨天/今天/最近/当时/上个月"——o 必须逐字填**，没有才留空）；d=按【会话日期】锚定解析（显式日期优先；相对表达按会话日期推算；推不出就留空），格式 YYYY-MM-DD 或 YYYY-MM 或 YYYY；g=粒度（日/周/月/年/无）。
6. 类型 y 从 {fact_types} 中选。"事件"必须有三填日期。
7. 主题 tp 从词表选 1-2 个：{topics}。
8. 涉及实体 ev：该事实涉及的具体实体名（人/地点/组织/物品/活动，用规范名，含说话人）。泛指不填。
9. 出处 src：该事实依据的 dia_id（如 "D2:3"；跨多轮用分号连接 "D2:3;D2:5"）。
10. 只抽有意义的事实（事件/状态/偏好/观点/计划/关系/数量/背景），排除寒暄、问候、关于对话本身的元话。
11. 实体 e：本片段出现的重要实体，格式 [类型,名称,别名]，类型从 {etypes} 中选，别名没有就留空。
12. 人物关系 r：格式 [关系,人1,人2]，关系从 {relations} 中选（如 ["朋友","梅拉妮","卡罗琳"]）。没有就留空数组。

输出 JSON（键固定，值可为空数组/空串）：
{{"f":[{{"s":"主体","t":"陈述句（以主体开头）","y":"类型","d":"","g":"粒度","o":"","n":"","tp":["主题"],"ev":["实体名"],"src":"D2:3"}}],"e":[["人物","名称","别名"]],"r":[["关系","人1","人2"]]}}"""

REPAIR_TEMPLATE = """你上一轮输出的 JSON 无法解析。错误：{error}
请重新输出**只含一个合法 JSON 对象**的回答，格式与上次要求完全一致（键：f/e/r）。不要输出任何其他文字。"""

AUDIT_TEMPLATE = """对照原文检查已有原子事实清单，找出**遗漏**的事实。只补遗漏，不要重复已有内容，不要改写已有条目。

【会话序号】{session_no}　【会话日期】{session_date_iso}　【说话人】{speakers}
【已有事实清单】
{facts}

【对话片段】
{transcript}

重点检查：
- **描述性细节**：标志/告示/海报上写了什么、物品的具体样子、象征意义（"X代表温暖和快乐"）、书名/歌名/品牌逐字——这类最容易被漏。
- 人名/宠物名/数字/计数/年龄/日期/身份状态（单身、职业、学历）是否逐字保留；列表是否逐项抽取。
- 每条遗漏事实仍按原契约填 s/t/y/d/g/o/n/tp/ev/src。
若无遗漏，输出 {{"f":[],"e":[],"r":[]}}。

输出 JSON（同原契约）：{{"f":[...],"e":[...],"r":[...]}}"""


def render_extract_prompt(session, conv, topics: list[str] | None = None) -> str:
    transcript = "\n".join(
        f"[{t.dia_id}] {t.speaker}: {t.text}" for t in session.turns
    )
    return EXTRACT_TEMPLATE.format(
        session_no=session.no,
        session_date_iso=session.date_iso.isoformat() if session.date_iso else "未知",
        weekday=_weekday_cn(session.date_iso.weekday()) if session.date_iso else "未知",
        session_date_raw=session.date_raw or "未知",
        speakers="、".join(conv.speakers),
        transcript=transcript,
        fact_types="/".join(FACT_TYPES),
        topics="/".join(topics or DEFAULT_TOPICS),
        etypes="/".join(EXTRACTABLE_ETYPES),
        relations="/".join(EXTRACTABLE_RELATIONS),
    )


def _weekday_cn(wd: int) -> str:
    return "一二三四五六日"[wd]
