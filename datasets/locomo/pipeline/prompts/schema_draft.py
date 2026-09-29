"""P1 需求分析 / P2 schema 起草提示词（strong 模型，一次性）。

数据纪律：输入只有 问题文本 + 类别分布 + 对话原文抽样（禁 gold answer）。
"""
from __future__ import annotations

P1_SYSTEM = "你是本体需求分析师，只输出合法 JSON。"

P1_TEMPLATE = """任务：为"中文长对话记忆问答"设计本体（知识图谱 schema）的需求分析。

这是一个两人在 8 个月内多轮长对话的记忆基准。本体已固定以下八类实体骨架（不可增删改名）：
人物、地点、组织、物品、活动、主题、会话、原子事实。
原子事实的属性已固定：编号/陈述/主体/类型(事件|状态|偏好|观点|计划|关系|数量|背景|其他)/日期/日期粒度/日期原文/数值/主题/出处。
人物-人物/人物-地点关系已固定：亲属/朋友/同事/伴侣/居住于/就职于/就读于。

你的任务：只做需求分析——
1. 主题词表：给出 15-30 个覆盖下面所有问题所需的主题词（两字到四字中文短语，互斥、贴合题面用词）。
2. 每类题型的作答依赖分析：每类各需要本体提供什么（一两句）。

【题型分布】
单跳（直接查一个事实）、多跳（跨多条事实组合）、时间（日期定位/推算）、开放域推理（依据多条事实推断）、对抗（对话中不存在答案，问的往往是把主体换成另一个人）。

【问题样例】（共 {n_questions} 题，类别分布 {cat_dist}）
{questions}

【对话原文抽样】
{transcript_sample}

输出 JSON：
{{"topics":["主题1","主题2"],"category_needs":{{"单跳":"...","多跳":"...","时间":"...","开放域":"...","对抗":"..."}}}}"""

P2_SYSTEM = "你是本体工程师，只输出一个合法 YAML 代码块。"

P2_TEMPLATE = """基于以下需求分析，扩展现有中文本体骨架。

硬约束：
- 不得删除、改名八类骨架实体，不得删改骨架的固定关系（归属于/涉及*/属于主题/记录于/亲属/朋友/同事/伴侣/居住于/就职于/就读于/喜爱）。
- 只允许：（a）给骨架实体增加属性（name/dtype，dtype ∈ string|int|float|date|bool）；（b）新增实体类型（必须有 primary_key 且主键在 attributes 中）；（c）新增关系（必须声明 domain 与 range，range 只能是单个类型）。
- 围绕五类题型的作答需求做最小扩展，不要过度设计。

【现有骨架】
{skeleton_yaml}

【需求分析】
{p1_json}

输出：一个 ```yaml 代码块，内容为**完整**的扩展后 schema（含骨架全部内容+你的扩展）。"""


def render_p1(questions: list[str], cat_dist: dict[str, int], transcript_sample: str) -> str:
    return P1_TEMPLATE.format(
        n_questions=len(questions),
        cat_dist="、".join(f"{k}{v}题" for k, v in sorted(cat_dist.items())),
        questions="\n".join(f"- {q}" for q in questions),
        transcript_sample=transcript_sample,
    )


def render_p2(skeleton_yaml: str, p1_json: str) -> str:
    return P2_TEMPLATE.format(skeleton_yaml=skeleton_yaml, p1_json=p1_json)
