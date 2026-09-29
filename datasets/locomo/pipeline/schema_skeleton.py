"""手工中文本体骨架（v1）——九类实体收缩为八类（事件由"类型=事件"的
原子事实直接承载：事实自带日期三填，时间线工具直接在事实上工作）。

LLM 抽取契约只有三种结构：原子事实 / 实体 / 人-人(人-地)关系；
记录于/参与/喜爱 等边全部由 build.py 程序化派生（oak 的 derive 思想）。
"""
from __future__ import annotations

from oak.schema.model import Schema

DEFAULT_TOPICS = [
    "家庭", "子女教育", "收养", "心理健康", "性别身份", "职业发展", "教育学习",
    "阅读书籍", "运动健身", "旅行", "饮食", "宠物动物", "音乐艺术", "医疗健康",
    "人际关系", "爱好手工", "志愿公益", "购物物品", "居住搬家", "财务",
    "节日庆典", "交通驾驶", "技术设备", "宗教信仰", "服装外貌",
]

SKELETON_YAML = """
meta:
  domain: 长对话个人记忆问答（LoCoMo 中文）
  version: skeleton-v1

entity_types:
  人物:
    description: 对话双方及一切被提及的人（亲属、朋友、同事等第三方；第三方无名字时用"XX的Y"形式）
    primary_key: [姓名]
    attributes:
      - {name: 姓名, dtype: string}
      - {name: 别名, dtype: string}
      - {name: 身份, dtype: string}
  地点:
    primary_key: [名称]
    attributes:
      - {name: 名称, dtype: string}
      - {name: 类别, dtype: string}
  组织:
    primary_key: [名称]
    attributes:
      - {name: 名称, dtype: string}
      - {name: 类别, dtype: string}
  物品:
    description: 含宠物、礼物、收藏、书籍、首饰等
    primary_key: [名称]
    attributes:
      - {name: 名称, dtype: string}
      - {name: 类别, dtype: string}
  活动:
    description: 爱好、运动、课程、志愿、比赛等
    primary_key: [名称]
    attributes:
      - {name: 名称, dtype: string}
      - {name: 类别, dtype: string}
  主题:
    description: 受控话题词表（P1 按题面起草；抽取端从中选，代码接受词表外的词并建节点）
    primary_key: [名称]
    attributes:
      - {name: 名称, dtype: string}
  会话:
    description: 时间线锚点（每 session 程序化建一个）
    primary_key: [序号]
    attributes:
      - {name: 序号, dtype: int}
      - {name: 日期, dtype: string}
      - {name: 星期, dtype: string}
  原子事实:
    description: 一事一条的最小知识单元，全图内容主体；主体归因是对抗题的防御根基
    primary_key: [编号]
    attributes:
      - {name: 编号, dtype: string}
      - {name: 陈述, dtype: string}
      - {name: 主体, dtype: string}
      - {name: 类型, dtype: string}
      - {name: 日期, dtype: string}
      - {name: 日期粒度, dtype: string}
      - {name: 日期原文, dtype: string}
      - {name: 数值, dtype: string}
      - {name: 主题, dtype: string}
      - {name: 出处, dtype: string}

relation_types:
  归属于:
    domain: 原子事实
    range: 人物
    description: 事实的主体（谁的）——对抗题主防线；系统建（源自抽取的 s 字段）
  涉及人物:
    domain: [原子事实]
    range: 人物
    description: 系统建（源自抽取的 ev 字段）
  涉及地点:
    domain: [原子事实]
    range: 地点
    description: 系统建
  涉及组织:
    domain: [原子事实]
    range: 组织
    description: 系统建
  涉及物品:
    domain: [原子事实]
    range: 物品
    description: 系统建
  涉及活动:
    domain: [原子事实]
    range: 活动
    description: 系统建
  属于主题:
    domain: 原子事实
    range: 主题
    description: 系统建（源自抽取的 tp 字段）
  记录于:
    domain: 原子事实
    range: 会话
    description: 系统按出处 dia_id 派生
  亲属:
    domain: 人物
    range: 人物
    description: 明细（祖母/姐姐…）由关联的原子事实承载
  朋友: {domain: 人物, range: 人物}
  同事: {domain: 人物, range: 人物}
  伴侣: {domain: 人物, range: 人物}
  居住于: {domain: 人物, range: 地点}
  就职于: {domain: 人物, range: 组织}
  就读于: {domain: 人物, range: 组织}
  喜爱:
    domain: 人物
    range: 活动
    description: 系统从"类型=偏好"且态度为正的原子事实派生

axioms:
  - {kind: disjoint, classes: [人物, 地点, 组织, 物品, 活动, 主题, 会话, 原子事实]}
  - {kind: domain, relation: 归属于, class: 原子事实}
  - {kind: range, relation: 归属于, class: 人物}
"""

# 抽取端允许的实体类型白名单（会话/原子事实由系统建）
EXTRACTABLE_ETYPES = ["人物", "地点", "组织", "物品", "活动"]
# 抽取端允许的关系白名单（其余关系系统派生）
EXTRACTABLE_RELATIONS = ["亲属", "朋友", "同事", "伴侣", "居住于", "就职于", "就读于"]
FACT_TYPES = ["事件", "状态", "偏好", "观点", "计划", "关系", "数量", "背景", "其他"]
GRANULARITIES = ["日", "周", "月", "年", "无"]


def load_skeleton(topics: list[str] | None = None) -> Schema:
    """骨架 + 主题词表 → Schema。"""
    schema = Schema.from_yaml(SKELETON_YAML)
    errs = schema.validate()
    if errs:
        raise ValueError(f"骨架 schema 校验失败: {errs}")
    return schema


def merge_schema(draft_yaml: str) -> Schema:
    """骨架与 P2 草稿合并：同名实体取属性并集（主键以骨架为准）、关系取并集。
    草稿解析或校验失败 → 回退纯骨架。"""
    try:
        draft = Schema.from_yaml(draft_yaml)
        if draft.validate():
            return load_skeleton()
        base = load_skeleton()
        base_ent = {e.name: e for e in base.entities}
        for e in draft.entities:
            if e.name in base_ent:
                cur = base_ent[e.name]
                have = {a.name for a in cur.attributes}
                for a in e.attributes:
                    if a.name not in have and a.name not in cur.primary_key:
                        cur.attributes.append(a)
            else:
                base.entities.append(e)
        base_rel = {r.name for r in base.relations}
        for r in draft.relations:
            if r.name not in base_rel:
                base.relations.append(r)
        if base.validate():
            return base
        return load_skeleton()
    except Exception:
        return load_skeleton()
