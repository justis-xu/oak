"""P4 领域函数（S2 v1：手工种子 + fixture 门；S3 judicator 循环可再生成/扩展）。

分工（OaK 终态层思想）：**函数给骨架、模型管措辞**——列举/时间线/存在性三类
高频题型由确定性函数聚合全量候选（不漏项），终答模型只做选择与措辞。
零向量：全部基于图遍历与词法。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .tools import ToolBox

LIST_HINT = re.compile(r"有哪些|哪些|都有什么|什么活动|什么作品|什么书|参加过什么")
TIME_HINT = re.compile(r"什么时候|哪天|哪一年|几月|何时|哪一次")
CAT_WORDS = {
    "活动": ("活动", "爱好", "运动", "兴趣"),
    "书": ("书", "著作", "读物"),
    "作品": ("作品", "画", "陶罐", "碗", "创作"),
    "宠物": ("宠物", "猫", "狗"),
}


def question_kind(question: str) -> str:
    if LIST_HINT.search(question):
        return "列举"
    if TIME_HINT.search(question):
        return "时间"
    return ""


def infer_category(question: str) -> str:
    for cat, words in CAT_WORDS.items():
        if any(w in question for w in words):
            return cat
    return "活动"


@dataclass
class DomainFunctions:
    tb: ToolBox

    # ---- 列举：主体相关**要素级**全量聚合（按涉及实体去重；不漏项交给模型选）----
    def 列举(self, 主体: str, 类别: str = "活动") -> list[dict]:
        tb = self.tb
        主体 = tb._resolve_name(主体)
        subjects = [主体] + [s for s in tb.subject_index if 主体 and 主体 in s and s != 主体]
        fids: list[str] = []
        for s in subjects:
            fids += tb.subject_index.get(s, [])
        fids = sorted(set(fids))
        etype = {"活动": "活动", "书": "物品", "作品": "物品", "宠物": "物品"}.get(类别, "活动")
        kws = CAT_WORDS.get(类别, (类别,))

        def _row(fid: str, element: str) -> dict:
            r = tb.facts[fid]
            return {"要素": element, "编号": fid, "主体": r.get("主体"),
                    "陈述": r.get("陈述"), "类型": r.get("类型"),
                    "日期": r.get("日期"), "日期原文": r.get("日期原文"),
                    "出处": r.get("出处")}

        seen: dict[str, dict] = {}
        # 路径1：经"涉及X"边挂载的实体要素（活动/物品），按要素聚合取首个支撑事实
        for fid in fids:
            for _, tgt, ed in tb.g.out_edges(tb.facts[fid]["__id__"], data=True):
                rel = ed.get("relation", "")
                if rel in ("涉及活动", "涉及物品") and tb.g.nodes[tgt].get("etype") == etype:
                    from oak.kg.graph import node_view
                    name = str(node_view(tb.g.nodes[tgt]).get("名称") or "")
                    if name and name not in seen:
                        seen[name] = _row(fid, name)
        # 路径2：事件/计划类事实中词面含类别词的（未被实体覆盖）
        for fid in fids:
            r = tb.facts[fid]
            stmt = str(r.get("陈述", ""))
            if r.get("类型") in ("事件", "计划") and any(k in stmt for k in kws) \
                    and not any(v["编号"] == fid for v in seen.values()):
                key = stmt[:12]
                if key not in seen:
                    seen[key] = _row(fid, stmt[:20])
        out = list(seen.values())
        out.sort(key=lambda x: (str(x.get("日期") or ""), x["编号"]))
        return out

    # ---- 时间线：主体（可加关键词）的全部带日期事实按序 ----
    def 时间线(self, 主体: str, 关键词: str = "", 起: str = "", 止: str = "") -> list[dict]:
        tb = self.tb
        obs, _ = tb.tool_查询时间线(主体=主体, 关键词=关键词, 日期起=起, 日期止=止, 限=50)
        rows = []
        for line in obs.splitlines():
            m = re.match(r"\[(\d+-\d+)\]", line)
            if m:
                rows.append(tb.facts.get(m.group(1), {}))
        return [r for r in rows if r]

    # ---- 存在性：主体过滤的关键词存在检查（对抗题的可证伪基础）----
    def 存在性(self, 主体: str, 关键词: str) -> dict:
        tb = self.tb
        obs, fids = tb.tool_按主体查事实(主体=主体, 关键词=[关键词], 限=10)
        direct = bool(fids)
        return {"存在": direct, "命中": sorted(fids), "主体": tb._resolve_name(主体)}

    # ---- 发布门：fixture 必须非空（search 类防空壳，OaK 规矩）----
    def fixture_check(self) -> dict:
        ok = {}
        for subj in self.tb.subject_index:
            if len(self.tb.subject_index[subj]) >= 50:   # 找一个事实多的人做 fixture
                ok = {"列举": len(self.列举(subj, "活动")) > 0,
                      "时间线": len(self.时间线(subj)) > 0,
                      "存在性": self.存在性(subj, "的") is not None}
                return {"fixture_subject": subj, **ok}
        return {"fixture_subject": None}


def render_skeleton(kind: str, rows: list[dict], question: str) -> str:
    """函数骨架 → 终答提示词里的确定性聚合块。"""
    if not rows:
        return ""
    lines = [f"——以下为领域函数「{kind}」的确定性聚合结果（已去重，供你选择作答要素）——"]
    for r in rows[:60]:
        d = r.get("日期") or ""
        o = r.get("日期原文") or ""
        dt = f"{d}" + (f"（原文:{o}）" if o else "")
        lines.append(f"[{r.get('编号')}] {r.get('主体')}｜{r.get('陈述')}"
                     + (f"｜{dt}" if dt else ""))
    return "\n".join(lines)
