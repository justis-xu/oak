"""中文图算子库 + 词法索引（零向量）。

四层递进检索：按主体（单跳主力）→ 获取相关事实（多跳枢纽）→ 检索事实
（unigram+bigram 混合 IDF 兜底）→ 主题浏览。全部确定性/词法，无任何嵌入。

iter2 修复（iter1 归因：检索miss 50 / 误拒答 13）：
- 混合索引：单字 IDF 命中让"慈善长跑"↔"公益跑"（共享"跑"）这类译名漂移可召回
- 按主体查事实：关键词无命中时不再返回空——退化为词面相似度排序并提示；
  加 类型/页 参数，观察里报告总数，支持翻页收集全
"""
from __future__ import annotations

import math
import re
from datetime import date

import networkx as nx

from oak.kg.graph import node_view

from .dates import resolve_relative

FACT_ETYPE = "原子事实"
_LIMIT_DEFAULT = 25


def _fact_line(row: dict, anchor: str = "") -> str:
    d = row.get("日期") or ""
    g = row.get("日期粒度") or ""
    o = row.get("日期原文") or ""
    dt = f"{d}({g}" + (f"|原文:{o}" if o else "") + ")" if d else (f"原文:{o}" if o else "无日期")
    if o and anchor:
        dt += f"｜锚:{anchor}"           # 相对日期的推算基准（该事实所在会话的日期）
    return (f"[{row.get('编号')}] {row.get('主体')}｜{row.get('陈述')}"
            f"｜{row.get('类型')}｜{dt}｜出处:{row.get('出处', '')}")


def _terms(text: str) -> set[str]:
    """unigram + bigram 词面项（去空白）。"""
    t = re.sub(r"\s+", "", str(text))
    grams = set(t)
    grams |= {t[i:i + 2] for i in range(len(t) - 1)}
    return grams if t else set()


class LexicalIndex:
    """unigram+bigram 混合倒排 + IDF 打分。可索引文本 = 陈述 + 主体 + 主题 + 涉及实体名。"""

    def __init__(self, facts: list[dict], entity_names: dict[str, list[str]]):
        self.docs: dict[str, str] = {}
        for f in facts:
            extra = " ".join(entity_names.get(f["编号"], []))
            self.docs[f["编号"]] = f"{f['陈述']} {f.get('主体', '')} {f.get('主题', '')} {extra}"
        self.posting: dict[str, set[str]] = {}
        for fid, text in self.docs.items():
            for g in _terms(text):
                self.posting.setdefault(g, set()).add(fid)
        n = max(len(self.docs), 1)
        self.idf = {g: math.log(1 + n / len(ps)) * (2.0 if len(g) == 1 else 3.0)
                    for g, ps in self.posting.items()}

    def score(self, query: str) -> dict[str, float]:
        """query → {fid: 归一化得分}。"""
        scores: dict[str, float] = {}
        for g in _terms(query):
            for fid in self.posting.get(g, ()):
                scores[fid] = scores.get(fid, 0.0) + self.idf.get(g, 0.0)
        return scores

    def search(self, query: str, limit: int = 20) -> list[tuple[str, float]]:
        s = self.score(query)
        if not s:
            return []
        mx = max(s.values())
        out = sorted(((fid, v / mx) for fid, v in s.items()), key=lambda x: (-x[1], x[0]))
        return out[:limit]


class ToolBox:
    """作答 agent 的全部工具。execute() 返回 (observation_text, 涉及的事实编号集)。"""

    def __init__(self, graph: nx.MultiDiGraph):
        self.g = graph
        self.facts: dict[str, dict] = {}          # 编号 -> row
        self.entities: dict[str, dict] = {}       # 名称/姓名 -> {__id__, etype, props}
        self.alias: dict[str, str] = {}           # 别名 -> 规范名
        for nid, nd in graph.nodes(data=True):
            et = nd.get("etype")
            view = node_view(nd)
            if et == FACT_ETYPE:
                self.facts[view.get("编号", "")] = {"__id__": nid, **view}
            elif et:
                name = view.get("姓名") or view.get("名称") or view.get("序号")
                if name:
                    self.entities[str(name)] = {"__id__": nid, "etype": et, **view}
                    for a in re.split(r"[;；]", str(view.get("别名") or "")):
                        if a.strip():
                            self.alias[a.strip()] = str(name)
        # 事实 → 涉及实体名（索引文本用）
        self.fact_entities: dict[str, list[str]] = {fid: [] for fid in self.facts}
        for fid, row in self.facts.items():
            names = [str(row.get("主体", ""))]
            for _, tgt, ed in graph.out_edges(row["__id__"], data=True):
                if ed.get("relation", "").startswith("涉及"):
                    tv = node_view(graph.nodes[tgt])
                    names.append(str(tv.get("姓名") or tv.get("名称") or ""))
            self.fact_entities[fid] = [n for n in names if n]
        # 会话日期锚（相对日期的推算基准）：序号 -> iso
        self.session_dates: dict[int, str] = {}
        for nid, nd in graph.nodes(data=True):
            if nd.get("etype") == "会话":
                v = node_view(nd)
                try:
                    self.session_dates[int(v.get("序号"))] = str(v.get("日期") or "")
                except (TypeError, ValueError):
                    pass
        self.index = LexicalIndex(list(self.facts.values()), self.fact_entities)
        self.subject_index: dict[str, list[str]] = {}
        for fid in sorted(self.facts):
            s = str(self.facts[fid].get("主体", ""))
            if s:
                self.subject_index.setdefault(s, []).append(fid)

    # ---------------------------------------------------------------- 基础
    def _resolve_name(self, n: str) -> str:
        """直接名优先：名字本身是已知实体/主体时绝不走别名（防脏别名重定向）。"""
        n = str(n or "")
        if n in self.entities or n in self.subject_index:
            return n
        return self.alias.get(n, n)

    def _entity_row(self, name: str) -> dict | None:
        return self.entities.get(self._resolve_name(name))

    def _render(self, fids: list[str], limit: int, total: int | None = None) -> tuple[str, set[str]]:
        shown = fids[:limit]
        lines = [self.fact_line(f) for f in shown if f in self.facts]
        total = total if total is not None else len(fids)
        head = f"（共{total}条" + (f"，显示第1-{len(shown)}条，可用 页 参数翻页）"
                if total > len(shown) else "）")
        return (head + "\n" + ("\n".join(lines) if lines else "（无结果）"),
                {f for f in shown if f in self.facts})

    def fact_line(self, fid: str) -> str:
        row = self.facts.get(fid)
        if row is None:
            return ""
        anchor = ""
        m = re.match(r"D(\d+):", str(row.get("出处") or ""))
        if m:
            anchor = self.session_dates.get(int(m.group(1)), "")
        return _fact_line(row, anchor)

    @staticmethod
    def _date_ok(row: dict, lo: str, hi: str) -> bool:
        d = str(row.get("日期") or "")
        if lo and (not d or d < lo):
            return False
        if hi and (not d or d > hi):
            return False
        return True

    # ---------------------------------------------------------------- 工具
    def tool_查找实体(self, 名称: str = "", 类型: str = "") -> tuple[str, set[str]]:
        out = []
        for name, e in self.entities.items():
            if 名称 and 名称 not in name and 名称 not in str(e.get("别名", "")):
                continue
            if 类型 and 类型 != e["etype"]:
                continue
            props = "; ".join(f"{k}={v}" for k, v in e.items()
                              if not str(k).startswith("__") and k not in ("名称", "姓名", "etype")
                              and str(v) and k != "别名")
            out.append(f"〈{e['etype']}〉{name}" + (f"｜{props}" if props else ""))
        out = out[:30]
        return ("\n".join(out) if out else "（无结果）", set())

    def tool_按主体查事实(self, 主体: str = "", 关键词: list | None = None,
                         类型: str = "", 日期起: str = "", 日期止: str = "",
                         页: int = 1, 限: int = _LIMIT_DEFAULT) -> tuple[str, set[str]]:
        """单跳主力。关键词有直接命中→按词面得分排序；无命中→退化为主体事实
        相似度排序并提示（iter2 修复：不再返回空导致误拒答）。"""
        主体 = self._resolve_name(主体)
        kws = [str(k) for k in (关键词 or []) if str(k).strip()]
        fids = list(self.subject_index.get(主体, []))
        # 部分匹配双向：query⊂subject（"卡罗琳"⊂"卡罗琳的奶奶"）与 subject⊂query（"卡罗琳的祖母"⊃"卡罗琳"）
        partial = [f for s, fl in self.subject_index.items()
                   if 主体 and s != 主体 and (主体 in s or (len(主体) > 2 and s in 主体))
                   for f in fl]
        fids = fids + partial
        if 类型:
            fids = [f for f in fids if str(self.facts[f].get("类型", "")) == 类型]
        fids = [f for f in fids if self._date_ok(self.facts[f], 日期起, 日期止)]
        note = ""
        if kws:
            hit = [f for f in fids
                   if any(k in str(self.facts[f].get("陈述", "")) for k in kws)]
            if hit:
                scores = self.index.score(" ".join(kws))
                hit.sort(key=lambda f: (-scores.get(f, 0.0), f))
                fids = hit
            else:
                scores = self.index.score(" ".join(kws))
                ranked = sorted(fids, key=lambda f: (-scores.get(f, 0.0), f))
                keep = [f for f in ranked if scores.get(f, 0.0) > 0]
                fids = keep or ranked[:限]
                note = f"（关键词{kws}在主体事实中无直接命中；以下按词面相似度排序，注意甄别）\n"
        start = (max(1, int(页)) - 1) * int(限)
        page = fids[start:start + int(限)]
        text, got = self._render(page, int(限), total=len(fids))
        return note + text, got

    def tool_获取相关事实(self, 实体名: str = "", 限: int = 30,
                         页: int = 1) -> tuple[str, set[str]]:
        e = self._entity_row(实体名)
        if not e:
            return f"（未找到实体：{实体名}。可用 查找实体 确认名称）", set()
        fids = []
        for src, _, ed in self.g.in_edges(e["__id__"], data=True):
            if ed.get("relation", "").startswith("涉及") or ed.get("relation") == "归属于":
                sv = node_view(self.g.nodes[src])
                if sv.get("编号"):
                    fids.append(sv["编号"])
        fids.sort()
        start = (max(1, int(页)) - 1) * int(限)
        return self._render(fids[start:start + int(限)], int(限), total=len(fids))

    def tool_检索事实(self, 关键词: str = "", 主体: str = "", 限: int = _LIMIT_DEFAULT
                     ) -> tuple[str, set[str]]:
        if not str(关键词).strip():
            return "（关键词为空）", set()
        hits = self.index.search(str(关键词))
        fids = [fid for fid, _ in hits]
        if 主体:
            主体 = self._resolve_name(主体)
            subj_fids = set(self.subject_index.get(主体, []))
            fids = [f for f in fids if f in subj_fids] or fids
            fids = sorted(fids, key=lambda f: f not in subj_fids)
        text, got = self._render(fids, int(限), total=len(fids))
        return text, got

    def tool_查询时间线(self, 主体: str = "", 关键词: str = "",
                        日期起: str = "", 日期止: str = "",
                        限: int = _LIMIT_DEFAULT) -> tuple[str, set[str]]:
        主体 = self._resolve_name(主体)
        kws = [k for k in re.split(r"[、,，\s]+", str(关键词)) if k]
        rows = [r for r in self.facts.values() if str(r.get("日期"))]
        if 主体:
            rows = [r for r in rows if str(r.get("主体")) == 主体]
        if kws:
            scored = self.index.score(" ".join(kws))
            rows = [r for r in rows if scored.get(r["编号"], 0.0) > 0
                    or any(k in str(r.get("陈述", "")) for k in kws)]
            rows.sort(key=lambda r: (-scored.get(r["编号"], 0.0), str(r.get("日期"))))
        else:
            rows.sort(key=lambda r: (str(r.get("日期")), str(r.get("编号"))))
        rows = [r for r in rows if self._date_ok(r, 日期起, 日期止)]
        return self._render([r["编号"] for r in rows], int(限), total=len(rows))

    def tool_计算日期(self, 基准日期: str = "", 相对表达: str = "") -> tuple[str, set[str]]:
        m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", str(基准日期).strip())
        if not (m and 相对表达):
            return ("（参数格式：基准日期=YYYY-MM-DD；相对表达=中文相对时间，"
                    "如'上周日'、'3个月前'、'5月25日之前的那个周日'）", set())
        anchor = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        val, gran = resolve_relative(anchor, str(相对表达))
        if not val:
            return f"（无法解析：{相对表达}）", set()
        return f"{相对表达} → {val}（粒度:{gran}，基准 {基准日期}）", set()

    def tool_列出主题(self) -> tuple[str, set[str]]:
        cnt: dict[str, int] = {}
        for r in self.facts.values():
            for tp in re.split(r"[;；]", str(r.get("主题") or "")):
                if tp.strip():
                    cnt[tp.strip()] = cnt.get(tp.strip(), 0) + 1
        lines = [f"{k}（{v}条）" for k, v in sorted(cnt.items(), key=lambda x: -x[1])]
        return ("\n".join(lines) if lines else "（无主题）", set())

    def tool_按主题查事实(self, 主题: str = "", 主体: str = "", 限: int = 30
                         ) -> tuple[str, set[str]]:
        主体 = self._resolve_name(主体)
        fids = [r["编号"] for r in self.facts.values()
                if 主题 in str(r.get("主题") or "")
                and (not 主体 or str(r.get("主体")) == 主体)]
        return self._render(fids, int(限), total=len(fids))

    def tool_遍历关系(self, 实体名: str = "", 关系: str = "", 方向: str = "出",
                      跳数: int = 1) -> tuple[str, set[str]]:
        e = self._entity_row(实体名)
        if not e:
            return f"（未找到实体：{实体名}）", set()
        frontier, visited, fids = {e["__id__"]}, set(), []
        for _ in range(max(1, min(int(跳数), 2))):
            nxt = set()
            for nid in frontier:
                if 方向 == "入":
                    edges = list(self.g.in_edges(nid, data=True))
                elif 方向 == "双":
                    edges = list(self.g.out_edges(nid, data=True)) + \
                        list(self.g.in_edges(nid, data=True))
                else:
                    edges = list(self.g.out_edges(nid, data=True))
                for a, b, ed in edges:
                    if str(关系) and ed.get("relation") != str(关系):
                        continue
                    other = b if a == nid else a
                    if other in visited:
                        continue
                    nxt.add(other)
            visited |= nxt
            frontier = nxt
        lines = []
        for nid in visited:
            nd = self.g.nodes[nid]
            v = node_view(nd)
            if nd.get("etype") == FACT_ETYPE:
                fids.append(v.get("编号"))
                lines.append(self.fact_line(v.get("编号")))
            else:
                lines.append(f"〈{nd.get('etype')}〉{v.get('姓名') or v.get('名称') or v.get('序号')}")
        lines = lines[:40]
        return ("\n".join(lines) if lines else "（无结果）",
                {f for f in fids if f})

    # ---------------------------------------------------------------- 注册表
    TOOL_SPECS: list[dict] = [
        {"name": "查找实体", "sig": '查找实体(名称, 类型="")',
         "desc": "按名称/别名子串查实体（人物/地点/组织/物品/活动/主题）"},
        {"name": "按主体查事实", "sig": '按主体查事实(主体, 关键词=[], 类型="", 日期起="", 日期止="", 页=1, 限=25)',
         "desc": "【单跳主力】某人的事实。关键词无命中会自动按词面相似度排序；类型=事件/状态/偏好/观点/计划/关系/数量/背景 可过滤；可翻页收集全"},
        {"name": "获取相关事实", "sig": '获取相关事实(实体名, 限=30, 页=1)',
         "desc": "【多跳枢纽】与某实体（人/物/活动/地点/组织）相关的全部事实；可翻页"},
        {"name": "检索事实", "sig": '检索事实(关键词, 主体="", 限=25)',
         "desc": "【兜底】全图词法检索（单字+双字混合），换用内容词/同义词多次尝试"},
        {"name": "查询时间线", "sig": '查询时间线(主体="", 关键词="", 日期起="", 日期止="", 限=25)',
         "desc": "按日期排序/过滤的事实（时间题先查这个）；关键词支持词面相似度"},
        {"name": "计算日期", "sig": "计算日期(基准日期, 相对表达)",
         "desc": "日历推算：'上周日'/'3个月前'/'5月25日之前的那个周日' → 具体日期"},
        {"name": "列出主题", "sig": "列出主题()",
         "desc": "主题词表+各主题事实数（词法路由失败时浏览入口）"},
        {"name": "按主题查事实", "sig": '按主题查事实(主题, 主体="", 限=30)',
         "desc": "按主题过滤事实"},
        {"name": "遍历关系", "sig": '遍历关系(实体名, 关系, 方向="出", 跳数=1)',
         "desc": "沿关系遍历（亲属/朋友/同事/伴侣/喜爱/居住于等），方向 ∈ 出|入|双"},
    ]

    def execute(self, name: str, args: dict) -> tuple[str, set[str]]:
        fn = getattr(self, f"tool_{name}", None)
        if fn is None:
            valid = "、".join(t["name"] for t in self.TOOL_SPECS)
            return f"（未知工具：{name}。可用：{valid}）", set()
        clean = {}
        for k, v in args.items():
            clean[str(k).strip()] = v
        try:
            return fn(**clean)
        except TypeError as e:
            return f"（参数错误：{e}）", set()
        except Exception as e:  # noqa: BLE001
            return f"（工具执行错误：{e!r}）", set()


def render_tool_docs() -> str:
    return "\n".join(f"- {t['sig']}  # {t['desc']}" for t in ToolBox.TOOL_SPECS)
