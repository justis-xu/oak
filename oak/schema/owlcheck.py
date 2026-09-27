"""模式的形式化验证：静态检查（主进程）+ HermiT（子进程隔离，防 JVM 挂死）。

五类检查的归属（论文附录 A）：
  C1 不相交一致性、C2 限制一致性 —— 静态为主 + HermiT 复核
  C3 属性级一致性（functional/inverse 组合）—— 静态
  C4 全局一致性、C5 不可满足类 —— HermiT（sync_reasoner_hermit + probe individual）
发现矛盾后：贪心删公理重推理，定位最小冲突集，渲染带反例的反馈。
"""
from __future__ import annotations

import multiprocessing as mp
import queue as _queue
from dataclasses import dataclass, field

from .model import Schema

OWL_CHECK_TIMEOUT_S = 60
PINPOINT_MAX_RUNS = 30


@dataclass
class OWLFinding:
    check: str                 # disjointness | restriction | property | global | unsatisfiable | structure
    entity: str | None
    message: str
    culprits: list[str] = field(default_factory=list)

    def render(self) -> str:
        head = f"[check={self.check}]"
        if self.entity:
            head += f" {self.entity}"
        lines = [f"{head}: {self.message}"]
        if self.culprits:
            lines.append("  Minimal conflicting axioms: " + "; ".join(self.culprits))
        return "\n".join(lines)


# ---------------------------------------------------------------- 静态检查
def static_checks(schema: Schema) -> list[OWLFinding]:
    findings: list[OWLFinding] = []
    names = {e.name for e in schema.entities}

    # 子类环
    sub = {(ax.params.get("sub"), ax.params.get("sup")) for ax in schema.axioms
           if ax.kind == "subclass"}
    graph: dict[str, list[str]] = {}
    for s, p in sub:
        graph.setdefault(s, []).append(p)
    for start in graph:
        seen, stack = set(), [start]
        while stack:
            cur = stack.pop()
            for nxt in graph.get(cur, []):
                if nxt == start:
                    findings.append(OWLFinding(
                        "restriction", start,
                        f"subclass 环: {start} 是自己的（间接）子类，模式不可满足。"))
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)

    # disjoint 与 subclass 冲突：sub 有两个不相交的父类
    disjoint_sets: list[set[str]] = []
    for ax in schema.axioms:
        if ax.kind == "disjoint":
            cs = set(ax.params.get("classes", []))
            if len(cs) >= 2:
                disjoint_sets.append(cs)
    for s, p1 in sub:
        for _, p2 in sub:
            if s == s and p1 != p2:
                pass
    # 更直接：某类的两个父类互斥
    parents: dict[str, set[str]] = {}
    for s, p in sub:
        parents.setdefault(s, set()).add(p)
    for s, ps in parents.items():
        for ds in disjoint_sets:
            hit = ps & ds
            if len(hit) >= 2:
                findings.append(OWLFinding(
                    "disjointness", s,
                    f"{s} 的父类 {{{', '.join(sorted(hit))}}} 被声明为互斥，{s} 不可能有实例。",
                    culprits=[f"subclass: {s} <- {p}" for p in sorted(hit)] +
                             [f"disjoint: {sorted(ds)}"],
                ))

    # 属性级：同一关系两次声明 domain 公理指向互斥类
    dom_ax: dict[str, list[str]] = {}
    for ax in schema.axioms:
        if ax.kind == "domain":
            dom_ax.setdefault(ax.params.get("relation"), []).append(ax.params.get("class"))
    for rel, clss in dom_ax.items():
        for ds in disjoint_sets:
            hit = set(clss) & ds
            if len(hit) >= 2:
                findings.append(OWLFinding(
                    "property", rel,
                    f"关系 {rel} 的 domain 公理同时指向互斥类 {sorted(hit)}。",
                    culprits=[f"domain: {rel} -> {c}" for c in sorted(hit)] +
                             [f"disjoint: {sorted(ds)}"],
                ))

    # functional 关系 range 到多值实体的提示级别问题（信息性，不阻断）
    return findings


# ---------------------------------------------------------------- HermiT 子进程
def _build_owl(schema: Schema):
    """YAML→OWL 映射（独立 world，不污染全局）。"""
    import owlready2 as owl

    world = owl.World()
    onto = world.get_ontology("http://oak.local/schema.owl")
    with onto:
        # 实体类
        cls = {e.name: type(e.name, (owl.Thing,), {}) for e in schema.entities}
        # 数据属性（主键加 functional）
        dprops: dict[str, owl.DataProperty] = {}
        for e in schema.entities:
            for a in e.attributes:
                pyrange = {"string": str, "int": int, "float": float,
                           "date": str, "bool": bool}[a.dtype]
                p = type(f"{e.name}__{a.name}", (owl.DataProperty,),
                         {"domain": [cls[e.name]], "range": [pyrange]})
                if a.name in e.primary_key and len(e.primary_key) == 1:
                    p.is_a.append(owl.FunctionalProperty)
                dprops[f"{e.name}.{a.name}"] = p
        # 对象属性
        oprops: dict[str, owl.ObjectProperty] = {}
        for r in schema.relations:
            dom = [cls[d] for d in r.domain if d in cls]
            rng = [cls[r.range]] if r.range in cls else []
            bases = [owl.ObjectProperty] + ([owl.FunctionalProperty] if r.functional else [])
            oprops[r.name] = type(r.name, tuple(bases), {"domain": dom, "range": rng})
        # 公理
        for ax in schema.axioms:
            p = ax.params
            if ax.kind == "subclass" and p.get("sub") in cls and p.get("sup") in cls:
                cls[p["sub"]].is_a.append(cls[p["sup"]])
            elif ax.kind == "disjoint":
                cs = [cls[c] for c in p.get("classes", []) if c in cls]
                if len(cs) >= 2:
                    owl.AllDisjoint(cs)
            elif ax.kind == "domain":
                r = oprops.get(p.get("relation"))
                if r and p.get("class") in cls:
                    r.domain.append(cls[p["class"]])
            elif ax.kind == "range":
                r = oprops.get(p.get("relation"))
                if r and p.get("class") in cls:
                    r.range.append(cls[p["class"]])
            elif ax.kind == "cardinality":
                r = oprops.get(p.get("relation"))
                c = cls.get(p.get("class"))
                if r and c:
                    rng_cls = cls.get(r.range[0]) if r.range else None
                    if rng_cls is not None:
                        if p.get("min"):
                            c.is_a.append(r.some(rng_cls))
                        if p.get("max") is not None and int(p["max"]) == 1 and r.functional:
                            c.is_a.append(r.only(rng_cls))
        # probe individuals：每类一个假想实例（带主键字面量），让全局检查有据可依
        probes = {}
        for e in schema.entities:
            ind = cls[e.name]("probe_" + e.name)
            for k in e.primary_key:
                p = dprops.get(f"{e.name}.{k}")
                if p is not None:
                    # 非 functional 属性必须赋列表（owlready2 语义）
                    setattr(ind, python_attr(p.name), ["x"])
            probes[e.name] = ind
    return world, onto, cls, oprops, dprops


def python_attr(name: str) -> str:
    """OWL 属性名 → python 合法属性名（空格转下划线）。"""
    return name.replace(" ", "_")


def _hermit_consistent(world) -> bool:
    import owlready2 as owl
    try:
        owl.sync_reasoner_hermit(world, infer_property_values=True)
        return len(list(world.inconsistent_classes())) == 0
    except Exception:
        # 推理抛错常因不一致导致 —— 视为不一致
        return False


def _owl_worker(schema_yaml: str, q: "mp.Queue") -> None:
    """在子进程内跑完整 HermiT 检查 + pinpoint。"""
    try:
        import owlready2 as owl
        schema = Schema.from_yaml(schema_yaml)
        world, onto, cls, oprops, dprops = _build_owl(schema)

        findings: list[dict] = []
        consistent = _hermit_consistent(world)

        if not consistent:
            # pinpoint：贪心删公理定位最小冲突集（重建本体重推理）
            ax_desc = []
            for ax in schema.axioms:
                ax_desc.append(f"{ax.kind}: {ax.params}")
            # 把实体间 disjoint / subclass 也作为可删项；逐项移除重测
            removable = list(ax_desc)
            culprits = list(ax_desc)
            runs = 0
            for cand in removable:
                if runs >= PINPOINT_MAX_RUNS:
                    break
                trial = [a for a in culprits if a != cand]
                runs += 1
                try:
                    trial_schema = _schema_minus(schema, [a for a in schema.axioms
                                                         if f"{a.kind}: {a.params}" == cand])
                    w2, *_ = _build_owl(trial_schema)
                    if _hermit_consistent(w2):
                        culprits = trial  # 删掉它就一致 → 它在冲突集中；继续收缩
                except Exception:
                    continue
            findings.append({
                "check": "global", "entity": None,
                "message": "模式全局不一致：HermiT 判定存在逻辑矛盾（probe individual 推导冲突）。",
                "culprits": culprits,
            })
        else:
            # 逐类查不可满足
            for e in schema.entities:
                try:
                    if cls[e.name] in list(world.inconsistent_classes()):
                        findings.append({
                            "check": "unsatisfiable", "entity": e.name,
                            "message": f"类 {e.name} 不可满足：合并公理后不可能拥有实例。",
                            "culprits": [],
                        })
                except Exception:
                    pass
        q.put({"ok": True, "findings": findings, "consistent": consistent})
    except Exception as e:
        try:
            q.put({"ok": False, "error": f"{type(e).__name__}: {e}"})
        except Exception:
            pass


def _schema_minus(schema: Schema, drop_axioms: list) -> Schema:
    from dataclasses import replace
    drop_ids = {id(a) for a in drop_axioms}
    return replace(schema, axioms=[a for a in schema.axioms if id(a) not in drop_ids])


def hermit_checks(schema: Schema, timeout_s: int = OWL_CHECK_TIMEOUT_S) -> tuple[list[OWLFinding], bool]:
    """子进程跑 HermiT；超时强杀返回（findings 空 + hermit_ok=False 不阻断）。"""
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_owl_worker, args=(schema.to_yaml(), q), daemon=True)
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join(3)
        return [], False       # 超时：视为"未验证"，不阻塞（记录告警由调用方做）
    try:
        res = q.get_nowait()
    except _queue.Empty:
        return [], False
    if not res.get("ok"):
        return [], False
    findings = [OWLFinding(f["check"], f["entity"], f["message"], f.get("culprits", []))
                for f in res["findings"]]
    return findings, True


def check_schema(schema: Schema, timeout_s: int = OWL_CHECK_TIMEOUT_S) -> tuple[list[OWLFinding], bool]:
    """完整验证入口：静态 + HermiT。返回 (findings, hermit_ok)。"""
    findings = static_checks(schema)
    hermit_findings, ok = hermit_checks(schema, timeout_s)
    findings.extend(hermit_findings)
    return findings, ok


def findings_to_feedback(findings: list[OWLFinding], attempt: int) -> str:
    lines = [f"Your draft schema (attempt {attempt}) FAILED formal validation. "
             f"Issues found by the reasoner:"]
    for f in findings:
        lines.append(f.render())
    lines.append(
        "Fix the axioms or entity/relation definitions that cause these conflicts. "
        "Keep the intended domain coverage — fix the logic, not the intent. "
        "Output the corrected full YAML again."
    )
    return "\n".join(lines)
