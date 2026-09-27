"""模式（本体）的数据结构与 YAML 严格解析。

文法：
  entity_types:
    Flight:
      description: ...
      primary_key: [Flight Number, FlightDate]     # 有序，属性名保持语料原始大小写
      attributes:
        - {name: Flight Number, dtype: string}     # dtype ∈ string|int|float|date|bool
      attribute_aliases: {原始异名: 规范名}          # 可选
  relation_types:
    departs_from: {domain: Flight, range: City, functional: true, description: ...}
  axioms:
    - {kind: subclass, sub: A, sup: B}
    - {kind: disjoint, classes: [A, B, C]}
    - {kind: domain, relation: r, class: A}
    - {kind: range, relation: r, class: B}
    - {kind: cardinality, relation: r, class: A, min: 1, max: 1}
    - {kind: key_functional, entity: A, key: [k1, k2]}
"""
from __future__ import annotations

from dataclasses import dataclass, field

import yaml

DTYPES = {"string", "int", "float", "date", "bool"}
AXIOM_KINDS = {"subclass", "disjoint", "domain", "range", "cardinality", "key_functional"}


class SchemaParseError(ValueError):
    pass


@dataclass
class Attribute:
    name: str
    dtype: str = "string"

    @staticmethod
    def from_dict(d: dict) -> "Attribute":
        if not isinstance(d, dict) or "name" not in d:
            raise SchemaParseError(f"attribute 需要 {{name, dtype}}，得到: {d!r}")
        name = str(d["name"]).strip()
        dtype = str(d.get("dtype", "string")).strip()
        if dtype not in DTYPES:
            raise SchemaParseError(f"attribute {name!r} 的 dtype 非法: {dtype!r}（允许 {DTYPES}）")
        return Attribute(name=name, dtype=dtype)


@dataclass
class EntityType:
    name: str
    primary_key: list[str]
    attributes: list[Attribute] = field(default_factory=list)
    description: str = ""
    attribute_aliases: dict[str, str] = field(default_factory=dict)

    def attr_names(self) -> set[str]:
        return {a.name for a in self.attributes}

    def canonical_attr(self, raw: str) -> str | None:
        """语料异名 → 规范属性名；未注册则原样返回（若恰是规范名）。"""
        if raw in self.attr_names():
            return raw
        return self.attribute_aliases.get(raw)


@dataclass
class RelationType:
    name: str
    domain: list[str]           # 允许 list（union）
    range: str
    functional: bool = False
    inverse_of: str | None = None
    description: str = ""
    derive: dict | None = None  # {"attr": <字段>} —— 从属性值派生该关系（LLM 不抽）


@dataclass
class Axiom:
    kind: str
    params: dict

    @staticmethod
    def from_dict(d: dict) -> "Axiom":
        if not isinstance(d, dict) or "kind" not in d:
            raise SchemaParseError(f"axiom 需要 kind，得到: {d!r}")
        kind = str(d["kind"]).strip()
        if kind not in AXIOM_KINDS:
            raise SchemaParseError(f"未知 axiom kind: {kind!r}（允许 {AXIOM_KINDS}）")
        params = {k: v for k, v in d.items() if k != "kind"}
        return Axiom(kind=kind, params=params)


@dataclass
class Schema:
    meta: dict
    entities: list[EntityType]
    relations: list[RelationType]
    axioms: list[Axiom]

    # ---------- 序列化 ----------
    def to_yaml(self) -> str:
        data: dict = {"meta": self.meta}
        ents: dict = {}
        for e in self.entities:
            ents[e.name] = {
                "description": e.description,
                "primary_key": e.primary_key,
                "attributes": [{"name": a.name, "dtype": a.dtype} for a in e.attributes],
                **({"attribute_aliases": e.attribute_aliases} if e.attribute_aliases else {}),
            }
        data["entity_types"] = ents
        rels: dict = {}
        for r in self.relations:
            item: dict = {
                "domain": r.domain if len(r.domain) > 1 else r.domain[0],
                "range": r.range,
            }
            if r.functional:
                item["functional"] = True
            if r.inverse_of:
                item["inverse_of"] = r.inverse_of
            if r.derive:
                item["derive"] = r.derive
            if r.description:
                item["description"] = r.description
            rels[r.name] = item
        data["relation_types"] = rels
        data["axioms"] = [{"kind": ax.kind, **ax.params} for ax in self.axioms]
        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=100)

    @classmethod
    def from_yaml(cls, text: str) -> "Schema":
        # 剥可能的代码栅栏
        t = text.strip()
        if t.startswith("```"):
            t = t.split("\n", 1)[1] if "\n" in t else ""
            if t.rstrip().endswith("```"):
                t = t.rstrip()[:-3]
        try:
            data = yaml.safe_load(t)
        except yaml.YAMLError as e:
            raise SchemaParseError(f"YAML 解析失败: {e}")
        if not isinstance(data, dict):
            raise SchemaParseError(f"YAML 顶层应是映射，得到 {type(data).__name__}")

        ents_raw = data.get("entity_types") or {}
        rels_raw = data.get("relation_types") or {}
        axs_raw = data.get("axioms") or []

        if not ents_raw:
            raise SchemaParseError("entity_types 为空——至少需要一个实体类型")

        entities: list[EntityType] = []
        for name, body in ents_raw.items():
            if not isinstance(body, dict):
                raise SchemaParseError(f"实体 {name} 的定义应是映射")
            pk = body.get("primary_key") or []
            if isinstance(pk, str):
                pk = [pk]
            pk = [str(k).strip() for k in pk]
            if not pk:
                raise SchemaParseError(f"实体 {name!r} 缺 primary_key（论文硬规定：每个类型必须有主键）")
            attrs_raw = body.get("attributes") or []
            attrs = [Attribute.from_dict(a) for a in attrs_raw]
            aliases = {str(k): str(v) for k, v in (body.get("attribute_aliases") or {}).items()}
            entities.append(EntityType(
                name=str(name), primary_key=pk, attributes=attrs,
                description=str(body.get("description", "")), attribute_aliases=aliases,
            ))

        relations: list[RelationType] = []
        for name, body in rels_raw.items():
            if not isinstance(body, dict):
                raise SchemaParseError(f"关系 {name} 的定义应是映射")
            dom = body.get("domain")
            dom = [dom] if isinstance(dom, str) else list(dom or [])
            rng = str(body.get("range", "")).strip()
            if not dom or not rng:
                raise SchemaParseError(f"关系 {name!r} 必须声明 domain 与 range（论文硬规定）")
            relations.append(RelationType(
                name=str(name), domain=[str(d).strip() for d in dom], range=rng,
                functional=bool(body.get("functional", False)),
                inverse_of=body.get("inverse_of"),
                description=str(body.get("description", "")),
                derive=body.get("derive") if isinstance(body.get("derive"), dict) else None,
            ))

        axioms = [Axiom.from_dict(a) for a in axs_raw]
        return cls(meta=data.get("meta") or {}, entities=entities,
                   relations=relations, axioms=axioms)

    # ---------- 校验 ----------
    def entity(self, name: str) -> EntityType | None:
        return next((e for e in self.entities if e.name == name), None)

    def validate(self) -> list[str]:
        """引用完整性检查，返回错误列表（空 = 通过）。"""
        errs: list[str] = []
        names = [e.name for e in self.entities]
        if len(names) != len(set(names)):
            errs.append("实体类型名重复")
        for e in self.entities:
            attrset = e.attr_names()
            for k in e.primary_key:
                if k not in attrset:
                    errs.append(f"实体 {e.name}: 主键字段 {k!r} 不在 attributes 中")
            for raw in e.attribute_aliases:
                if raw in attrset:
                    errs.append(f"实体 {e.name}: 别名 {raw!r} 与规范属性名冲突")
        rnames = [r.name for r in self.relations]
        if len(rnames) != len(set(rnames)):
            errs.append("关系名重复")
        for r in self.relations:
            for d in r.domain:
                if d not in names:
                    errs.append(f"关系 {r.name}: domain 引用不存在的类型 {d!r}")
            if r.range not in names:
                errs.append(f"关系 {r.name}: range 引用不存在的类型 {r.range!r}")
            if r.inverse_of and r.inverse_of not in rnames:
                errs.append(f"关系 {r.name}: inverse_of 引用不存在的关系 {r.inverse_of!r}")
        for ax in self.axioms:
            p = ax.params
            if ax.kind == "subclass":
                if p.get("sub") not in names or p.get("sup") not in names:
                    errs.append(f"axiom subclass: 引用不存在的类 {p.get('sub')!r}/{p.get('sup')!r}")
            elif ax.kind == "disjoint":
                for c in p.get("classes", []):
                    if c not in names:
                        errs.append(f"axiom disjoint: 引用不存在的类 {c!r}")
            elif ax.kind in ("domain", "range"):
                if p.get("relation") not in rnames:
                    errs.append(f"axiom {ax.kind}: 引用不存在的关系 {p.get('relation')!r}")
                if p.get("class") not in names:
                    errs.append(f"axiom {ax.kind}: 引用不存在的类 {p.get('class')!r}")
            elif ax.kind == "cardinality":
                if p.get("relation") not in rnames or p.get("class") not in names:
                    errs.append(f"axiom cardinality: 引用不存在的关系/类")
            elif ax.kind == "key_functional":
                e = self.entity(p.get("entity", ""))
                if e is None:
                    errs.append(f"axiom key_functional: 引用不存在的实体 {p.get('entity')!r}")
        return errs

    # ---------- 渲染 ----------
    def render_brief(self, with_attrs: bool = True) -> str:
        lines: list[str] = ["ENTITY TYPES:"]
        for e in self.entities:
            pk = ", ".join(e.primary_key)
            lines.append(f"- {e.name} (primary key: {pk})")
            if with_attrs and e.attributes:
                lines.append(f"    attrs: " + "; ".join(f"{a.name}:{a.dtype}" for a in e.attributes))
            if e.attribute_aliases:
                lines.append(f"    aliases: " + "; ".join(f"{k}->{v}" for k, v in e.attribute_aliases.items()))
        lines.append("RELATION TYPES:")
        for r in self.relations:
            dom = "|".join(r.domain)
            fn = " [functional]" if r.functional else ""
            dv = " [DERIVED-SYSTEM: do not extract]" if r.derive else ""
            lines.append(f"- {r.name}: {dom} -> {r.range}{fn}{dv}")
        if self.axioms:
            lines.append("AXIOMS:")
            for ax in self.axioms:
                lines.append(f"- {ax.kind}: {ax.params}")
        return "\n".join(lines)

    def pk_signature(self, etype: str) -> list[str]:
        e = self.entity(etype)
        return list(e.primary_key) if e else []
