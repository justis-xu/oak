"""步骤① 模式构建：需求分析 → 草拟 → 验证（失败带反例重试 ≤4 次，降级不阻塞）。"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from oak.config import Config
from oak.llm.client import LLMClient
from .data.queries import Query
from oak.schema.model import Schema, SchemaParseError
from oak.schema import owlcheck
from .prompts import requirement_analysis as P1
from .prompts import schema_draft as P2
from .prompts.requirement_analysis import TASK_DESCRIPTION

log = logging.getLogger("oak.schema")

# 语料表头样例（与 corpus.py 的源对应；字段名逐字来自真实 csv）
CORPUS_HEADERS = {
    "flights": "Flight Number,Price,DepTime,ArrTime,ActualElapsedTime,FlightDate,OriginCityName,DestCityName,Distance",
    "accommodations": "NAME,room type,price,minimum nights,review rate number,house_rules,maximum occupancy,city",
    "restaurants": "Name,City,Cuisines,Average Cost,Aggregate Rating",
    "attractions": "Name,Latitude,Longitude,Address,Phone,Website,City",
    "distances (intercity)": "origin,destination,duration,distance",
}


async def analyze_requirements(client: LLMClient, round_queries: list[Query],
                               namespace: str) -> str:
    views = [q.query for q in round_queries]
    prompt = P1.build(TASK_DESCRIPTION, views, CORPUS_HEADERS)
    res = await client.chat(role="schema", messages=[
        {"role": "system", "content": P1.SYSTEM},
        {"role": "user", "content": prompt},
    ], temperature=0.2, namespace=namespace)
    return res.content


def _parse_yaml(raw: str) -> Schema:
    return Schema.from_yaml(raw)


async def build_schema(client: LLMClient, cfg: Config,
                       round_queries: list[Query],
                       prev_schema: Schema | None,
                       psi_s: list[dict],
                       round_dir: Path, namespace: str) -> tuple[Schema, dict]:
    """返回 (schema, 过程记录)。最多 cfg.schema_attempts 次草拟；全败则剔除冲突公理降级。"""
    spec = await analyze_requirements(client, round_queries, namespace)
    (round_dir / "requirement_spec.md").write_text(spec)

    prev_yaml = prev_schema.to_yaml() if prev_schema else None
    corpus_fields_text = "\n".join(
        f"- {src}: {hdr}" for src, hdr in CORPUS_HEADERS.items())
    # 全局字段白名单（四源真实字段 + 值类字段）
    whitelist: set[str] = set()
    for hdr in CORPUS_HEADERS.values():
        whitelist |= {f.strip() for f in hdr.split(",")}
    whitelist |= {"name"}

    def _field_check(s: Schema) -> list[str]:
        """语料实体严格白名单；概念/值类实体放行（属性上限 6）。"""
        import re
        corpus_kw = re.compile(
            r"flight|hotel|lodg|accom|restaur|food|cuisin|attraction|city|distance",
            re.I)
        errs = []
        for e in s.entities:
            bad = [a.name for a in e.attributes if a.name not in whitelist]
            if not bad:
                continue
            if corpus_kw.search(e.name):
                errs.append(
                    f"Entity {e.name}: attributes {bad} are NOT real corpus fields. "
                    f"Only these exist (verbatim casing): {sorted(whitelist)}")
            elif len(e.attributes) > 6:
                errs.append(
                    f"Entity {e.name} is a concept/value type: keep at most 6 "
                    f"attributes (has {len(e.attributes)}).")
        return errs

    failures: list[str] = []
    last_schema: Schema | None = None
    attempts_log = []

    for attempt in range(1, cfg.schema_attempts + 1):
        prompt = P2.build(spec, prev_yaml, psi_s, "\n\n".join(failures) or None,
                          corpus_fields=corpus_fields_text)
        res = await client.chat(role="schema", messages=[
            {"role": "system", "content": P2.SYSTEM},
            {"role": "user", "content": prompt},
        ], temperature=0.2, max_tokens=8192, namespace=namespace)

        # 解析
        try:
            schema = _parse_yaml(res.content)
            last_schema = schema
        except SchemaParseError as e:
            failures.append(f"[attempt {attempt}] YAML parse error: {e}\nRe-output valid YAML.")
            attempts_log.append({"attempt": attempt, "stage": "parse", "error": str(e)})
            _save_attempt(round_dir, attempt, res.content, None)
            continue

        # 引用完整性 + 字段白名单核对
        errs = schema.validate() + _field_check(schema)
        if errs:
            failures.append(f"[attempt {attempt}] Schema reference errors:\n" +
                            "\n".join(f"- {e}" for e in errs))
            attempts_log.append({"attempt": attempt, "stage": "validate", "errors": errs})
            _save_attempt(round_dir, attempt, res.content, None)
            continue

        # 静态 + HermiT
        findings, hermit_ok = owlcheck.check_schema(schema)
        _save_attempt(round_dir, attempt, res.content,
                      [f.render() for f in findings] or ["(passed)"])
        if not findings:
            log.info("schema attempt %d accepted (hermit_ok=%s)", attempt, hermit_ok)
            return schema, {"attempts": attempts_log, "accepted_attempt": attempt,
                            "hermit_ok": hermit_ok}
        failures.append(owlcheck.findings_to_feedback(findings, attempt))
        attempts_log.append({"attempt": attempt, "stage": "owlcheck",
                             "findings": [f"{f.check}:{f.entity}" for f in findings]})

    # 全部尝试失败：取最后一次解析成功的 schema，剔除非法字段与全部公理（保守降级）
    if last_schema is not None:
        log.warning("schema 降级接受：剔除非法字段与全部公理（%d 次 attempt 均未过验证）",
                    cfg.schema_attempts)
        import dataclasses
        cleaned_entities = []
        for e in last_schema.entities:
            attrs = [a for a in e.attributes if a.name in whitelist]
            pk = [k for k in e.primary_key if k in {a.name for a in attrs}]
            if not pk and attrs:
                pk = [attrs[0].name]
            if pk:
                cleaned_entities.append(dataclasses.replace(
                    e, attributes=attrs, primary_key=pk,
                    attribute_aliases={}))
        degraded = dataclasses.replace(last_schema, entities=cleaned_entities,
                                       axioms=[])
        return degraded, {"attempts": attempts_log, "accepted_attempt": None,
                          "degraded": True}
    raise RuntimeError("schema 构建失败：所有 attempt 都无法解析出合法 YAML")


def _save_attempt(round_dir: Path, attempt: int, raw: str, findings) -> None:
    d = round_dir / "schema_attempts"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"attempt_{attempt}.yaml").write_text(raw)
    if findings is not None:
        (d / f"attempt_{attempt}.findings.json").write_text(json.dumps(findings, indent=1))
