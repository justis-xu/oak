"""P2 模式草拟（glm-5.3）：SPEC + 上轮模式 + 反馈 + 验证失败 → YAML。"""

SYSTEM = "You are a schema architect. Output ONLY one YAML code block, nothing else."

YAML_GRAMMAR = """\
entity_types:
  <TypeName>:
    description: <one line>
    primary_key: [<field>, ...]              # REQUIRED, ordered; fields must exist in attributes
    attributes:
      - {name: <field>, dtype: string|int|float|date|bool}   # keep corpus casing verbatim
    attribute_aliases: {<corpus variant>: <canonical>}       # optional
relation_types:
  <relname>:
    domain: <TypeName or [T1, T2]>
    range: <TypeName>                          # REQUIRED
    functional: true                           # optional: one subject has at most one target
    inverse_of: <other relation>               # optional
    derive: {attr: <domain attribute>}         # optional: link is derived from this attribute
                                               # value (matched against range type's `name`);
                                               # DERIVED relations are NOT extracted by the LLM —
                                               # use for e.g. located_in (attr: city)
    # if the domain types use DIFFERENTLY-CASED fields for the same meaning, use:
    # derive: {attr_by_domain: {Accommodation: city, Restaurant: City, Attraction: City}}
    description: <one line>
axioms:                                        # optional; kinds: subclass|disjoint|domain|range|cardinality|key_functional
  - {kind: disjoint, classes: [Flight, Accommodation, Restaurant, Attraction]}
  - {kind: subclass, sub: A, sup: B}
  - {kind: cardinality, relation: r, class: A, min: 1, max: 1}
"""

EXAMPLE = """\
```yaml
meta:
  domain: travel-planning
entity_types:
  Flight:
    description: A scheduled flight between two cities on a date.
    primary_key: [Flight Number, FlightDate]
    attributes:
      - {name: Flight Number, dtype: string}
      - {name: FlightDate, dtype: date}
      - {name: Price, dtype: float}
      - {name: OriginCityName, dtype: string}
      - {name: DestCityName, dtype: string}
      - {name: DepTime, dtype: string}
      - {name: ArrTime, dtype: string}
  City:
    primary_key: [name]
    attributes:
      - {name: name, dtype: string}
relation_types:
  departs_from:
    domain: Flight
    range: City
    functional: true
  arrives_at:
    domain: Flight
    range: City
    functional: true
axioms:
  - {kind: disjoint, classes: [Flight, City]}
```
"""

TEMPLATE = """\
=== SPEC (requirement analysis) ===
{spec}

=== CORPUS FIELDS (the ONLY attribute names you may use — casing is verbatim) ===
{corpus_fields}

=== YAML GRAMMAR (follow exactly) ===
{grammar}

=== MINIMAL EXAMPLE ===
{example}

{prev_section}
{feedback_section}
{failures_section}
=== OUTPUT ===
Output ONLY one complete YAML code block (```yaml ... ```) implementing the SPEC.
HARD CONSTRAINTS:
1. Entity types must cover the four corpora (flights, accommodations, restaurants,
   attractions) plus value types (e.g. City).
2. Attribute names for corpus entity types MUST be copied verbatim from the CORPUS
   FIELDS section of the matching source. NEVER invent, rename, or re-case fields.
   (e.g. accommodations uses NAME / city / price / room type / minimum nights /
   maximum occupancy exactly — NOT PropertyName / PricePerNight / StarRating.)
3. Value-type entities (City, Cuisine, ...) may only have a `name` attribute.
4. Every relation must declare domain and range. Every entity must have a primary key
   chosen from its own attributes.
"""


def build(spec: str, prev_yaml: str | None, psi_s: list[dict] | None,
          owl_failures: str | None, corpus_fields: str | None = None) -> str:
    prev = ""
    if prev_yaml:
        prev = ("=== PREVIOUS SCHEMA (evolve from this) ===\n```yaml\n"
                + prev_yaml + "\n```\n")
    fb = ""
    if psi_s:
        lines = []
        for f in psi_s:
            lines.append(f"- u={f.get('u')} a={f.get('a')} delta={f.get('delta')} "
                         f"why={f.get('rho')}")
        fb = ("=== SCHEMA FEEDBACK from last round's judge (MUST apply) ===\n"
              + "\n".join(lines) + "\n")
    fail = ""
    if owl_failures:
        fail = f"=== VALIDATION FAILURES (fix these) ===\n{owl_failures}\n"
    return TEMPLATE.format(spec=spec, grammar=YAML_GRAMMAR, example=EXAMPLE,
                           corpus_fields=corpus_fields or "",
                           prev_section=prev, feedback_section=fb, failures_section=fail)
