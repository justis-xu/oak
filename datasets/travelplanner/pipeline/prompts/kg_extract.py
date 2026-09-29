"""P3 KG 抽取（glm-5.3-flash）：模式简表 + 单 chunk → 实体/关系 JSON（紧凑格式）。"""

SYSTEM = "You are a precise information extractor. Output ONLY compact JSON."

TEMPLATE = """\
=== SCHEMA ===
{schema_brief}

=== DATA CHUNK ===
{chunk}

=== OUTPUT CONTRACT (compact; e_i means index i of your entities array) ===
Output one JSON object, nothing else:
{{"e":[["<EntityTypeName>",{{"pk_field":"value",...}},{{"attr":"value",...}}],
       ["City",{{"name":"Moab"}},{{}}]],
  "r":[["<RelationName>",e_0,e_1],
       ["arrives_at",e_0,e_1]]}}

HARD RULES:
1. Each entity = [type, primary-key dict (EXACTLY the pk fields), other attributes dict
   (only declared attribute names)]. Type MUST come from the SCHEMA.
2. Relations reference entities BY INDEX (e_0 = first entity). Both endpoints must be
   in the entities array. Relation names MUST come from the SCHEMA.
3. Relations marked [DERIVED-SYSTEM: do not extract] in the SCHEMA are derived
   automatically by the system from attributes — NEVER output them in "r".
4. Copy values VERBATIM from the text (no inference, no translation, keep casing).
   Numbers: plain digits/decimals, no units, no thousands separators.
5. One entity per data row. Do NOT output City/value-type entities that only appear
   as another row's attribute value (e.g. a city column) — the system derives them.
   If nothing matches: {{"e":[],"r":[]}}.
6. Keep output COMPACT: single line if possible, no whitespace padding, no echoing input.
"""


def build(schema_brief: str, chunk_text: str) -> str:
    return TEMPLATE.format(schema_brief=schema_brief, chunk=chunk_text)
