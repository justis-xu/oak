"""P6 评判器（glm-5.3）：五输入 → σ 四元组反馈（json_mode）。"""

SYSTEM = "You are the ontology evaluator. Diagnose failures and prescribe minimal fixes."

TEMPLATE = """\
=== 1. SCHEMA (current ontology) ===
{schema_yaml}

=== 2. KNOWLEDGE GRAPH ===
{graph_section}

=== 3. FUNCTIONS (catalog + last trial status) ===
{functions_section}

=== 4. TRAJECTORIES (per query: calls made, constraints failed with expected vs actual) ===
{trajectories_section}

=== 5. SCORES ===
{scores_section}

=== YOUR JOB ===
Diagnose WHY queries failed and attribute each failure to a fixable kernel element
(entity type, relation, or function). Prescribe minimal fixes.

Output STRICT JSON:
{{"feedback": [
  {{"u": "entity_type|relation|function",      // target kind
    "name": "<entity/relation/function name>",  // target name; null for new (add)
    "a": "add|delete|modify",
    "delta": {{...}},                             // schema patch: {{name, primary_key, attributes:[{{name,dtype}}], domain, range}}
                                                   // function patch: {{"function": "<name>", "instruction": "<precise change>"}}
    "rho": "<diagnosis — MUST cite failed query idx or score numbers>"}}
]}}

RULES:
- At most 6 feedback items; prioritize by number of failed queries affected.
- Attribute each failure to the EARLIEST controllable layer. If the official
  message shows a rendering/completeness problem (missing meals, placeholder
  strings, wrong formats), target function/planner behavior — do NOT invent
  schema changes for it.
- Only target things that exist in the SCHEMA/FUNCTIONS (for modify/delete),
  or well-motivated additions grounded in corpus fields.
- A missing schema FIELD that a failed constraint needed (e.g. occupancy) is an
  entity_type modify (add attribute).
- If a function never used a schema field it should have (constraint failures show
  the filter was never applied), that is a function modify with a precise instruction.
- Do NOT propose new functions unless a recurring pattern has no coverage at all.
"""


def build(schema_yaml: str, graph_section: str, functions_section: str,
          trajectories_section: str, scores_section: str) -> str:
    return TEMPLATE.format(
        schema_yaml=schema_yaml,
        graph_section=graph_section,
        functions_section=functions_section,
        trajectories_section=trajectories_section,
        scores_section=scores_section,
    )
