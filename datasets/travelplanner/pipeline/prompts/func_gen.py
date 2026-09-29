"""P4 函数生成（glm-5.3）：算子文档 + 模式 + 查询模式 + 反馈 → 单个领域函数。"""

SYSTEM = "You are a Python engineer. Output ONLY one Python code block defining ONE function."

TEMPLATE = """\
=== GENERIC OPERATORS (the only building blocks you may call) ===
{operators}

=== SCHEMA (field names below are the ONLY field strings you may hardcode) ===
{schema_brief}

=== QUERY PATTERNS (what the agent repeatedly needs to do) ===
{query_patterns}

{existing_section}{feedback_section}{failure_section}
=== TASK ===
{task}

=== CODING RULES ===
1. Define exactly ONE function with this shape:
   - Value parameters (budget, people_count, ...) are explicit function parameters WITH default values.
   - Field names appear ONLY as string literals that exist in the SCHEMA.
   - The knowledge graph is accessed ONLY through the nine operators above —
     do NOT reference any global variable G.
   - Allowed builtins: len, min, max, sum, sorted, set, dict, list, tuple, str,
     int, float, round, abs, enumerate, zip, range, bool, any, all, isinstance, ceil.
   - Allowed methods: .lower .upper .strip .lstrip .rstrip .split .rsplit
     .splitlines .join .startswith .endswith .replace .format .title .capitalize
     .isdigit .isalpha .items .keys .values .get .copy .append .extend .insert
     .add .update .pop .remove
   - Anything else (imports, eval/exec, underscore attributes, other calls) is rejected.
2. Include a docstring: one-line purpose, Args, Returns, and one Example call.
3. Return JSON-serializable data (list of dicts / dict / number).
4. Keep it small: chain lookup -> filter -> project/aggregate as needed.

{planning_section}=== OUTPUT ===
Output ONLY one ```python code block.
"""


def build(operators_doc: str, schema_brief: str, query_patterns: str, task: str,
          existing_fn_source: str | None = None,
          judge_delta: dict | None = None,
          failure_note: str | None = None,
          planning_output: str | None = None) -> str:
    existing = ""
    if existing_fn_source:
        existing = ("=== EXISTING FUNCTION (modify this) ===\n```python\n"
                    + existing_fn_source + "\n```\n")
    fb = ""
    if judge_delta:
        fb = ("=== JUDGE INSTRUCTION (apply exactly) ===\n"
              + str(judge_delta) + "\n")
    fail = ""
    if failure_note:
        fail = f"=== PREVIOUS ATTEMPT FAILED (fix this) ===\n{failure_note}\n"
    planning = ""
    if planning_output:
        planning = f"=== CAPABILITY PLAN (follow this list) ===\n{planning_output}\n"
    return TEMPLATE.format(
        operators=operators_doc, schema_brief=schema_brief,
        query_patterns=query_patterns, task=task,
        existing_section=existing, feedback_section=fb, failure_section=fail,
        planning_section=planning,
    )


PLAN_TEMPLATE = """\
=== QUERY PATTERNS ===
{query_patterns}

=== SCHEMA ===
{schema_brief}

=== TASK ===
List the domain functions the agent will need, one per line, as:
name(param1, param2=default, ...) -> what it returns; why needed (pattern it serves).
Aim for 6-10 functions. The list MUST cover ALL of:
1. FLIGHT candidates by origin/dest city and date (from Flight entities) — trips
   normally start and end with flights between distant cities.
2. Ground-transport (intercity) connections between two given cities.
3. Restaurant candidates (by city / cuisine).
4. Accommodation candidates (by city / occupancy / price / room type / house rule).
5. Attraction candidates (by city).
6. Cities within a destination state/region (for multi-city trips).
7. Total plan cost using the OFFICIAL formula: flights Price x people; meals
   Average Cost x people; hotels price x ceil(people/maximum occupancy) per
   night; self-driving int(km x 0.05) x ceil(people/5); taxi int(km) x
   ceil(people/4) — ground-leg km comes from the distance field.
8. Budget check helpers.
Output ONLY the list, no prose.
"""


def build_planning(query_patterns: str, schema_brief: str) -> str:
    return PLAN_TEMPLATE.format(query_patterns=query_patterns, schema_brief=schema_brief)
