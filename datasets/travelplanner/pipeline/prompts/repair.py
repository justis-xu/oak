"""P7 计划修复/组装（fast 档）：机器 issue + 精确候选 → {"plan":[...]}。

与 json_mode=json_object 顶层一致；候选一律来自 CANDIDATES_JSON，禁止编造。
"""

REPAIR_SYSTEM = """\
You repair or assemble a TravelPlanner plan from machine-produced issues and
exact candidates. Return exactly one JSON object: {"plan":[...]} or {"plan":[]}
when no grounded candidate can be produced. Do not add commentary.

If CURRENT_PLAN_JSON is non-null, preserve every field not implicated by an
issue. Any changed city, flight, ground leg, restaurant, attraction, or
accommodation must be copied verbatim from CANDIDATES_JSON. Warnings, errors,
empty results, relaxed mismatches, and suggestions are not candidates.

A meal cell contains one "Name, City" entity; split entity strings at the last
comma. Attractions are individual "Name, City; " items. On moving days meals
and attractions may use either endpoint and accommodation uses the destination;
on stay days all entities use the current city. Keep the closed route, exact
city count, uniqueness, local constraints, transport compatibility, and budget
valid. Never invent a replacement.
"""

REPAIR_USER = """\
QUERY_JSON={query_json}
CONSTRAINTS_JSON={constraints_json}
PLAN_SCHEMA_JSON={plan_schema_json}
CURRENT_PLAN_JSON={current_plan_json}
ISSUES_JSON={issues_json}
CANDIDATES_JSON={candidates_json}
"""

PLAN_SCHEMA = {
    "day_object_keys": ["days", "current_city", "transportation", "breakfast",
                        "attraction", "lunch", "dinner", "accommodation"],
    "flight": "Flight Number: <F>, from <A> to <B>, Departure Time: <HH:MM>, Arrival Time: <HH:MM>",
    "ground": "Taxi, from <A> to <B>  |  Self-driving, from <A> to <B>",
    "meal_or_accommodation": "<exact name>, <exact city>   (no cost suffix)",
    "attractions": "\"<name>, <city>; \" per item, joined, ALWAYS ending with \"; \"",
    "current_city": "\"from A to B\" on moving days, else \"CityName\"",
    "placeholder": "- only for: stay-day nothing (never), travel-day meals, last-day accommodation",
}


def build(query_json: str, constraints_json: str, current_plan_json: str,
          issues_json: str, candidates_json: str) -> str:
    import json as _json
    return REPAIR_USER.format(
        query_json=query_json,
        constraints_json=constraints_json,
        plan_schema_json=_json.dumps(PLAN_SCHEMA, separators=(",", ":"), ensure_ascii=False),
        current_plan_json=current_plan_json,
        issues_json=issues_json,
        candidates_json=candidates_json,
    )
