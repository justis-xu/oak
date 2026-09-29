"""P1 需求分析（glm-5.3）：任务描述 + 本轮样本 + 语料表头 → 需求规范 R。"""

TASK_DESCRIPTION = """\
TASK: Build the ontology (schema) for a travel-planning domain.
Given a user's multi-day trip request (origin, destination, dates, party size, budget,
local constraints), the system must produce a day-by-day plan covering intercity
transportation, three meals per day, attractions, and accommodation, within budget.
The data comes from four structured corpora: flights, accommodations (Airbnb-style),
restaurants, and attractions; plus an intercity ground-transport distance matrix.
"""

SYSTEM = "You are a knowledge engineer analyzing requirements for a task-specific ontology."

TEMPLATE = """\
=== TASK ===
{task}

=== QUERIES (this round's {n} samples) ===
{queries}

=== CORPUS HEADERS (verbatim field names — casing must be preserved) ===
{headers}

=== OUTPUT FORMAT (markdown, exactly these sections) ===
## Core Entities
(entity types the ontology must declare, each with: candidate primary key field(s) from the corpus, key attributes, why needed)
## Value-like Entities
(cross-source connector types, e.g. City / Cuisine — primary key is the name itself)
## Core Relations
(each: name, domain -> range, whether one instance can have at most one such link)
## Query Decomposition Patterns
(3-5 recurring reasoning patterns you see across the samples above, e.g. "filter hotels by occupancy >= party size AND minimum nights <= stay")
## Field Glossary
(corpus field name -> meaning; keep the EXACT casing/spelling of corpus fields like "Flight Number", "room type", "NAME")
## Out of Scope
(what this ontology deliberately does NOT model)

Rules:
- Derive everything from the TASK and the SAMPLES. Do not invent corpus fields that are not in the headers.
- Primary keys must be fields that actually appear in the corpora.
- The knowledge graph is instantiated ONLY from corpus data: entity types must be
  things that appear as rows of the four corpora, plus value types (City, Cuisine...).
  Do NOT model the plan/task structure itself (no TripRequest/DayPlan/MealSlot classes).
"""


def build(task: str, query_views: list[str], corpus_headers: dict[str, str]) -> str:
    return TEMPLATE.format(
        task=task,
        n=len(query_views),
        queries="\n\n".join(f"[sample {i+1}]\n{q}" for i, q in enumerate(query_views)),
        headers="\n\n".join(f"### {src}\n{h}" for src, h in corpus_headers.items()),
    )
