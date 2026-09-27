"""计划终态编排：非破坏 normalize → 全量校验 → 确定性修复 → LLM 修复 → 预算降级。

设计原则（OPTIMIZATION_LOG 第二轮）：
- 候选支配性回滚：repair/fill/budget 候选必须严格优于当前 best 才接受；
- 确定性修复绝不跨城补齐（同城无合法候选保留 issue）；
- LLM 修复只拿 issue-scoped 精确候选（CANDIDATES_JSON），顶层 {"plan":[...]}；
- 最终仍有 blocking issue 也保留 best 候选交付（如实记录），不伪造完整。
"""
from __future__ import annotations

import copy
import json
import math

from ..config import Config
from ..llm.client import LLMClient
from ..data.queries import Query, query_view
from .graph_index import (GraphIndex, compute_plan_cost, parse_from_to,
                          parse_name_city, split_attractions, strip_paren,
                          first_flight_number, transportation_mode)
from .validator import (DAY_KEYS, MEALS, PlanIssue, ValidationReport,
                        validate_plan_full, activity_cities, lodging_city)
from ..prompts import repair as P7

FLIGHT_FMT = "Flight Number: {num}, from {o} to {d}, Departure Time: {dep}, Arrival Time: {arr}"

ROOM_MAP = {"entire room": "Entire home/apt", "private room": "Private room",
            "shared room": "Shared room"}


# ---------------------------------------------------------------- payload 与 normalize
def parse_plan_payload(raw) -> list | None:
    """str/dict/list → plan list。兼容 {"plan":[...]} wrapper、裸数组、Final Plan 文本。"""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw["plan"] if isinstance(raw.get("plan"), list) else None
    if isinstance(raw, list):
        return raw
    s = str(raw).strip()
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            d = json.loads(s[i:j + 1])
            if isinstance(d, dict) and isinstance(d.get("plan"), list):
                return d["plan"]
            if isinstance(d, list):
                return d
        except Exception:
            pass
    i, j = s.find("["), s.rfind("]")
    if i >= 0 and j > i:
        try:
            v = json.loads(s[i:j + 1])
            if isinstance(v, list):
                return v
        except Exception:
            pass
    return None


def _scalarize(v) -> str:
    """字段值规范化：列表解包（单元素取值、多元素 '; ' 连接），None/空 → '-'。

    修 anchor q15/q25 实证问题：repair 模型返回列表值被 str() 成 "['...']" 垃圾串。
    """
    if isinstance(v, list):
        vs = [str(x).strip() for x in v if str(x).strip() and str(x).strip() != "None"]
        if not vs:
            return "-"
        return "; ".join(vs) if len(vs) > 1 else vs[0]
    return str(v).strip() if v is not None else "-"


def normalize_plan(raw, q: Query) -> tuple[list[dict] | None, list[str]]:
    """非破坏规范化：键白名单、标量化、占位归一。

    不截断多余天、不补空天、不覆盖 days（validator 报 issue，local_repair/LLM 负责）。
    """
    payload = parse_plan_payload(raw)
    if not isinstance(payload, list) or not payload:
        return None, ["raw plan is not a non-empty list"]
    plan: list[dict] = []
    for i, day in enumerate(payload):
        if not isinstance(day, dict):
            day = {}
        nd = {}
        for k in DAY_KEYS:
            v = _scalarize(day.get(k, "-"))
            if v in ("", "None", "null", "-;"):
                v = "-"
            nd[k] = v
        try:
            nd["days"] = int(day.get("days", i + 1))
        except Exception:
            nd["days"] = i + 1
        plan.append(nd)
    return plan, []


# ---------------------------------------------------------------- 候选支配性
def _dominates(new_rep: ValidationReport, old_rep: ValidationReport) -> bool:
    nb, ob = len(new_rep.blocking), len(old_rep.blocking)
    if nb < ob:
        return True
    if nb > ob:
        return False
    old_codes = {i.code for i in old_rep.blocking}
    new_codes = {i.code for i in new_rep.blocking}
    if new_codes - old_codes:
        return False                      # 引入了新类型的问题
    nc = (new_rep.cost or {}).get("total", math.inf)
    oc = (old_rep.cost or {}).get("total", math.inf)
    return nc < oc - 0.5                  # 同等问题数下成本严格下降


# ---------------------------------------------------------------- 确定性修复（无跨城）
def _rest_str(r: dict) -> str:
    return f"{r.get('Name')}, {r.get('City')}"


def _acc_str(a: dict) -> str:
    return f"{a.get('NAME')}, {a.get('city')}"


def _attr_str(a: dict) -> str:
    return f"{a.get('Name')}, {a.get('City')}"


def _num(v) -> float:
    try:
        if v in (None, ""):
            return math.inf
        return float(str(v).replace(",", "").replace("$", "").strip())
    except Exception:
        return math.inf


def _acc_ok(a: dict, q: Query, block_len: int) -> bool:
    lc = q.local_constraint or {}
    rt = lc.get("room type")
    if rt:
        rv = str(a.get("room type") or "")
        if rt == "not shared room":
            if rv == "Shared room":
                return False
        elif rv != ROOM_MAP.get(str(rt).lower(), rt):
            return False
    hr = lc.get("house rule")
    if hr and f"No {hr}" in str(a.get("house_rules") or ""):
        return False
    mn = _num(a.get("minimum nights"))
    if mn != math.inf and mn > block_len:
        return False
    return True


def _rooms(a: dict, people: int) -> int:
    occ = _num(a.get("maximum occupancy"))
    occ = 1 if occ == math.inf else max(1, int(occ))
    return math.ceil(people / occ)


def local_repair(plan: list[dict] | None, q: Query, idx: GraphIndex) -> list[dict] | None:
    """约束感知确定性修复：同城图池回填缺位/替换坏项/住宿块/菜系覆盖。

    铁律：候选只来自当天允许城市（移动日两端点 / stay 日当前城 / 住宿=终点城），
    池空保留 issue——绝不跨城。末日晚清空住宿（官方口径不计、省预算）。
    """
    if not plan:
        return plan
    plan = copy.deepcopy(plan)[:q.days]
    for i, day in enumerate(plan):
        day["days"] = i + 1
    # ---- 交通修复：图外/错路航班 → 同腿有效航班；无航班 → Taxi（保模式相容） ----
    # （anchor q30 实证：LLM 编的航班号不在图内，validator 抓到但 repair 没换）
    modes_in_plan = {transportation_mode(d["transportation"])
                     for d in plan if d["transportation"] not in ("-", "")}
    for day in plan:
        trans = day["transportation"]
        if "Flight Number" not in trans:
            continue
        # 复合串（Taxi 去机场 + Flight）官方按 taxi 优先解析、from 是住宿名 → 全废；
        # 取 Flight 段的 from→to 作事实源，重建纯航班串
        seg = trans[trans.index("Flight Number"):]
        o, d = parse_from_to(seg)
        if not (o and d):
            o, d = parse_from_to(trans)
        if not (o and d):
            continue
        o2, d2 = strip_paren(o).strip(), strip_paren(d).strip()
        num = first_flight_number(seg)
        row = idx.flight_by_num.get(num or "")
        if row is not None and str(row.get("OriginCityName")) == o2 \
                and str(row.get("DestCityName")) == d2:
            if seg.strip() != day["transportation"].strip() or ";" in trans:
                day["transportation"] = FLIGHT_FMT.format(
                    num=num, o=o, d=d, dep=row.get("DepTime"), arr=row.get("ArrTime"))
            continue                                   # 已有效（重建复合串）
        cands = sorted(idx.flights_by_route.get((o2, d2), []),
                       key=lambda f: _num(f.get("Price")))
        if cands:
            f = cands[0]
            day["transportation"] = FLIGHT_FMT.format(
                num=f.get("Flight Number"), o=o, d=d,
                dep=f.get("DepTime"), arr=f.get("ArrTime"))
        elif idx.ground_valid(o2, d2) and "Self-driving" not in modes_in_plan:
            day["transportation"] = f"Taxi, from {o} to {d}"   # Flight+Taxi 官方允许

    # ---- 移动日 current_city 重写：交通已声明 from A to B 而 current_city 是 stay 日
    # （anchor q15/q25 实证：官方要求移动日 "from X to Y"，交通串才是事实来源）
    for day in plan:
        trans = day["transportation"]
        cc = day["current_city"]
        if trans and trans != "-":
            o, d = parse_from_to(trans)
            if o and d and not any(parse_from_to(cc)):
                stay = strip_paren(cc).strip()
                if stay in (strip_paren(o).strip(), strip_paren(d).strip(), ""):
                    day["current_city"] = f"from {o} to {d}"
    last = q.days - 1
    if last >= 0 and last < len(plan):
        plan[last]["accommodation"] = "-"
    people = max(1, int(q.people_number or 1))

    used_rests: set[str] = set()
    used_attrs: set[str] = set()
    for day in plan:
        for meal in MEALS:
            v = day[meal]
            if v and v != "-":
                used_rests.add(v)
        for item in split_attractions(day["attraction"]):
            used_attrs.add(item)

    def rest_pool(cities: list[str]):
        rows = []
        for c in cities:
            rows += idx.rests_by_city.get(c, [])
        rows.sort(key=lambda r: _num(r.get("Average Cost")))
        return rows

    def attr_pool(cities: list[str]):
        rows = []
        for c in cities:
            rows += idx.attrs_by_city.get(c, [])
        return rows

    # ---- 景点串归一（'-;'/尾分号/空段） ----
    for day in plan:
        att = day["attraction"]
        if att and att != "-":
            items = split_attractions(att)
            day["attraction"] = ("; ".join(items) + "; ") if items else "-"

    # ---- 餐位修复（官方口径：breakfast/lunch/dinner 全表一个列表去重） ----
    seen_meals: set[str] = set()      # 本轮按序保留的串（跨列去重，官方语义）
    for i, day in enumerate(plan):
        endpoints = activity_cities(i, plan)
        moving = len(endpoints) >= 2
        if not endpoints:
            continue
        for meal in MEALS:
            v = day[meal]
            need = False
            if v in ("-", ""):
                need = not moving            # 移动日餐可 '-'
            else:
                if ";" in v:
                    need = True
                else:
                    nm, ct = parse_name_city(v)
                    if nm == "-" or idx.find_restaurant(nm, ct) is None:
                        need = True
                    elif not any(c and c in v for c in endpoints):
                        need = True
                    elif v in seen_meals:
                        need = True          # 跨列重复（首次已保留）
            if need:
                for r in rest_pool(endpoints):
                    s = _rest_str(r)
                    if s not in used_rests and s not in seen_meals:
                        day[meal] = s
                        used_rests.add(s)
                        seen_meals.add(s)
                        break
            elif v not in ("-", ""):
                seen_meals.add(v)

    # 回收不再使用的旧串，保持 used 与计划一致
    cur = {day[m] for day in plan for m in MEALS if day[m] != "-"}
    used_rests &= cur
    seen_meals &= cur

    # ---- 景点修复（逐项；跨天按序去重，官方 split(';')[:-1] 全表判重） ----
    kept_attrs: set[str] = set()
    for i, day in enumerate(plan):
        endpoints = activity_cities(i, plan)
        moving = len(endpoints) >= 2
        items = split_attractions(day["attraction"]) if day["attraction"] != "-" else []
        if not items and not moving:
            items = ["__FILL__"]
        new_items: list[str] = []
        for it in items:
            if it == "__FILL__" or it in kept_attrs or _attr_bad(it, endpoints, idx):
                pool = attr_pool(endpoints)
                rep = None
                for a in pool:
                    s = _attr_str(a)
                    if s not in used_attrs and s not in kept_attrs \
                            and any(c and c in s for c in endpoints):
                        rep = s
                        break
                if rep:
                    used_attrs.add(rep)
                    kept_attrs.add(rep)
                    new_items.append(rep)
            else:
                new_items.append(it)
                kept_attrs.add(it)
        if new_items:
            day["attraction"] = "; ".join(new_items) + "; "
        else:
            day["attraction"] = "-"
    cur_att = {it for day in plan for it in split_attractions(day["attraction"])}
    used_attrs &= cur_att
    kept_attrs &= cur_att

    # ---- 住宿块修复 ----
    if q.days >= 2:
        lodging = [lodging_city(i, plan) for i in range(q.days - 1)]
        blocks: list[tuple[int, int, str | None]] = []
        i = 0
        while i < len(lodging):
            j = i
            while j + 1 < len(lodging) and lodging[j + 1] == lodging[i]:
                j += 1
            blocks.append((i, j, lodging[i]))
            i = j + 1
        for bi, bj, city in blocks:
            if not city:
                continue
            cur_vals = [plan[k]["accommodation"] for k in range(bi, bj + 1)]
            one_val = len(set(cur_vals)) == 1 and cur_vals[0] not in ("-", "")
            valid = False
            if one_val:
                nm, ct = parse_name_city(cur_vals[0])
                row = idx.find_accommodation(nm, ct)
                valid = (row is not None and ct == city
                         and _acc_ok(row, q, bj - bi + 1))
            if not valid:
                cands = [a for a in idx.accs_by_city.get(city, [])
                         if _acc_ok(a, q, bj - bi + 1)]
                cands.sort(key=lambda a: _num(a.get("price")) * (bj - bi + 1) * _rooms(a, people))
                if cands:
                    s = _acc_str(cands[0])
                    for k in range(bi, bj + 1):
                        plan[k]["accommodation"] = s

    # ---- 菜系覆盖（在路线城市内替换） ----
    lc = q.local_constraint or {}
    want = lc.get("cuisine") or []
    if isinstance(want, str):
        want = [want]
    route_cities = [c for c in dict.fromkeys(
        c for i in range(q.days) for c in activity_cities(i, plan)) if c and c != q.org]
    for cui in want:
        have = any(cui.lower() in str(r.get("Cuisines") or "").lower()
                   for s, r in ((day[m], idx.find_restaurant(*parse_name_city(day[m])))
                                for day in plan for m in MEALS if day[m] != "-")
                   if r is not None)
        if have:
            continue
        done = False
        for city in route_cities:
            cover = [r for r in idx.rests_by_city.get(city, [])
                     if cui.lower() in str(r.get("Cuisines") or "").lower()]
            if not cover:
                continue
            # 找该城一个可替换餐位（同日端点含 city），优先替换最贵的
            slots = [(i, m) for i in range(q.days)
                     for m in MEALS
                     if plan[i][m] != "-" and city in activity_cities(i, plan)]
            slots.sort(key=lambda im: -_num(
                (idx.find_restaurant(*parse_name_city(plan[im[0]][im[1]])) or {}).get("Average Cost")))
            for r in sorted(cover, key=lambda r: _num(r.get("Average Cost"))):
                s = _rest_str(r)
                if s in {plan[i][m] for i, m in slots} or s in used_rests:
                    continue
                if slots:
                    i, m = slots[0]
                    old = plan[i][m]
                    plan[i][m] = s
                    used_rests.discard(old)
                    used_rests.add(s)
                    done = True
                    break
            if done:
                break
    return plan


def _attr_bad(item: str, endpoints: list[str], idx: GraphIndex) -> bool:
    nm, ct = parse_name_city(item)
    if nm == "-" or ct == "-":
        return True
    if idx.find_attraction(nm, ct) is None:
        return True
    if not any(c and c in item for c in endpoints):
        return True
    return False


# ---------------------------------------------------------------- 预算降级（保守）
def budget_downgrade(plan: list[dict] | None, q: Query, idx: GraphIndex) -> list[dict] | None:
    """只做同城/同约束/同路线的降价替换：住宿块 → 餐 → 航班。每次替换后复算。"""
    if not plan:
        return plan
    plan = copy.deepcopy(plan)
    people = max(1, int(q.people_number or 1))
    cost = compute_plan_cost(plan, q, idx)
    if cost["total"] <= q.budget:
        return plan

    # A. 住宿块换最便宜合规同城候选（逐块尝试，不早退——q125 实证：换第一块仍超预算
    #    就 return 会漏掉后续可降的块）
    if q.days >= 2:
        lodging = [lodging_city(i, plan) for i in range(q.days - 1)]
        blocks = []
        i = 0
        while i < len(lodging):
            j = i
            while j + 1 < len(lodging) and lodging[j + 1] == lodging[i]:
                j += 1
            blocks.append((i, j, lodging[i]))
            i = j + 1

        def _block_cost(city: str, block: int) -> list[dict]:
            cands = [a for a in idx.accs_by_city.get(city, []) if _acc_ok(a, q, block)]
            cands.sort(key=lambda a: _num(a.get("price")) * block * _rooms(a, people))
            return cands

        for i, j, city in blocks:
            if not city:
                continue
            block = j - i + 1
            cur = plan[i]["accommodation"]
            row = idx.find_accommodation(*parse_name_city(cur)) if cur not in ("-", "") else None
            cur_cost = (_num(row.get("price")) * block * _rooms(row, people)
                        if row else math.inf)
            cands = [a for a in _block_cost(city, block)
                     if _num(a.get("price")) * block * _rooms(a, people) < cur_cost - 0.5]
            if not cands:
                continue
            s = _acc_str(cands[0])
            for k in range(i, j + 1):
                plan[k]["accommodation"] = s
            cost = compute_plan_cost(plan, q, idx)
            if cost["total"] <= q.budget:
                return plan

    # B. 餐换同城更便宜（保唯一 + 菜系覆盖不回退）
    slots = [(i, m) for i in range(len(plan)) for m in MEALS if plan[i][m] != "-"]
    rows = {}
    for i, m in slots:
        rows[(i, m)] = idx.find_restaurant(*parse_name_city(plan[i][m]))
    slots.sort(key=lambda im: -_num((rows.get(im) or {}).get("Average Cost")))
    needed_cuisines = (q.local_constraint or {}).get("cuisine") or []
    if isinstance(needed_cuisines, str):
        needed_cuisines = [needed_cuisines]
    for i, m in slots:
        cur_row = rows.get((i, m))
        if cur_row is None:
            continue
        endpoints = activity_cities(i, plan)
        cur_cui = str(cur_row.get("Cuisines") or "")
        covers_needed = any(c in cur_cui for c in needed_cuisines)
        cands = [r for r in rest_pool_cities(idx, endpoints)
                 if _num(r.get("Average Cost")) < _num(cur_row.get("Average Cost")) - 0.5]
        for r in sorted(cands, key=lambda r: _num(r.get("Average Cost"))):
            s = _rest_str(r)
            if s in {plan[k][mm] for k in range(len(plan)) for mm in MEALS}:
                continue
            if covers_needed and not any(c in str(r.get("Cuisines") or "") for c in needed_cuisines):
                continue      # 不撤走覆盖必需菜系的最后一家
            plan[i][m] = s
            cost = compute_plan_cost(plan, q, idx)
            if cost["total"] <= q.budget:
                return plan
            break

    # C. 航班换同腿更便宜
    for i, day in enumerate(plan):
        trans = day["transportation"]
        if "Flight Number" not in trans:
            continue
        o, d = parse_from_to(trans)
        if not o or not d:
            continue
        num = first_flight_number(trans)
        row = idx.flight_by_num.get(num or "")
        if row is None:
            continue
        cur_price = _num(row.get("Price"))
        cands = [f for f in idx.flights_by_route.get((o, d), [])
                 if _num(f.get("Price")) < cur_price - 0.5]
        cands.sort(key=lambda f: _num(f.get("Price")))
        if cands:
            f = cands[0]
            plan[i]["transportation"] = FLIGHT_FMT.format(
                num=f.get("Flight Number"), o=o, d=d,
                dep=f.get("DepTime"), arr=f.get("ArrTime"))
            cost = compute_plan_cost(plan, q, idx)
            if cost["total"] <= q.budget:
                return plan
    return plan


def rest_pool_cities(idx: GraphIndex, cities: list[str]) -> list[dict]:
    rows = []
    for c in cities:
        rows += idx.rests_by_city.get(c, [])
    return rows


# ---------------------------------------------------------------- 修复证据（issue-scoped）
def build_evidence(plan: list[dict] | None, q: Query, idx: GraphIndex,
                   rep: ValidationReport, per_bucket: int = 5) -> dict:
    """结构化候选 JSON：只给 issue 涉及的 route/day/city，先按硬约束过滤再按价排序。"""
    n_days = q.days
    route_days = []
    cities_in_route: list[str] = []
    if plan:
        for i in range(n_days):
            endpoints = activity_cities(i, plan)
            lg = lodging_city(i, plan)
            route_days.append({"day": i + 1, "activity_cities": endpoints,
                               "lodging_city": lg})
            for c in endpoints:
                if c and c not in cities_in_route:
                    cities_in_route.append(c)
    else:
        # no-plan salvage：给全量 covered 候选与 org
        cities_in_route = [q.org]

    want_cuisines = (q.local_constraint or {}).get("cuisine") or []
    if isinstance(want_cuisines, str):
        want_cuisines = [want_cuisines]

    ev: dict = {"route": {
        "org": q.org,
        "covered_city_candidates": [
            {"name": c["name"], "state": c["state"]} for c in idx.covered_cities()],
        "allowed_by_day": route_days,
    }}

    restaurants: dict[str, list] = {}
    accommodations: dict[str, list] = {}
    attractions: dict[str, list] = {}
    cities_for_fill = cities_in_route if plan else \
        [c["name"] for c in idx.covered_cities()]
    for city in dict.fromkeys(cities_for_fill):
        rs = sorted(idx.rests_by_city.get(city, []),
                    key=lambda r: _num(r.get("Average Cost")))
        picked = rs[:per_bucket]
        for cui in want_cuisines:
            for r in rs:
                if cui.lower() in str(r.get("Cuisines") or "").lower():
                    s = _rest_str(r)
                    if s not in {_rest_str(x) for x in picked}:
                        picked.append(r)
                    break
        restaurants[city] = [{"name": r.get("Name"), "city": r.get("City"),
                              "average_cost": r.get("Average Cost"),
                              "cuisines": r.get("Cuisines")} for r in picked[:per_bucket + 3]]
        as_ = sorted((a for a in idx.accs_by_city.get(city, [])
                      if _acc_ok(a, q, max(1, q.days - 1))),
                     key=lambda a: _num(a.get("price")))
        accommodations[city] = [{"name": a.get("NAME"), "city": a.get("city"),
                                  "price": a.get("price"),
                                  "minimum_nights": a.get("minimum nights"),
                                  "maximum_occupancy": a.get("maximum occupancy"),
                                  "room_type": a.get("room type"),
                                  "house_rules": a.get("house_rules")}
                                 for a in as_[:per_bucket]]
        ats = idx.attrs_by_city.get(city, [])[:per_bucket]
        attractions[city] = [{"name": a.get("Name"), "city": a.get("City")} for a in ats]

    ev["restaurants_by_city"] = restaurants
    ev["accommodations_by_city"] = accommodations
    ev["attractions_by_city"] = attractions

    legs: dict[str, list] = {}
    ground: dict[str, dict] = {}
    if plan:
        for i in range(min(len(plan), n_days)):
            o, d = parse_from_to(plan[i].get("current_city") or "")
            if o and d:
                key = f"{strip_paren(o)}->{strip_paren(d)}"
                if key not in legs:
                    fs = sorted(idx.flights_by_route.get((strip_paren(o), strip_paren(d)), []),
                                key=lambda f: _num(f.get("Price")))
                    legs[key] = [{"flight_number": f.get("Flight Number"),
                                  "date": f.get("FlightDate"),
                                  "price": f.get("Price"),
                                  "departure_time": f.get("DepTime"),
                                  "arrival_time": f.get("ArrTime")}
                                 for f in fs[:3]]
                    rec = idx.ground.get((strip_paren(o), strip_paren(d)))
                    if rec:
                        ground[key] = {"duration": rec["duration"],
                                       "distance": rec["distance"], "km": rec["km"],
                                       "valid": rec["valid"]}
    ev["flights_by_leg"] = legs
    ev["ground_by_leg"] = ground

    if not plan:
        # no-plan salvage：补 org↔covered 候选城市的航班/地面腿（q35：语料无航班时
        # 模型按"不编造"输出空计划，但证据必须给它地面交通选项）
        legs2: dict[str, list] = {}
        ground2: dict[str, dict] = {}
        for c in ev["route"]["covered_city_candidates"][:6]:
            city = c["name"]
            for a, b in ((q.org, city), (city, q.org)):
                key = f"{a}->{b}"
                fs = sorted(idx.flights_by_route.get((a, b), []),
                            key=lambda f: _num(f.get("Price")))
                legs2[key] = [{"flight_number": f.get("Flight Number"),
                               "date": f.get("FlightDate"), "price": f.get("Price"),
                               "departure_time": f.get("DepTime"),
                               "arrival_time": f.get("ArrTime")} for f in fs[:3]]
                rec = idx.ground.get((a, b))
                if rec:
                    ground2[key] = {"duration": rec["duration"],
                                    "distance": rec["distance"], "km": rec["km"],
                                    "valid": rec["valid"]}
        ev["flights_by_leg"] = legs2
        ev["ground_by_leg"] = ground2

    ev["cost"] = {"budget": q.budget, "people": q.people_number}
    if rep.cost:
        ev["cost"]["current_total"] = round(rep.cost["total"])
        ev["cost"]["overspend"] = max(0, round(rep.cost["total"] - q.budget))
    return ev


def _constraints_json(q: Query) -> dict:
    lc = q.local_constraint or {}
    out: dict = {}
    if lc.get("cuisine"):
        cs = lc["cuisine"] if isinstance(lc["cuisine"], list) else [lc["cuisine"]]
        out["cuisine"] = {"required": cs,
                          "rule": f"each must appear in the Cuisines of at least one "
                                  f"chosen restaurant OUTSIDE {q.org}"}
    if lc.get("room type"):
        rt = lc["room type"]
        want = ROOM_MAP.get(str(rt).lower(), rt) if rt != "not shared room" else "not 'Shared room'"
        out["room_type"] = {"required": want}
    if lc.get("house rule"):
        out["house_rule"] = {"forbidden_substring": f"No {lc['house rule']}"}
    if lc.get("transportation"):
        out["transportation"] = {"ban": lc["transportation"]}
    out["budget"] = {"max": q.budget,
                     "formula": "flights Price x people; meals AverageCost x people; "
                                "hotels price x ceil(people/occupancy) per night; "
                                "self-driving int(km x 0.05) x ceil(people/5); "
                                "taxi int(km) x ceil(people/4)"}
    out["city_count"] = {"distinct_cities_besides_org": q.visiting_city_number}
    return out


# ---------------------------------------------------------------- LLM 修复
async def llm_repair(client: LLMClient, plan: list[dict] | None, q: Query,
                     idx: GraphIndex, rep: ValidationReport, namespace: str
                     ) -> list[dict] | None:
    issues = [{"code": i.code, "day": i.day, "field": i.field, "message": i.message}
              for i in rep.blocking[:40]]
    evidence = build_evidence(plan, q, idx, rep)
    prompt = P7.build(
        query_json=json.dumps(query_view(q), ensure_ascii=False,
                              separators=(",", ":")),
        constraints_json=json.dumps(_constraints_json(q), ensure_ascii=False,
                                    separators=(",", ":")),
        current_plan_json=json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
        if plan is not None else "null",
        issues_json=json.dumps(issues, ensure_ascii=False, separators=(",", ":")),
        candidates_json=json.dumps(evidence, ensure_ascii=False,
                                   separators=(",", ":")),
    )
    res = await client.chat(role="plan_repair", messages=[
        {"role": "system", "content": P7.REPAIR_SYSTEM},
        {"role": "user", "content": prompt},
    ], temperature=0.1, json_mode=True, namespace=namespace)
    cand, _ = normalize_plan(res.content, q)
    return cand


# ---------------------------------------------------------------- 单一终态入口
async def finalize_plan(client: LLMClient, cfg: Config, q: Query, raw_plan,
                        g, namespace: str) -> tuple[list[dict] | None, ValidationReport, dict]:
    """normalize → 全量校验 → 确定性修复 →（≤cfg.plan_repair_attempts 轮）LLM 修复
    +确定性收尾 → 预算降级；候选支配性回滚；最终保留 best 候选（如实记录 issue）。
    """
    idx = GraphIndex(g, cfg.tp_root)
    stats = {"repairs": 0, "rejected": 0, "salvage": 0}

    plan, _ = normalize_plan(raw_plan, q) if raw_plan else (None, ["no plan"])
    if plan is None:
        # no-plan salvage：结构化组装一次（旧实现 6 题空计划完全不修）
        rep0 = ValidationReport(issues=[PlanIssue("plan.empty", message="no plan produced")])
        cand = await llm_repair(client, None, q, idx, rep0, namespace)
        stats["salvage"] = 1
        if cand:
            plan = cand
    if plan is None:
        return None, validate_plan_full(None, q, idx), stats

    best_plan = plan
    best_rep = validate_plan_full(best_plan, q, idx)

    def accept(cand) -> bool:
        nonlocal best_plan, best_rep
        if not cand:
            return False
        cand_rep = validate_plan_full(cand, q, idx)
        if _dominates(cand_rep, best_rep):
            best_plan, best_rep = cand, cand_rep
            return True
        stats["rejected"] += 1
        return False

    if not best_rep.ok:
        accept(local_repair(best_plan, q, idx))

    for _ in range(max(1, cfg.plan_repair_attempts)):
        if best_rep.ok:
            break
        cand = await llm_repair(client, best_plan, q, idx, best_rep, namespace)
        stats["repairs"] += 1
        if cand is None:
            break
        if not accept(local_repair(cand, q, idx)):
            accept(cand)

    if not best_rep.ok:
        cand = budget_downgrade(best_plan, q, idx)
        if cand is not best_plan:
            accept(cand)

    return best_plan, best_rep, stats


def to_plan_record(q_idx: int, plan: list[dict] | None) -> dict:
    """评测提交行（官方只读 'plan' 键）。"""
    return {"query_idx": q_idx, "plan": plan or []}
