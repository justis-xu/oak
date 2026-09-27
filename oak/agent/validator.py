"""结构化全量计划校验：一次跑完官方 CS/HC 全部判定，产出可定位 PlanIssue。

设计原则：
- 官方语义逐条镜像（不严不松），标注官方函数出处；
- 本地图比官方 DB 小（每题语料），图存在性检查比官方 sandbox 更严——
  计划值全部来自图时这是良性收紧；
- 每条 issue 带 code/day/field/blocking，供 repair/fill/回滚决策。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..data.queries import Query
from .graph_index import (GraphIndex, compute_plan_cost, parse_from_to,
                          parse_name_city, split_attractions, strip_paren,
                          transportation_mode, first_flight_number)

DAY_KEYS = ["days", "current_city", "transportation", "breakfast",
            "attraction", "lunch", "dinner", "accommodation"]
MEALS = ("breakfast", "lunch", "dinner")


@dataclass
class PlanIssue:
    code: str
    day: int | None = None
    field: str | None = None
    message: str = ""
    blocking: bool = True

    def render(self) -> str:
        loc = f"day[{self.day}].{self.field}" if self.day is not None else self.code
        return f"{loc}: {self.message}"


@dataclass
class ValidationReport:
    issues: list[PlanIssue] = field(default_factory=list)
    cost: dict | None = None

    @property
    def ok(self) -> bool:
        return not any(i.blocking for i in self.issues)

    @property
    def blocking(self) -> list[PlanIssue]:
        return [i for i in self.issues if i.blocking]

    def summary(self) -> dict:
        from collections import Counter
        return {"n_issues": len(self.issues),
                "n_blocking": len(self.blocking),
                "by_code": dict(Counter(i.code for i in self.issues)),
                "cost": self.cost}


def _valid_city_sequence(city_list: list[str]) -> bool:
    """官方 is_valid_city_sequence：中间城市必须连续块（≥2 项）、离城不回访。"""
    if len(city_list) < 3:
        return False
    visited: set[str] = set()
    i = 0
    while i < len(city_list):
        city = city_list[i]
        if city in visited and (i != 0 and i != len(city_list) - 1):
            return False
        count = 0
        while i < len(city_list) and city_list[i] == city:
            count += 1
            i += 1
        if count == 1 and 0 < i - 1 < len(city_list) - 1:
            return False
        visited.add(city)
    return True


def _route_of(plan: list[dict]) -> list[tuple[str | None, str | None]]:
    """每天的 (origin, destination)：移动日 from→to，stay 日 (None, city)。"""
    out = []
    for unit in plan:
        cc = str(unit.get("current_city") or "")
        o, d = parse_from_to(cc)
        out.append((strip_paren(o).strip() if o else None,
                    strip_paren(d).strip() if d else None))
    return out


def activity_cities(i: int, plan: list[dict]) -> list[str]:
    """当天允许的餐/景点城市（官方 current_city 检查语义：移动日两端点，stay 日当前城）。"""
    cc = str(plan[i].get("current_city") or "")
    o, d = parse_from_to(cc)
    if o or d:
        return [c for c in (strip_paren(o).strip(), strip_paren(d).strip()) if c]
    v = strip_paren(cc.strip())
    return [v] if v else []


def lodging_city(i: int, plan: list[dict]) -> str | None:
    """当天住宿必须所在的城市（官方 final_city_list[-1]）。"""
    cc = str(plan[i].get("current_city") or "")
    o, d = parse_from_to(cc)
    if o or d:
        return strip_paren(d).strip() if d else None
    v = strip_paren(cc.strip())
    return v or None


def validate_plan_full(plan: list[dict] | None, q: Query, idx: GraphIndex) -> ValidationReport:
    rep = ValidationReport()
    if not plan:
        rep.issues.append(PlanIssue("plan.empty", message="plan is empty"))
        return rep
    issues = rep.issues
    last = q.days - 1

    if len(plan) != q.days:
        issues.append(PlanIssue("days.count",
                                message=f"plan has {len(plan)} days, expected {q.days}"))

    for i, unit in enumerate(plan[:q.days]):
        if unit.get("days") != i + 1:
            issues.append(PlanIssue("days.seq", day=i, field="days",
                                    message=f"'days' should be {i+1}, got {unit.get('days')!r}"))
        missing = [k for k in DAY_KEYS if k not in unit]
        if missing:
            issues.append(PlanIssue("day.keys", day=i, message=f"missing keys: {missing}"))
            continue
        cc = str(unit.get("current_city") or "")
        trans = str(unit.get("transportation") or "")
        moving = any(parse_from_to(cc))
        if not moving and ("from " in cc or " to " in cc):
            issues.append(PlanIssue("current_city.malformed", day=i, field="current_city",
                                    message=f"cannot parse route: {cc[:60]!r}"))
        if not moving and not cc.strip():
            issues.append(PlanIssue("current_city.empty", day=i, field="current_city"))

    route = _route_of(plan[:q.days])

    # ---- 路线（官方 is_reasonable_visiting_city） ----
    city_list: list[str] = []
    for i, (o, d) in enumerate(route):
        if o or d:
            city_list += [o, d]
        else:
            v = lodging_city(i, plan)
            if v:
                city_list.append(v)
    if q.days >= 1 and len(city_list) >= 2:
        if route[0][0] != q.org:
            issues.append(PlanIssue("route.start",
                                    message=f"day 1 must depart from {q.org}, got {route[0][0]!r}"))
        if city_list[0] != city_list[-1]:
            issues.append(PlanIssue("route.loop",
                                    message=f"trip not closed: starts {city_list[0]!r} ends {city_list[-1]!r}"))
    if not _valid_city_sequence(city_list):
        issues.append(PlanIssue("route.sequence",
                                message="city sequence invalid (middle singleton or revisit)"))
    for j, city in enumerate(city_list):
        if city and city not in idx.city_state_map:
            issues.append(PlanIssue("route.city_invalid", day=j // 2,
                                    message=f"{city!r} is not a valid city"))
    if q.days > 3:
        for j, city in enumerate(city_list):
            if city and j not in (0, len(city_list) - 1):
                st = idx.city_state_map.get(city)
                if st and st != q.dest:
                    issues.append(PlanIssue("route.state", day=j // 2,
                                            message=f"{city!r} is in {st}, not in {q.dest}"))
                    break
    # 本地增量：中间停留城市必须有业务数据（covered）
    for j, city in enumerate(city_list):
        if city and j not in (0, len(city_list) - 1):
            info = idx.cities.get(city)
            if info is not None and not info["covered"]:
                issues.append(PlanIssue("city.uncovered", day=j // 2,
                                        message=f"stop city {city!r} has no restaurant/"
                                                f"accommodation/attraction data in this query's corpus"))

    # ---- 城市数（官方 is_valid_visiting_city_number） ----
    seen_cities = {c for c in city_list if c}
    seen_cities.discard(q.org)
    if len(seen_cities) != q.visiting_city_number:
        issues.append(PlanIssue("route.city_count",
                                message=f"{len(seen_cities)} distinct visited cities "
                                        f"{sorted(seen_cities)}, expected {q.visiting_city_number}"))

    # ---- 交通（官方 is_valid_transportation / current_city / sandbox） ----
    modes: list[str | None] = []
    for i, unit in enumerate(plan[:q.days]):
        trans = str(unit.get("transportation") or "")
        cc = str(unit.get("current_city") or "")
        endpoints = activity_cities(i, plan)
        o, d = route[i]
        if trans and trans != "-":
            mode = transportation_mode(trans)
            modes.append(mode)
            if mode is None:
                issues.append(PlanIssue("transport.template", day=i, field="transportation",
                                        message=f"no recognizable mode: {trans[:60]!r}"))
            elif mode == "Flight":
                num = first_flight_number(trans)
                # 官方 sandbox：先取交通串自身 from→to，缺省才看 current_city
                to_, td_ = parse_from_to(trans)
                if not (to_ and td_):
                    to_, td_ = (o, d)
                row = idx.flight_by_num.get(num or "")
                if row is None or str(row.get("OriginCityName")) != (strip_paren(to_ or "") .strip()) \
                        or str(row.get("DestCityName")) != (strip_paren(td_ or "").strip()):
                    issues.append(PlanIssue("transport.flight_not_in_graph", day=i,
                                            field="transportation",
                                            message=f"flight {num} {to_}->{td_} not in graph"))
            elif mode in ("Taxi", "Self-driving"):
                if not idx.ground_valid(o or "", d or ""):
                    issues.append(PlanIssue("transport.ground_not_in_graph", day=i,
                                            field="transportation",
                                            message=f"{mode} leg {o}->{d} not valid in distance matrix"))
            # 官方 current_city：两端点城市必须是 trans 的子串
            for c in endpoints:
                if c and c not in trans:
                    issues.append(PlanIssue("transport.endpoint_mismatch", day=i,
                                            field="transportation",
                                            message=f"{c!r} not mentioned in transportation"))
        else:
            modes.append(None)
            if o or d:     # 移动日必须有交通（官方 is_not_absent）
                issues.append(PlanIssue("transport.missing", day=i, field="transportation",
                                        message="moving day has no transportation"))
    mode_set = {m for m in modes if m}
    if ("Self-driving" in mode_set and "Flight" in mode_set) or \
       ("Taxi" in mode_set and "Self-driving" in mode_set):
        issues.append(PlanIssue("transport.mixed",
                                message=f"conflicting transport modes: {sorted(mode_set)}"))
    lc_transport = (q.local_constraint or {}).get("transportation")
    if lc_transport:
        for i, unit in enumerate(plan[:q.days]):
            trans = str(unit.get("transportation") or "")
            if trans and trans != "-":
                if lc_transport == "no flight" and "Flight" in trans:
                    issues.append(PlanIssue("constraint.transport", day=i, field="transportation",
                                            message="no-flight constraint violated"))
                elif lc_transport == "no self-driving" and "Self-driving" in trans:
                    issues.append(PlanIssue("constraint.transport", day=i, field="transportation",
                                            message="no-self-driving constraint violated"))

    # ---- 餐 / 景点 / 住宿（官方 current_city + sandbox + is_not_absent） ----
    used_rests: dict[str, int] = {}
    used_attrs: dict[str, int] = {}
    cuisine_set: set[str] = set()
    want_cuisines = (q.local_constraint or {}).get("cuisine") or []
    if isinstance(want_cuisines, str):
        want_cuisines = [want_cuisines]

    for i, unit in enumerate(plan[:q.days]):
        endpoints = activity_cities(i, plan)
        moving = len(endpoints) >= 2

        for meal in MEALS:
            v = str(unit.get(meal) or "").strip()
            if v in ("-", ""):
                if not moving:
                    issues.append(PlanIssue("completeness.meal", day=i, field=meal,
                                            message=f"{meal} required on non-moving day"))
                continue
            if ";" in v:
                issues.append(PlanIssue("meal.multi_entity", day=i, field=meal,
                                        message=f"meal cell must be ONE 'Name, City': {v[:50]!r}"))
                continue
            nm, ct = parse_name_city(v)
            if nm == "-" or ct == "-":
                issues.append(PlanIssue("meal.format", day=i, field=meal,
                                        message=f"cannot parse 'Name, City': {v[:50]!r}"))
                continue
            row = idx.find_restaurant(nm, ct)
            if row is None:
                issues.append(PlanIssue("meal.not_in_graph", day=i, field=meal,
                                        message=f"restaurant {v[:50]!r} not in graph"))
                continue
            if not any(c and c in v for c in endpoints):
                issues.append(PlanIssue("meal.city_mismatch", day=i, field=meal,
                                        message=f"{v[:50]!r} not in {endpoints}"))
            used_rests[v] = used_rests.get(v, 0) + 1
            if used_rests[v] > 1:
                issues.append(PlanIssue("meal.repeat", day=i, field=meal,
                                        message=f"restaurant repeated: {v[:50]!r}"))
            if ct != q.org:
                cuis = str(row.get("Cuisines") or "")
                for c in want_cuisines:
                    if c in cuis:
                        cuisine_set.add(c)

        att = str(unit.get("attraction") or "").strip()
        if att in ("-", ""):
            if not moving:
                issues.append(PlanIssue("completeness.attraction", day=i, field="attraction",
                                        message="attraction required on non-moving day"))
        else:
            if not att.rstrip().endswith(";"):
                issues.append(PlanIssue("attraction.format", day=i, field="attraction",
                                        message=f"must end with ';': {att[:50]!r}"))
            for item in split_attractions(att):
                nm, ct = parse_name_city(item)
                if nm == "-" or ct == "-":
                    issues.append(PlanIssue("attraction.format", day=i, field="attraction",
                                            message=f"cannot parse item: {item[:50]!r}"))
                    continue
                if idx.find_attraction(nm, ct) is None:
                    issues.append(PlanIssue("attraction.not_in_graph", day=i, field="attraction",
                                            message=f"attraction {item[:50]!r} not in graph"))
                if not any(c and c in item for c in endpoints):
                    issues.append(PlanIssue("attraction.city_mismatch", day=i, field="attraction",
                                            message=f"{item[:40]!r} not in {endpoints}"))
                used_attrs[item] = used_attrs.get(item, 0) + 1
                if used_attrs[item] > 1:
                    issues.append(PlanIssue("attraction.repeat", day=i, field="attraction",
                                            message=f"attraction repeated: {item[:40]!r}"))

        acc = str(unit.get("accommodation") or "").strip()
        if acc in ("-", ""):
            if i != last:
                issues.append(PlanIssue("completeness.accommodation", day=i, field="accommodation",
                                        message="accommodation required (only last day may omit)"))
        else:
            nm, ct = parse_name_city(acc)
            row = idx.find_accommodation(nm, ct)
            if row is None:
                issues.append(PlanIssue("accom.not_in_graph", day=i, field="accommodation",
                                        message=f"accommodation {acc[:50]!r} not in graph"))
            else:
                lc = q.local_constraint or {}
                want_rt = lc.get("room type")
                if want_rt:
                    rt = str(row.get("room type") or "")
                    bad = ((want_rt == "not shared room" and rt == "Shared room") or
                           (want_rt == "shared room" and rt != "Shared room") or
                           (want_rt == "private room" and rt != "Private room") or
                           (want_rt == "entire room" and rt != "Entire home/apt"))
                    if bad:
                        issues.append(PlanIssue("constraint.room_type", day=i, field="accommodation",
                                                message=f"room type {rt!r} violates {want_rt!r}"))
                hr = lc.get("house rule")
                if hr and f"No {hr}" in str(row.get("house_rules") or ""):
                    issues.append(PlanIssue("constraint.house_rule", day=i, field="accommodation",
                                            message=f"house rule 'No {hr}' violated"))
            lg = lodging_city(i, plan)
            if lg and lg not in acc:
                issues.append(PlanIssue("accom.city_mismatch", day=i, field="accommodation",
                                        message=f"accommodation must be in {lg!r}: {acc[:50]!r}"))

    # ---- minimum nights（官方 count_consecutive_values 语义） ----
    i = 0
    acc_seq = [str(u.get("accommodation") or "-") for u in plan[:q.days]]
    while i < len(acc_seq):
        j = i
        while j + 1 < len(acc_seq) and acc_seq[j + 1] == acc_seq[i]:
            j += 1
        run = acc_seq[i]
        if run not in ("-", ""):
            nm, ct = parse_name_city(run)
            row = idx.find_accommodation(nm, ct)
            if row is not None:
                mn = row.get("minimum nights")
                try:
                    mn = int(float(mn)) if mn not in (None, "") else 0
                except Exception:
                    mn = 0
                if j - i + 1 < mn:
                    issues.append(PlanIssue("accom.min_nights", day=i, field="accommodation",
                                            message=f"{run[:40]!r} needs {mn} nights, block is {j - i + 1}"))
        i = j + 1

    # ---- 菜系覆盖（官方 is_valid_cuisine） ----
    for c in want_cuisines:
        if c not in cuisine_set:
            issues.append(PlanIssue("constraint.cuisine",
                                    message=f"cuisine {c!r} not covered outside {q.org}"))

    # ---- 信息量 ≥50%（官方 is_not_absent 尾部口径） ----
    n_valid = 0
    for unit in plan[:q.days]:
        for k in DAY_KEYS:
            v = unit.get(k)
            if v and v != "-":
                n_valid += 1
    if n_valid < 0.5 * 6 * q.days:
        issues.append(PlanIssue("completeness.absent",
                                message=f"valid info {n_valid}/{6*q.days} < 50%"))

    # ---- 成本（官方 get_total_cost + budget） ----
    rep.cost = compute_plan_cost(plan, q, idx)
    if rep.cost["unknown"]:
        issues.append(PlanIssue("cost.unknown", blocking=False,
                                message=f"{rep.cost['unknown']} cost items unresolvable "
                                        f"(total may be understated)"))
    if rep.cost["total"] > q.budget:
        issues.append(PlanIssue("budget.over",
                                message=f"total cost {rep.cost['total']:.0f} > budget {q.budget} "
                                        f"(flights {rep.cost['flights']:.0f} meals {rep.cost['meals']:.0f} "
                                        f"hotels {rep.cost['hotels']:.0f} ground {rep.cost['ground']:.0f})"))
    return rep
