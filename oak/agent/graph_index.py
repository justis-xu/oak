"""GraphIndex：一次物化全图行视图 + 官方语义解析/成本，供 validator/fill/evidence 共用。

所有解析函数严格镜像官方 evaluation 语义（utils/func.py、commonsense_constraint.py、
hard_constraint.py、googleDistanceMatrix/apis.py），不做更严或更松的自创规则。
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

from ..kg.graph import node_view

# ---------------------------------------------------------------- 官方解析语义
FROM_TO_RE = re.compile(r"from\s+(.+?)\s+to\s+([^,]+)(?=[,\s]|$)")      # 官方 extract_from_to
PAREN_RE = re.compile(r"^(.*?)\([^)]*\)")                                # 官方 extract_before_parenthesis
NAME_CITY_RE = re.compile(r"(.*?),\s*([^,]+)(\(\w[\w\s]*\))?$")          # 官方 get_valid_name_city
FLIGHT_NUM_RE = re.compile(r"Flight Number:\s*(F\d+)")


def strip_paren(s: str) -> str:
    m = PAREN_RE.search(s)
    return m.group(1) if m else s


def parse_from_to(text: str) -> tuple[str | None, str | None]:
    m = FROM_TO_RE.search(text or "")
    return (m.group(1), m.group(2)) if m else (None, None)


def parse_name_city(info: str) -> tuple[str, str]:
    """官方 get_valid_name_city：最后一个逗号拆分（保留名称内部逗号），city 去括注。"""
    m = NAME_CITY_RE.search(str(info or "").strip())
    if m:
        return m.group(1).strip(), strip_paren(m.group(2).strip()).strip()
    return "-", "-"


def split_attractions(s: str) -> list[str]:
    """官方 split(';')[:-1]：逐项景点实体（输入必须尾分号，否则丢最后一段）。"""
    return [x.strip() for x in str(s or "").split(";")[:-1] if x.strip()]


def transportation_mode(text: str) -> str | None:
    """官方 transportation_match 优先级：taxi > self-driving > flight。"""
    t = (text or "").lower()
    if "taxi" in t:
        return "Taxi"
    if "self-driving" in t:
        return "Self-driving"
    if "flight" in t:
        return "Flight"
    return None


def first_flight_number(trans: str) -> str | None:
    """官方只取字符串中第一个 'Flight Number: X'。"""
    m = FLIGHT_NUM_RE.search(trans or "")
    return m.group(1) if m else None


def _f(v) -> float | None:
    try:
        if v is None or v == "":
            return None
        return float(str(v).replace(",", "").replace("$", "").strip())
    except Exception:
        return None


def _i(v) -> int | None:
    f = _f(v)
    return int(f) if f is not None else None


# ---------------------------------------------------------------- 官方库保真门
from functools import lru_cache


@lru_cache(maxsize=4)
def official_index(tp_root_str: str) -> dict:
    """官方环境库索引（与 distance.csv 同级的静态事实）。

    两件事：
    1. 名字存在性（保真门）——语料 repr 个别行损坏（q40：名称跨行粘连的住宿
       官方库不存在），图忠实投影了坏行必须拦截；
    2. 权威属性行——图的属性是 LLM 抽的，会整字段丢失（q115 图 house_rules=None
       但官方是 'No parties'；q125 图 room type=None 但官方是 'Private room'），
       而官方评测全按官方库判。故属性以官方库为准。
    """
    tp = Path(tp_root_str)

    def rows_of(path, name_col, city_col) -> dict[str, list[dict]]:
        m: dict[str, list[dict]] = {}
        if path.exists():
            with path.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    c = (row.get(city_col) or "").strip()
                    n = (row.get(name_col) or "").strip()
                    if c and n:
                        m.setdefault(c, []).append(row)
        return m

    out = {
        "accommodations": rows_of(
            tp / "database/accommodations/clean_accommodations_2022.csv", "NAME", "city"),
        "restaurants": rows_of(
            tp / "database/restaurants/clean_restaurant_2022.csv", "Name", "City"),
        "attractions": rows_of(
            tp / "database/attractions/attractions.csv", "Name", "City"),
    }

    def first_match(kind: str, name: str, city: str, name_col: str) -> dict | None:
        """官方语义：NAME.contains(escape(name)) & city == → 第一行。"""
        for row in out[kind].get(city, []):
            if name and name in str(row.get(name_col) or ""):
                return row
        return None

    out["_first"] = first_match
    fl: dict[str, set] = {}
    fp = tp / "database/flights/clean_Flights_2022.csv"
    if fp.exists():
        with fp.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                n = (row.get("Flight Number") or "").strip()
                if n:
                    fl.setdefault(n, set()).add(
                        ((row.get("OriginCityName") or "").strip(),
                         (row.get("DestCityName") or "").strip()))
    out["flights"] = fl
    return out


# ---------------------------------------------------------------- GraphIndex
class GraphIndex:
    def __init__(self, g, tp_root, fidelity_gate: bool = True):
        """fidelity_gate=False 供合成图单测绕过官方库过滤。"""
        self.g = g
        self.tp_root = Path(tp_root)
        offi = official_index(str(self.tp_root))

        def rows(etype):
            return [node_view(nd) for _, nd in g.nodes(data=True)
                    if nd.get("etype") == etype]

        # 官方库保真过滤：餐/住/景 (name substring, city 等值)；航班 (num, o, d)
        # 官方 sandbox 全按官方库判 —— 图里损坏行（语料 repr 跨行粘连等）在此拦截
        self.dropped_official = {"Restaurant": 0, "Accommodation": 0,
                                 "Attraction": 0, "Flight": 0}
        first = offi["_first"]

        def offi_row(kind: str, name: str, city: str, name_col: str) -> dict | None:
            return first(kind, name, city, name_col)

        def merge_official(row: dict, off: dict | None) -> dict:
            """官方库非空属性覆盖图值（图抽取会整字段丢失；官方评测以官方库为准）。"""
            if not off:
                return row
            out = dict(row)
            for k, v in off.items():
                if v not in (None, "") and str(v).strip() not in ("nan", "NaT"):
                    out[k] = v
            return out

        _rests = rows("Restaurant")
        self.restaurants = []
        for r in _rests:
            off = offi_row("restaurants", str(r.get("Name") or ""),
                           str(r.get("City") or "").strip(), "Name")
            if off or not fidelity_gate:
                self.restaurants.append(merge_official(r, off))
        self.dropped_official["Restaurant"] = len(_rests) - len(self.restaurants)

        _accs = rows("Accommodation")
        self.accommodations = []
        for a in _accs:
            off = offi_row("accommodations", str(a.get("NAME") or ""),
                           str(a.get("city") or "").strip(), "NAME")
            if off or not fidelity_gate:
                self.accommodations.append(merge_official(a, off))
        self.dropped_official["Accommodation"] = len(_accs) - len(self.accommodations)

        _attrs = rows("Attraction")
        self.attractions = []
        for a in _attrs:
            off = offi_row("attractions", str(a.get("Name") or ""),
                           str(a.get("City") or "").strip(), "Name")
            if off or not fidelity_gate:
                self.attractions.append(merge_official(a, off))
        self.dropped_official["Attraction"] = len(_attrs) - len(self.attractions)

        _flights = rows("Flight")
        self.flights = []
        for f in _flights:
            routes = offi["flights"].get(str(f.get("Flight Number") or "").strip())
            if not fidelity_gate or (routes and ((str(f.get("OriginCityName") or ""), str(f.get("DestCityName") or "")) in routes)):
                self.flights.append(f)
        self.dropped_official["Flight"] = len(_flights) - len(self.flights)

        # (name, city) 索引（值规范化小写比较；contains 语义在查找时做）
        self.rest_by_key: dict[tuple[str, str], dict] = {}
        self.rests_by_city: dict[str, list[dict]] = {}
        for r in self.restaurants:
            nm, ct = str(r.get("Name") or ""), str(r.get("City") or "")
            self.rest_by_key[(nm.lower(), ct.lower())] = r
            self.rests_by_city.setdefault(ct, []).append(r)
        self.acc_by_key: dict[tuple[str, str], dict] = {}
        self.accs_by_city: dict[str, list[dict]] = {}
        for a in self.accommodations:
            nm, ct = str(a.get("NAME") or ""), str(a.get("city") or "")
            self.acc_by_key[(nm.lower(), ct.lower())] = a
            self.accs_by_city.setdefault(ct, []).append(a)
        self.attr_by_key: dict[tuple[str, str], dict] = {}
        self.attrs_by_city: dict[str, list[dict]] = {}
        for a in self.attractions:
            nm, ct = str(a.get("Name") or ""), str(a.get("City") or "")
            self.attr_by_key[(nm.lower(), ct.lower())] = a
            self.attrs_by_city.setdefault(ct, []).append(a)

        self.flight_by_num: dict[str, dict] = {}
        self.flights_by_route: dict[tuple[str, str], list[dict]] = {}
        for f in self.flights:
            num = str(f.get("Flight Number") or "")
            if num:
                self.flight_by_num.setdefault(num, f)
            o, d = str(f.get("OriginCityName") or ""), str(f.get("DestCityName") or "")
            self.flights_by_route.setdefault((o, d), []).append(f)

        # City 节点（state/covered 来自 enrich_city_nodes）
        self.cities: dict[str, dict] = {}
        for _, nd in g.nodes(data=True):
            if nd.get("etype") == "City":
                nm = str(node_view(nd).get("name") or "")
                if nm:
                    self.cities[nm] = {"state": nd.get("state", ""),
                                       "covered": bool(nd.get("covered")),
                                       "restaurants": nd.get("restaurant_count", 0),
                                       "accommodations": nd.get("accommodation_count", 0),
                                       "attractions": nd.get("attraction_count", 0)}

        # 官方州映射（is_reasonable_visiting_city 的城市合法性来源）
        self.city_state_map: dict[str, str] = {}
        cs_file = self.tp_root / "database" / "background" / "citySet_with_states.txt"
        if cs_file.exists():
            for ln in cs_file.read_text().splitlines():
                if "\t" in ln:
                    c, s = ln.split("\t", 1)
                    self.city_state_map[c.strip()] = s.strip()

        # 地面交通：直接读官方 distance.csv（GroundTransportLink 图节点不含成本）
        # valid = 行存在且 duration/distance 非 nan 且 duration 不含 'day'（官方 cost=None 语义）
        self.ground: dict[tuple[str, str], dict] = {}
        dist_file = self.tp_root / "database" / "googleDistanceMatrix" / "distance.csv"
        if dist_file.exists():
            with dist_file.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    o = strip_paren((row.get("origin") or "").strip())
                    d = strip_paren((row.get("destination") or "").strip())
                    dur = (row.get("duration") or "").strip()
                    dist = (row.get("distance") or "").strip()
                    km = _f(dist.replace("km", "")) if dist else None
                    valid = bool(dur and dist and km is not None and "day" not in dur)
                    self.ground[(o, d)] = {
                        "km": km, "duration": dur, "distance": dist, "valid": valid}

    # ---- 查找（官方 contains 语义：name 子串 + city 等值） ----
    def find_restaurant(self, name: str, city: str) -> dict | None:
        if name in ("-", "") or city in ("-", ""):
            return None
        r = self.rest_by_key.get((name.lower(), city.lower()))
        if r:
            return r
        nl, cl = name.lower(), city.lower()
        for r in self.rests_by_city.get(city, []):
            if nl in str(r.get("Name") or "").lower():
                return r
        return None

    def find_accommodation(self, name: str, city: str) -> dict | None:
        if name in ("-", "") or city in ("-", ""):
            return None
        a = self.acc_by_key.get((name.lower(), city.lower()))
        if a:
            return a
        nl, cl = name.lower(), city.lower()
        for a in self.accs_by_city.get(city, []):
            if nl in str(a.get("NAME") or "").lower():
                return a
        return None

    def find_attraction(self, name: str, city: str) -> dict | None:
        if name in ("-", "") or city in ("-", ""):
            return None
        a = self.attr_by_key.get((name.lower(), city.lower()))
        if a:
            return a
        nl, cl = name.lower(), city.lower()
        for a in self.attrs_by_city.get(city, []):
            if nl in str(a.get("Name") or "").lower():
                return a
        return None

    def ground_valid(self, org: str, dest: str) -> bool:
        rec = self.ground.get((strip_paren((org or "").strip()),
                               strip_paren((dest or "").strip())))
        return bool(rec and rec["valid"])

    def ground_km(self, org: str, dest: str) -> float | None:
        rec = self.ground.get((strip_paren((org or "").strip()),
                               strip_paren((dest or "").strip())))
        return rec["km"] if rec and rec["valid"] else None

    def covered_cities(self, state: str | None = None) -> list[dict]:
        out = [{"name": nm, **info} for nm, info in self.cities.items()
               if info["covered"] and (state is None or info["state"] == state)]
        out.sort(key=lambda c: c["name"])
        return out


# ---------------------------------------------------------------- 官方成本公式
def compute_plan_cost(plan: list[dict], q, idx: GraphIndex) -> dict:
    """镜像官方 get_total_cost（含首航段、每住宿日、每腿独立 mode 语义）。

    返回 {"total", "flights", "meals", "hotels", "ground", "unknown"}；
    unknown = 无法解析的项数（成本被低估的信号，validator 会报 issue）。
    """
    people = max(1, int(q.people_number or 1))
    flights_c = meals_c = hotels_c = ground_c = 0.0
    unknown = 0

    for i, unit in enumerate(plan[:q.days]):
        trans = str(unit.get("transportation") or "")
        if trans and trans != "-":
            org, dest = parse_from_to(trans)
            if org is None or dest is None:
                org, dest = parse_from_to(str(unit.get("current_city") or ""))
            mode = transportation_mode(trans)
            if mode == "Flight" and org and dest:
                num = first_flight_number(trans)
                row = idx.flight_by_num.get(num or "")
                if row and _f(row.get("Price")) is not None:
                    flights_c += _f(row["Price"]) * people
                else:
                    unknown += 1
            elif mode in ("Taxi", "Self-driving") and org and dest:
                km = idx.ground_km(org, dest)
                if km is not None:
                    per = int(km * 0.05) if mode == "Self-driving" else int(km)
                    veh = math.ceil(people / 5) if mode == "Self-driving" else math.ceil(people / 4)
                    ground_c += per * veh
                else:
                    unknown += 1
            elif mode is None:
                unknown += 1

        for meal in ("breakfast", "lunch", "dinner"):
            v = str(unit.get(meal) or "")
            if v and v != "-":
                nm, ct = parse_name_city(v)
                r = idx.find_restaurant(nm, ct)
                if r and _f(r.get("Average Cost")) is not None:
                    meals_c += _f(r["Average Cost"]) * people
                else:
                    unknown += 1

        acc = str(unit.get("accommodation") or "")
        if acc and acc != "-":
            nm, ct = parse_name_city(acc)
            a = idx.find_accommodation(nm, ct)
            if a and _f(a.get("price")) is not None:
                occ = _i(a.get("maximum occupancy")) or 1
                hotels_c += _f(a["price"]) * math.ceil(people / max(1, occ))
            else:
                unknown += 1

    return {"total": flights_c + meals_c + hotels_c + ground_c,
            "flights": flights_c, "meals": meals_c, "hotels": hotels_c,
            "ground": ground_c, "unknown": unknown}
