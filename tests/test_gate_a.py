"""第二轮终态层单元测试（Gate A，纯离线，不调 LLM）。

运行: uv run python -m unittest tests.test_gate_a -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx

from oak.config import load_config
from oak.data.queries import Query, parse_dates, applicable_hc_keys
from oak.kg.graph import node_view, enrich_city_nodes, build_graph
from oak.kg.extract import EntityCandidate, _entity_grounded, _chunk_blob
from oak.data.corpus import Chunk
from oak.agent.graph_index import (GraphIndex, compute_plan_cost, parse_from_to,
                                    parse_name_city, split_attractions,
                                    transportation_mode, first_flight_number)
from oak.agent.validator import validate_plan_full
from oak.agent.planner import parse_plan_payload, normalize_plan, local_repair
from oak.agent.react import _envelope


def _q(**kw) -> Query:
    base = dict(idx=0, org="Austin", dest="Texas", days=3, date=["2022-03-01"],
                people_number=1, local_constraint={}, budget=100000, query="",
                level="easy", visiting_city_number=1)
    base.update(kw)
    return Query(**base)


def _mini_graph() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    def add(nid, etype, key, **props):
        import json as _j
        g.add_node(nid, etype=etype, __key__=_j.dumps(key), **props)
    add("r1", "Restaurant", {"Name": "Rome's Pizza", "City": "Houston"},
        Cuisines="Italian", **{"Average Cost": 20})
    add("r2", "Restaurant", {"Name": "Sushi Bar", "City": "Houston"},
        Cuisines="Japanese", **{"Average Cost": 30})
    add("r3", "Restaurant", {"Name": "Taco Stand", "City": "Austin"},
        Cuisines="Mexican", **{"Average Cost": 10})
    add("r4", "Restaurant", {"Name": "Pho Saigon", "City": "Houston"},
        Cuisines="Vietnamese", **{"Average Cost": 15})
    add("r5", "Restaurant", {"Name": "Pappas BBQ", "City": "Houston"},
        Cuisines="American BBQ", **{"Average Cost": 25})
    add("r6", "Restaurant", {"Name": "La Balanza", "City": "Houston"},
        Cuisines="Peruvian", **{"Average Cost": 18})
    add("a1", "Accommodation", {"NAME": "Huge, New and Green Oasis", "city": "Houston"},
        price=100, **{"room type": "Private room", "minimum nights": 2,
                      "maximum occupancy": 2, "house_rules": "No smoking"})
    add("a2", "Accommodation", {"NAME": "Budget Inn", "city": "Houston"},
        price=40, **{"room type": "Entire home/apt", "minimum nights": 1,
                     "maximum occupancy": 2, "house_rules": ""})
    add("t1", "Attraction", {"Name": "Space Center", "City": "Houston", "Address": "1 Road"})
    add("t2", "Attraction", {"Name": "Museum", "City": "Houston", "Address": "2 Road"})
    add("f1", "Flight", {"Flight Number": "F1", "FlightDate": "2022-03-01"},
        Price=100, DepTime="08:00", ArrTime="09:00",
        OriginCityName="Austin", DestCityName="Houston")
    add("c1", "City", {"name": "Houston"})
    add("c2", "City", {"name": "Austin"})
    return g


def _idx(g) -> GraphIndex:
    cfg = load_config()
    from oak.kg.graph import enrich_city_nodes
    enrich_city_nodes(g, cfg.tp_root)
    return GraphIndex(g, cfg.tp_root, fidelity_gate=False)


class TestParsing(unittest.TestCase):
    def test_parse_dates_forms(self):
        self.assertEqual(parse_dates(["2022-03-01"]), ["2022-03-01"])
        self.assertEqual(parse_dates("['2022-03-01']"), ["2022-03-01"])
        self.assertEqual(parse_dates(list("['2022-03-01']")), ["2022-03-01"])
        self.assertEqual(parse_dates("2022-03-01"), ["2022-03-01"])
        self.assertEqual(parse_dates(None), [])

    def test_applicable_hc_keys(self):
        lc = {"cuisine": ["Italian"], "transportation": "no flight"}
        self.assertEqual(applicable_hc_keys("easy", {}), ["valid_cost"])
        self.assertEqual(applicable_hc_keys("medium", lc),
                         ["valid_cost", "valid_cuisine"])     # medium 不计 transportation
        self.assertEqual(applicable_hc_keys("hard", lc),
                         ["valid_cost", "valid_cuisine", "valid_transportation"])

    def test_name_city_last_comma(self):
        nm, ct = parse_name_city("Huge, New and Green Oasis, Tucson")
        self.assertEqual(nm, "Huge, New and Green Oasis")
        self.assertEqual(ct, "Tucson")
        nm, ct = parse_name_city("Rome's Pizza, Houston (Texas)")
        self.assertEqual(ct, "Houston")

    def test_split_attractions_needs_trailing_semicolon(self):
        self.assertEqual(split_attractions("A, X; B, Y; "), ["A, X", "B, Y"])
        self.assertEqual(split_attractions("A, X; B, Y"), ["A, X"])   # 无尾分号丢最后段

    def test_transport_mode_priority(self):
        self.assertEqual(transportation_mode("Taxi, from A to B"), "Taxi")
        self.assertEqual(transportation_mode("Self-driving, from A to B"), "Self-driving")
        self.assertEqual(transportation_mode("Flight Number: F1, ..."), "Flight")

    def test_first_flight_number(self):
        self.assertEqual(first_flight_number(
            "Flight Number: F1, from A to B; Flight Number: F2, from B to C"), "F1")

    def test_node_view(self):
        nv = node_view({"__key__": '{"NAME": "H1", "city": "Tucson"}',
                        "etype": "Accommodation", "price": 912, "__merged__": 2})
        self.assertEqual((nv["NAME"], nv["city"], nv["price"]), ("H1", "Tucson", 912))
        self.assertNotIn("__merged__", nv)


class TestGrounding(unittest.TestCase):
    def test_exact_grounded(self):
        ch = Chunk("restaurants", 0, "Name,City,Cuisines", "Parrot's,Miami,Chinese")
        blob = _chunk_blob(ch)
        self.assertTrue(_entity_grounded({"Name": "Parrot's", "City": "Miami"}, blob))
        self.assertFalse(_entity_grounded({"Name": "Italiize", "City": "Miami"}, blob))

    def test_numeric_pk_skipped(self):
        ch = Chunk("flights", 0, "num", "1")
        self.assertTrue(_entity_grounded({"num": 1.0}, _chunk_blob(ch)))


class TestEnvelope(unittest.TestCase):
    def test_warning_dict_not_candidate(self):
        env = _envelope({"warning": "no flights found"})
        self.assertEqual(env["status"], "empty")

    def test_results_unwrap_and_sentinel_rows(self):
        env = _envelope({"results": [{}, {"Name": "X", "City": "Y"}, None][:2]})
        self.assertEqual(env["status"], "ok")
        self.assertEqual(len(env["results"]), 1)

    def test_match_note_goes_to_suggestions(self):
        env = _envelope([{"Name": "X", "match_note": "date_relaxed"}])
        self.assertEqual(env["status"], "empty")
        self.assertEqual(env["suggestions"][0]["Name"], "X")

    def test_scalar_value(self):
        self.assertEqual(_envelope(42), {"status": "ok", "value": 42})


class TestPayload(unittest.TestCase):
    def test_wrapper(self):
        self.assertEqual(parse_plan_payload('{"plan": [{"days": 1}]}'), [{"days": 1}])

    def test_bare_array_text(self):
        self.assertEqual(parse_plan_payload('junk [ {"days": 1} ] junk'), [{"days": 1}])

    def test_none(self):
        self.assertIsNone(parse_plan_payload(None))
        self.assertIsNone(parse_plan_payload("no json"))

    def test_scalarize_list_values(self):
        from oak.agent.planner import _scalarize
        self.assertEqual(_scalarize(["Flight Number: F1, from A to B"]),
                         "Flight Number: F1, from A to B")
        self.assertEqual(_scalarize(["-"]), "-")
        self.assertEqual(_scalarize([]), "-")
        self.assertEqual(_scalarize(None), "-")
        self.assertEqual(_scalarize(["a", "b"]), "a; b")

    def test_normalize_unwraps_list_day_values(self):
        plan, _ = normalize_plan(
            [{"days": 1, "current_city": "Norfolk",
              "transportation": ["Flight Number: F1, from A to B, Departure Time: 08:00, Arrival Time: 09:00"],
              "breakfast": "-", "lunch": "-", "dinner": "-", "attraction": "-",
              "accommodation": "-"}], _q())
        self.assertFalse(plan[0]["transportation"].startswith("["))

    def test_local_repair_rewrites_moving_current_city(self):
        q = _q()
        idx = _idx(_mini_graph())
        plan = [{"days": 1, "current_city": "Houston",
                 "transportation": "Flight Number: F1, from Austin to Houston, "
                                   "Departure Time: 08:00, Arrival Time: 09:00",
                 "breakfast": "-", "lunch": "-", "dinner": "-", "attraction": "-",
                 "accommodation": "-"},
                {"days": 2, "current_city": "Houston", "transportation": "-",
                 "breakfast": "-", "lunch": "-", "dinner": "-", "attraction": "-",
                 "accommodation": "-"},
                {"days": 3, "current_city": "from Houston to Austin",
                 "transportation": "Flight Number: F9, from Houston to Austin, "
                                   "Departure Time: 08:00, Arrival Time: 09:00",
                 "breakfast": "-", "lunch": "-", "dinner": "-", "attraction": "-",
                 "accommodation": "-"}]
        out = local_repair(json_loads_dumps(plan), q, idx)
        self.assertEqual(out[0]["current_city"], "from Austin to Houston")


class TestValidator(unittest.TestCase):
    def _good_plan(self, q):
        return [
            {"days": 1, "current_city": "from Austin to Houston",
             "transportation": "Flight Number: F1, from Austin to Houston, "
                               "Departure Time: 08:00, Arrival Time: 09:00",
             "breakfast": "-", "lunch": "Sushi Bar, Houston",
             "dinner": "Rome's Pizza, Houston",
             "attraction": "Space Center, Houston; ",
             "accommodation": "Huge, New and Green Oasis, Houston"},
            {"days": 2, "current_city": "Houston", "transportation": "-",
             "breakfast": "Pho Saigon, Houston", "lunch": "Pappas BBQ, Houston",
             "dinner": "La Balanza, Houston",
             "attraction": "Museum, Houston; ",
             "accommodation": "Huge, New and Green Oasis, Houston"},
            {"days": 3, "current_city": "from Houston to Austin",
             "transportation": "Flight Number: F1, from Austin to Houston, "
                               "Departure Time: 08:00, Arrival Time: 09:00",
             "breakfast": "-", "lunch": "-", "dinner": "-",
             "attraction": "-", "accommodation": "-"},
        ]

    def test_good_plan_passes(self):
        q = _q()
        idx = _idx(_mini_graph())
        rep = validate_plan_full(self._good_plan(q), q, idx)
        self.assertEqual(rep.blocking, [], [i.render() for i in rep.blocking])

    def test_wrong_city_restaurant(self):
        q = _q()
        idx = _idx(_mini_graph())
        plan = self._good_plan(q)
        plan[1]["lunch"] = "Taco Stand, Austin"      # stay day 在 Houston
        rep = validate_plan_full(plan, q, idx)
        codes = [i.code for i in rep.blocking]
        self.assertIn("meal.city_mismatch", codes)

    def test_repeat_restaurant(self):
        q = _q()
        idx = _idx(_mini_graph())
        plan = self._good_plan(q)
        plan[1]["dinner"] = "Sushi Bar, Houston"       # day1 lunch 已用
        rep = validate_plan_full(plan, q, idx)
        self.assertIn("meal.repeat", [i.code for i in rep.blocking])

    def test_min_nights(self):
        q = _q(days=3)
        idx = _idx(_mini_graph())
        plan = self._good_plan(q)
        plan[1]["accommodation"] = "-"                  # 块只剩 1 晚 < minimum 2
        rep = validate_plan_full(plan, q, idx)
        self.assertIn("accom.min_nights", [i.code for i in rep.blocking])

    def test_budget(self):
        q = _q(budget=10)
        idx = _idx(_mini_graph())
        rep = validate_plan_full(self._good_plan(q), q, idx)
        self.assertIn("budget.over", [i.code for i in rep.blocking])

    def test_transport_mixed(self):
        q = _q()
        idx = _idx(_mini_graph())
        plan = self._good_plan(q)
        plan[2]["transportation"] = "Self-driving, from Houston to Austin"
        rep = validate_plan_full(plan, q, idx)
        self.assertIn("transport.mixed", [i.code for i in rep.blocking])


class TestCost(unittest.TestCase):
    def test_formula_people_rooms(self):
        g = _mini_graph()
        idx = _idx(g)
        q = _q(people_number=5)
        plan = [{"transportation": "Flight Number: F1, from Austin to Houston, "
                                   "Departure Time: 08:00, Arrival Time: 09:00",
                 "current_city": "from Austin to Houston",
                 "breakfast": "-", "lunch": "Sushi Bar, Houston",
                 "dinner": "Rome's Pizza, Houston", "attraction": "-",
                 "accommodation": "Huge, New and Green Oasis, Houston", "days": 1}]
        c = compute_plan_cost(plan, q, idx)
        # flight 100*5 + meals (30+20)*5 + hotel 100*ceil(5/2)=300
        self.assertAlmostEqual(c["total"], 500 + 250 + 300)

    def test_ground_modes(self):
        cfg = load_config()
        idx = GraphIndex(nx.MultiDiGraph(), cfg.tp_root)
        # 官方 distance.csv：取一对真实有效城市
        pair = next((k for k, v in idx.ground.items() if v["valid"]), None)
        if pair is None:
            self.skipTest("no valid ground pair")
        (o, d), rec = pair, idx.ground[next(k for k, v in idx.ground.items() if v["valid"])]
        q = _q(people_number=6, org=o, dest=d)
        plan = [{"transportation": f"Taxi, from {o} to {d}",
                 "current_city": f"from {o} to {d}", "breakfast": "-", "lunch": "-",
                 "dinner": "-", "attraction": "-", "accommodation": "-", "days": 1}]
        c1 = compute_plan_cost(plan, q, idx)
        plan[0]["transportation"] = f"Self-driving, from {o} to {d}"
        c2 = compute_plan_cost(plan, q, idx)
        self.assertAlmostEqual(c1["ground"], int(rec["km"]) * 2)          # ceil(6/4)=2
        self.assertAlmostEqual(c2["ground"], int(rec["km"] * 0.05) * 2)   # ceil(6/5)=2


class TestLocalRepair(unittest.TestCase):
    def test_no_cross_city_fallback(self):
        q = _q()
        idx = _idx(_mini_graph())
        plan = [{"days": 1, "current_city": "from Austin to Houston",
                 "transportation": "Flight Number: F1, from Austin to Houston, "
                                   "Departure Time: 08:00, Arrival Time: 09:00",
                 "breakfast": "-", "lunch": "-", "dinner": "-",
                 "attraction": "-", "accommodation": "-"},
                {"days": 2, "current_city": "Houston", "transportation": "-",
                 "breakfast": "-", "lunch": "-", "dinner": "-",
                 "attraction": "-", "accommodation": "-"},
                {"days": 3, "current_city": "from Houston to Austin",
                 "transportation": "Flight Number: F9, from Houston to Austin, "
                                   "Departure Time: 08:00, Arrival Time: 09:00",
                 "breakfast": "-", "lunch": "-", "dinner": "-",
                 "attraction": "-", "accommodation": "-"}]
        out = local_repair(json_loads_dumps(plan), q, idx)
        # day2（stay Houston）三餐应从 Houston 池回填
        self.assertNotEqual(out[1]["lunch"], "-")
        self.assertIn("Houston", out[1]["lunch"])
        # Austin 只有 Taco Stand，但 day2 是 Houston stay day —— 绝不跨城
        self.assertNotEqual(out[1]["dinner"], "Taco Stand, Austin")
        # 住宿块（day1-2，Houston）应回填同城
        self.assertIn("Houston", out[0]["accommodation"])


def json_loads_dumps(x):
    import json
    return json.loads(json.dumps(x))


if __name__ == "__main__":
    unittest.main(verbosity=2)
