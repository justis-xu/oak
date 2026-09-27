def search_flights(origin_city="", dest_city="", flight_date="", max_price=None, limit=50):
    """Find Flights origin->dest on a date with progressive relaxation so empty exact matches still return real, PK-carrying rows.

    Args:
        origin_city (str): departure city name (substring match on OriginCityName).
        dest_city (str): destination city name (substring match on DestCityName).
        flight_date (str): departure date string, e.g. "2022-03-23" (matches FlightDate).
        max_price (float | None): optional upper bound on Price.
        limit (int): maximum number of candidate rows returned.

    Returns:
        list[dict]: up to `limit` flight rows, each guaranteed to carry the Flight
        primary key ('Flight Number', 'FlightDate') plus Price, DepTime, ArrTime,
        OriginCityName and DestCityName, together with a 'match_note' field
        ('exact' | 'date_relaxed' | 'dest_relaxed') and a 'suggestion' string listing
        the destination cities the origin actually flies to (derived from
        has_destination edges). Returns [] only when even the relaxed lookups
        (origin-only / dest-only) match nothing, so the planner can pivot to ground
        legs instead of inventing a flight number.

    Example:
        search_flights("Houston", "Utah", "2022-03-23")
    """
    exact_filters = {}
    if origin_city:
        exact_filters["OriginCityName"] = origin_city
    if dest_city:
        exact_filters["DestCityName"] = dest_city
    if flight_date:
        exact_filters["FlightDate"] = flight_date

    rows = lookup_entities("Flight", exact_filters if exact_filters else None, limit=limit)
    match_note = "exact"

    # Relaxation 1: exact match empty -> drop flight_date, keep origin/dest constraints.
    if not rows and flight_date:
        relaxed = dict(exact_filters)
        relaxed.pop("FlightDate", None)
        rows = lookup_entities("Flight", relaxed if relaxed else None, limit=limit)
        match_note = "date_relaxed"

    # Relaxation 2: still empty -> query origin-only and dest-only pools.
    if not rows and (origin_city or dest_city):
        pool = []
        if origin_city:
            pool.extend(lookup_entities("Flight", {"OriginCityName": origin_city}, limit=limit))
        if dest_city:
            pool.extend(lookup_entities("Flight", {"DestCityName": dest_city}, limit=limit))
        rows = pool
        match_note = "dest_relaxed"

    if not rows:
        return []

    # Suggestion: dest cities the origin actually flies to, via has_destination edges.
    suggestion = ""
    if match_note in ("date_relaxed", "dest_relaxed") and origin_city:
        origin_pool = lookup_entities("Flight", {"OriginCityName": origin_city}, limit=limit)
        city_names = []
        seen_cities = set()
        for f in origin_pool:
            fid = f.get("__id__")
            if fid is None:
                continue
            for target in traverse_relations(fid, "has_destination", direction="out", max_hops=1):
                cname = target.get("name")
                if cname and cname not in seen_cities:
                    seen_cities.add(cname)
                    city_names.append(cname)
        if not city_names:  # defensive fallback to the DestCityName attribute
            for f in origin_pool:
                cname = f.get("DestCityName")
                if cname and cname not in seen_cities:
                    seen_cities.add(cname)
                    city_names.append(cname)
        if city_names:
            suggestion = "Flights from {0} fly to: {1}".format(
                origin_city, ", ".join(city_names[:15])
            )

    # Hard guarantee: keep only rows that explicitly carry BOTH primary-key attributes.
    identified = [
        r for r in rows
        if r.get("Flight Number") is not None and r.get("FlightDate") is not None
    ]

    if max_price is not None:
        identified = filter_numeric(identified, "Price", "<=", max_price)

    # Deduplicate on the Flight primary key (Flight Number, FlightDate).
    seen = set()
    unique = []
    for r in identified:
        key = (r.get("Flight Number"), r.get("FlightDate"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    # Cheapest first; non-numeric prices sort as 0.0 rather than crashing.
    unique = sorted(
        unique,
        key=lambda r: r.get("Price") if isinstance(r.get("Price"), (int, float)) else 0.0,
    )

    if isinstance(limit, int) and limit >= 0:
        unique = unique[:limit]

    out = project_properties(
        unique,
        [
            "Flight Number",
            "FlightDate",
            "Price",
            "DepTime",
            "ArrTime",
            "OriginCityName",
            "DestCityName",
        ],
    )
    for row in out:
        row["match_note"] = match_note
        row["suggestion"] = suggestion
    return out

def find_restaurants(city, cuisines=None, max_avg_cost=None):
    """Find Restaurants in a REQUIRED city (exact case-insensitive City match, graph-verified via located_in -> City) with optional cuisine/cost filters; empty matches return a loud warning dict instead of [].

    Args:
        city (str): REQUIRED. Exact (case-insensitive) equality on the Restaurant
            'City' attribute; surviving rows must also be Restaurant nodes
            connected to a City node through the 'located_in' relation. If city
            is missing/empty, a warning dict is returned (never an unfiltered list).
        cuisines (list[str] | None): A row is kept if any comma-split token of its
            'Cuisines' field overlaps a requested cuisine (case-insensitive).
        max_avg_cost (float | None): Keep rows with 'Average Cost' <= value.

    Returns:
        list[dict]: Matching restaurants projected to Name, City, Cuisines,
        Average Cost, Aggregate Rating (Name and City present on every row,
        City set to the graph-resolved city name so breakfast/lunch/dinner
        picks can be cross-checked against the day's current city), best rated
        first (ties broken by lower cost). If nothing matches, returns
        {'warning': '<LOUD MESSAGE>', 'city': ..., 'cuisines': [...],
        'max_avg_cost': ..., 'restaurants': []} so the planner re-plans rather
        than inventing a restaurant.

    Example:
        find_restaurants("Fort Lauderdale", cuisines=["American", "French"], max_avg_cost=40)
    """
    props = ["Name", "City", "Cuisines", "Average Cost", "Aggregate Rating"]

    needle = str(city).strip() if city is not None else ""

    wanted = []
    if cuisines:
        for c in cuisines:
            cs = str(c).strip()
            if cs:
                wanted.append(cs)

    cap = max_avg_cost
    if cap is not None and not isinstance(cap, (int, float)):
        s = str(cap).strip().replace("$", "").replace(",", "")
        cap = float(s) if s.replace(".", "", 1).isdigit() else None

    # --- Guard: city is a required parameter (never return an unfiltered list) ---
    if not needle:
        return {
            "warning": "REQUIRED PARAMETER MISSING: 'city' is required for find_restaurants; RE-PLAN with an explicit city — DO NOT invent a restaurant.",
            "city": needle,
            "cuisines": wanted,
            "max_avg_cost": cap,
            "restaurants": [],
        }

    # --- Stage 1: strict case-insensitive City equality on real Restaurant nodes ---
    rows = lookup_entities("Restaurant", {"City": needle}, limit=500)
    if not rows:
        # Broad fetch only as retrieval aid; strict equality re-applied below.
        rows = lookup_entities("Restaurant", None, limit=2000)
    low = needle.lower()
    rows = [r for r in rows if str(r.get("City", "")).strip().lower() == low]
    if not rows:
        return {
            "warning": "NO RESTAURANTS IN CITY: no Restaurant node has City == '" + needle + "' (exact, case-insensitive); RE-PLAN (change city or relax filters) — DO NOT invent a restaurant.",
            "city": needle,
            "cuisines": wanted,
            "max_avg_cost": cap,
            "restaurants": [],
        }

    # --- Stage 2: graph verification — rows must be located_in a City node ---
    rows = filter_relation_connected(rows, "located_in", "City", None)
    if not rows:
        return {
            "warning": "NO VERIFIED RESTAURANTS: city-matching Restaurant nodes lack a 'located_in' link to a City node in the sandbox; RE-PLAN — DO NOT invent a restaurant.",
            "city": needle,
            "cuisines": wanted,
            "max_avg_cost": cap,
            "restaurants": [],
        }

    # --- Resolved city name taken from surviving graph-verified nodes ---
    resolved = needle
    for r in rows:
        rc = str(r.get("City", "")).strip()
        if rc:
            resolved = rc
        break

    # --- Stage 3: cuisine filter via multi-value set overlap ---
    if wanted:
        rows = filter_set_overlap(rows, "Cuisines", wanted)
        if not rows:
            return {
                "warning": "NO RESTAURANTS FOR CUISINES: no Restaurant in '" + resolved + "' serves " + ", ".join(wanted) + "; RE-PLAN (relax cuisines) — DO NOT invent a restaurant.",
                "city": resolved,
                "cuisines": wanted,
                "max_avg_cost": cap,
                "restaurants": [],
            }

    # --- Stage 4: strict cost cap (Average Cost <= value) ---
    if cap is not None:
        rows = filter_numeric(rows, "Average Cost", "<=", cap)
        if not rows:
            return {
                "warning": "NO RESTAURANTS UNDER BUDGET: no Restaurant in '" + resolved + "' has Average Cost <= " + str(cap) + "; RE-PLAN (raise budget) — DO NOT invent a restaurant.",
                "city": resolved,
                "cuisines": wanted,
                "max_avg_cost": cap,
                "restaurants": [],
            }

    # --- Stage 5: rank by rating desc then lower cost; project and stamp resolved City ---
    ranked = []
    for r in rows:
        rating = r.get("Aggregate Rating", 0)
        cost = r.get("Average Cost", 0)
        if not isinstance(rating, (int, float)):
            rating = 0.0
        if not isinstance(cost, (int, float)):
            cost = 0.0
        ranked.append((rating, -cost, r))

    ranked = sorted(ranked, key=lambda t: (t[0], t[1]), reverse=True)

    out = project_properties([t[2] for t in ranked], props)
    if not out:
        return {
            "warning": "NO PROJECTED RESTAURANT ROWS: verified matches produced no projected fields; RE-PLAN — DO NOT invent a restaurant.",
            "city": resolved,
            "cuisines": wanted,
            "max_avg_cost": cap,
            "restaurants": [],
        }
    for row in out:
        row["City"] = resolved
    return out

def find_accommodations(city="", max_price=None, min_review=None, room_type="", min_occupancy=None, limit=50, visiting_city=""):
    """Anchor lodging to the trip's current (visiting) city via the located_in edge, filter by
    price, review rating, room type and occupancy, and always return the resolved city and NAME
    on every row; empty results come back as one explanatory row with the city's Accommodation count.

    Args:
        city (str or dict): exact case-insensitive match on 'city' (the located_in target);
            a payload dict like {"city": ..., "max_price": ...} is also tolerated.
        max_price (float, optional): keep rows with 'price' <= max_price.
        min_review (float, optional): keep rows with 'review rate number' >= min_review.
        room_type (str, optional): exact match on 'room type', e.g. "Private room".
        min_occupancy (int, optional): keep rows with 'maximum occupancy' >= min_occupancy.
        limit (int): maximum number of rows returned (default 50).
        visiting_city (str, optional): current day's visiting city; when set, rows are filtered
            through the located_in edge to that exact city and mismatches are dropped.

    Returns:
        dict: {"results": rows with ALL Accommodation fields, each always carrying the resolved
        'city' and 'NAME', sorted by price ascending, "dropped": rows whose city/located_in did
        not equal the anchored city, "count": number of Accommodations anchored to that city,
        "warning": "" on success or an explanatory message}. On 0 matches, results holds one
        explanatory row with the Accommodation count for that city instead of a bare [].
        Never raises.

    Example:
        find_accommodations(city="Houston", max_price=200, min_review=4.0, room_type="Private room", min_occupancy=2)
    """
    ALL_FIELDS = ["NAME", "city", "room type", "price", "minimum nights",
                  "review rate number", "house_rules", "maximum occupancy"]
    warnings = []
    dropped = []

    # ---- tolerate a payload dict passed as the first argument ----
    if isinstance(city, dict):
        payload = city
        city = payload.get("city") or payload.get("destination") or ""
        if max_price is None:
            max_price = payload.get("max_price") or payload.get("budget")
        if min_review is None:
            min_review = payload.get("min_review") or payload.get("min_rating")
        if not room_type:
            room_type = payload.get("room_type") or ""
        if min_occupancy is None:
            min_occupancy = payload.get("min_occupancy") or payload.get("people") or payload.get("people_number")
        if payload.get("limit"):
            limit = payload.get("limit")
        if not visiting_city:
            visiting_city = payload.get("visiting_city") or payload.get("day_city") or payload.get("current_city") or ""

    # ---- normalize limit (default 50 on anything unusable) ----
    lim = 50
    if isinstance(limit, bool):
        lim = 50
    elif isinstance(limit, (int, float)) and limit >= 1:
        lim = int(limit)
    elif isinstance(limit, str) and limit.strip().isdigit() and int(limit.strip()) >= 1:
        lim = int(limit.strip())

    # ---- normalize max_price (tolerate "200", "$1,200" style strings) ----
    price_val = None
    if isinstance(max_price, bool):
        price_val = None
    elif isinstance(max_price, (int, float)):
        price_val = float(max_price)
    elif max_price is not None:
        s = str(max_price).strip().replace(",", "").replace("$", "")
        if s and s.replace(".", "", 1).isdigit():
            price_val = float(s)

    # ---- normalize min_review ----
    review_val = None
    if isinstance(min_review, bool):
        review_val = None
    elif isinstance(min_review, (int, float)):
        review_val = float(min_review)
    elif min_review is not None:
        s = str(min_review).strip().replace(",", "")
        if s and s.replace(".", "", 1).isdigit():
            review_val = float(s)

    # ---- normalize min_occupancy ----
    occ_val = None
    if isinstance(min_occupancy, bool):
        occ_val = None
    elif isinstance(min_occupancy, (int, float)) and min_occupancy >= 1:
        occ_val = int(min_occupancy)
    elif isinstance(min_occupancy, str) and min_occupancy.strip().isdigit() and int(min_occupancy.strip()) >= 1:
        occ_val = int(min_occupancy.strip())

    rt = str(room_type).strip() if room_type else ""

    # ---- resolve the anchored city (the visiting city wins on conflict) ----
    raw = "" if city is None else str(city).strip()
    vc = "" if visiting_city is None else str(visiting_city).strip()
    if not raw and vc:
        raw = vc
    if raw and vc and raw.lower() != vc.lower():
        warnings.append("Requested city '{}' differs from the current day's visiting city '{}'; anchoring lodging to the visiting city.".format(raw, vc))
        raw = vc

    # ---- REQUIRE a city: no lodging search without an itinerary city ----
    if not raw:
        msg = "No city provided: accommodations must be anchored to the current day's visiting city; the planner should select a city before booking lodging."
        return {"warning": msg, "dropped": [], "count": 0,
                "results": [{"NAME": "", "city": "", "Accommodation count": 0, "note": msg}]}

    # ---- fetch candidates (substring probes widen lookup; equality enforced below) ----
    target = raw.lower()
    probes = []
    for p in (raw, raw.title(), raw.lower(), raw.upper()):
        if p and p not in probes:
            probes.append(p)
    seen = set()
    cand = []
    for p in probes:
        for r in lookup_entities("Accommodation", {"city": p}, limit=1000):
            key = (str(r.get("NAME", "")), str(r.get("city", "")))
            if key not in seen:
                seen.add(key)
                cand.append(r)

    # ---- verify each row's city attribute equals the anchored city; drop/flag mismatches ----
    rows = []
    for r in cand:
        if str(r.get("city", "")).strip().lower() == target:
            rows.append(r)
        else:
            dropped.append(r)

    # ---- located_in edge anchor: exact visiting city when set, else the resolved city ----
    anchor = vc if vc else raw
    verified = filter_relation_connected(rows, "located_in", "City", {"name": anchor})
    if vc or verified:
        kept = set((str(v.get("NAME", "")), str(v.get("city", ""))) for v in verified)
        still = []
        for r in rows:
            if (str(r.get("NAME", "")), str(r.get("city", ""))) in kept:
                still.append(r)
            else:
                dropped.append(r)
        rows = still

    # ---- Accommodation count for that city (before optional preference filters) ----
    city_count = len(rows)

    # ---- strict filter cascade: occupancy -> price -> review rating -> room type ----
    if occ_val is not None:
        rows = filter_numeric(rows, "maximum occupancy", ">=", occ_val)
    if price_val is not None:
        rows = filter_numeric(rows, "price", "<=", price_val)
    if review_val is not None:
        rows = filter_numeric(rows, "review rate number", ">=", review_val)
    if rt:
        rows = filter_categorical(rows, "room type", [rt])

    # ---- sort by price ascending (unparseable prices sink to the end) ----
    rows = sorted(rows, key=lambda r: r.get("price") if isinstance(r.get("price"), (int, float)) else 1e9)

    results = project_properties(rows[:lim], ALL_FIELDS)

    # ---- guarantee the resolved city and NAME on every returned row ----
    for r in results:
        r["NAME"] = str(r.get("NAME", ""))
        r["city"] = raw

    # ---- never a bare []: explanatory row with the Accommodation count for that city ----
    if not results:
        note = "0 accommodations in '{}' matched the filters; {} Accommodation(s) exist for that city (located_in '{}'). The planner must not invent a listing.".format(raw, city_count, anchor)
        warnings.append(note)
        results = [{"NAME": "", "city": raw, "Accommodation count": city_count, "note": note}]

    return {"warning": " ".join(warnings), "dropped": project_properties(dropped, ALL_FIELDS), "count": city_count, "results": results}

def find_attractions(city="", exclude_names=None):
    """Find Attraction candidates in a city while blocking repeats of already-visited attractions.

    Args:
        city (str): Name of the city whose attractions should be retrieved
                    (substring match, case handled by the lookup operator).
        exclude_names (list[str] | None): Full running list of already-visited
                    attraction names accumulated across all calls within the
                    trip; any row whose Name matches an entry (case-insensitive)
                    is blocked so the planner cannot re-pick it. A single string
                    is accepted and treated as a one-element list.

    Returns:
        dict: JSON-serializable object with:
            - "results": list of unique attraction rows (deduplicated within the
              call on Name + resolved City), each with keys Name, City (resolved
              to the requested city when missing, so the planner can verify the
              attraction matches the day's current city), Address, Phone,
              Latitude, Longitude (plus __id__/__type__ kept by the projection);
            - "duplicate_blocked": sorted list of names suppressed this call
              because they were already served (present in exclude_names) or
              appeared as duplicates within the result set (empty if none).

    Example:
        find_attractions("Houston", ["Space Center Houston"])
    """
    served = set()
    if isinstance(exclude_names, str):
        exclude_names = [exclude_names]
    if exclude_names:
        for n in exclude_names:
            key = str(n).strip().lower()
            if key:
                served.add(key)

    rows = lookup_entities("Attraction", {"City": city} if city else None, limit=50)
    rows = project_properties(rows, ["Name", "City", "Address", "Phone", "Latitude", "Longitude"])

    results = []
    blocked = set()
    seen = set()
    for r in rows:
        name = str(r.get("Name", "")).strip()
        name_key = name.lower()
        resolved_city = str(r.get("City") or city or "").strip()
        row_key = name_key + "|" + resolved_city.lower()
        if name_key in served:
            blocked.add(name)
            continue
        if row_key in seen:
            blocked.add(name)
            continue
        seen.add(row_key)
        row = dict(r)
        row["Name"] = name
        row["City"] = resolved_city
        results.append(row)

    return {"results": results, "duplicate_blocked": sorted(blocked)}

def cities_in_state(state="Texas", lookup_limit=300):
    """Return the sorted list of City names located inside a given U.S. state/region.

    Args:
        state: State given as full name or 2-letter abbreviation, case-insensitive
            (e.g. "Texas", "tx", "MN", "minnesota").
        lookup_limit: Maximum rows requested per entity lookup.

    Returns:
        JSON-serializable sorted list of unique city-name strings within the
        state; when the primary City lookup yields 0 rows, candidates are
        derived from GroundTransportLink origin/destination values and Flight
        OriginCityName/DestCityName values so a valid US state never yields []
        silently.

    Example:
        cities_in_state("TX")  # -> ["Dallas", "Houston", "San Antonio", ...]
    """
    abbr = {
        "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
        "California": "CA", "Colorado": "CO", "Connecticut": "CT",
        "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
        "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
        "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
        "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
        "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
        "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
        "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
        "New York": "NY", "North Carolina": "NC", "North Dakota": "ND",
        "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
        "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
        "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
        "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
        "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    }
    code_to_name = {}
    for full, short in abbr.items():
        code_to_name[short] = full

    # --- Normalize state input (case-insensitive; full name or 2-letter code) ---
    s = str(state).strip()
    up = s.upper()
    if up in code_to_name:                 # e.g. "tx" -> code TX, full name Texas
        code = up
        key = code_to_name[up]
    else:
        key = s.title()                    # e.g. "minnesota" -> "Minnesota"
        code = abbr.get(key, up)

    # Substring patterns tied to the state; the bare 2-letter code is
    # deliberately excluded as a lookup pattern (too noisy, e.g. "OR" inside
    # "Orlando") and is instead used only for strict suffix parsing below.
    patterns = [", " + code, "," + code, key, key.upper()]

    names = set()

    # 1) Primary: City entities whose stored name carries the state token.
    for pat in patterns:
        rows = lookup_entities("City", {"name": pat}, limit=lookup_limit)
        if rows:
            for r in rows:
                v = r.get("name")
                if isinstance(v, str) and v.strip():
                    names.add(v.split(",")[0].strip())
            break

    # 2) Fallback A: GroundTransportLink endpoints of the form "City, ST".
    if not names:
        for field in ["origin", "destination"]:
            rows = []
            for pat in patterns:
                rows = lookup_entities("GroundTransportLink", {field: pat},
                                       limit=lookup_limit)
                if rows:
                    break
            for r in project_properties(rows, [field]):
                v = r.get(field)
                if isinstance(v, str) and v.strip():
                    names.add(v.split(",")[0].strip())

    # 3) Fallback B: Flight endpoints mentioning the state.
    if not names:
        for field in ["OriginCityName", "DestCityName"]:
            rows = []
            for pat in patterns:
                rows = lookup_entities("Flight", {field: pat}, limit=lookup_limit)
                if rows:
                    break
            for r in project_properties(rows, [field]):
                v = r.get(field)
                if isinstance(v, str) and v.strip():
                    names.add(v.split(",")[0].strip())

    # 4) Last resort: broad scans parsing "<City>, <ST>" suffixes manually.
    if not names:
        scans = [
            ("GroundTransportLink", ["origin", "destination"]),
            ("Flight", ["OriginCityName", "DestCityName"]),
        ]
        for etype, fields in scans:
            for r in lookup_entities(etype, limit=lookup_limit):
                for field in fields:
                    v = r.get(field)
                    if not isinstance(v, str):
                        continue
                    parts = v.rsplit(",", 1)
                    if len(parts) == 2:
                        tail = parts[1].strip().upper()
                        if tail == code or tail == key.upper():
                            head = parts[0].strip()
                            if head:
                                names.add(head)
            if names:
                break

    # 5) Prefer names that exist as actual City entities, but never drop a
    #    valid state down to an empty result during verification.
    if names:
        verified = [n for n in sorted(names)
                    if lookup_entities("City", {"name": n}, limit=5)]
        if verified:
            return verified
    return sorted(n for n in names if n)

def nights_between(start_date="2022-03-23", end_date="2022-03-29"):
    """Compute the number of nights between two calendar date strings (e.g., 'Mar 23-29, 2022' -> 6 nights).

    Args:
        start_date: check-in date string; accepted forms include '2022-03-23',
            '03/23/2022', 'March 23, 2022', or '23rd of March 2022'.
        end_date: check-out date string in any of the same accepted forms.

    Returns:
        int: number of nights spanned (whole days from start_date to end_date).

    Example:
        nights_between("2022-03-23", "2022-03-29")  # -> 6
    """
    months = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
              "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
    cum = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    ordinals = []
    for text in (start_date, end_date):
        t = str(text).strip().lower().replace(",", " ").replace("/", " ").replace("-", " ")
        year, month, day = 2022, 0, 0
        nums = []
        for tok in t.split():
            if tok == "of":
                continue
            digits = ""
            for ch in tok:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                val = int(digits)
                if len(digits) >= 4:
                    year = val
                else:
                    nums.append(val)
            elif len(tok) >= 3 and tok.isalpha():
                code = months.get(tok[:3])
                if code is not None:
                    month = code
        if month == 0:
            if len(nums) >= 2:
                if nums[0] > 12:
                    day, month = nums[0], nums[1]
                else:
                    month, day = nums[0], nums[1]
            elif nums:
                day, month = nums[0], 1
        elif day == 0:
            day = nums[0] if nums else 1
        month = max(1, min(12, month))
        day = max(1, min(31, day))
        leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        doy = cum[month - 1] + day + (1 if month > 2 and leap else 0)
        y0 = year - 1
        ordinals.append(y0 * 365 + y0 // 4 - y0 // 100 + y0 // 400 + doy)
    return ordinals[1] - ordinals[0]

def compute_trip_cost(flight_prices=[], meal_costs=[], hotel_stays=[], leg_km=[], people=1, ground_mode='self-driving'):
    """Itemize and total the cost of a candidate travel plan per the official pricing formula.

    Args:
        flight_prices: list of flight Price values (one entry per flight leg, per person).
        meal_costs: list of restaurant Average Cost values (one entry per person per meal).
        hotel_stays: list of (price_per_night, nights, max_occupancy) tuples, one per hotel stay.
        leg_km: list of ground-transport leg distances in kilometers.
        people: number of travelers (default 1).
        ground_mode: 'self-driving' or 'taxi' (default 'self-driving').

    Returns:
        dict with per-category costs, per-stay / per-leg detail, and the grand total.
        Formula: flights = Price * people; meals = Average Cost * people;
        hotels = price * nights * ceil(people / max occupancy);
        self-driving = int(km * 0.05) * ceil(people / 5) per leg;
        taxi = int(km) * ceil(people / 4) per leg.

    Example:
        compute_trip_cost([120.0], [30.0, 25.0], [(90.0, 2, 2)], [150.0], people=2, ground_mode='taxi')
    """
    p = max(1, int(people))

    # Flights: Price x people
    flight_cost = round(sum(float(x) for x in flight_prices) * p, 2)

    # Meals: Average Cost x people
    meal_cost = round(sum(float(x) for x in meal_costs) * p, 2)

    # Hotels: price x ceil(people / max occupancy) per night
    hotel_cost = 0.0
    hotel_detail = []
    for stay in hotel_stays:
        price = float(stay[0])
        nights = max(1, int(stay[1]))
        max_occ = max(1, int(stay[2]))
        rooms = ceil(p / max_occ)
        cost = round(price * nights * rooms, 2)
        hotel_detail.append({"per_night": price, "nights": nights, "rooms": rooms, "cost": cost})
        hotel_cost += cost
    hotel_cost = round(hotel_cost, 2)

    # Ground transport per leg
    if ground_mode == 'self-driving':
        vehicles = ceil(p / 5)          # 5 people per car
    else:
        vehicles = ceil(p / 4)          # 4 people per taxi
    ground_cost = 0
    ground_detail = []
    for km in leg_km:
        if ground_mode == 'self-driving':
            per_vehicle = int(float(km) * 0.05)
        else:
            per_vehicle = int(float(km))
        cost = per_vehicle * vehicles
        ground_detail.append({"km": float(km), "rate_per_vehicle": per_vehicle, "vehicles": vehicles, "cost": cost})
        ground_cost += cost

    total = round(flight_cost + meal_cost + hotel_cost + ground_cost, 2)

    return {
        "people": p,
        "ground_mode": ground_mode,
        "flights_cost": flight_cost,
        "meals_cost": meal_cost,
        "hotels_cost": hotel_cost,
        "hotel_detail": hotel_detail,
        "ground_cost": ground_cost,
        "ground_detail": ground_detail,
        "total_cost": total,
    }

def check_budget(total_cost, budget=2300.0, tolerance=0.01):
    """Validate a plan's total cost against a stated budget and report headroom to guide cheaper re-selection.

    Args:
        total_cost: Numeric total spend, or a list of numeric component costs to be summed.
        budget: Maximum allowed spend as stated by the user (e.g., 7700, 2300, 7100).
        tolerance: Small slack absorbed for floating-point rounding (default 0.01).

    Returns:
        dict: {
            "total_cost": float,          # summed cost of the plan
            "budget": float,              # budget ceiling used
            "is_within_budget": bool,     # True if cost fits within budget
            "remaining": float,           # budget - cost (negative when over budget)
            "overspend": float,           # amount over budget (0 when within)
            "max_candidate_price": float, # price ceiling when re-selecting cheaper candidates
            "guidance": str               # next-step hint for re-selection
        }

    Example:
        check_budget([980.0, 615.5, 555.0], budget=2300)
    """
    if isinstance(total_cost, (list, tuple)):
        cost = float(sum(float(c) for c in total_cost))
    else:
        cost = float(total_cost)
    cap = float(budget)
    is_within = bool(cost <= cap + tolerance)
    remaining = round(cap - cost, 2)
    overspend = round(max(0.0, cost - cap), 2)
    if is_within:
        guidance = (
            "Within budget; up to {0} may still be spent or used as the ceiling "
            "when swapping in a more expensive candidate.".format(remaining)
        )
    else:
        guidance = (
            "Over budget by {0}; re-select cheaper candidates whose combined "
            "price drops total_cost by at least {0} (target ceiling {1}).".format(
                overspend, round(cap, 2)
            )
        )
    return {
        "total_cost": round(cost, 2),
        "budget": round(cap, 2),
        "is_within_budget": is_within,
        "remaining": remaining,
        "overspend": overspend,
        "max_candidate_price": remaining,
        "guidance": guidance,
    }