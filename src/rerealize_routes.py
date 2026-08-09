"""Re-realize fixed trip sequences: recompute optimal charging from scratch.

Recorded plans (GIRO seeds, MATCHING covers) fail injection for reasons that
are artifacts of the recording, not of the trip sequences themselves: Hastus
minute-rounding implies >nominal charge rates, literal sequences lack model
arcs for long unrecharged gaps, and covers built under one physics regime
plan charging another regime cannot deliver. Patching recorded windows hits
a cascade ceiling (extending one window breaks downstream trip timing).

This utility keeps only the trip SEQUENCE of each input route and re-derives
everything else by optimization under the target pool physics and tariff:
per gap it chooses direct travel or one station interposition (arcs must
exist in the restricted model graph), and per station visit it chooses
hour-aligned charging segments (delayed start native), amounts bounded by
window x charger power, SOC kept within [reserve, G]. Each route solves as
an independent small MILP (scipy/HiGHS); the emitted plan is replayed
through run_exact_pool_mip.validate_injected_route before writing, so every
output route is injection-valid by construction — or reported infeasible
with the binding reason, which is then a statement about the sequence under
that physics, not about the recording.

This is also Tier-2 of the savings decomposition: fixed (GIRO) routes,
charging re-optimized under a tariff.

    python rerealize_routes.py --routes results/giro_seeds/X_giro_seed.json \
        --instance duty_unions/X.csv --g-kwh 240 --charge-kw 220 \
        --reserve 0.2 --prices hourly_prices_single_peak_12.csv
    python rerealize_routes.py --routes cover.json --physics-from pool.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from audit_giro_known_columns import DEPOT, HORIZON_MIN, STATIONS, build_problem
from config import CHARGE_RATE_KW, CHARGE_START_COST, CHARGING_STATIONS, \
    TRAVEL_COST_FACTOR, charge_cost_premium
from run_exact_pool_mip import validate_injected_route
from utils_v2 import base_station_name, load_station_hourly_prices

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATION_TIEBREAK = 1e-3  # prefer direct arcs over zero-charge station waits
SOC_EPS_KWH = 1e-3  # solve strictly inside [reserve, G] so solver tolerance
                    # and kWh rounding never trip the validator's 1e-6 checks


def _arc_map(problem):
    return {(u, v): (travel, dh)
            for u, arcs in problem.adjacency.items()
            for v, travel, dh, _kind in arcs}


def _segments(window_start, window_end, prices, charge_kw):
    """Hour-aligned charging segments inside [window_start, window_end]."""
    segs = []
    t = float(window_start)
    max_hour = max(prices)
    while t < window_end - 1e-9:
        nxt = min((int(t // 60) + 1) * 60.0, float(window_end))
        hour = min(int(t // 60), max_hour)
        segs.append({"start": t, "end": nxt, "price": float(prices[hour]),
                     "cap": (nxt - t) * charge_kw / 60.0})
        t = nxt
    return segs


def rerealize_route(trip_seq, problem, arc, prices, g_kwh, charge_kw,
                    reserve_kwh, grace_min=1.0):
    """Optimal charging plan for one fixed trip sequence.

    Returns (record, planned_cost, None) or (None, None, reason).
    """
    n = len(trip_seq)
    if n == 0:
        return None, None, "no trips in route"
    reserve_kwh = reserve_kwh + SOC_EPS_KWH
    g_kwh = g_kwh - SOC_EPS_KWH
    for t in trip_seq:
        if (DEPOT, trip_seq[0]) not in arc and t == trip_seq[0]:
            return None, None, f"no depot arc to first trip {t}"
    if (trip_seq[-1], DEPOT) not in arc:
        return None, None, f"no arc from last trip {trip_seq[-1]} to depot"

    # Gap options: gaps[i] is between trip_seq[i] and trip_seq[i+1].
    gaps = []
    for i in range(n - 1):
        a, b = trip_seq[i], trip_seq[i + 1]
        options = []
        if (a, b) in arc:
            options.append({"kind": "direct", "dh": arc[(a, b)][1]})
        for s in STATIONS:
            if (a, s) not in arc or (s, b) not in arc:
                continue
            arrive = problem.end_min[a] + arc[(a, s)][0]
            depart = problem.start_min[b] - arc[(s, b)][0]
            if arrive > depart + grace_min:
                continue
            options.append({
                "kind": "station", "station": s,
                "dh": arc[(a, s)][1] + arc[(s, b)][1],
                "arrive": arrive, "depart": depart,
                "segments": _segments(arrive, depart,
                                      prices[base_station_name(s)], charge_kw),
            })
        if not options:
            return None, None, (f"gap {a}->{b}: no direct arc and no "
                                f"feasible station interposition")
        gaps.append(options)

    # Variables: y (one binary per gap option), then per station segment a
    # continuous x (kWh) and a binary z (stop opened, pays start cost), then
    # S_1..S_n (SOC after each trip's energy draw).
    idx = 0
    y_idx, x_idx, z_idx = {}, {}, {}
    for i, options in enumerate(gaps):
        for o, opt in enumerate(options):
            y_idx[i, o] = idx
            idx += 1
            if opt["kind"] == "station":
                for j in range(len(opt["segments"])):
                    x_idx[i, o, j] = idx
                    idx += 1
                    z_idx[i, o, j] = idx
                    idx += 1
    s_idx = {i: idx + i for i in range(n)}
    nvar = idx + n

    cost = np.zeros(nvar)
    integrality = np.zeros(nvar)
    lb = np.zeros(nvar)
    ub = np.full(nvar, np.inf)
    for (i, o), k in y_idx.items():
        opt = gaps[i][o]
        integrality[k] = 1
        ub[k] = 1
        cost[k] = TRAVEL_COST_FACTOR * opt["dh"]
        if opt["kind"] == "station":
            cost[k] += STATION_TIEBREAK
    for (i, o, j), k in x_idx.items():
        seg = gaps[i][o]["segments"][j]
        ub[k] = seg["cap"]
        cost[k] = seg["price"] * charge_cost_premium
    for k in z_idx.values():
        integrality[k] = 1
        ub[k] = 1
        cost[k] = CHARGE_START_COST
    for i in range(n):
        lb[s_idx[i]] = reserve_kwh
        ub[s_idx[i]] = g_kwh

    rows, cols, vals, con_lb, con_ub = [], [], [], [], []

    def add_row(entries, lo, hi):
        r = len(con_lb)
        for k, v in entries:
            rows.append(r)
            cols.append(k)
            vals.append(v)
        con_lb.append(lo)
        con_ub.append(hi)

    # Exactly one option per gap.
    for i, options in enumerate(gaps):
        add_row([(y_idx[i, o], 1.0) for o in range(len(options))], 1.0, 1.0)

    # x active only under its option; z opens a stop for any positive x.
    for (i, o, j), k in x_idx.items():
        cap = gaps[i][o]["segments"][j]["cap"]
        add_row([(k, 1.0), (y_idx[i, o], -cap)], -np.inf, 0.0)
        add_row([(k, 1.0), (z_idx[i, o, j], -cap)], -np.inf, 0.0)

    # SOC recursion. S_0 = G - dh(depot->t1) - E_1 is a constant.
    s0 = g_kwh - arc[(DEPOT, trip_seq[0])][1] - problem.trip_energy[trip_seq[0]]
    if s0 < reserve_kwh - 1e-6:
        return None, None, (f"first trip {trip_seq[0]} alone drives SOC to "
                            f"{s0:.1f} < reserve {reserve_kwh:.1f}")
    add_row([(s_idx[0], 1.0)], s0, s0)
    for i in range(n - 1):
        entries = [(s_idx[i + 1], 1.0), (s_idx[i], -1.0)]
        for o, opt in enumerate(gaps[i]):
            entries.append((y_idx[i, o], opt["dh"]))
        for (gi, o, j), k in x_idx.items():
            if gi == i:
                entries.append((k, -1.0))
        e_next = problem.trip_energy[trip_seq[i + 1]]
        add_row(entries, -e_next, -e_next)

    # Station reserve on arrival and battery cap at end of charging (big-M).
    for i, options in enumerate(gaps):
        for o, opt in enumerate(options):
            if opt["kind"] != "station":
                continue
            dh1 = arc[(trip_seq[i], opt["station"])][1]
            # S_i - dh1 >= reserve when y=1:  S_i - G*y >= reserve + dh1 - G
            add_row([(s_idx[i], 1.0), (y_idx[i, o], -g_kwh)],
                    reserve_kwh + dh1 - g_kwh, np.inf)
            # S_i - dh1 + X <= G when y=1:  S_i + X + G*y <= 2G + dh1
            entries = [(s_idx[i], 1.0), (y_idx[i, o], g_kwh)]
            entries += [(x_idx[i, o, j], 1.0)
                        for j in range(len(opt["segments"]))]
            add_row(entries, -np.inf, 2.0 * g_kwh + dh1)

    # Final depot leg.
    add_row([(s_idx[n - 1], 1.0)],
            reserve_kwh + arc[(trip_seq[-1], DEPOT)][1], np.inf)

    a_matrix = csr_matrix((vals, (rows, cols)), shape=(len(con_lb), nvar))
    res = milp(c=cost, integrality=integrality, bounds=Bounds(lb, ub),
               constraints=LinearConstraint(a_matrix, con_lb, con_ub))
    if not res.success:
        return None, None, "MILP infeasible under physics"

    # Emit the runner-format record: one stop per used segment (hour-pure
    # pricing), zero-kWh stop for wait-only station visits so the validator's
    # stop consumption stays aligned with route_nodes.
    x = res.x
    route_nodes = [DEPOT]
    stops = {"stations": [], "cst": [], "cet": [], "kwh": []}
    deadhead = arc[(DEPOT, trip_seq[0])][1]
    for i, t in enumerate(trip_seq):
        route_nodes.append(t)
        if i == n - 1:
            break
        o = next(o for o in range(len(gaps[i])) if x[y_idx[i, o]] > 0.5)
        opt = gaps[i][o]
        deadhead += opt["dh"]
        if opt["kind"] == "direct":
            continue
        used = [(j, x[x_idx[i, o, j]]) for j in range(len(opt["segments"]))
                if x[x_idx[i, o, j]] > 1e-6]
        if not used:
            used = [(0, 0.0)]  # pure wait: record one zero-kWh stop
        for j, kwh in used:
            seg = opt["segments"][j]
            cst = round(seg["start"], 2)
            cet = round(min(seg["end"], cst + kwh * 60.0 / charge_kw + 0.01), 2)
            route_nodes.append(opt["station"])
            stops["stations"].append(opt["station"])
            stops["cst"].append(cst)
            stops["cet"].append(max(cet, cst))
            stops["kwh"].append(round(kwh, 6))
    route_nodes.append(DEPOT)
    deadhead += arc[(trip_seq[-1], DEPOT)][1]
    record = {
        "route": route_nodes,
        "charging_stops": stops,
        "charging_activities": len(stops["stations"]),
        "type": "truck",
        "deadhead_kwh": round(deadhead, 6),
        "_rc": 0.0,
        "desc": f"[rerealized] {n} trips, {len(stops['stations'])} stops, "
                f"{sum(stops['kwh']):.1f} kWh",
    }
    return record, float(res.fun), None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, required=True,
                        help="Runner-format routes JSON (seed or cover).")
    parser.add_argument("--instance", type=str, default=None,
                        help="Instance CSV relative to data/.")
    parser.add_argument("--physics-from", type=Path, default=None,
                        help="Pool status JSON supplying g_kwh/charge_kw/"
                             "min_soc_frac/csv (CLI flags override).")
    parser.add_argument("--g-kwh", type=float, default=None)
    parser.add_argument("--charge-kw", type=float, default=None)
    parser.add_argument("--reserve", type=float, default=None,
                        help="Reserve as a fraction of G.")
    parser.add_argument("--prices", type=str, default="hourly_prices_flat.csv")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    g_kwh, charge_kw, reserve_frac, instance = 300.0, float(CHARGE_RATE_KW), 0.0, None
    if args.physics_from:
        with open(args.physics_from) as fh:
            status = json.load(fh)
        g_kwh = float(status.get("g_kwh", g_kwh))
        charge_kw = float(status.get("charge_kw", charge_kw))
        reserve_frac = float(status.get("min_soc_frac", reserve_frac))
        instance = status.get("csv")
    if args.g_kwh is not None:
        g_kwh = args.g_kwh
    if args.charge_kw is not None:
        charge_kw = args.charge_kw
    if args.reserve is not None:
        reserve_frac = args.reserve
    if args.instance:
        instance = args.instance
    if not instance:
        raise SystemExit("need --instance or --physics-from with a csv field")
    reserve_kwh = reserve_frac * g_kwh

    problem = build_problem(DATA_DIR, instance,
                            max_station_to_trip_wait_min=HORIZON_MIN)
    arc = _arc_map(problem)
    prices = load_station_hourly_prices(DATA_DIR / Path(args.prices).name,
                                        CHARGING_STATIONS)

    with open(args.routes) as fh:
        payload = json.load(fh)
    print(f"[RRZ] {args.routes.name}: {len(payload.get('routes', []))} routes; "
          f"G={g_kwh:.0f} kWh, {charge_kw:.0f} kW, reserve {reserve_frac:.0%}, "
          f"prices {Path(args.prices).name}")

    out_routes, infeasible, planned = [], [], 0.0
    for r_i, route in enumerate(payload.get("routes", [])):
        trip_seq = [nd for nd in route.get("route", []) if isinstance(nd, int)]
        try:
            record, cost_val, reason = rerealize_route(
                trip_seq, problem, arc, prices, g_kwh, charge_kw, reserve_kwh)
        except Exception as exc:  # isolate: one bad route must not kill the file
            import traceback
            traceback.print_exc()
            record, cost_val = None, None
            reason = f"exception: {type(exc).__name__}: {exc}"
        if reason is not None:
            if reason == "MILP infeasible under physics":
                # Probe the frontier: minimal battery or charger relaxation
                # that admits ANY charging plan for this sequence.
                for dg in (10, 20, 40, 60, 120):
                    ok, _, why = rerealize_route(
                        trip_seq, problem, arc, prices, g_kwh + dg,
                        charge_kw, reserve_kwh)
                    if ok is not None:
                        reason += f"; feasible at G+{dg} kWh"
                        break
                for dkw in (10, 30, 50, 80):
                    ok, _, why = rerealize_route(
                        trip_seq, problem, arc, prices, g_kwh,
                        charge_kw + dkw, reserve_kwh)
                    if ok is not None:
                        reason += f"; feasible at rate+{dkw} kW"
                        break
            infeasible.append({"index": r_i, "trips": trip_seq,
                               "reason": reason})
            print(f"[RRZ]   route {r_i} ({len(trip_seq)} trips) INFEASIBLE: "
                  f"{reason}")
            continue
        verdict = validate_injected_route(
            problem, {"route_nodes": record["route"],
                      "charging_stops": record["charging_stops"]},
            g_kwh, charge_kw, reserve_kwh, HORIZON_MIN)
        if verdict is not None:
            raise SystemExit(f"internal error: rerealized route {r_i} fails "
                             f"validation: {verdict}")
        out_routes.append(record)
        planned += cost_val

    suffix = f"_rrz_g{int(g_kwh)}_{Path(args.prices).stem.replace('hourly_prices_', '')}"
    out = args.out or args.routes.with_name(args.routes.stem + suffix + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"routes": out_routes, "source": "rerealized",
                   "from": str(args.routes), "instance_csv": instance,
                   "physics": {"g_kwh": g_kwh, "charge_kw": charge_kw,
                               "reserve_frac": reserve_frac},
                   "prices_csv": Path(args.prices).name,
                   "planned_charging_cost": round(planned, 4),
                   "infeasible": infeasible}, fh, indent=1)
    print(f"[RRZ] wrote {out}: {len(out_routes)} valid routes "
          f"(validator-replayed), {len(infeasible)} infeasible, planned "
          f"charging+stop cost {planned:.2f}")
    return 0 if not infeasible else 3


if __name__ == "__main__":
    raise SystemExit(main())
