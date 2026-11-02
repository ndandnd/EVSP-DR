"""Price a single continuous charging interval over all hourly tariff changes."""
import itertools
import json
import math


def solve(j, arm, fee, seconds, threads, seed):
    gp = j.gp
    routes = j.REPLAY[arm]['routes']
    perm = max(itertools.permutations(range(len(routes))),
               key=lambda p: sum(len(set(r['trips']) & set(j.ORIG[k]['trips'])) for r, k in zip(routes, p)))
    duties = [j.ORIG[k]['duty_id'] for k in perm]
    m = gp.Model('continuous_tariff_hours_' + arm)
    m.Params.Threads = threads
    m.Params.TimeLimit = seconds
    m.Params.Seed = seed
    m.Params.NumericFocus = 2
    m.Params.FeasibilityTol = 1e-8
    m.Params.IntFeasTol = 1e-9
    m.Params.MIPGap = .0001
    m.Params.LogFile = str(j.P / (arm + '.gurobi.log'))
    def battery():
        return m.addVar(lb=j.reserve, ub=j.C)
    def add_charge_segment(before, site, lo, hi):
        use = m.addVar(vtype=gp.GRB.BINARY)
        start = m.addVar(lb=lo, ub=hi)
        dt = m.addVar(ub=hi-lo)
        q = m.addVar(ub=j.C)
        pre, peak, after = battery(), battery(), battery()
        m.addConstr(dt <= (hi-lo)*use)
        m.addConstr(start+dt <= hi)
        m.addConstr(pre == before - j.idle*(start-lo))
        m.addConstr(peak == pre + q)
        m.addConstr(after == peak - j.idle*(hi-start-dt))
        xx, yy = j.f.potential(j.pr, site)
        h1 = m.addVar(ub=yy[-1])
        h2 = m.addVar(ub=yy[-1])
        m.addGenConstrPWL(pre, h1, xx, yy)
        m.addGenConstrPWL(peak, h2, xx, yy)
        m.addConstr(dt == h2-h1)
        return after, (use, start, dt, pre, peak, q)
    visits, ends, obj = [], [], gp.LinExpr()
    for ri, route in enumerate(routes):
        soc = battery()
        m.addConstr(soc == j.C - route['actions'][0]['deadhead_kwh'])
        for a in route['actions'][1:]:
            aftertrip = battery()
            m.addConstr(aftertrip == soc - j.TR[a['from_trip']]['energy'])
            if a['kind'] == 'direct':
                soc = battery()
                m.addConstr(soc == aftertrip-a['deadhead_kwh']-a['idle_kwh'])
                continue
            arrival, latest = a['arrival_min'], a['latest_departure_min']
            site = a['station']
            soc = battery()
            m.addConstr(soc == aftertrip - a['inbound_kwh'])
            use = m.addVar(vtype=gp.GRB.BINARY)
            start = m.addVar(lb=arrival, ub=latest)
            end = m.addVar(lb=arrival, ub=latest)
            pieces = []
            for hour in range(math.floor(arrival/60), math.ceil(latest/60)):
                lo, hi = max(arrival, hour*60), min(latest, (hour+1)*60)
                soc, piece = add_charge_segment(soc, site, lo, hi)
                z, s, dt, pre, peak, q = piece
                m.addConstr(z <= use)
                m.addConstr(s >= start - (latest-lo)*(1-z))
                m.addConstr(s+dt <= end + (hi-arrival)*(1-z))
                obj += j.f.prices[hour] * q
                pieces.append(piece)
            # Each hourly piece lies inside [start,end]. Their disjoint
            # lengths sum to the full interval, which forces contiguous power.
            m.addConstr(end-start == gp.quicksum(p[2] for p in pieces))
            m.addConstr(end-start >= j.pr.minimum_recharge_duration_min*use)
            obj += fee*use
            departure = battery()
            m.addConstr(departure == soc-a['outbound_kwh'])
            soc = departure
            visits.append((ri, a, use, start, end, pieces))
        m.addConstr(soc >= j.f.TARGET[duties[ri]])
        ends.append(soc)
    for x,y in itertools.combinations(visits,2):
        rx, ax, zx, sx, ex, _ = x
        ry, ay, zy, sy, ey, _ = y
        if rx == ry or ax['station'] != ay['station'] or ax['station'] == 'PARX':
            continue
        mx = ax['latest_departure_min']-ay['arrival_min']
        my = ay['latest_departure_min']-ax['arrival_min']
        if min(mx,my) <= 0:
            continue
        order = m.addVar(vtype=gp.GRB.BINARY)
        m.addConstr(ex <= sy + mx*(1-order+2-zx-zy))
        m.addConstr(ey <= sx + my*(order+2-zx-zy))
    m.setObjective(obj, gp.GRB.MINIMIZE)
    m.optimize()
    result = dict(arm=arm, fee=fee, status=m.Status, solutions=m.SolCount, objective=m.ObjVal if m.SolCount else None,
                  bound=m.ObjBound if abs(m.ObjBound)<1e50 else None, wall_seconds=m.Runtime, original_terminal_match=duties,
                  routes=[], model_scope='fixed trip orders and mandatory station path; one optional contiguous charge per visit; exact hourly tariff')
    if m.SolCount:
        for ri,r in enumerate(routes):
            charges = []
            for index,a,y,start,end,pieces in visits:
                if index != ri or y.X <= .5:
                    continue
                active = [p for p in pieces if p[2].X > 1e-8]
                assert active
                before, after = active[0][3].X, active[-1][4].X
                cost, points = j.f.cost_profile(j.pr, a['station'], start.X, before, after)
                charges.append(dict(station=a['station'], start=start.X, end=end.X, kwh=sum(p[5].X for p in active),
                                    points=points, cost=cost))
            result['routes'].append(dict(source_index=ri, trips=r['trips'], charges=charges, starts=len(charges),
                                         electricity_cost=sum(c['cost'] for c in charges), terminal_kwh=ends[ri].X,
                                         target_kwh=j.f.TARGET[duties[ri]]))
    (j.P / (arm+'.json')).write_text(json.dumps(result, indent=2)+'\n')
    m.write(str(j.P / (arm+'.lp')))
    m.dispose()
    return result
