"""Fixed ordered-sequence charging sensitivity; no scheduler or CG entry point.

Uses the frozen strict driver's event transitions, retaining the FULL instance's
event times while restricting only the allowed trip sequence. A no-path result
certifies this event representation, never the continuous physical model.
"""
from __future__ import annotations
import contextlib, dataclasses, hashlib, json, signal, time

ARMS = {
    'baseline': {'battery_kwh': 240., 'reserve_kwh': 0., 'parx_kw': 240., 'segregate': False},
    'parx60_only': {'battery_kwh': 240., 'reserve_kwh': 0., 'parx_kw': 60., 'segregate': False},
    'reserve15_only': {'battery_kwh': 240., 'reserve_kwh': 36., 'parx_kw': 240., 'segregate': False},
    'battery236p44_only': {'battery_kwh': 236.44, 'reserve_kwh': 0., 'parx_kw': 240., 'segregate': False},
    'battery239p01_only': {'battery_kwh': 239.01, 'reserve_kwh': 0., 'parx_kw': 240., 'segregate': False},
    'segregation_only': {'battery_kwh': 240., 'reserve_kwh': 0., 'parx_kw': 240., 'segregate': True},
}

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()

def canonical(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def ordered_sequence(route):
    trips=route['trips']
    if any(type(t) is not int for t in trips) or len(set(trips))!=len(trips):
        raise ValueError('invalid source trip incidence')
    seq=[v for v in route['route_nodes'] if type(v) is int]
    if len(seq)!=len(trips) or set(seq)!=set(trips):
        raise ValueError('route-node sequence and trip incidence disagree')
    return tuple(seq)

class SequenceTimeout(TimeoutError): pass

@contextlib.contextmanager
def deadline(seconds):
    """Bound even graph construction; timeout means unknown, not infeasible."""
    if seconds<=0: raise ValueError('positive per-sequence time limit required')
    def stop(*_): raise SequenceTimeout('per-sequence replay deadline')
    old_handler=signal.signal(signal.SIGALRM,stop)
    old_timer=signal.setitimer(signal.ITIMER_REAL,seconds)
    try: yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old_handler)
        if old_timer[0]: signal.setitimer(signal.ITIMER_REAL,*old_timer)

def restricted_network_class():
    from event_pricer_network import EventExpandedNetwork
    class FixedSequenceNetwork(EventExpandedNetwork):
        def __init__(self,problem,prices,sequence,full_events,**kwargs):
            seq=tuple(sequence)
            self.sequence=seq; self.successor=dict(zip(seq,(*seq[1:],None)))
            self.full_events=full_events
            allowed=set(seq); adjacency={}
            for source,arcs in problem.adjacency.items():
                if type(source) is int and source not in allowed: continue
                kept=[]
                for target,travel,energy,kind in arcs:
                    if type(target) is int and target not in allowed: continue
                    if kind=='depot_trip' and target!=seq[0]: continue
                    if kind=='trip_trip' and target!=self.successor[source]: continue
                    if kind=='trip_depot' and source!=seq[-1]: continue
                    kept.append((target,travel,energy,kind))
                adjacency[source]=kept
            restricted=dataclasses.replace(problem,trips=seq,adjacency=adjacency)
            super().__init__(restricted,prices,arc_mode='explicit',**kwargs)
        def _build_nodes(self):
            # Preserve every full-instance event breakpoint, not a smaller
            # sequence-derived time lattice that could confound the treatment.
            self.events=self.full_events
            super()._build_nodes()
        def _charge_candidates(self,trip,soc_exit,depart):
            nxt=self.successor[trip]
            saved_trips,saved_depot=self.station_trip,self.station_depot
            self.station_trip={s:({nxt:dest[nxt]} if nxt in dest else {}) for s,dest in saved_trips.items()}
            self.station_depot=saved_depot if nxt is None else {}
            try: yield from super()._charge_candidates(trip,soc_exit,depart)
            finally: self.station_trip,self.station_depot=saved_trips,saved_depot
    return FixedSequenceNetwork

def replay(problem,prices,full_events,sequence,physics,groups,seconds=120):
    started=time.perf_counter(); seq=tuple(sequence)
    result={'sequence_sha256':canonical(seq),'trip_sequence':list(seq),'status':None,
            'physical_replay_validated':False,'charging_optimal_in_fixed_sequence_event_model':False,
            'full_model_pricing_certificate':False}
    if physics['segregate'] and len({groups[t] for t in seq})>1:
        result.update(status='structurally_excluded_mixed_groups',elapsed_s=time.perf_counter()-started)
        return result,None
    try:
        with deadline(seconds):
            net=restricted_network_class()(problem,prices,seq,full_events,soc_step=2.5,block_min=5,
                g_kwh=physics['battery_kwh'],charge_kw=240.,reserve_kwh=physics['reserve_kwh'],
                station_charge_kw={'PARX':physics['parx_kw']},strict_tariff_coverage=False)
            route=net.fixed_sequence_record(seq)
            result['event_lattice_sha256']=net.metrics()['event_lattice_sha256']
            if route is None:
                result['status']='infeasible_in_fixed_sequence_event_model'
                return result,None
            if ordered_sequence(route)!=seq: raise ValueError('optimizer changed trip sequence')
            from run_exact_pool_mip import validate_injected_route
            from audit_giro_known_columns import HORIZON_MIN
            reason=validate_injected_route(problem,route,physics['battery_kwh'],240.,physics['reserve_kwh'],
                HORIZON_MIN,arrival_grace_min=0.,station_charge_kw={'PARX':physics['parx_kw']})
            if reason is not None:
                result.update(status='unknown_physical_validation_failure',reason=str(reason))
                return result,None
            result.update(status='feasible',physical_replay_validated=True,
                charging_optimal_in_fixed_sequence_event_model=True,reoptimized_cost=route['cost'])
            return result,route
    except SequenceTimeout as exc:
        result.update(status='unknown_timeout',reason=str(exc)); return result,None
    except Exception as exc:
        result.update(status='unknown_error',reason=f'{type(exc).__name__}: {exc}'); return result,None
    finally: result['elapsed_s']=time.perf_counter()-started

def coverage_gate(records,trip_ids):
    covered={t for rec in records if rec['status']=='feasible' for t in rec['trip_sequence']}
    missing=sorted(set(trip_ids)-covered)
    return {'all_trips_covered_by_physical_routes':not missing,'missing_trip_ids':missing,
            'artificials_are_not_physical_buses':True,'full_model_bound_available':False}
