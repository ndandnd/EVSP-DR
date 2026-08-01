import os

from gurobipy import Model, Column, LinExpr, GRB
from utils_v2 import calculate_truck_route_cost_accurate
from config import (
    THREADS, NODEFILE_START, NODEFILE_DIR,
    MASTER_TIMELIMIT, MASTER_MIPGAP,
    BIG_M_PENALTY,
    CHARGE_START_COST, CHARGE_RATE_KW,
)

def _apply_master_params(rmp: Model, *, mip_mode=False):
    # common
    rmp.Params.OutputFlag = 1
    rmp.Params.Threads = THREADS
    rmp.Params.NodefileStart = NODEFILE_START
    rmp.Params.NodefileDir = (
        NODEFILE_DIR
        or os.environ.get("SLURM_TMPDIR")
        or os.environ.get("TMPDIR")
        or "/tmp"
    )
    # LP (CG) vs MIP (final)
    if not mip_mode:
        rmp.Params.Method = 1         # dual simplex for LPs
        rmp.Params.TimeLimit = MASTER_TIMELIMIT
        rmp.Params.BarHomogeneous = 1
    else:
        rmp.Params.MIPGap = MASTER_MIPGAP
        rmp.Params.MIPFocus = 1
        rmp.Params.Heuristics = 0.5
        rmp.Params.Cuts = 1

def build_master(
    R_truck,
    T,
    charging_cost_data,
    bus_cost,
    binary=False,
    station_hourly_prices=None,   # NEW
):

    rmp = Model("RMP_EVSP")
    _apply_master_params(rmp, mip_mode=binary)

    # 1) Coverage constraints for all trips
    trip_cov = {}
    for i in T:
        trip_cov[i] = rmp.addConstr(LinExpr() >= 1, name=f"trip_coverage_{i}")


    ### Dummy variables to ensure feasibility for empty initial R'
    q_vars = {}
    for i in T:
        col = Column()
        col.addTerms(1.0, trip_cov[i]) # Add q_i to the coverage constraint for trip i

        q_vars[i] = rmp.addVar(
            obj=BIG_M_PENALTY,  # High penalty
            lb=0,
            ub=GRB.INFINITY,    # In the end bounded by 1 because constraint is \geq 1.
            vtype=GRB.CONTINUOUS,
            column=col,
            name=f"q_{i}"
        )


    # 2) Decision variables (one per truck route)
    vtype = GRB.INTEGER if binary else GRB.CONTINUOUS
    a = {}

    # 3) Objective coefficients (route costs)
    for idx, route in enumerate(R_truck):
        if route.get("dummy", False):
            cost = float(route.get("dummy_cost", 1e7))
        else:
            # Hour-split charging cost — must match the DP pricer's rc math
            cost = calculate_truck_route_cost_accurate(
                route, bus_cost, charging_cost_data,
                charge_rate_kw=CHARGE_RATE_KW,
                station_hourly_prices=station_hourly_prices,
                charge_start_cost=CHARGE_START_COST,
            )

        col = Column()
        for i in route.get("route", []):
            if i in T:
                col.addTerms(1.0, trip_cov[i])

        # LP columns must be unbounded above: with ub=1, a column at its bound
        # can price negative (rc = -mu of the bound) under dual degeneracy, so
        # the pricer keeps regenerating existing routes and CG stalls on
        # "no new columns". Over-selection never helps set covering, so the
        # bound is redundant in the LP; keep it only for the integer model.
        a[idx] = rmp.addVar(
            obj=cost,
            lb=0, ub=1 if binary else GRB.INFINITY,
            vtype=vtype,
            column=col,
            name=f"a[{idx}]"
        )

    rmp.update()
    rmp.modelSense = GRB.MINIMIZE
    return rmp, a, trip_cov


# def init_master(R_truck, T, charging_cost_data, bus_cost, binary=False):
#     return build_master(R_truck, T, charging_cost_data, bus_cost, binary=binary)

def init_master(R_truck, T, charging_cost_data, bus_cost, binary=False, station_hourly_prices=None):
    return build_master(R_truck, T, charging_cost_data, bus_cost, binary=binary,
                        station_hourly_prices=station_hourly_prices)


def solve_master(R_truck, T, charging_cost_data, bus_cost, binary=False,  station_hourly_prices=None):
    rmp, a, trip_cov = build_master(
        R_truck=R_truck,
        T=T,
        charging_cost_data=charging_cost_data,
        bus_cost=bus_cost,
        binary=binary,

        station_hourly_prices=station_hourly_prices,
    )
    rmp.optimize()
    return rmp, a
