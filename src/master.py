from gurobipy import Model, Column, LinExpr, GRB
from utils import calculate_truck_route_cost
from config import (
    THREADS, NODEFILE_START, NODEFILE_DIR,
    MASTER_TIMELIMIT, MASTER_MIPGAP
)

def _apply_master_params(rmp: Model, *, mip_mode=False):
    # common
    rmp.Params.OutputFlag = 1
    rmp.Params.Threads = THREADS
    rmp.Params.NodefileStart = NODEFILE_START
    if NODEFILE_DIR:
        rmp.Params.NodefileDir = NODEFILE_DIR
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
    binary=False
):
    rmp = Model("RMP_EVSP")
    _apply_master_params(rmp, mip_mode=binary)

    # 1) Coverage constraints for all trips
    trip_cov = {}
    for i in T:
        trip_cov[i] = rmp.addConstr(LinExpr() >= 1, name=f"trip_coverage_{i}")

    # 2) Decision variables (one per truck route)
    vtype = GRB.INTEGER if binary else GRB.CONTINUOUS
    a = {}

    # 3) Objective coefficients (route costs)
    for idx, route in enumerate(R_truck):
        if route.get("dummy", False):
            cost = float(route.get("dummy_cost", 1e7))
        else:
            cost = calculate_truck_route_cost(route, bus_cost, charging_cost_data)

        col = Column()
        for i in route.get("route", []):
            if i in T:
                col.addTerms(1.0, trip_cov[i])

        a[idx] = rmp.addVar(
            obj=cost,
            lb=0, ub=1,
            vtype=vtype,
            column=col,
            name=f"a[{idx}]"
        )

    rmp.update()
    rmp.modelSense = GRB.MINIMIZE
    return rmp, a, trip_cov


def init_master(R_truck, T, charging_cost_data, bus_cost, binary=False):
    return build_master(R_truck, T, charging_cost_data, bus_cost, binary=binary)


def solve_master(R_truck, T, charging_cost_data, bus_cost, binary=False):
    rmp, a, trip_cov = build_master(
        R_truck=R_truck,
        T=T,
        charging_cost_data=charging_cost_data,
        bus_cost=bus_cost,
        binary=binary
    )
    rmp.optimize()
    return rmp, a
