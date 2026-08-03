"""Fast, read-only validation for EVSP Unicorn jobs."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import scipy

from config import CHARGING_STATIONS, DEPOT_NAME, STATION_COPIES
from utils_v2 import load_station_hourly_prices, select_unique_station_copies


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SUPPORTED_PYTHON_MAJOR_MINOR = (3, 12)


def validate_python_runtime(
    version_info: tuple[int, ...] | None = None,
    version_text: str | None = None,
) -> tuple[int, int]:
    """Require the repository's tested CPython 3.12 runtime line."""

    info = sys.version_info if version_info is None else version_info
    major_minor = (int(info[0]), int(info[1]))
    if major_minor != SUPPORTED_PYTHON_MAJOR_MINOR:
        found = sys.version.split()[0] if version_text is None else version_text
        raise RuntimeError(
            "Python 3.12.x required; "
            f"found {found}. Build the pinned environment with "
            "'bash src/bootstrap_python312.sh'."
        )
    return major_minor


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_gurobi() -> str:
    import gurobipy as gp

    model = gp.Model("evsp_preflight")
    model.Params.OutputFlag = 0
    variable = model.addVar(lb=0.0, obj=1.0)
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL or abs(variable.X) > 1e-9:
        raise RuntimeError(f"Unexpected Gurobi preflight status: {model.Status}")
    return ".".join(str(part) for part in gp.gurobi.version())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Instance filename under data/")
    parser.add_argument("--prices_csv", default="hourly_prices_flat.csv")
    parser.add_argument(
        "--mode",
        choices=["NO_CHEAT", "CHEAT", "GREEDY", "MATCHING"],
        default="MATCHING",
    )
    parser.add_argument("--allow_dirty", action="store_true")
    parser.add_argument("--skip_gurobi", action="store_true", help="For local data-only checks")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    validate_python_runtime()

    instance = Path(args.csv)
    if not instance.is_absolute():
        instance = DATA / instance
    prices = Path(args.prices_csv)
    if not prices.is_absolute():
        prices = DATA / prices

    required_files = [
        instance,
        prices,
        DATA / "par_ref_dhd.csv",
        DATA / "Ref_dict.csv",
        DATA / "Par_VehicleDetails_Updated.csv",
    ]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {missing_files}")

    dirty = git_output("status", "--porcelain")
    if dirty and not args.allow_dirty:
        raise RuntimeError(
            "Checkout is dirty; commit/stash changes or pass --allow_dirty deliberately:\n" + dirty
        )

    trips = pd.read_csv(instance)
    acceptable_columns = [
        {"From1", "Start1", "End1", "To1", "Usage kWh"},
        {"SL", "ST", "ET", "EL", "Energy used"},
    ]
    if not any(columns.issubset(trips.columns) for columns in acceptable_columns):
        raise ValueError(f"Unrecognized trip schema in {instance}: {list(trips.columns)}")
    if trips.empty:
        raise ValueError(f"Instance is empty: {instance}")

    historical_task_count = None
    if args.mode == "CHEAT":
        if "Ordered_Trip_ID" not in trips.columns:
            raise ValueError("CHEAT mode requires Ordered_Trip_ID")
        master = pd.read_csv(DATA / "Par_VehicleDetails_Updated.csv")
        regular = master[master["Identifier"] == "Regular"].copy()
        regular_ids = set(pd.to_numeric(regular["Ordered_Trip_ID"], errors="raise").astype(int))
        instance_ids = set(pd.to_numeric(trips["Ordered_Trip_ID"], errors="raise").astype(int))
        unmapped = sorted(instance_ids - regular_ids)
        if unmapped:
            raise ValueError(f"CHEAT mapping is missing Ordered_Trip_ID values: {unmapped[:20]}")
        task_by_trip = regular.set_index(
            pd.to_numeric(regular["Ordered_Trip_ID"], errors="raise").astype(int)
        )["VehicleTask"]
        historical_task_count = task_by_trip.loc[list(instance_ids)].astype(str).nunique()

    curves = load_station_hourly_prices(prices, CHARGING_STATIONS)
    copies = [
        f"{station}_{copy}"
        for station in CHARGING_STATIONS
        for copy in range(STATION_COPIES.get(station, 1))
    ]
    selected = select_unique_station_copies(copies, f"{DEPOT_NAME}_0")

    gurobi_version = "skipped"
    if not args.skip_gurobi:
        gurobi_version = check_gurobi()

    print("EVSP Unicorn preflight: PASS")
    print(f"  commit       : {git_output('rev-parse', 'HEAD')}")
    print(f"  branch       : {git_output('branch', '--show-current')}")
    print(f"  python       : {sys.version.split()[0]}")
    print(f"  pandas       : {pd.__version__}")
    print(f"  numpy        : {np.__version__}")
    print(f"  scipy        : {scipy.__version__}")
    print(f"  gurobi       : {gurobi_version}")
    print(f"  license env  : {os.environ.get('GRB_LICENSE_FILE', '(not set)')}")
    print(f"  instance     : {instance.name} ({len(trips)} trips, sha256={sha256(instance)[:12]})")
    print(f"  prices       : {prices.name} ({len(next(iter(curves.values())))} hours)")
    print(f"  pricing nodes: {selected}")
    print(f"  mode         : {args.mode}")
    if historical_task_count is not None:
        print(f"  CHEAT tasks  : {historical_task_count} historical VehicleTask values")
        print("  CHEAT caveat : coverage mapping only; route time/SOC feasibility is not validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
