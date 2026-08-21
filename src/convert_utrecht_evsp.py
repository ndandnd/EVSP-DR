"""Convert Utrecht's public E-VSP text format into portable ProblemData JSON.

The upstream repository is https://github.com/UtrechtUniversity/evsp-instances.
No upstream files are redistributed here because that repository declares no
license.  Clone it separately and point this converter at one instance folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


SCHEMA = "evsp-dr-portable-problem-v1"
UPSTREAM_URL = "https://github.com/UtrechtUniversity/evsp-instances"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(path: Path, prefix: str) -> list[list[str]]:
    return [
        line.rstrip("\r\n").split(";")
        for line in path.read_text(encoding="latin-1").splitlines()
        if line.startswith(prefix + ";")
    ]


def convert_instance(
    source_dir: Path,
    *,
    name: str,
    upstream_commit: str,
) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    paths = {
        filename: source_dir / filename
        for filename in ("parameters.txt", "trips.txt", "dhd.txt")
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"benchmark files missing: {missing}")

    parameter_rows = _records(paths["parameters.txt"], "U")
    depot_rows = _records(paths["parameters.txt"], "G")
    station_rows = _records(paths["parameters.txt"], "E")
    if len(parameter_rows) != 1 or len(depot_rows) != 1 or not station_rows:
        raise ValueError("expected one U row, one G row, and at least one E row")
    vehicle = parameter_rows[0]
    if len(vehicle) < 13:
        raise ValueError("vehicle parameter row is incomplete")
    energy_per_km = float(vehicle[9])
    max_charge_kwh_per_min = float(vehicle[10])
    battery_kwh = float(vehicle[12])
    depot = depot_rows[0][1]
    stations = []
    station_parameters = {}
    for row in station_rows:
        if len(row) < 5:
            raise ValueError(f"charging-location row is incomplete: {row}")
        location = row[1]
        stations.append(location)
        station_parameters[location] = {
            "vehicle_capacity": int(row[2]),
            "setup_min": float(row[3]),
            "infrastructure_charge_kwh_per_min": float(row[4]),
            "max_bus_soc_fraction": (
                float(row[5]) if len(row) > 5 and row[5] else None
            ),
        }

    raw_trips = _records(paths["trips.txt"], "T")
    if not raw_trips:
        raise ValueError("instance contains no T records")
    parsed_trips = []
    for ordinal, row in enumerate(raw_trips):
        if len(row) < 18:
            raise ValueError(f"trip row {ordinal + 1} is incomplete")
        distance_km = float(row[17])
        parsed_trips.append({
            "source_ordinal": ordinal,
            "external_id": f"{row[1]}:{row[2]}",
            "line": row[1],
            "trip_number": row[2],
            "from": row[5],
            "start_min": int(row[6]),
            "end_min": int(row[7]),
            "to": row[8],
            "distance_km": distance_km,
            "energy_kwh": distance_km * energy_per_km,
        })
    parsed_trips.sort(key=lambda trip: (
        trip["start_min"], trip["end_min"], trip["external_id"],
        trip["source_ordinal"],
    ))
    for local_id, trip in enumerate(parsed_trips):
        trip["id"] = local_id

    windows = []
    for row in _records(paths["dhd.txt"], "G"):
        if len(row) != 4:
            raise ValueError(f"deadhead time-window row is invalid: {row}")
        windows.append({
            "version": int(row[1]),
            "start_min": int(row[2]),
            "end_min": int(row[3]),
        })
    windows.sort(key=lambda window: window["start_min"])
    deadheads = []
    for row in _records(paths["dhd.txt"], "D"):
        if len(row) < 7 or "-" not in row[1]:
            raise ValueError(f"deadhead row is invalid: {row}")
        source, target = row[1].split("-", 1)
        distance_km = float(row[6])
        deadheads.append({
            "from": source,
            "to": target,
            "travel_min_by_version": [int(value) for value in row[2:6]],
            "distance_km": distance_km,
            "energy_kwh": distance_km * energy_per_km,
        })
    horizon_min = max(
        max(trip["end_min"] for trip in parsed_trips),
        max(window["end_min"] + 1 for window in windows),
    )
    return {
        "schema": SCHEMA,
        "name": name,
        "horizon_min": horizon_min,
        "depot": depot,
        "stations": list(dict.fromkeys(stations)),
        "vehicle": {
            "battery_kwh": battery_kwh,
            "energy_kwh_per_km": energy_per_km,
            "max_charge_kwh_per_min": max_charge_kwh_per_min,
        },
        "station_parameters": station_parameters,
        "time_windows": windows,
        "deadheads": deadheads,
        "trips": parsed_trips,
        "source": {
            "repository": UPSTREAM_URL,
            "commit": upstream_commit,
            "instance_directory": source_dir.name,
            "files": {
                filename: {
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
                for filename, path in paths.items()
            },
        },
        "conversion": {
            "trip_energy": "published trip distance times U-row kWh/km",
            "deadhead_energy":
                "published deadhead distance times U-row kWh/km",
            "deadhead_time": "published departure-time version profile",
            "charging_model_changed": True,
        },
    }


@dataclass
class ConvertedProblemData:
    frame: pd.DataFrame
    trips: tuple[int, ...]
    adjacency: dict[Any, list[tuple[Any, float, float, str]]]
    start_min: dict[int, float]
    end_min: dict[int, float]
    trip_energy: dict[int, float]
    depot: str
    stations: tuple[str, ...]
    horizon_min: float
    start_location: dict[int, str]
    end_location: dict[int, str]
    profiles: dict[tuple[str, str], tuple[tuple[int, ...], float]]
    time_windows: tuple[tuple[int, int, int], ...]

    def _location(self, node, *, departing: bool) -> str:
        if isinstance(node, int) and not isinstance(node, bool):
            return (
                self.end_location[node]
                if departing else self.start_location[node]
            )
        return str(node)

    def transition_at(self, source, target, departure_min):
        left = self._location(source, departing=True)
        right = self._location(target, departing=False)
        if left == right:
            return 0.0, 0.0
        profile = self.profiles.get((left, right))
        if profile is None:
            return None
        times, energy = profile
        minute = int(math.floor(float(departure_min) + 1e-9))
        version = next((
            version for version, start, end in self.time_windows
            if start <= minute <= end
        ), None)
        if version is None or version >= len(times):
            return None
        return float(times[version]), float(energy)

    def arc_map_for_route(self, route: dict) -> dict:
        nodes = list(route.get("route_nodes") or [])
        stops = route.get("charging_stops") or {}
        stations = list(stops.get("stations", []))
        cets = list(stops.get("cet", []))
        stop_index = 0
        time_now = None
        result = {}
        previous = nodes[0]
        for position, node in enumerate(nodes[1:], 1):
            is_last = position == len(nodes) - 1
            departure = (
                0.0 if time_now is None else float(time_now)
            )
            transition = self.transition_at(previous, node, departure)
            if transition is None:
                raise ValueError(f"missing dynamic transition {previous}->{node}")
            result[(previous, node)] = transition
            if isinstance(node, int) and not isinstance(node, bool):
                time_now = self.end_min[node]
            elif not is_last and stop_index < len(stations):
                if node != stations[stop_index]:
                    raise ValueError("charging-stop route order mismatch")
                time_now = float(cets[stop_index])
                stop_index += 1
            previous = node
        return result


def _latest_pullout(problem: ConvertedProblemData, trip: int):
    start = int(problem.start_min[trip])
    for departure in range(start, -1, -1):
        transition = problem.transition_at(problem.depot, trip, departure)
        if transition is not None and departure + transition[0] <= start:
            return transition
    return None


def load_problem(payload_or_path: dict | Path) -> ConvertedProblemData:
    payload = (
        payload_or_path
        if isinstance(payload_or_path, dict)
        else json.loads(Path(payload_or_path).read_text())
    )
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported portable problem schema: {payload.get('schema')}")
    trips = tuple(int(trip["id"]) for trip in payload["trips"])
    if trips != tuple(range(len(trips))):
        raise ValueError("portable trip ids must be dense and ordered")
    start_min = {trip["id"]: float(trip["start_min"]) for trip in payload["trips"]}
    end_min = {trip["id"]: float(trip["end_min"]) for trip in payload["trips"]}
    trip_energy = {trip["id"]: float(trip["energy_kwh"]) for trip in payload["trips"]}
    start_location = {trip["id"]: trip["from"] for trip in payload["trips"]}
    end_location = {trip["id"]: trip["to"] for trip in payload["trips"]}
    profiles = {
        (deadhead["from"], deadhead["to"]): (
            tuple(int(value) for value in deadhead["travel_min_by_version"]),
            float(deadhead["energy_kwh"]),
        )
        for deadhead in payload["deadheads"]
    }
    problem = ConvertedProblemData(
        frame=pd.DataFrame(payload["trips"]),
        trips=trips,
        adjacency={},
        start_min=start_min,
        end_min=end_min,
        trip_energy=trip_energy,
        depot=payload["depot"],
        stations=tuple(payload["stations"]),
        horizon_min=float(payload["horizon_min"]),
        start_location=start_location,
        end_location=end_location,
        profiles=profiles,
        time_windows=tuple(
            (int(window["version"]), int(window["start_min"]),
             int(window["end_min"]))
            for window in payload["time_windows"]
        ),
    )
    adjacency: dict[Any, list] = {
        problem.depot: [],
        **{trip: [] for trip in trips},
        **{station: [] for station in problem.stations},
    }

    def put(source, target, departure, kind, transition=None):
        resolved = (
            problem.transition_at(source, target, departure)
            if transition is None else transition
        )
        if resolved is not None:
            adjacency[source].append((target, *resolved, kind))

    for trip in trips:
        pullout = _latest_pullout(problem, trip)
        if pullout is not None:
            put(problem.depot, trip, 0, "depot_trip", pullout)
        put(trip, problem.depot, end_min[trip], "trip_depot")
        for following in trips:
            if trip == following:
                continue
            transition = problem.transition_at(trip, following, end_min[trip])
            if (
                transition is not None
                and end_min[trip] + transition[0] <= start_min[following]
            ):
                put(trip, following, end_min[trip], "trip_trip", transition)
        for station in problem.stations:
            put(trip, station, end_min[trip], "trip_station")
            if any(
                (transition := problem.transition_at(station, trip, minute))
                is not None
                and minute + transition[0] <= start_min[trip]
                for minute in range(0, int(start_min[trip]) + 1)
            ):
                put(station, trip, 0, "station_trip", transition)
    for station in problem.stations:
        put(station, problem.depot, 0, "station_depot")
    problem.adjacency = adjacency
    return problem


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = convert_instance(
        args.source_dir, name=args.name,
        upstream_commit=args.upstream_commit,
    )
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(args.out.resolve()),
        "name": payload["name"],
        "trips": len(payload["trips"]),
        "deadheads": len(payload["deadheads"]),
        "battery_kwh": payload["vehicle"]["battery_kwh"],
        "source_commit": payload["source"]["commit"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
