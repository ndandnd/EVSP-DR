#!/usr/bin/env python3
"""Build and validate the immutable tariff-response pilot manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "tariff_response"
STATIONS = ("PARX", "JON_A", "2190L", "4808", "3127L", "7880C")
ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0)
HORIZON_END_HOUR = 26
MANIFEST_FIELDS = (
    "tariff_id", "relative_path", "sha256", "format",
    "spatial", "currency", "peak_hour", "alpha", "alpha_family",
    "peak_window_start_hour", "peak_window_end_hour",
    "solar_station", "solar_start_hour", "solar_end_hour",
    "source_flat_sha256", "source_peak_sha256",
    "source_tariff_sha256", "coverage_end_hour", "extension_policy",
    "has_negative_prices", "negative_price_policy", "availability",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_temporal(
    path: Path, *, required_end_hour: int = HORIZON_END_HOUR
) -> dict[int, float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if set(rows[0] if rows else ()) != {"time_block", "cost"}:
        raise ValueError(f"invalid temporal tariff schema: {path}")
    curve = {}
    for row in rows:
        hour = int(float(row["time_block"]))
        price = float(row["cost"])
        if hour in curve or not math.isfinite(price):
            raise ValueError(f"invalid temporal tariff row: {path}")
        curve[hour] = price
    if not set(range(required_end_hour + 1)) <= set(curve) or set(curve) != set(
        range(max(curve) + 1)
    ):
        raise ValueError(
            "tariff must define contiguous hours through "
            f"{required_end_hour}: {path}"
        )
    return curve


def write_temporal(path: Path, curve: dict[int, float]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("time_block", "cost"))
        for hour in range(max(curve) + 1):
            writer.writerow((hour, format(curve[hour], ".17g")))
        handle.flush()
        os.fsync(handle.fileno())


def write_spatial_solar(
    path: Path,
    flat: dict[int, float],
    *,
    station: str,
    start_hour: int,
    end_hour: int,
    midday_price: float,
) -> None:
    if station not in STATIONS or not 0 <= start_hour < end_hour <= 25:
        raise ValueError("invalid spatial-solar definition")
    with path.open("x", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("time_block", "station", "cost"))
        for hour in range(HORIZON_END_HOUR + 1):
            for candidate in STATIONS:
                price = (
                    midday_price
                    if candidate == station and start_hour <= hour < end_hour
                    else flat[hour]
                )
                writer.writerow((
                    hour, candidate, format(float(price), ".17g")
                ))
        handle.flush()
        os.fsync(handle.fileno())


def validate_spatial(path: Path) -> None:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if set(rows[0] if rows else ()) != {
        "time_block", "station", "cost"
    }:
        raise ValueError("invalid spatial tariff schema")
    keys = set()
    for row in rows:
        key = (int(float(row["time_block"])), row["station"])
        price = float(row["cost"])
        if (
            key in keys
            or key[0] not in range(HORIZON_END_HOUR + 1)
            or key[1] not in STATIONS
            or not math.isfinite(price)
        ):
            raise ValueError("invalid spatial tariff row")
        keys.add(key)
    if keys != {
        (hour, station)
        for hour in range(HORIZON_END_HOUR + 1)
        for station in STATIONS
    }:
        raise ValueError("spatial tariff is incomplete")


def build(output_dir: Path = OUTPUT_DIR) -> tuple[Path, list[dict]]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and (
        not output_dir.is_dir()
        or (output_dir / "tariff_manifest.csv").exists()
    ):
        raise FileExistsError(f"tariff output exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    source_definitions = {
        "flat": ("hourly_prices_flat.csv", "model_currency_per_kwh", ""),
        "peak08": (
            "hourly_prices_single_peak_08.csv",
            "model_currency_per_kwh", 8,
        ),
        "peak12": (
            "hourly_prices_single_peak_12.csv",
            "model_currency_per_kwh", 12,
        ),
        "peak18": (
            "hourly_prices_single_peak_18.csv",
            "model_currency_per_kwh", 18,
        ),
        "sek": ("hourly_prices_transdev_sek.csv", "SEK_per_kwh", ""),
    }
    normalized_sources = {}
    source_hashes = {}
    for tariff_id, (relative, _currency, _peak) in source_definitions.items():
        source_path = DATA_DIR / relative
        source_curve = read_temporal(
            source_path, required_end_hour=24
        )
        last = source_curve[max(source_curve)]
        normalized = {
            hour: source_curve.get(hour, last)
            for hour in range(HORIZON_END_HOUR + 1)
        }
        normalized_path = output_dir / f"{tariff_id}_h26.csv"
        if normalized_path.exists():
            observed = read_temporal(normalized_path)
            if observed != normalized:
                raise ValueError(
                    f"partial normalized tariff differs: {normalized_path}"
                )
        else:
            write_temporal(normalized_path, normalized)
        normalized_sources[tariff_id] = (
            normalized_path, normalized
        )
        source_hashes[tariff_id] = sha256_file(source_path)
    flat_path, flat = normalized_sources["flat"]
    peak12_path, peak12 = normalized_sources["peak12"]
    flat_sha = sha256_file(flat_path)
    peak12_sha = sha256_file(peak12_path)

    generated = []
    for alpha in ALPHAS:
        alpha_label = str(alpha).replace(".", "p")
        path = output_dir / f"peak12_alpha_{alpha_label}_h26.csv"
        curve = {
            hour: flat[hour] + alpha * (peak12[hour] - flat[hour])
            for hour in range(HORIZON_END_HOUR + 1)
        }
        if path.exists():
            observed = read_temporal(path)
            if any(
                not math.isclose(
                    observed.get(hour, math.nan), curve[hour],
                    rel_tol=0.0, abs_tol=1e-15,
                )
                for hour in range(HORIZON_END_HOUR + 1)
            ):
                raise ValueError(f"partial tariff differs: {path}")
        else:
            write_temporal(path, curve)
        generated.append({
            "tariff_id": f"peak12_alpha_{alpha_label}",
            "path": path,
            "spatial": False,
            "currency": "model_currency_per_kwh",
            "peak_hour": 12,
            "alpha": alpha,
            "alpha_family": "peak12",
            "peak_window_start_hour": 12,
            "peak_window_end_hour": 13,
            "solar_station": "",
            "solar_start_hour": "",
            "solar_end_hour": "",
            "source_flat_sha256": flat_sha,
            "source_peak_sha256": peak12_sha,
            "source_tariff_sha256": "",
            "coverage_end_hour": HORIZON_END_HOUR,
            "extension_policy":
                "explicit_last_source_hour_through_hour_26",
            "has_negative_prices": any(
                value < 0 for value in curve.values()
            ),
            "negative_price_policy":
                "allow_feasible_consumption_no_export",
        })
    solar_path = output_dir / "spatial_solar_parx_midday_free_h26.csv"
    if not solar_path.exists():
        write_spatial_solar(
            solar_path,
            flat,
            station="PARX",
            start_hour=11,
            end_hour=15,
            midday_price=0.0,
        )
    validate_spatial(solar_path)

    rows = []
    for tariff_id, (_relative, currency, peak) in (
        source_definitions.items()
    ):
        path, curve = normalized_sources[tariff_id]
        rows.append({
            "tariff_id": tariff_id,
            "relative_path": str(path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(path),
            "format": "temporal_hourly",
            "spatial": False,
            "currency": currency,
            "peak_hour": peak,
            "alpha": "",
            "alpha_family": "",
            "peak_window_start_hour": peak if peak != "" else "",
            "peak_window_end_hour": (
                int(peak) + 1 if peak != "" else ""
            ),
            "solar_station": "",
            "solar_start_hour": "",
            "solar_end_hour": "",
            "source_flat_sha256": (
                flat_sha if tariff_id == "flat" else ""
            ),
            "source_peak_sha256": "",
            "source_tariff_sha256": source_hashes[tariff_id],
            "coverage_end_hour": HORIZON_END_HOUR,
            "extension_policy":
                "explicit_last_source_hour_through_hour_26",
            "has_negative_prices": any(
                value < 0 for value in curve.values()
            ),
            "negative_price_policy":
                "allow_feasible_consumption_no_export",
            "availability": "available",
        })
    rows.append({
        "tariff_id": "solar_parx_midday_free",
        "relative_path": str(solar_path.relative_to(REPO_ROOT)),
        "sha256": sha256_file(solar_path),
        "format": "station_hourly",
        "spatial": True,
        "currency": "model_currency_per_kwh",
        "peak_hour": "",
        "alpha": "",
        "alpha_family": "",
        "peak_window_start_hour": 11,
        "peak_window_end_hour": 15,
        "solar_station": "PARX",
        "solar_start_hour": 11,
        "solar_end_hour": 15,
        "source_flat_sha256": flat_sha,
        "source_peak_sha256": "",
        "source_tariff_sha256": "",
        "coverage_end_hour": HORIZON_END_HOUR,
        "extension_policy":
            "explicit_flat_base_with_named_station_solar_override",
        "has_negative_prices": False,
        "negative_price_policy": "allow_feasible_consumption_no_export",
        "availability": "available",
    })
    for item in generated:
        rows.append({
            "tariff_id": item["tariff_id"],
            "relative_path": str(item["path"].relative_to(REPO_ROOT)),
            "sha256": sha256_file(item["path"]),
            "format": "temporal_hourly",
            "spatial": item["spatial"],
            "currency": item["currency"],
            "peak_hour": item["peak_hour"],
            "alpha": item["alpha"],
            "alpha_family": item["alpha_family"],
            "peak_window_start_hour":
                item["peak_window_start_hour"],
            "peak_window_end_hour": item["peak_window_end_hour"],
            "solar_station": item["solar_station"],
            "solar_start_hour": item["solar_start_hour"],
            "solar_end_hour": item["solar_end_hour"],
            "source_flat_sha256": item["source_flat_sha256"],
            "source_peak_sha256": item["source_peak_sha256"],
            "source_tariff_sha256": item["source_tariff_sha256"],
            "coverage_end_hour": item["coverage_end_hour"],
            "extension_policy": item["extension_policy"],
            "has_negative_prices": item["has_negative_prices"],
            "negative_price_policy": item["negative_price_policy"],
            "availability": "available",
        })
    rows.sort(key=lambda row: row["tariff_id"])
    manifest = output_dir / "tariff_manifest.csv"
    with manifest.open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    return manifest, rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest, rows = build(args.out_dir)
    print(f"{manifest} rows={len(rows)} sha256={sha256_file(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
