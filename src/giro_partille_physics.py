"""Documented Partille battery and charging physics for recovery audits.

This module deliberately contains only single-vehicle physics.  Charger-count,
platform/FIFO, interlining-preference, and crew constraints couple routes and
belong in a later master-level experiment.  Keeping that boundary explicit
prevents a successful duty replay from being reported as full operational
feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass


TOL = 1e-9


@dataclass(frozen=True)
class ChargeBand:
    maximum_soc_fraction: float
    power_kw: float


@dataclass(frozen=True)
class VehicleProfile:
    name: str
    usable_capacity_kwh: float
    reserve_fraction: float
    idle_kw: float
    opportunity_setup_min: float
    minimum_recharge_duration_min: float
    allowed_opportunity_sites: tuple[str, ...]
    opportunity_curve: tuple[ChargeBand, ...]

    @property
    def reserve_kwh(self) -> float:
        return self.usable_capacity_kwh * self.reserve_fraction

    @property
    def transformed_usable_kwh(self) -> float:
        """Energy above the reserve, useful for zero-based formulations."""

        return self.usable_capacity_kwh - self.reserve_kwh


_PARTILLE_BANDS = tuple(
    ChargeBand(maximum_soc_fraction=fraction, power_kw=power)
    for fraction, power in (
        (0.10, 371.5),
        (0.20, 357.0),
        (0.30, 342.5),
        (0.40, 328.0),
        (0.50, 313.5),
        (0.60, 299.0),
        (0.70, 284.5),
        (0.80, 270.0),
        (0.90, 150.0),
        (1.00, 120.0),
    )
)


PARTILLE_PROFILES = {
    "18E1": VehicleProfile(
        name="18E1",
        usable_capacity_kwh=236.44,
        reserve_fraction=0.15,
        idle_kw=0.10,
        opportunity_setup_min=0.0,
        minimum_recharge_duration_min=3.0,
        allowed_opportunity_sites=("2190L", "4808"),
        opportunity_curve=_PARTILLE_BANDS,
    ),
    "18E2": VehicleProfile(
        name="18E2",
        usable_capacity_kwh=239.01,
        reserve_fraction=0.15,
        idle_kw=0.10,
        opportunity_setup_min=0.75,
        minimum_recharge_duration_min=3.0,
        allowed_opportunity_sites=("3127L", "7880C", "JON_A"),
        opportunity_curve=_PARTILLE_BANDS,
    ),
}


def base_site(station: object) -> str:
    value = str(station)
    left, separator, right = value.rpartition("_")
    return left if separator and right.isdigit() else value


def profile_for_duty(duty_id: object) -> VehicleProfile:
    value = str(duty_id)
    if value.startswith("133"):
        return PARTILLE_PROFILES["18E2"]
    if value.startswith("134"):
        return PARTILLE_PROFILES["18E1"]
    raise ValueError(f"Partille vehicle group is unknown for duty {value!r}")


def _power_at_soc(
    profile: VehicleProfile,
    station: object,
    soc_kwh: float,
) -> float:
    fraction = min(1.0, max(0.0, soc_kwh / profile.usable_capacity_kwh))
    for band in profile.opportunity_curve:
        if fraction < band.maximum_soc_fraction - TOL:
            # Par_Notes lists 271 kW through 80% at 3127L for 18E2,
            # versus 270 kW at the other two E2 opportunity sites.
            if (
                profile.name == "18E2"
                and base_site(station) == "3127L"
                and abs(band.maximum_soc_fraction - 0.80) <= TOL
            ):
                return 271.0
            return band.power_kw
    return profile.opportunity_curve[-1].power_kw


def charge_soc_after_minutes(
    profile: VehicleProfile,
    station: object,
    start_soc_kwh: float,
    active_minutes: float,
) -> float:
    """Integrate the documented taper curve for one active charge period.

    ``active_minutes`` excludes setup.  PARX is the 60 kW depot; opportunity
    sites use the SOC-dependent curve.  The result is battery-side energy,
    matching the usable-capacity convention recovered from VehicleDetails.
    The source does not define a separate grid-to-battery loss convention.
    """

    capacity = profile.usable_capacity_kwh
    soc = min(capacity, max(0.0, float(start_soc_kwh)))
    remaining = max(0.0, float(active_minutes))
    site = base_site(station)
    if site == "PARX":
        return min(capacity, soc + remaining * 60.0 / 60.0)
    if site not in profile.allowed_opportunity_sites:
        raise ValueError(
            f"{profile.name} is not documented to charge at {site}"
        )
    while remaining > TOL and soc < capacity - TOL:
        fraction = soc / capacity
        band = next(
            (row for row in profile.opportunity_curve
             if fraction < row.maximum_soc_fraction - TOL),
            profile.opportunity_curve[-1],
        )
        power_kw = _power_at_soc(profile, site, soc)
        threshold = band.maximum_soc_fraction * capacity
        energy_to_threshold = max(0.0, threshold - soc)
        minutes_to_threshold = energy_to_threshold * 60.0 / power_kw
        if remaining + TOL < minutes_to_threshold:
            soc += remaining * power_kw / 60.0
            remaining = 0.0
        else:
            soc = min(capacity, threshold)
            remaining -= minutes_to_threshold
    return min(capacity, soc)


def minutes_to_full(
    profile: VehicleProfile,
    station: object,
    start_soc_kwh: float,
) -> float:
    """Return active power-delivery minutes needed to reach usable capacity."""

    capacity = profile.usable_capacity_kwh
    soc = min(capacity, max(0.0, float(start_soc_kwh)))
    site = base_site(station)
    if site == "PARX":
        return capacity - soc
    if site not in profile.allowed_opportunity_sites:
        raise ValueError(
            f"{profile.name} is not documented to charge at {site}"
        )
    minutes = 0.0
    while soc < capacity - TOL:
        fraction = soc / capacity
        band = next(
            (row for row in profile.opportunity_curve
             if fraction < row.maximum_soc_fraction - TOL),
            profile.opportunity_curve[-1],
        )
        threshold = band.maximum_soc_fraction * capacity
        power_kw = _power_at_soc(profile, site, soc)
        minutes += max(0.0, threshold - soc) * 60.0 / power_kw
        soc = threshold
    return minutes


def charge_window(
    profile: VehicleProfile,
    station: object,
    start_soc_kwh: float,
    available_minutes: float,
) -> dict | None:
    """Return the maximum feasible charge in a fixed idle window."""

    site = base_site(station)
    setup = 0.0 if site == "PARX" else profile.opportunity_setup_min
    available_connected = max(0.0, float(available_minutes) - setup)
    minimum = profile.minimum_recharge_duration_min
    if available_connected + TOL < minimum:
        return None
    # Setup is a noncharging interval; account for the documented idle draw.
    charge_start_soc = float(start_soc_kwh) - setup * profile.idle_kw / 60.0
    if charge_start_soc < profile.reserve_kwh - TOL:
        return None
    power_delivery = min(
        available_connected,
        minutes_to_full(profile, site, charge_start_soc),
    )
    # Keep the vehicle connected for the full feasible window.  If it reaches
    # usable capacity early, the charger is assumed to maintain that level
    # against the tiny idle load.  This avoids inventing an unmodeled idle
    # interval and is conservative for the later charger-overlap audit.
    connected = available_connected
    end_soc = charge_soc_after_minutes(
        profile, site, charge_start_soc, power_delivery
    )
    return {
        "station": site,
        "setup_min": setup,
        "power_delivery_min": power_delivery,
        "connected_min": connected,
        "minimum_recharge_duration_min": minimum,
        "arrival_soc_kwh": float(start_soc_kwh),
        "start_soc_kwh": charge_start_soc,
        "end_soc_kwh": end_soc,
        "setup_idle_kwh": float(start_soc_kwh) - charge_start_soc,
        "delivered_kwh": end_soc - charge_start_soc,
    }


def documented_single_vehicle_scope() -> dict:
    return {
        "implemented": [
            "usable capacity by Partille vehicle group",
            "15 percent SOC floor at every modeled movement boundary",
            "60 kW PARX depot charging",
            "SOC-dependent Partille opportunity charging",
            "18E2 opportunity-charge setup of 45 seconds",
            "three-minute minimum recharge duration",
            "0.10 kW idle draw",
            "vehicle-group-specific opportunity sites",
        ],
        "not_implemented_in_single_vehicle_audit": [
            "shared charger counts across selected duties",
            "JON_A and 2190L departure-platform blocking",
            "4808 first-in-first-out movement",
            "complete interlining and preferred-layover rules",
            "driver and crew constraints",
            "directed time-dependent deadhead intervals (the gate uses the repository's static symmetric reference pairs)",
        ],
        "interpretation_only": [
            "65 percent is a recharge-activity target, not a proven hard terminal SOC",
            "charger-side efficiency convention is unspecified; usable-energy values are used directly",
            "three-minute minimum recharge duration is applied after setup; power delivery may stop at full SOC",
            "a vehicle remains connected through the available window; after full SOC the charger is assumed to maintain SOC",
        ],
    }
