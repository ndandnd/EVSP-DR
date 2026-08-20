"""Durable state and hash-bound root identity for branch-and-price."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from durable_io import atomic_write_json, flush_and_fsync, read_jsonl_records
from exact_pricer_expanded import _file_sha256, load_column_pool


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_MANIFEST = (
    REPO_ROOT / "analysis/branch_and_price_experiment_20260820/"
    "ROOT_BASELINES.json"
)


def baseline_identity(args, provenance, error_type=RuntimeError):
    payload = json.loads(BASELINE_MANIFEST.read_text())
    if args.prices_csv != "hourly_prices_flat.csv":
        raise error_type("G1 baseline requires hourly_prices_flat.csv")
    for key, expected in payload["fixed_input_hashes"].items():
        if provenance.get(key) != expected:
            raise error_type(
                f"G1 identity mismatch for {key}: {provenance.get(key)}"
            )
    fields = ("csv", "soc_step", "block_min", "g_kwh",
              "charge_kw", "min_soc_frac")
    matches = [row for row in payload["baselines"]
               if all(row[key] == getattr(args, key) for key in fields)]
    if not matches:
        if args.expected_root_weight is not None:
            raise error_type("G1 has no hash-bound baseline row")
        return None
    row = matches[0]
    if row["instance_sha256"] != provenance["instance_sha256"]:
        raise error_type("G1 baseline instance hash mismatch")
    source = payload["sources"][row["source"]]
    if _file_sha256(REPO_ROOT / source["path"]) != source["sha256"]:
        raise error_type("G1 baseline source hash mismatch")
    if (args.expected_root_weight is not None
            and not math.isclose(args.expected_root_weight,
                                 row["route_weight"], abs_tol=1e-12)):
        raise error_type("G1 CLI value disagrees with bound baseline")
    return {
        **row, "source_path": source["path"],
        "source_sha256": source["sha256"],
        "manifest_sha256": _file_sha256(BASELINE_MANIFEST),
    }


def fleet_bound_closes(lower_bound, incumbent_fleet, tolerance=1e-7):
    return math.ceil(lower_bound - tolerance) >= incumbent_fleet


class DurableStateMixin:
    """Append-only columns/events plus an atomic resumable tree checkpoint."""

    def _setup_io(self):
        self.out = self.args.out.expanduser().resolve()
        self.journal_path = Path(str(self.out) + ".columns.jsonl")
        self.ledger_path = Path(str(self.out) + ".nodes.jsonl")
        self.out.parent.mkdir(parents=True, exist_ok=True)
        paths = (self.out, self.journal_path, self.ledger_path)
        if self.args.resume:
            if not all(path.exists() for path in paths):
                raise FileNotFoundError("resume requires status, columns, ledger")
            saved = json.loads(self.out.read_text())
            if saved.get("run_identity") != self.run_identity:
                raise RuntimeError("resume identity mismatch")
            self.elapsed_offset = float(saved.get("wall_s", 0.0))
            self.pool = load_column_pool(
                read_jsonl_records(self.journal_path, repair_trailing=True),
                self.trips,
            )
            state = saved["search_checkpoint"]
            self.stack = [self._node_from_payload(row) for row in state["stack"]]
            self.frontier_bounds = list(map(float, state["frontier_bounds"]))
            for key in self._checkpoint_fields:
                setattr(self, key, state.get(key, getattr(self, key)))
            self.incumbent = state.get("incumbent")
        elif any(path.exists() for path in paths):
            raise FileExistsError("fresh run refuses existing durable artifacts")
        self.journal = self.journal_path.open("a" if self.args.resume else "x")
        self.ledger = self.ledger_path.open("a" if self.args.resume else "x")
        if not self.args.resume:
            for record in self.pool.values():
                self.journal.write(json.dumps(record, sort_keys=True) + "\n")
            flush_and_fsync(self.journal)
        self._event("resumed" if self.args.resume else "initialized")
        self._checkpoint()

    def _event(self, event, **detail):
        self.ledger_events += 1
        self.ledger.write(json.dumps({
            "event_index": self.ledger_events, "event": event,
            "elapsed_s": self._elapsed_s(), **detail,
        }, sort_keys=True) + "\n")
        flush_and_fsync(self.ledger)

    def _append_column(self, record):
        self.journal.write(json.dumps(record, sort_keys=True) + "\n")
        flush_and_fsync(self.journal)

    def _prune_open_by_incumbent(self):
        if self.incumbent is None:
            return
        fleet = self.incumbent["fleet"]
        before = len(self.stack) + len(self.frontier_bounds)
        self.stack = [node for node in self.stack if not fleet_bound_closes(
            node.lower_bound, fleet, self.args.bound_tolerance
        )]
        self.frontier_bounds = [
            bound for bound in self.frontier_bounds
            if not fleet_bound_closes(
                bound, fleet, self.args.bound_tolerance
            )
        ]
        after = len(self.stack) + len(self.frontier_bounds)
        if before > after:
            self._event(
                "open_nodes_pruned_by_integer_fleet_bound",
                pruned=before - after, incumbent_fleet=fleet,
            )

    def _elapsed_s(self):
        return self.elapsed_offset + time.monotonic() - self.started

    def _base_payload(self):
        return {
            "schema": "evsp-dr-exact-branch-and-price-v2",
            "csv": self.args.csv, "prices_csv": self.args.prices_csv,
            "soc_step": self.args.soc_step, "block_min": self.args.block_min,
            "g_kwh": self.args.g_kwh, "charge_kw": self.args.charge_kw,
            "min_soc_frac": self.args.min_soc_frac, "master_sense": "partition",
            "column_pool_treatment": "RAW",
            "target_fleet": self.args.target_fleet, "trip_ids": self.trips,
            "pricing_certificate_scope":
                "conservative_expanded_grid_model_only",
            "master_objective": "phase_1_artificial_then_phase_2_fleet_only",
            "global_lower_bound_units": "fractional_fleet_route_count",
            "provenance": self.provenance, "run_identity": self.run_identity,
            "root_baseline": self.baseline,
        }

    def _search_checkpoint(self):
        return {
            "stack": [self._node_payload(node) for node in self.stack],
            "frontier_bounds": self.frontier_bounds,
            "incumbent": self.incumbent,
            **{key: getattr(self, key) for key in self._checkpoint_fields},
        }

    def _payload(self, search_complete=False):
        open_bounds = list(self.frontier_bounds) + [
            node.lower_bound for node in self.stack
        ]
        proven = bool(
            search_complete and not open_bounds and self.incumbent is not None
            and not self.args.root_only
        )
        bound = (
            float(self.incumbent["fleet"]) if proven
            else min(open_bounds) if open_bounds else self.root_lower_bound
        )
        fleet = self.incumbent["fleet"] if self.incumbent else None
        gap = (max(0.0, fleet - bound) / max(1.0, fleet)
               if fleet is not None and bound is not None else None)
        payload = {
            **self._base_payload(), "root_certified": self.root_solved,
            "root_lp": self.root_record, "best_integer_fleet": fleet,
            "best_integer_cost": self.incumbent["cost"] if self.incumbent else None,
            "best_integer_source":
                self.incumbent["source"] if self.incumbent else None,
            "best_integer_routes":
                self.incumbent["routes"] if self.incumbent else [],
            "global_lower_bound": bound, "lower_bound_valid": self.root_solved,
            "gap": gap, "proven_optimal": proven,
            "proven_optimal_scope": "fleet_only" if proven else None,
            "nodes_explored": self.nodes_explored,
            "nodes_depth_capped": self.nodes_depth_capped,
            "nodes_infeasible_certified": self.infeasible_certificates,
            "pricing_solves": self.pricing_solves,
            "pricing_calls": self.pricing_calls,
            "pricing_wall_s": self.pricing_wall_s,
            "master_solves": self.master_solves, "wall_s": self._elapsed_s(),
            "network_build_s": self.network_build_s,
            "interrupted_reason": self.interrupted_reason,
            "kill_criterion_triggered":
                self.interrupted_reason == "pricing_slowdown_kill_criterion",
            "open_frontier_nodes": len(open_bounds), "columns": len(self.pool),
            "columns_journal": str(self.journal_path),
            "node_ledger": str(self.ledger_path),
            "search_checkpoint": self._search_checkpoint(),
            "validation": {
                "G1": "pass" if self.root_solved else "pending",
                "G2_bound_assertions": self.bound_assertions,
                "G4_integrality_certificates": self.integrality_certificates,
                "G5_integer_audits": self.integer_audits,
            },
        }
        if search_complete:
            payload["columns_journal_sha256"] = _file_sha256(self.journal_path)
            payload["node_ledger_sha256"] = _file_sha256(self.ledger_path)
        return payload

    def _checkpoint(self, search_complete=False):
        flush_and_fsync(self.journal)
        flush_and_fsync(self.ledger)
        payload = self._payload(search_complete)
        atomic_write_json(self.out, payload)
        return payload

    def close(self):
        for handle in (getattr(self, "journal", None),
                       getattr(self, "ledger", None)):
            if handle is not None and not handle.closed:
                handle.close()
