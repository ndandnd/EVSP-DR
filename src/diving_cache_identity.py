"""Identity-verified reuse of an event-network pickle across two commits.

The production loader (``exact_pricer_expanded._load_event_network_cache``)
compares the whole recorded identity dictionary, including ``git_commit``.  A
cache produced at one commit therefore cannot be loaded by a later commit even
when graph construction is byte-identical, and rebuilding a k=8 event graph
costs 2,520-7,279 s (``outputs/cumulative_budget_20260913/audit/budgets.csv``).

This module keeps the strict check as the default and adds one explicit,
recorded bridge.  The bridge never weakens the substantive guarantees:

* every non-``git_commit`` identity field must match exactly;
* the pickle sha256 must match the producer manifest;
* ``network.metrics()`` recomputed under the consumer commit must equal the
  metrics recorded by the producer (node count, arc count, arc mode, packed
  arc bytes and the event-lattice sha256);
* every method of ``EventExpandedNetwork`` that participates in graph
  construction or pricing must have identical source text under both commits.

Only the last check is new.  It mirrors, programmatically, the manual audit
already recorded in
``outputs/cumulative_budget_20260913/cache_compatibility.json``.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import time
from pathlib import Path


CACHE_BRIDGE_SCHEMA = "evsp-dr-diving-cache-bridge-v1"

# Methods whose source must be identical between the producing and consuming
# commit.  Graph construction plus every pricing/replay entry point used by the
# diving pilot.  ``__init__``/``__getstate__``/``__setstate__`` are excluded on
# purpose: they are the constructor/runtime-index plumbing that the recorded
# campaign audit already classified as non-graph, and the pickle hash plus the
# metrics comparison cover what they could affect.
GRAPH_CRITICAL_METHODS = (
    "_action_recipe",
    "_add",
    "_build_arcs",
    "_build_nodes",
    "_charge_arcs",
    "_charge_candidates",
    "_direct_arcs",
    "_direct_candidates",
    "_edge_action",
    "_finalize_source",
    "_iter_arcs",
    "_min_reduced_cost_route_lazy",
    "_record",
    "_source_candidates",
    "_split_arcs",
    "_walk",
    "metrics",
    "min_reduced_cost_route",
    "sink_predecessor_route_batch",
)

GRAPH_SOURCE_FILES = (
    "src/event_pricer_network.py",
    "src/audit_giro_known_columns.py",
    "src/expanded_path_realization.py",
)


class CacheIdentityError(RuntimeError):
    """The requested cache cannot be trusted under this commit."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_show(repo: Path, commit: str, relative_path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=repo, capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise CacheIdentityError(
            f"cannot read {relative_path} at {commit}: "
            f"{result.stderr.decode('utf-8', 'replace').strip()}"
        )
    return result.stdout


def _method_sources(module_text: str, class_name: str) -> dict:
    """Return ``{method_name: dedented source}`` for one class in a file."""

    import ast
    import textwrap

    tree = ast.parse(module_text)
    lines = module_text.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            output = {}
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = (item.decorator_list[0].lineno - 1
                             if item.decorator_list else item.lineno - 1)
                    body = "\n".join(lines[start:item.end_lineno])
                    output[item.name] = textwrap.dedent(body)
            return output
    raise CacheIdentityError(
        f"class {class_name} is absent from the compared source"
    )


def compare_graph_methods(repo: Path, producer_commit: str) -> dict:
    """Compare ``EventExpandedNetwork`` methods across producer and HEAD."""

    from event_pricer_network import EventExpandedNetwork

    producer_text = _git_show(
        repo, producer_commit, "src/event_pricer_network.py"
    ).decode("utf-8")
    producer_methods = _method_sources(producer_text, "EventExpandedNetwork")

    differing = []
    missing = []
    for name in GRAPH_CRITICAL_METHODS:
        consumer = getattr(EventExpandedNetwork, name, None)
        if consumer is None or name not in producer_methods:
            missing.append(name)
            continue
        import textwrap
        consumer_source = textwrap.dedent(
            inspect.getsource(consumer)
        ).strip()
        if consumer_source != producer_methods[name].strip():
            differing.append(name)
    return {
        "graph_critical_methods": list(GRAPH_CRITICAL_METHODS),
        "missing_methods": missing,
        "differing_methods": differing,
        "identical": not missing and not differing,
        "producer_source_sha256": {
            path: _sha256_bytes(_git_show(repo, producer_commit, path))
            for path in GRAPH_SOURCE_FILES
        },
        "consumer_source_sha256": {
            path: file_sha256(repo / path) for path in GRAPH_SOURCE_FILES
        },
    }


def load_verified_network(
    cache_path: Path,
    expected_identity: dict,
    *,
    repo: Path,
    allow_commit_bridge: bool = False,
    verify_pickle_sha256: bool = True,
):
    """Load a cached ``EventExpandedNetwork`` under a recorded identity audit.

    ``expected_identity`` must be produced by
    ``exact_pricer_expanded._event_network_cache_identity`` for *this* run.
    Returns ``(network, audit)``.
    """

    import pickle

    from event_pricer_network import EventExpandedNetwork
    from exact_pricer_expanded import _event_network_cache_manifest_path

    cache_path = Path(cache_path).expanduser().resolve()
    manifest_path = _event_network_cache_manifest_path(cache_path)
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        raise CacheIdentityError(
            f"event-network cache manifest is unavailable: {manifest_path}"
        ) from exc

    recorded = manifest.get("identity")
    if not isinstance(recorded, dict):
        raise CacheIdentityError(
            f"event-network cache manifest has no identity: {manifest_path}"
        )

    audit = {
        "schema": CACHE_BRIDGE_SCHEMA,
        "cache_path": str(cache_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "recorded_identity": recorded,
        "expected_identity": expected_identity,
        "commit_bridge_requested": bool(allow_commit_bridge),
        "commit_bridge_used": False,
        "pickle_sha256_verified": bool(verify_pickle_sha256),
    }

    differing = sorted(
        key for key in set(recorded) | set(expected_identity)
        if recorded.get(key) != expected_identity.get(key)
    )
    audit["differing_identity_fields"] = differing

    if differing and differing != ["git_commit"]:
        raise CacheIdentityError(
            f"event-network cache identity mismatch on {differing}: "
            f"{manifest_path}"
        )
    if differing == ["git_commit"]:
        if not allow_commit_bridge:
            raise CacheIdentityError(
                "event-network cache was produced at commit "
                f"{recorded.get('git_commit')} but this run is "
                f"{expected_identity.get('git_commit')}; pass "
                "--cache-commit-bridge to request a recorded source audit"
            )
        method_audit = compare_graph_methods(
            repo, str(recorded.get("git_commit"))
        )
        audit["method_audit"] = method_audit
        if not method_audit["identical"]:
            raise CacheIdentityError(
                "event-network cache commit bridge refused: graph-critical "
                "methods differ between "
                f"{recorded.get('git_commit')} and HEAD: "
                f"missing={method_audit['missing_methods']} "
                f"differing={method_audit['differing_methods']}"
            )
        audit["commit_bridge_used"] = True

    started = time.time()
    if verify_pickle_sha256:
        observed = file_sha256(cache_path)
        audit["observed_pickle_sha256"] = observed
        if observed != manifest.get("pickle_sha256"):
            raise CacheIdentityError(
                f"event-network cache hash mismatch: {cache_path}"
            )
    audit["hash_s"] = time.time() - started

    started = time.time()
    with cache_path.open("rb") as handle:
        network = pickle.load(handle)
    audit["unpickle_s"] = time.time() - started
    if not isinstance(network, EventExpandedNetwork):
        raise CacheIdentityError(
            f"event-network cache has unexpected object type: {cache_path}"
        )
    metrics = network.metrics()
    audit["network_metrics"] = metrics
    if metrics != manifest.get("network_metrics"):
        raise CacheIdentityError(
            f"event-network cache metrics mismatch: {cache_path}"
        )
    audit["pickle_bytes"] = cache_path.stat().st_size
    audit["original_build_s"] = manifest.get("original_build_s")
    audit["cache_io_s"] = audit["hash_s"] + audit["unpickle_s"]
    return network, audit
