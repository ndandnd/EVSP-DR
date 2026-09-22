"""Independent worker-control tests; no graph, solver, SSH or scheduler call."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(sys.argv[1]).resolve() / "scripts"))
import validate_strict_graph_artifact as gate


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    (root / "model_code").mkdir()
    (root / "wrapper_source.tar").write_bytes(b"mock-wrapper")
    (root / "fedf_since_357.bundle").write_bytes(b"mock-bundle")
    proof = root / "lineage" / "gate_result.json"
    put(proof, {"status": "passed", "all_routes_physically_replayed": 8397,
                "inherited_routes_replayed": 8343})
    manifest = {"wrapper_commit": "mock-review", "wrapper_source_sha256": {},
                "wrapper_archive_sha256": gate.sha(root / "wrapper_source.tar"),
                "model_bundle_sha256": gate.sha(root / "fedf_since_357.bundle"),
                "native_lineage_attempt": str(proof.parent),
                "native_lineage_sha256": {"gate_result.json": gate.sha(proof)},
                "minimum_disk_free_bytes": 0,
                "limits": {"cold_process_s": 22500, "reload_process_s": 4800}}
    put(root / "manifest.json", manifest)
    calls = []

    def phase(command, **kwargs):
        name = command[command.index("--phase") + 1]
        output = Path(command[command.index("--out") + 1])
        attempt = root / "attempts" / (os.environ["SLURM_JOB_ID"] + "_r0")
        assert (attempt / "commands.json").exists(), "commands must predate execution"
        calls.append(name)
        if name == "cold":
            (root / "artifacts" / "k19_fedf4214.graph.cache").write_bytes(b"mock-cache")
            put(output, {"schema": gate.SCHEMA, "phase": "cold", "status": "passed",
                         "manifest_sha256": gate.sha(root / "manifest.json"),
                         "wrapper_commit": manifest["wrapper_commit"],
                         "model_provenance": {"git_commit": gate.MODEL_COMMIT},
                         "initial_pool": {"initial_pool_columns": 8397}})
        else:
            assert (root / "artifacts" / "COLD_COMPLETE.json").exists()
            put(output, {"cold_reload_parity": True})
        return SimpleNamespace(returncode=0)

    with patch.object(gate.subprocess, "run", side_effect=phase), patch.dict(os.environ, {"SLURM_JOB_ID": "first", "SLURM_RESTART_COUNT": "0"}):
        gate.worker(root, manifest)
    assert calls == ["cold", "reload"]
    assert (root / "attempts" / "first_r0" / "COMPLETE.json").exists()
    calls.clear()
    with patch.object(gate.subprocess, "run", side_effect=phase), patch.dict(os.environ, {"SLURM_JOB_ID": "resumed", "SLURM_RESTART_COUNT": "0"}):
        gate.worker(root, manifest)
    assert calls == ["reload"]
    calls.clear()
    (root / "artifacts" / "cold.json").write_text("{}")
    with patch.object(gate.subprocess, "run", side_effect=phase), patch.dict(os.environ, {"SLURM_JOB_ID": "corrupt", "SLURM_RESTART_COUNT": "0"}):
        try:
            gate.worker(root, manifest)
        except ValueError as error:
            assert "seal mismatch" in str(error)
        else:
            raise AssertionError("corrupt cold baseline accepted")
    assert calls == []
    assert (root / "attempts" / "corrupt_r0" / "FAILED.json").exists()
print(json.dumps({"status": "passed", "checks": ["commands before sequential cold/reload", "sealed resume reload-only", "corrupt baseline fails before subprocess"], "scope": "mock worker orchestration; no production graph validation"}, indent=2))
