#!/usr/bin/env python3
"""Copy this check's explicit, small evidence allowlist into the evidence worktree.

Does not stage, commit, push, remove files, or copy binaries/raw route journals.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / ".codex-work/week-evidence-20260921"
TREES = [
    "outputs/research_management_20260922/monitor_20260922T155736Z",
    "outputs/research_management_20260922/strict_graph_reuse/production_gate",
]
EXPLICIT = [
    "outputs/research_management_20260922/strict_graph_reuse/native_preflight/README.md",
    "outputs/research_register/entries/20260922_1558_baseline_and_native_lineage.md",
    "outputs/research_register/README.md",
    "outputs/research_register/ACTIVE_MONITOR_PLAN.md",
    "outputs/research_management_20260921/MANAGER_PLAN.md",
    "outputs/research_management_20260921/MONITOR_CONTINUATION.md",
]
EXTENSIONS = {".md", ".json", ".csv", ".py", ".log", ".txt", ".out", ".err", ".diff", ".html"}
PRIVATE = re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{25,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)")


def selected(path):
    return (path.is_file() and path.suffix in EXTENSIONS
            and not {"input_artifacts", "__pycache__"}.intersection(path.parts)
            and ".local." not in path.name)


def main():
    assert (DEST / ".git").exists(), "expected isolated evidence worktree"
    paths = {ROOT / name for name in EXPLICIT}
    for name in TREES:
        paths.update(p for p in (ROOT / name).rglob("*") if selected(p))
    records = []
    for path in sorted(paths):
        data = path.read_bytes()
        assert len(data) < 1_000_000, f"unexpected large evidence file: {path}"
        assert not PRIVATE.search(data), f"credential-pattern review needed: {path}"
        relative = path.relative_to(ROOT)
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        digest = hashlib.sha256(data).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        records.append({"path": str(relative), "bytes": len(data), "sha256": digest})
    receipt = {"files": records, "count": len(records), "bytes": sum(r["bytes"] for r in records),
               "destination": str(DEST), "binaries_and_raw_journals_excluded": True,
               "credential_pattern_hits": 0, "staged_or_committed": False}
    (Path(__file__).parent / "publication/evidence_allowlist.local.json").write_text(
        json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "files"}))


if __name__ == "__main__":
    main()
