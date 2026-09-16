"""Rebuild the current dashboard from the audited, human-reviewed source.
Edit the current HTML and its cited evidence together; historical snapshots remain unchanged.
"""
from pathlib import Path
import json, hashlib
root = Path(__file__).resolve().parents[2]
source = root / "outputs/independent_review_20260916/execution/doc/current.html"
out = Path(__file__).resolve().parent
(out / "dashboard.html").write_bytes(source.read_bytes())
(out / "evidence.json").write_text(json.dumps({
    "current_source": str(source.relative_to(root)),
    "current_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "audit": "outputs/independent_review_20260916/execution/README.md",
    "chain_audit": "outputs/independent_review_20260916/execution/audited_chain_results.csv",
    "original_chain_snapshot": "20260916T194843Z",
    "status": "P0 verified/refuted findings; new campaign outcomes require independent checks",
}, indent=2) + "\n")
print("Dashboard synchronized with independent-review corrections.")
