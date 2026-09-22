"""Collect this specific read-only review without exporting private reasoning."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess

BASE = Path(__file__).resolve().parent
SESSION = "701b774a-2ebe-4d6f-93fa-27c55cb9943b"
receipt = json.loads((BASE / "model_receipt.json").read_text())
process = subprocess.run(
    ["/Users/nadan/.local/bin/claude", "agents", "--json", "--all"],
    capture_output=True, text=True, check=True,
)
matching = [x for x in json.loads(process.stdout) if x.get("sessionId") == SESSION]
status = {
    "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "session": matching,
    "report_exported": False,
}
messages = []
for filename in receipt["transcript_paths"]:
    for line in Path(filename).read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue  # Last append may still be incomplete.
        if row.get("type") != "assistant":
            continue
        message = row.get("message", {})
        blocks = message.get("content", [])
        if any(x.get("type") == "tool_use" for x in blocks):
            continue
        text = "\n\n".join(x["text"] for x in blocks if x.get("type") == "text")
        if text:
            messages.append((text, message.get("model")))

# Require positive session-state evidence of idle/completed, never infer it
# merely because the agent disappeared from a process list.
done = matching and all(x.get("status") in {"idle", "completed", "exited"}
                        for x in matching)
if done and messages and len(messages[-1][0]) > 1000:
    report, model = messages[-1]
    if model != "claude-opus-5-5":
        raise RuntimeError(f"Unexpected model: {model}")
    (BASE / "REVIEW.md").write_text(report + "\n")
    status.update(report_exported=True, model=model,
                  sha256=hashlib.sha256((report + "\n").encode()).hexdigest())
else:
    status["note"] = "Review still running, unavailable, or needs inspection before export."
(BASE / "status.json").write_text(json.dumps(status, indent=2) + "\n")
print(json.dumps(status, indent=2))
