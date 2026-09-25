"""
Persistent, append-only log of every decision the agent makes — what the
Ampcome article calls the audit trail that becomes "a learning resource that
continuously improves the governance rules." Each entry is one JSON line.
"""
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_PATH = Path(__file__).parent.parent.parent.parent / "outputs" / "agent_audit_log.jsonl"


@dataclass
class AuditLog:
    path: Path = field(default_factory=lambda: _DEFAULT_PATH)
    entries: list = field(default_factory=list)

    def record(self, phase: str, action: str, detail: str, **extra) -> dict:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "action": action,
            "detail": detail,
            **extra,
        }
        self.entries.append(entry)
        self._append_to_disk(entry)
        return entry

    def _append_to_disk(self, entry: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def tail(self, n: int = 20) -> list:
        return self.entries[-n:]
