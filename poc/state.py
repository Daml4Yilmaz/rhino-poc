"""Atomic case manifest and conservative stage-resume semantics."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def capture_fingerprint(capture_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in ("odometry.csv", "camera_matrix.csv", "rgb.mp4"):
        path = capture_dir / name
        stat = path.stat()
        digest.update(name.encode())
        digest.update(str(stat.st_size).encode())
        if name != "rgb.mp4":
            digest.update(path.read_bytes())
    for folder in ("depth", "confidence"):
        paths = sorted((capture_dir / folder).glob("*.png"))
        digest.update(f"{folder}:{len(paths)}".encode())
        if paths:
            digest.update(paths[0].name.encode())
            digest.update(paths[-1].name.encode())
    return digest.hexdigest()


def stage_signature(
    stage: str, capture_hash: str, parameters: dict[str, Any], dependency: str = ""
) -> str:
    payload = {
        "stage": stage,
        "capture": capture_hash,
        "parameters": parameters,
        "dependency": dependency,
        "software_version": __version__,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class CaseManifest:
    def __init__(self, path: Path, capture_dir: Path):
        self.path = path
        self.capture_dir = capture_dir.resolve()
        self.capture_hash = capture_fingerprint(self.capture_dir)
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            previous_hash = self.data.get("capture", {}).get("fingerprint")
            if previous_hash and previous_hash != self.capture_hash:
                raise RuntimeError(
                    "The output directory belongs to a different or modified capture. Use a new "
                    "output directory instead of mixing case data."
                )
        else:
            self.data = {
                "schema_version": 1,
                "software_version": __version__,
                "created_at": _utc_now(),
                "capture": {
                    "path": str(self.capture_dir),
                    "fingerprint": self.capture_hash,
                },
                "stages": {},
            }
            self._write()

    def _write(self) -> None:
        self.data["updated_at"] = _utc_now()
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    def previous_signature(self, stage: str) -> str:
        return self.data.get("stages", {}).get(stage, {}).get("signature", "")

    def is_current(self, stage: str, signature: str, outputs: list[Path]) -> bool:
        record = self.data.get("stages", {}).get(stage, {})
        return (
            record.get("status") == "complete"
            and record.get("signature") == signature
            and all(path.exists() for path in outputs)
        )

    def has_stale_record(self, stage: str, signature: str) -> bool:
        record = self.data.get("stages", {}).get(stage)
        return bool(record and record.get("signature") != signature)

    def start(self, stage: str, signature: str, parameters: dict[str, Any]) -> None:
        self.data["stages"][stage] = {
            "status": "running",
            "signature": signature,
            "parameters": parameters,
            "started_at": _utc_now(),
        }
        self._write()

    def complete(self, stage: str, metadata: dict[str, Any] | None = None) -> None:
        record = self.data["stages"][stage]
        record["status"] = "complete"
        record["completed_at"] = _utc_now()
        if metadata:
            record["metadata"] = metadata
        self._write()

    def fail(self, stage: str, error: Exception) -> None:
        record = self.data["stages"].setdefault(stage, {})
        record["status"] = "failed"
        record["failed_at"] = _utc_now()
        record["error"] = str(error)
        self._write()
