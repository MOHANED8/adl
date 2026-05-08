from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class JsonlLogger:
    """Append structured records to a UTF-8 JSONL file."""

    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("ts", time.time())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class NullLogger:
    """No-op structured logger used for disabled outputs."""

    def log(self, record: dict[str, Any]) -> None:
        return


def make_logger(out_dir: str | Path, name: str = "events.jsonl") -> JsonlLogger:
    """Create a JSONL logger inside the experiment output directory."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return JsonlLogger(out_dir / name)


def configure_logging(level: str = "INFO") -> None:
    """Configure process-wide console logging once."""

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=False,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a standard library logger for the given module name."""

    return logging.getLogger(name)

