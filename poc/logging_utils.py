"""Consistent console, file, and progress logging for long-running stages."""

from __future__ import annotations

import logging
import re
import selectors
import shlex
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

LOGGER_NAME = "rhino_poc"


def configure_logging(log_file: Path | None = None, verbose: bool = False) -> logging.Logger:
    """Configure one concise console handler and an optional detailed file handler."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    )
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        detailed = logging.FileHandler(log_file, encoding="utf-8")
        detailed.setLevel(logging.DEBUG)
        detailed.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(detailed)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def format_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


@dataclass
class ProgressReporter:
    """Rate-limited progress messages with elapsed time and ETA."""

    label: str
    total: int | None = None
    interval_seconds: float = 15.0

    def __post_init__(self) -> None:
        self.started_at = time.monotonic()
        self.last_report_at = 0.0
        self.current = 0

    def update(self, current: int, *, force: bool = False, detail: str = "") -> None:
        self.current = current
        now = time.monotonic()
        if not force and now - self.last_report_at < self.interval_seconds:
            return
        self.last_report_at = now
        elapsed = now - self.started_at
        suffix = f" | {detail}" if detail else ""
        if self.total:
            percent = min(100.0, 100.0 * current / self.total)
            eta = elapsed / current * (self.total - current) if current else 0.0
            get_logger().info(
                "%s | %d/%d (%.0f%%) | elapsed %s | ETA %s%s",
                self.label,
                current,
                self.total,
                percent,
                format_duration(elapsed),
                format_duration(max(0.0, eta)),
                suffix,
            )
        else:
            get_logger().info(
                "%s | %d | elapsed %s%s", self.label, current, format_duration(elapsed), suffix
            )

    def heartbeat(self, detail: str = "") -> None:
        self.update(self.current, detail=detail)

    def finish(self, detail: str = "") -> None:
        if self.total is not None:
            self.current = self.total
        self.update(self.current, force=True, detail=detail or "complete")


ProgressProbe = Callable[[], tuple[int, int | None, str] | None]


def run_command(
    command: Sequence[str],
    *,
    stage: str,
    raw_log_file: Path,
    progress_probe: ProgressProbe | None = None,
    progress_patterns: Sequence[re.Pattern[str]] = (),
    heartbeat_seconds: float = 15.0,
) -> None:
    """Run a subprocess while streaming raw output and emitting periodic progress.

    A selector is used instead of a blocking ``readline`` loop. Progress therefore
    remains visible even when COLMAP produces no output for several minutes.
    """
    logger = get_logger()
    raw_log_file.parent.mkdir(parents=True, exist_ok=True)
    quoted = shlex.join([str(part) for part in command])
    logger.info("%s | started", stage)
    logger.debug("%s | command: %s", stage, quoted)

    started_at = time.monotonic()
    last_heartbeat = 0.0
    reporter = ProgressReporter(stage, interval_seconds=heartbeat_seconds)
    with raw_log_file.open("a", encoding="utf-8") as raw_log:
        raw_log.write(f"\n$ {quoted}\n")
        raw_log.flush()
        process = subprocess.Popen(
            [str(part) for part in command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)

        while process.poll() is None:
            events = selector.select(timeout=0.5)
            for key, _ in events:
                line = key.fileobj.readline()
                if not line:
                    continue
                raw_log.write(line)
                raw_log.flush()
                logger.debug("%s | %s", stage, line.rstrip())
                for pattern in progress_patterns:
                    match = pattern.search(line)
                    if match:
                        reporter.total = int(match.group("total"))
                        reporter.update(int(match.group("current")))
                        break

            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_seconds:
                last_heartbeat = now
                probed = progress_probe() if progress_probe else None
                if probed:
                    current, total, detail = probed
                    reporter.total = total
                    reporter.update(current, force=True, detail=detail)
                else:
                    reporter.update(
                        reporter.current,
                        force=True,
                        detail=f"running for {format_duration(now - started_at)}",
                    )

        for line in process.stdout:
            raw_log.write(line)
            logger.debug("%s | %s", stage, line.rstrip())
        return_code = process.wait()

    elapsed = time.monotonic() - started_at
    if return_code != 0:
        logger.error(
            "%s | failed with exit code %d after %s | details: %s",
            stage,
            return_code,
            format_duration(elapsed),
            raw_log_file,
        )
        raise subprocess.CalledProcessError(return_code, command)
    reporter.finish(detail=f"completed in {format_duration(elapsed)}")
