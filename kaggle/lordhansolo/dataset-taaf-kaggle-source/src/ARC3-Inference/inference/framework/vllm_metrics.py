"""Periodic scrape of the vLLM Prometheus endpoint into a run-level JSONL.

One record per server per tick, written to ``<job_dir>/metrics/vllm_server.jsonl``
(deliberately outside the per-game ``*_metrics.jsonl`` pattern) so it can be joined
by ``ts`` with the per-game metrics the agent and solver write.
Every ``vllm:``-prefixed series is kept (summed across label sets, histogram
buckets dropped, ``_sum``/``_count`` retained), which sidesteps version-specific
metric names: consumers pick what they need. Scraping is pure observability, so
every failure is logged and swallowed.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

log = logging.getLogger(__name__)

VLLM_METRICS_FILENAME = "vllm_server.jsonl"
VLLM_METRICS_SCRAPE_INTERVAL_SECONDS = 15.0
_SCRAPE_TIMEOUT_SECONDS = 5.0
_SERIES_PREFIX = "vllm:"


@dataclass(frozen=True)
class VllmMetricsTarget:
    server_index: int
    metrics_url: str
    api_key: str = ""


def metrics_url_from_base_url(base_url: str) -> str:
    """``http://host:port/v1`` -> ``http://host:port/metrics``."""
    root = base_url.strip().rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]

    return f"{root}/metrics"


def parse_vllm_series(exposition_text: str) -> dict[str, float]:
    """Sum every ``vllm:`` series across its label sets, keyed by the series name
    without the prefix. Histogram ``_bucket`` samples are skipped."""
    totals: dict[str, float] = {}
    for line in exposition_text.splitlines():
        if not line.startswith(_SERIES_PREFIX):
            continue
        label_start = line.find("{")
        if label_start != -1:
            name = line[len(_SERIES_PREFIX):label_start]
            value_text = line[line.rfind("}") + 1:].split()
        else:
            name, _, remainder = line[len(_SERIES_PREFIX):].partition(" ")
            value_text = remainder.split()
        if not name or name.endswith("_bucket") or not value_text:
            continue
        try:
            value = float(value_text[0])
        except ValueError:
            continue
        totals[name] = totals.get(name, 0.0) + value

    return totals


@dataclass
class VllmMetricsScraper:
    targets: list[VllmMetricsTarget]
    output_path: Path
    interval_seconds: float = VLLM_METRICS_SCRAPE_INTERVAL_SECONDS
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _failed_targets: set[int] = field(default_factory=set, init=False, repr=False)

    def start(self) -> None:
        if self._thread is not None or not self.targets:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="vllm-metrics-scraper", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=_SCRAPE_TIMEOUT_SECONDS + 1.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.scrape_once()
            self._stop_event.wait(self.interval_seconds)

    def scrape_once(self) -> None:
        for target in self.targets:
            record = self._scrape_target(target)
            if record is not None:
                self._append_record(record)

    def _scrape_target(self, target: VllmMetricsTarget) -> dict[str, object] | None:
        headers = {"Authorization": f"Bearer {target.api_key}"} if target.api_key else {}
        started_at = time.monotonic()
        try:
            response = requests.get(target.metrics_url, headers=headers, timeout=_SCRAPE_TIMEOUT_SECONDS)
            response.raise_for_status()
            series = parse_vllm_series(response.text)
        except Exception as exc:  # noqa: BLE001
            if target.server_index not in self._failed_targets:
                log.warning("vLLM metrics scrape failed for %s: %s", target.metrics_url, exc)
                self._failed_targets.add(target.server_index)

            return None
        if target.server_index in self._failed_targets:
            log.warning("vLLM metrics scrape recovered for %s", target.metrics_url)
            self._failed_targets.discard(target.server_index)

        return {
            "ts": round(time.time(), 3),
            "server_index": target.server_index,
            "scrape_seconds": round(time.monotonic() - started_at, 3),
            "metrics": series,
        }

    def _append_record(self, record: dict[str, object]) -> None:
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, "a", encoding="utf-8") as output_file:
                output_file.write(json.dumps(record, ensure_ascii=True))
                output_file.write("\n")
        except Exception as exc:  # noqa: BLE001
            log.warning("vLLM metrics write failed: %s", exc)
