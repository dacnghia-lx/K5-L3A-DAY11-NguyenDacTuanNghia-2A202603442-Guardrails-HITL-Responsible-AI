"""
Assignment 11 — Monitoring & Alerts starter (TODO).

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def default_metrics_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0

    # Bộ đếm phụ (không bắt buộc) — chi tiết theo lớp chặn
    input_blocks: int = 0
    output_redactions: int = 0
    output_withheld: int = 0
    llm_errors: int = 0

    def record(self, *, blocked: bool, layer: str | None = None) -> None:
        """Cập nhật bộ đếm sau mỗi request (gọi từ pipeline)."""
        self.total_requests += 1
        if blocked:
            self.blocked_requests += 1
        if layer == "rate_limiter":
            self.rate_limit_hits += 1
        elif layer == "input_guardrail":
            self.input_blocks += 1
        elif layer == "output_guardrail":
            if blocked:
                self.output_withheld += 1
            else:
                self.output_redactions += 1
        elif layer == "llm_error":
            self.llm_errors += 1

    def record_judge(self, *, passed: bool) -> None:
        self.judge_checks += 1
        if not passed:
            self.judge_fails += 1

    def check_metrics(self) -> list[Alert]:
        """Compute rates, append Alert objects when thresholds exceeded.

        Mỗi metric chỉ phát một alert (không nhân bản khi gọi nhiều lần).
        """
        snap = self.snapshot()
        existing = {a.metric for a in self.alerts}
        new_alerts: list[Alert] = []

        def _fire(metric: str, value: float, threshold: float, message: str) -> None:
            if metric in existing:
                return
            alert = Alert(metric=metric, value=value, threshold=threshold, message=message)
            self.alerts.append(alert)
            new_alerts.append(alert)

        if self.total_requests and snap["block_rate"] > self.block_rate_threshold:
            _fire(
                "block_rate",
                round(snap["block_rate"], 4),
                self.block_rate_threshold,
                f"Block rate {snap['block_rate']:.0%} > {self.block_rate_threshold:.0%} "
                "— possible attack campaign or over-blocking (check false positives).",
            )
        if self.rate_limit_hits >= self.rate_limit_hit_threshold:
            _fire(
                "rate_limit_hits",
                float(self.rate_limit_hits),
                float(self.rate_limit_hit_threshold),
                f"{self.rate_limit_hits} rate-limit hits (≥ {self.rate_limit_hit_threshold}) "
                "— possible flooding / cost attack.",
            )
        if self.judge_checks and snap["judge_fail_rate"] > self.judge_fail_rate_threshold:
            _fire(
                "judge_fail_rate",
                round(snap["judge_fail_rate"], 4),
                self.judge_fail_rate_threshold,
                f"Judge fail rate {snap['judge_fail_rate']:.0%} > "
                f"{self.judge_fail_rate_threshold:.0%} — model output quality/safety degraded.",
            )
        if self.llm_errors and self.total_requests and (
            self.llm_errors / self.total_requests > 0.2
        ):
            _fire(
                "llm_error_rate",
                round(self.llm_errors / self.total_requests, 4),
                0.2,
                "LLM provider errors > 20% — check API key / quota / provider status.",
            )
        return new_alerts

    def export_json(self, filepath: str | None = None):
        """Write metrics + alerts to JSON under repo-root ``outputs/`` by default."""
        self.check_metrics()
        path = Path(filepath or default_metrics_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.snapshot()
        payload["by_layer"] = {
            "rate_limiter": self.rate_limit_hits,
            "input_guardrail": self.input_blocks,
            "output_redacted": self.output_redactions,
            "output_withheld": self.output_withheld,
            "llm_error": self.llm_errors,
        }
        payload["thresholds"] = {
            "block_rate": self.block_rate_threshold,
            "rate_limit_hits": self.rate_limit_hit_threshold,
            "judge_fail_rate": self.judge_fail_rate_threshold,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
