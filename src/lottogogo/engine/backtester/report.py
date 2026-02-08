"""Backtest report generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .metrics import compare_summaries, summarize_results
from .walk_forward import RoundBacktestResult


@dataclass(frozen=True)
class BacktestReport:
    """Combined JSON and Markdown backtest report."""

    json_report: dict[str, Any]
    markdown_report: str


def generate_backtest_report(
    *,
    strategy_results: list[RoundBacktestResult],
    baseline_results: list[RoundBacktestResult] | None = None,
    config_snapshot: dict[str, Any] | None = None,
) -> BacktestReport:
    """Generate report payloads for programmatic and human consumption."""
    strategy_summary = summarize_results(strategy_results)
    baseline_summary = summarize_results(baseline_results or [])
    comparison = compare_summaries(strategy_summary, baseline_summary)

    report_json: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy_summary,
        "baseline": baseline_summary,
        "comparison": comparison,
        "config_snapshot": config_snapshot or {},
    }
    markdown = _to_markdown(report_json)
    return BacktestReport(json_report=report_json, markdown_report=markdown)


def _to_markdown(report_json: dict[str, Any]) -> str:
    strategy = report_json["strategy"]
    baseline = report_json["baseline"]
    comparison = report_json["comparison"]

    return "\n".join(
        [
            "## Backtest Summary",
            "",
            f"- Generated At: {report_json['generated_at']}",
            f"- Total Rounds: {strategy['total_rounds']}",
            f"- Total Recommendations: {strategy['total_recommendations']}",
            "",
            "## Strategy Metrics",
            f"- P(match>=3): {strategy['p_match_ge_3']:.4f}",
            f"- P(match>=4): {strategy['p_match_ge_4']:.4f}",
            f"- Avg Match Count: {strategy['average_match_count']:.4f}",
            f"- Std Match Count: {strategy['std_match_count']:.4f}",
            "",
            "## Baseline Metrics",
            f"- P(match>=3): {baseline['p_match_ge_3']:.4f}",
            f"- P(match>=4): {baseline['p_match_ge_4']:.4f}",
            f"- Avg Match Count: {baseline['average_match_count']:.4f}",
            "",
            "## Comparison",
            f"- Δ P(match>=3): {comparison['p_match_ge_3_diff']:.4f}",
            f"- Δ P(match>=4): {comparison['p_match_ge_4_diff']:.4f}",
            f"- Δ Avg Match Count: {comparison['average_match_count_diff']:.4f}",
        ]
    )

