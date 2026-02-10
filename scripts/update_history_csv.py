#!/usr/bin/env python3
"""Incrementally update history.csv with only newly published lotto rounds."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lottogogo.data.fetcher import LottoHistoryFetcher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append only new lotto rounds to history.csv (no full backfill by default)."
    )
    parser.add_argument("--csv", default="history.csv", help="Path to history CSV file.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel fetch workers.")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout seconds.")
    parser.add_argument("--max-retries", type=int, default=2, help="HTTP retry count.")
    parser.add_argument("--retry-delay", type=float, default=0.4, help="HTTP retry base delay seconds.")
    parser.add_argument(
        "--verify-ssl",
        action="store_true",
        help="Enable TLS certificate verification for dhlottery endpoint.",
    )
    parser.add_argument(
        "--allow-bootstrap",
        action="store_true",
        help="Allow fetching from round 1 when CSV is missing or empty.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)

    if not csv_path.exists() and not args.allow_bootstrap:
        raise SystemExit(
            f"{csv_path} does not exist. Refusing full bootstrap. "
            "Restore the file or run with --allow-bootstrap."
        )

    before_round = LottoHistoryFetcher.latest_round_from_history_csv(csv_path)
    if before_round <= 0 and not args.allow_bootstrap:
        raise SystemExit(
            f"{csv_path} has no valid round data. Refusing full bootstrap. "
            "Fix the CSV or run with --allow-bootstrap."
        )

    fetcher = LottoHistoryFetcher(
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        verify_ssl=args.verify_ssl,
    )
    merged = fetcher.update_history_csv(csv_path, workers=args.workers)

    after_round = LottoHistoryFetcher.latest_round_from_history_csv(csv_path)
    added = max(0, after_round - before_round)

    print(f"history path   : {csv_path}")
    print(f"round before   : {before_round}")
    print(f"round after    : {after_round}")
    print(f"added rounds   : {added}")
    print(f"total rows     : {len(merged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
