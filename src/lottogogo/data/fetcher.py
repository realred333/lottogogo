"""HTTP fetcher for lotto winning-number data."""

from __future__ import annotations

import json
import ssl
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

API_URL = "https://dhlottery.co.kr/lt645/selectPstLt645Info.do"
HISTORY_COLUMNS = ["round", "n1", "n2", "n3", "n4", "n5", "n6"]


def _create_ssl_context(verify: bool = True) -> ssl.SSLContext:
    """Create SSL context with optional certificate verification."""
    if verify:
        return ssl.create_default_context()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class LottoFetchError(RuntimeError):
    """Raised when remote lotto data cannot be fetched or parsed."""


class LottoRoundNotFoundError(LottoFetchError):
    """Raised when a requested round is not available in remote response."""


@dataclass(frozen=True)
class LottoRoundResult:
    """One lotto round winning-number result."""

    round: int
    n1: int
    n2: int
    n3: int
    n4: int
    n5: int
    n6: int
    bonus: int
    draw_date: str

    def as_dict(self) -> dict[str, int | str]:
        """Convert result to dictionary."""
        return {
            "round": self.round,
            "n1": self.n1,
            "n2": self.n2,
            "n3": self.n3,
            "n4": self.n4,
            "n5": self.n5,
            "n6": self.n6,
            "bonus": self.bonus,
            "draw_date": self.draw_date,
        }

    def as_history_dict(self) -> dict[str, int]:
        """Convert to the history.csv schema (round + six numbers)."""
        return {
            "round": self.round,
            "n1": self.n1,
            "n2": self.n2,
            "n3": self.n3,
            "n4": self.n4,
            "n5": self.n5,
            "n6": self.n6,
        }


class LottoHistoryFetcher:
    """Fetch lotto winning-number data from the official endpoint."""

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        api_url: str = API_URL,
        max_retries: int = 2,
        retry_delay: float = 0.4,
        verify_ssl: bool = False,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be > 0.")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0.")
        if retry_delay < 0:
            raise ValueError("retry_delay must be >= 0.")
        self.timeout = float(timeout)
        self.api_url = api_url
        self.max_retries = int(max_retries)
        self.retry_delay = float(retry_delay)
        self._ssl_context = _create_ssl_context(verify=verify_ssl)

    def fetch_round(self, round_id: int) -> LottoRoundResult:
        """Fetch and parse one round."""
        if round_id <= 0:
            raise ValueError("round_id must be > 0.")
        payload = self._request_json(round_id=round_id)
        return self.parse_payload(payload, expected_round=round_id)

    def fetch_rounds(self, round_ids: list[int]) -> list[LottoRoundResult]:
        """Fetch multiple rounds in order."""
        return [self.fetch_round(round_id) for round_id in round_ids]

    def fetch_rounds_parallel(self, round_ids: list[int], *, workers: int = 8) -> list[LottoRoundResult]:
        """Fetch multiple rounds concurrently while preserving input order."""
        if workers <= 0:
            raise ValueError("workers must be > 0.")
        if not round_ids:
            return []
        if workers == 1 or len(round_ids) == 1:
            return self.fetch_rounds(round_ids)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(self.fetch_round, round_ids))

    def fetch_as_dataframe(self, round_ids: list[int]) -> pd.DataFrame:
        """Fetch multiple rounds and return normalized dataframe."""
        rows = [result.as_dict() for result in self.fetch_rounds(round_ids)]
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values("round").reset_index(drop=True)

    def fetch_latest_available_round(self, *, start_round: int) -> int:
        """Find latest available round by exponential + binary search.

        Returns ``start_round - 1`` when ``start_round`` is not available.
        """
        if start_round <= 0:
            raise ValueError("start_round must be > 0.")
        if not self._round_exists(start_round):
            return start_round - 1

        low = start_round
        step = 1
        while self._round_exists(low + step):
            low = low + step
            step *= 2

        upper = low + step
        left = low + 1
        right = upper - 1
        best = low
        while left <= right:
            mid = (left + right) // 2
            if self._round_exists(mid):
                best = mid
                left = mid + 1
            else:
                right = mid - 1
        return best

    def fetch_new_rounds_since(self, last_round: int, *, workers: int = 8) -> list[LottoRoundResult]:
        """Fetch rounds newer than ``last_round``."""
        if last_round < 0:
            raise ValueError("last_round must be >= 0.")
        start_round = last_round + 1
        latest_round = self.fetch_latest_available_round(start_round=start_round)
        if latest_round <= last_round:
            return []
        round_ids = list(range(start_round, latest_round + 1))
        return self.fetch_rounds_parallel(round_ids, workers=workers)

    @staticmethod
    def latest_round_from_history_csv(
        csv_path: str | Path,
        *,
        encoding: str = "utf-8",
    ) -> int:
        """Return max round from existing history CSV, or 0 if file is empty/missing."""
        path = Path(csv_path)
        if not path.exists():
            return 0
        frame = pd.read_csv(path, encoding=encoding)
        if frame.empty:
            return 0
        if "round" not in frame.columns:
            raise LottoFetchError("history csv must include 'round' column.")
        rounds = pd.to_numeric(frame["round"], errors="coerce")
        if rounds.isna().all():
            raise LottoFetchError("history csv has invalid 'round' values.")
        return int(rounds.max())

    def update_history_csv(
        self,
        csv_path: str | Path,
        *,
        workers: int = 8,
        encoding: str = "utf-8",
    ) -> pd.DataFrame:
        """Append only missing rounds to history CSV and return merged dataframe."""
        path = Path(csv_path)
        if path.exists():
            existing = pd.read_csv(path, encoding=encoding)
            if existing.empty:
                existing_core = pd.DataFrame(columns=HISTORY_COLUMNS)
            else:
                missing = [column for column in HISTORY_COLUMNS if column not in existing.columns]
                if missing:
                    raise LottoFetchError(f"history csv missing required columns: {missing}")
                existing_core = existing[HISTORY_COLUMNS].copy()
        else:
            existing_core = pd.DataFrame(columns=HISTORY_COLUMNS)

        last_round = self.latest_round_from_history_csv(path, encoding=encoding)
        new_rounds = self.fetch_new_rounds_since(last_round, workers=workers)
        if not new_rounds:
            merged = existing_core.copy()
            if "round" in merged.columns and not merged.empty:
                merged["round"] = pd.to_numeric(merged["round"], errors="raise").astype(int)
                merged = merged.sort_values("round").reset_index(drop=True)
            return merged

        new_frame = pd.DataFrame([result.as_history_dict() for result in new_rounds], columns=HISTORY_COLUMNS)
        merged = pd.concat([existing_core, new_frame], ignore_index=True)
        merged["round"] = pd.to_numeric(merged["round"], errors="raise").astype(int)
        for column in HISTORY_COLUMNS[1:]:
            merged[column] = pd.to_numeric(merged[column], errors="raise").astype(int)
        merged = merged.drop_duplicates(subset=["round"], keep="last").sort_values("round").reset_index(drop=True)

        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(path, index=False, encoding=encoding)
        return merged

    def _request_json(self, *, round_id: int) -> dict[str, Any]:
        params = urlencode({"srchLtEpsd": round_id})
        url = f"{self.api_url}?{params}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "LottoGoGo/0.1",
            },
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout, context=self._ssl_context) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    raw = response.read().decode(charset, errors="replace")
                break
            except (HTTPError, URLError, TimeoutError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise LottoFetchError(f"Failed to fetch round {round_id}: {exc}") from exc
                if self.retry_delay > 0:
                    time.sleep(self.retry_delay * (2**attempt))
        else:
            raise LottoFetchError(f"Failed to fetch round {round_id}: {last_error}")

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LottoFetchError(f"Invalid JSON response for round {round_id}.") from exc
        if not isinstance(payload, dict):
            raise LottoFetchError(f"Unexpected response shape for round {round_id}.")
        return payload

    @staticmethod
    def parse_payload(payload: dict[str, Any], expected_round: int | None = None) -> LottoRoundResult:
        """Parse endpoint payload.

        Some responses may contain ``resultCode``/``resultMessage`` as ``null`` while
        still returning valid data in ``data.list``. This parser trusts data presence first.
        """
        try:
            data = payload["data"]
            items = data["list"]
        except KeyError as exc:
            raise LottoRoundNotFoundError("Missing required fields: data.list") from exc

        if not isinstance(items, list) or not items:
            raise LottoRoundNotFoundError("Response does not include winning-number rows.")

        row = items[0]
        if not isinstance(row, dict):
            raise LottoFetchError("Invalid row shape in response.")

        try:
            round_id = int(row["ltEpsd"])
            result = LottoRoundResult(
                round=round_id,
                n1=int(row["tm1WnNo"]),
                n2=int(row["tm2WnNo"]),
                n3=int(row["tm3WnNo"]),
                n4=int(row["tm4WnNo"]),
                n5=int(row["tm5WnNo"]),
                n6=int(row["tm6WnNo"]),
                bonus=int(row["bnsWnNo"]),
                draw_date=str(row["ltRflYmd"]),
            )
        except KeyError as exc:
            raise LottoFetchError(f"Missing winning-number field: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise LottoFetchError("Winning-number field types are invalid.") from exc

        if expected_round is not None and result.round != expected_round:
            raise LottoRoundNotFoundError(
                f"Requested round {expected_round} but received round {result.round}."
            )

        return result

    def _round_exists(self, round_id: int) -> bool:
        try:
            self.fetch_round(round_id)
            return True
        except LottoRoundNotFoundError:
            return False
