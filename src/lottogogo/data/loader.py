"""CSV loader and validator for lotto history data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["round", "n1", "n2", "n3", "n4", "n5", "n6"]
NUMBER_COLUMNS = ["n1", "n2", "n3", "n4", "n5", "n6"]
ROUND_ALIASES = {"round_id": "round"}


class DataValidationError(ValueError):
    """Raised when lotto history data fails validation."""


class LottoHistoryLoader:
    """Load and validate lotto history CSV data."""

    def load_csv(self, path: str | Path, encoding: str = "utf-8") -> pd.DataFrame:
        """Load CSV and normalize column names."""
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        dataframe = pd.read_csv(csv_path, encoding=encoding)
        return self._normalize_columns(dataframe)

    def validate(self, dataframe: pd.DataFrame) -> None:
        """Validate schema, number range, and per-row uniqueness."""
        missing = [col for col in REQUIRED_COLUMNS if col not in dataframe.columns]
        if missing:
            raise DataValidationError(f"Missing required columns: {missing}")

        for column in REQUIRED_COLUMNS:
            try:
                pd.to_numeric(dataframe[column], errors="raise")
            except Exception as exc:
                raise DataValidationError(f"Column '{column}' must be numeric.") from exc

        numeric_df = dataframe[REQUIRED_COLUMNS].astype(int)

        for column in NUMBER_COLUMNS:
            out_of_range = (numeric_df[column] < 1) | (numeric_df[column] > 45)
            if out_of_range.any():
                invalid_rows = numeric_df.loc[out_of_range, "round"].tolist()
                raise DataValidationError(
                    f"Column '{column}' has values outside 1~45 at rounds: {invalid_rows}"
                )

        duplicate_rows = numeric_df[NUMBER_COLUMNS].nunique(axis=1) != len(NUMBER_COLUMNS)
        if duplicate_rows.any():
            invalid_rounds = numeric_df.loc[duplicate_rows, "round"].tolist()
            raise DataValidationError(
                f"Duplicate lotto numbers detected in rounds: {invalid_rounds}"
            )

    def index_by_round(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Sort by round and set round as index."""
        self.validate(dataframe)
        indexed = dataframe.copy()
        if "round" in indexed.index.names:
            indexed = indexed.reset_index(drop=True)
        indexed["round"] = indexed["round"].astype(int)
        indexed = indexed.sort_values("round").set_index("round", drop=False)
        return indexed

    def get_recent_rounds(self, dataframe: pd.DataFrame, recent_n: int) -> pd.DataFrame:
        """Return latest N rounds in ascending round order."""
        if recent_n <= 0:
            raise ValueError("recent_n must be greater than 0.")

        indexed = self.index_by_round(dataframe)
        return indexed.tail(recent_n)

    def load_and_validate(
        self, path: str | Path, recent_n: int | None = None, encoding: str = "utf-8"
    ) -> pd.DataFrame:
        """Load CSV, validate it, and optionally keep only recent N rounds."""
        dataframe = self.load_csv(path, encoding=encoding)
        indexed = self.index_by_round(dataframe)
        if recent_n is not None:
            return self.get_recent_rounds(indexed, recent_n=recent_n)
        return indexed

    @staticmethod
    def _normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
        normalized = {}
        for column in dataframe.columns:
            new_name = str(column).strip().lower()
            normalized[column] = ROUND_ALIASES.get(new_name, new_name)
        return dataframe.rename(columns=normalized)
