from __future__ import annotations

import pytest

from lottogogo.data.loader import DataValidationError, LottoHistoryLoader


def test_load_csv_and_index_recent_rounds(tmp_path):
    csv_path = tmp_path / "history.csv"
    csv_path.write_text(
        "Round,N1,N2,N3,N4,N5,N6,memo\n"
        "3,1,2,3,4,5,6,안녕\n"
        "1,7,8,9,10,11,12,테스트\n"
        "2,13,14,15,16,17,18,값\n",
        encoding="utf-8",
    )

    loader = LottoHistoryLoader()
    full_df = loader.load_and_validate(csv_path)
    recent_df = loader.load_and_validate(csv_path, recent_n=2)

    assert full_df.index.tolist() == [1, 2, 3]
    assert recent_df.index.tolist() == [2, 3]
    assert "round" in full_df.columns


def test_missing_required_column_raises(tmp_path):
    csv_path = tmp_path / "missing.csv"
    csv_path.write_text(
        "round,n1,n2,n3,n4,n5\n" "1,1,2,3,4,5\n",
        encoding="utf-8",
    )

    loader = LottoHistoryLoader()
    dataframe = loader.load_csv(csv_path)
    with pytest.raises(DataValidationError, match="Missing required columns"):
        loader.validate(dataframe)


def test_number_out_of_range_raises(tmp_path):
    csv_path = tmp_path / "range.csv"
    csv_path.write_text(
        "round,n1,n2,n3,n4,n5,n6\n" "1,1,2,3,4,5,46\n",
        encoding="utf-8",
    )

    loader = LottoHistoryLoader()
    dataframe = loader.load_csv(csv_path)
    with pytest.raises(DataValidationError, match="outside 1~45"):
        loader.validate(dataframe)


def test_duplicate_numbers_in_row_raises(tmp_path):
    csv_path = tmp_path / "duplicate.csv"
    csv_path.write_text(
        "round,n1,n2,n3,n4,n5,n6\n" "1,1,1,2,3,4,5\n",
        encoding="utf-8",
    )

    loader = LottoHistoryLoader()
    dataframe = loader.load_csv(csv_path)
    with pytest.raises(DataValidationError, match="Duplicate lotto numbers"):
        loader.validate(dataframe)

