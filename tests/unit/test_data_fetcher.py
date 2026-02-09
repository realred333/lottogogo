from __future__ import annotations

from pathlib import Path

import pytest

from lottogogo.data.fetcher import (
    LottoFetchError,
    LottoHistoryFetcher,
    LottoRoundNotFoundError,
    LottoRoundResult,
)


def _payload(round_id: int = 1205) -> dict:
    return {
        "resultCode": None,
        "resultMessage": None,
        "data": {
            "list": [
                {
                    "ltEpsd": round_id,
                    "tm1WnNo": 1,
                    "tm2WnNo": 4,
                    "tm3WnNo": 16,
                    "tm4WnNo": 23,
                    "tm5WnNo": 31,
                    "tm6WnNo": 41,
                    "bnsWnNo": 2,
                    "ltRflYmd": "20260103",
                }
            ]
        },
    }


def test_parse_payload_accepts_null_result_code_when_data_exists():
    result = LottoHistoryFetcher.parse_payload(_payload(), expected_round=1205)

    assert result.round == 1205
    assert (result.n1, result.n2, result.n3, result.n4, result.n5, result.n6) == (
        1,
        4,
        16,
        23,
        31,
        41,
    )
    assert result.bonus == 2
    assert result.draw_date == "20260103"


def test_parse_payload_raises_for_empty_list():
    with pytest.raises(LottoRoundNotFoundError, match="winning-number rows"):
        LottoHistoryFetcher.parse_payload({"data": {"list": []}})


def test_parse_payload_raises_for_round_mismatch():
    with pytest.raises(LottoRoundNotFoundError, match="Requested round"):
        LottoHistoryFetcher.parse_payload(_payload(round_id=1204), expected_round=1205)


def test_fetch_round_uses_request_json(monkeypatch):
    fetcher = LottoHistoryFetcher()

    def fake_request_json(*, round_id: int):
        assert round_id == 1205
        return _payload(round_id=1205)

    monkeypatch.setattr(fetcher, "_request_json", fake_request_json)
    result = fetcher.fetch_round(1205)
    assert result.round == 1205


def test_fetch_new_rounds_since_uses_only_missing_rounds(monkeypatch):
    fetcher = LottoHistoryFetcher()
    called: dict[str, object] = {}

    def fake_latest(*, start_round: int):
        called["start_round"] = start_round
        return 1208

    def fake_parallel(round_ids: list[int], *, workers: int):
        called["round_ids"] = list(round_ids)
        called["workers"] = workers
        return [
            LottoRoundResult(
                round=round_id,
                n1=1,
                n2=2,
                n3=3,
                n4=4,
                n5=5,
                n6=6,
                bonus=7,
                draw_date="20260101",
            )
            for round_id in round_ids
        ]

    monkeypatch.setattr(fetcher, "fetch_latest_available_round", fake_latest)
    monkeypatch.setattr(fetcher, "fetch_rounds_parallel", fake_parallel)

    results = fetcher.fetch_new_rounds_since(1205, workers=4)
    assert called["start_round"] == 1206
    assert called["round_ids"] == [1206, 1207, 1208]
    assert called["workers"] == 4
    assert [result.round for result in results] == [1206, 1207, 1208]


def test_fetch_new_rounds_since_returns_empty_when_latest_is_same(monkeypatch):
    fetcher = LottoHistoryFetcher()
    monkeypatch.setattr(fetcher, "fetch_latest_available_round", lambda *, start_round: start_round - 1)

    results = fetcher.fetch_new_rounds_since(1209, workers=8)
    assert results == []


def test_latest_round_from_history_csv_reads_max_round(tmp_path: Path):
    csv_path = tmp_path / "history.csv"
    csv_path.write_text(
        "round,n1,n2,n3,n4,n5,n6\n"
        "1208,1,2,3,4,5,6\n"
        "1209,7,8,9,10,11,12\n",
        encoding="utf-8",
    )
    assert LottoHistoryFetcher.latest_round_from_history_csv(csv_path) == 1209


def test_update_history_csv_appends_only_new_rows(tmp_path: Path, monkeypatch):
    csv_path = tmp_path / "history.csv"
    csv_path.write_text(
        "round,n1,n2,n3,n4,n5,n6\n"
        "1208,1,2,3,4,5,6\n"
        "1209,7,8,9,10,11,12\n",
        encoding="utf-8",
    )
    fetcher = LottoHistoryFetcher()

    monkeypatch.setattr(
        fetcher,
        "fetch_new_rounds_since",
        lambda last_round, workers=8: [
            LottoRoundResult(
                round=1210,
                n1=13,
                n2=14,
                n3=15,
                n4=16,
                n5=17,
                n6=18,
                bonus=19,
                draw_date="20260110",
            )
        ],
    )

    merged = fetcher.update_history_csv(csv_path, workers=5)
    assert merged["round"].tolist() == [1208, 1209, 1210]

    saved = csv_path.read_text(encoding="utf-8")
    assert "1210,13,14,15,16,17,18" in saved
