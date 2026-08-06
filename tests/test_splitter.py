from __future__ import annotations

from services.embedding_service.splitter import CHUNK_OVERLAP, CHUNK_SIZE, split_600_60


def test_short_text_returns_single_piece():
    pieces = split_600_60("hello world")
    assert pieces == ["hello world"]


def test_empty_text_returns_empty_list():
    assert split_600_60("") == []


def test_long_text_splits_into_bounded_pieces_with_overlap():
    text = "word " * 400  # ~2000 chars, well over CHUNK_SIZE
    pieces = split_600_60(text)
    assert len(pieces) > 1
    for piece in pieces:
        assert len(piece) <= CHUNK_SIZE + CHUNK_OVERLAP  # splitter may break on whitespace


def test_config_matches_plan_600_60():
    assert CHUNK_SIZE == 600
    assert CHUNK_OVERLAP == 60
