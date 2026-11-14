"""Times the model reads are in the asker's zone (production: Myte said 5:30 PM for 11 PM IST)."""

from services.rag_service.myte.timefmt import localize


def test_utc_timestamps_become_local_words():
    line = (
        "[1] meeting | Feature Update - Backend | "
        "start 2026-11-12T17:30:00Z, end 2026-11-12T18:00:00+00:00"
    )
    out = localize(line, "Asia/Kolkata")
    assert "Thursday, November 12, 11 PM" in out and "Thursday, November 12, 11:30 PM" in out
    assert "17:30" not in out


def test_naive_times_are_utc_and_plain_dates_stay():
    assert localize("due 2026-11-18, at 2026-11-13T09:05", "America/New_York") == (
        "due 2026-11-18, at Friday, November 13, 4:05 AM"
    )


def test_unknown_zone_keeps_utc_and_says_so():
    assert localize("2026-11-13T17:30:00Z", None) == "Friday, November 13, 5:30 PM UTC"
    assert localize("2026-11-13T17:30:00Z", "Not/AZone") == "Friday, November 13, 5:30 PM UTC"


def test_offsets_are_respected():
    assert localize("2026-11-13T23:00:00+05:30", "Asia/Kolkata") == "Friday, November 13, 11 PM"
