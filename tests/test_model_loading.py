from __future__ import annotations

from typing import Any

from libs.shared.model_loading import load_local_first


def test_uses_the_local_cache_and_never_goes_online_when_the_files_exist() -> None:
    calls: list[dict[str, Any]] = []

    def factory(name: str, **kwargs: Any) -> str:
        calls.append(kwargs)
        return f"loaded:{name}"

    assert load_local_first(factory, "m", device="cpu") == "loaded:m"
    assert calls == [{"local_files_only": True, "device": "cpu"}]


def test_falls_back_to_a_download_only_when_the_files_are_missing() -> None:
    calls: list[dict[str, Any]] = []

    def factory(name: str, **kwargs: Any) -> str:
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise OSError("not in cache")
        return "downloaded"

    assert load_local_first(factory, "m", device="cpu") == "downloaded"
    assert calls == [{"local_files_only": True, "device": "cpu"}, {"device": "cpu"}]
