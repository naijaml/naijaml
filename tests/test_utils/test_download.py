"""Tests for the model download helper. The network is replaced with a fake."""
from __future__ import annotations

import pytest
import requests


class FakeDownload:
    def __init__(self, chunks=(b"{}",), status_code: int = 200, fail_after_first_chunk: bool = False):
        self.chunks = list(chunks)
        self.status_code = status_code
        self.fail_after_first_chunk = fail_after_first_chunk
        self.headers = {"content-length": str(sum(len(c) for c in self.chunks))}

    def iter_content(self, chunk_size: int = 8192):
        for index, chunk in enumerate(self.chunks):
            if self.fail_after_first_chunk and index == 1:
                raise requests.ConnectionError("connection dropped")
            yield chunk


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NAIJAML_CACHE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def fake_get(monkeypatch):
    state = {"urls": [], "response": FakeDownload()}

    def get(url, **kwargs):
        state["urls"].append(url)
        return state["response"]

    monkeypatch.setattr(requests, "get", get)
    return state


class TestGetModelsCacheDir:
    def test_follows_env_var_and_creates_the_directory(self, cache_dir):
        from naijaml.utils.download import get_models_cache_dir

        models_dir = get_models_cache_dir()

        assert models_dir == cache_dir / "models"
        assert models_dir.is_dir()


class TestGetModelPath:
    def test_cached_file_needs_no_network(self, cache_dir, fake_get):
        from naijaml.utils.download import get_model_path, get_models_cache_dir

        cached = get_models_cache_dir() / "lang_model.json"
        cached.write_text("{}", encoding="utf-8")

        assert get_model_path("lang_model.json") == cached
        assert fake_get["urls"] == []

    def test_downloads_missing_file(self, cache_dir, fake_get):
        from naijaml.utils.download import get_model_path

        fake_get["response"] = FakeDownload(chunks=[b'{"a": ', b"1}"])

        path = get_model_path("word_diacritic_model.json")

        assert path == cache_dir / "models" / "word_diacritic_model.json"
        assert path.read_bytes() == b'{"a": 1}'
        assert fake_get["urls"] == [
            "https://huggingface.co/naijaml/naijaml-models/resolve/main/word_diacritic_model.json"
        ]
        assert [p.name for p in path.parent.iterdir()] == ["word_diacritic_model.json"]

    def test_custom_repo(self, cache_dir, fake_get):
        from naijaml.utils.download import get_model_path

        get_model_path("model.json", repo="someone/other-models")

        assert fake_get["urls"] == ["https://huggingface.co/someone/other-models/resolve/main/model.json"]

    def test_http_error_raises_and_leaves_nothing_behind(self, cache_dir, fake_get):
        from naijaml.utils.download import get_model_path

        fake_get["response"] = FakeDownload(status_code=404)

        with pytest.raises(RuntimeError, match="HTTP 404"):
            get_model_path("missing.json")
        assert list((cache_dir / "models").iterdir()) == []

    def test_interrupted_download_leaves_nothing_behind(self, cache_dir, fake_get):
        from naijaml.utils.download import get_model_path

        fake_get["response"] = FakeDownload(chunks=[b"partial", b"never sent"], fail_after_first_chunk=True)

        with pytest.raises(requests.ConnectionError):
            get_model_path("model.json")
        assert list((cache_dir / "models").iterdir()) == []
