"""Tests for the dataset registry, cache and loaders.

No test here touches the network: ``requests.get`` is replaced with a fake
and the cache is redirected to a temporary directory.
"""
from __future__ import annotations

import pytest
import requests

NAIJASENTI_TSV = (
    "tweet\tlabel\n"
    "mo kí àrẹ wa alàgbà kú àbọ̀ o\tpositive\n"
    "obodo à wu igwe\tneutral\n"
    "\tnegative\n"
    "tweet with no label\t\n"
)

MASAKHANER_CONLL = (
    "Ndị B-ORG\n"
    "uweojii I-ORG\n"
    "na O\n"
    "Edo B-LOC\n"
    "\n"
    "malformed\n"
    "Ọ O\n"
    "bụ O\n"
    "Lagos B-LOC"
)

MASAKHANEWS_TSV = (
    "category\theadline\ttext\turl\n"
    "sports\tSuper Eagles ti borí\tẸgbẹ́ agbábọ́ọ̀lù Nàìjíríà ti borí\thttps://example.org/a\n"
    "\tNo category\tThis row is skipped\thttps://example.org/b\n"
    "health\tAkụkọ ahụike\t\thttps://example.org/c\n"
)


class FakeResponse:
    def __init__(self, text: str = "", status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("HTTP %d" % self.status_code)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Point the NaijaML cache at an empty temporary directory."""
    monkeypatch.setenv("NAIJAML_CACHE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def fake_get(monkeypatch):
    """Replace requests.get; returns the list of requested URLs and a setter for the response."""
    state = {"urls": [], "response": FakeResponse()}

    def get(url, **kwargs):
        state["urls"].append(url)
        return state["response"]

    monkeypatch.setattr(requests, "get", get)
    return state


class TestRegistry:
    def test_list_datasets_is_sorted(self):
        from naijaml.data import list_datasets

        names = list_datasets()
        assert names == sorted(names)
        assert {"naijasenti", "masakhaner", "masakhanews"} <= set(names)

    def test_dataset_info_returns_a_copy(self):
        from naijaml.data import dataset_info

        info = dataset_info("naijasenti")
        assert info["hf_id"] == "HausaNLP/NaijaSenti-Twitter"
        info["hf_id"] = "changed"
        assert dataset_info("naijasenti")["hf_id"] == "HausaNLP/NaijaSenti-Twitter"

    def test_unknown_dataset(self):
        from naijaml.data import dataset_info

        with pytest.raises(ValueError, match="Unknown dataset 'nope'"):
            dataset_info("nope")

    def test_get_hf_id(self):
        from naijaml.data.registry import get_hf_id

        assert get_hf_id("masakhanews") == "masakhane/masakhanews"

    def test_validate_lang(self):
        from naijaml.data.registry import validate_lang

        validate_lang("naijasenti", "yor")
        validate_lang("naijasenti", None)
        with pytest.raises(ValueError, match="not supported"):
            validate_lang("masakhaner", "pcm")

    def test_validate_split(self):
        from naijaml.data.registry import validate_split

        validate_split("naijasenti", "test")
        with pytest.raises(ValueError):
            validate_split("naijasenti", "holdout")

    def test_every_dataset_has_the_documented_fields(self):
        from naijaml.data import dataset_info, list_datasets

        for name in list_datasets():
            info = dataset_info(name)
            for field in ("name", "description", "languages", "task", "splits", "hf_id", "citation"):
                assert info[field], "%s is missing %s" % (name, field)


class TestCache:
    def test_cache_dir_follows_env_var(self, cache_dir):
        from naijaml.data.cache import get_cache_dir

        assert get_cache_dir() == cache_dir

    def test_round_trip_keeps_diacritics(self, cache_dir):
        from naijaml.data.cache import is_cached, load_from_cache, save_to_cache

        records = [{"text": "Ọjọ́ dára púpọ̀", "label": "positive"}]
        assert not is_cached("naijasenti", "yor", "test")

        path = save_to_cache(records, "naijasenti", "yor", "test")

        assert path == cache_dir / "naijasenti" / "naijasenti_yor_test.json"
        assert "Ọjọ́" in path.read_text(encoding="utf-8")
        assert is_cached("naijasenti", "yor", "test")
        assert load_from_cache("naijasenti", "yor", "test") == records

    def test_all_languages_use_a_separate_file(self, cache_dir):
        from naijaml.data.cache import is_cached, save_to_cache

        path = save_to_cache([], "naijasenti", None, "train")

        assert path.name == "naijasenti_train.json"
        assert not is_cached("naijasenti", "yor", "train")

    def test_load_missing_raises(self, cache_dir):
        from naijaml.data.cache import load_from_cache

        with pytest.raises(FileNotFoundError, match="not cached"):
            load_from_cache("naijasenti", "yor", "train")

    def test_clear_one_dataset(self, cache_dir):
        from naijaml.data import clear_cache
        from naijaml.data.cache import is_cached, save_to_cache

        save_to_cache([], "naijasenti", "yor", "train")
        save_to_cache([], "masakhaner", "yor", "train")

        clear_cache("naijasenti")
        clear_cache("never_downloaded")

        assert not is_cached("naijasenti", "yor", "train")
        assert is_cached("masakhaner", "yor", "train")

    def test_clear_everything(self, cache_dir):
        from naijaml.data import clear_cache
        from naijaml.data.cache import save_to_cache

        save_to_cache([], "naijasenti", "yor", "train")
        (cache_dir / "stray.txt").write_text("x", encoding="utf-8")

        clear_cache()

        assert list(cache_dir.iterdir()) == []


class TestLoadDataset:
    def test_rejects_unknown_dataset(self, cache_dir):
        from naijaml.data import load_dataset

        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nope")

    def test_rejects_unsupported_language_before_downloading(self, cache_dir, fake_get):
        from naijaml.data import load_dataset

        with pytest.raises(ValueError, match="not supported"):
            load_dataset("masakhaner", lang="pcm")
        assert fake_get["urls"] == []

    def test_rejects_unknown_split_before_downloading(self, cache_dir, fake_get):
        from naijaml.data import load_dataset

        with pytest.raises(ValueError):
            load_dataset("naijasenti", lang="yor", split="holdout")
        assert fake_get["urls"] == []

    def test_registered_dataset_without_loader(self, cache_dir, fake_get):
        from naijaml.data import load_dataset

        with pytest.raises(ValueError, match="does not have a loader yet"):
            load_dataset("menyo20k")
        assert fake_get["urls"] == []


class TestNaijaSenti:
    def test_parses_rows_and_skips_incomplete_ones(self, cache_dir, fake_get):
        from naijaml.data import load_dataset

        fake_get["response"] = FakeResponse(NAIJASENTI_TSV)

        data = load_dataset("naijasenti", lang="yor", split="test")

        assert data == [
            {"text": "mo kí àrẹ wa alàgbà kú àbọ̀ o", "label": "positive"},
            {"text": "obodo à wu igwe", "label": "neutral"},
        ]
        assert fake_get["urls"] == [
            "https://raw.githubusercontent.com/hausanlp/NaijaSenti/main/data/annotated_tweets/yor/test.tsv"
        ]

    def test_validation_split_is_the_dev_file(self, cache_dir, fake_get):
        from naijaml.data import load_dataset

        fake_get["response"] = FakeResponse(NAIJASENTI_TSV)

        load_dataset("naijasenti", lang="hau", split="validation")

        assert fake_get["urls"][0].endswith("/hau/dev.tsv")

    def test_second_load_comes_from_cache(self, cache_dir, fake_get):
        from naijaml.data import load_dataset

        fake_get["response"] = FakeResponse(NAIJASENTI_TSV)
        first = load_dataset("naijasenti", lang="ibo", split="train")
        fake_get["response"] = FakeResponse(status_code=500)

        second = load_dataset("naijasenti", lang="ibo", split="train")

        assert second == first
        assert len(fake_get["urls"]) == 1

    def test_no_language_loads_all_four(self, cache_dir, fake_get):
        from naijaml.data import load_dataset

        fake_get["response"] = FakeResponse(NAIJASENTI_TSV)

        data = load_dataset("naijasenti", split="train")

        assert len(data) == 8
        assert [url.split("/")[-2] for url in fake_get["urls"]] == ["yor", "hau", "ibo", "pcm"]

    def test_failed_download_is_not_cached(self, cache_dir, fake_get):
        from naijaml.data import load_dataset
        from naijaml.data.cache import is_cached

        fake_get["response"] = FakeResponse(status_code=404)

        with pytest.raises(requests.HTTPError):
            load_dataset("naijasenti", lang="pcm", split="train")
        assert not is_cached("naijasenti", "pcm", "train")


class TestMasakhaNER:
    def test_parses_conll_sentences(self, cache_dir, fake_get):
        from naijaml.data import load_dataset

        fake_get["response"] = FakeResponse(MASAKHANER_CONLL)

        data = load_dataset("masakhaner", lang="ibo", split="validation")

        assert data == [
            {"tokens": ["Ndị", "uweojii", "na", "Edo"], "ner_tags": ["B-ORG", "I-ORG", "O", "B-LOC"]},
            {"tokens": ["Ọ", "bụ", "Lagos"], "ner_tags": ["O", "O", "B-LOC"]},
        ]
        assert fake_get["urls"][0].endswith("MasakhaNER2.0/data/ibo/dev.txt")

    def test_no_language_loads_all_three(self, cache_dir, fake_get):
        from naijaml.data import load_dataset

        fake_get["response"] = FakeResponse(MASAKHANER_CONLL)

        data = load_dataset("masakhaner", split="test")

        assert len(data) == 6
        assert len(fake_get["urls"]) == 3

    def test_second_load_comes_from_cache(self, cache_dir, fake_get):
        from naijaml.data import load_dataset

        fake_get["response"] = FakeResponse(MASAKHANER_CONLL)
        first = load_dataset("masakhaner", lang="yor", split="train")

        assert load_dataset("masakhaner", lang="yor", split="train") == first
        assert len(fake_get["urls"]) == 1


class TestMasakhaNEWS:
    def test_parses_rows_and_skips_incomplete_ones(self, cache_dir, fake_get):
        from naijaml.data import load_dataset

        fake_get["response"] = FakeResponse(MASAKHANEWS_TSV)

        data = load_dataset("masakhanews", lang="yor", split="test")

        assert data == [{
            "text": "Ẹgbẹ́ agbábọ́ọ̀lù Nàìjíríà ti borí",
            "label": "sports",
            "headline": "Super Eagles ti borí",
            "url": "https://example.org/a",
        }]
        assert fake_get["urls"] == [
            "https://huggingface.co/datasets/masakhane/masakhanews/resolve/main/data/yor/test.tsv"
        ]

    def test_no_language_loads_all_four(self, cache_dir, fake_get):
        from naijaml.data import load_dataset

        fake_get["response"] = FakeResponse(MASAKHANEWS_TSV)

        data = load_dataset("masakhanews", split="validation")

        assert len(data) == 4
        assert all(url.endswith("/dev.tsv") for url in fake_get["urls"])

    def test_second_load_comes_from_cache(self, cache_dir, fake_get):
        from naijaml.data import load_dataset

        fake_get["response"] = FakeResponse(MASAKHANEWS_TSV)
        first = load_dataset("masakhanews", lang="pcm", split="train")

        assert load_dataset("masakhanews", lang="pcm", split="train") == first
        assert len(fake_get["urls"]) == 1
