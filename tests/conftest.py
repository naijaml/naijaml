"""Pytest configuration and shared fixtures for NaijaML tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def language_samples():
    """Load language sample texts for testing."""
    with open(FIXTURES_DIR / "language_samples.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def yoruba_samples(language_samples):
    """Yorùbá text samples."""
    return language_samples["yor"]


@pytest.fixture
def hausa_samples(language_samples):
    """Hausa text samples."""
    return language_samples["hau"]


@pytest.fixture
def igbo_samples(language_samples):
    """Igbo text samples."""
    return language_samples["ibo"]


@pytest.fixture
def pidgin_samples(language_samples):
    """Nigerian Pidgin text samples."""
    return language_samples["pcm"]


@pytest.fixture
def english_samples(language_samples):
    """English text samples."""
    return language_samples["eng"]
