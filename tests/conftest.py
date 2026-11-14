"""Shared small fixtures for fast, isolated tests."""

import pytest

from bikes_data import make_rentals


@pytest.fixture
def small_frame():
    return make_rentals(n=120, seed=2026)
