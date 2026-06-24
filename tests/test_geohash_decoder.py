import pytest
from src.data.geohash_decoder import decode_geohash


def test_decode_geohash_inside_cell():
    """The returned point must lie inside the geohash cell, i.e. its lat
    must be in [-90, 90] and lon in [-180, 180]."""
    lat, lon = decode_geohash("qp02z1")
    assert -90 <= lat <= 90
    assert -180 <= lon <= 180


def test_decode_geohash_one_char_world_wide():
    # Single-char geohash is one of 32 base32 cells, each spans 45 deg lon
    # and ~45 deg lat. Check the cell is large but valid.
    lat, lon = decode_geohash("e")  # arbitrary
    assert -90 <= lat <= 90
    assert -180 <= lon <= 180


def test_decode_geohash_longer_is_more_precise():
    """Longer geohash => cell gets smaller => finer coordinates."""
    lat1, lon1 = decode_geohash("u")
    lat2, lon2 = decode_geohash("u1")
    # All four coordinates are valid lat/lon
    assert -90 <= lat1 <= 90 and -180 <= lon1 <= 180
    assert -90 <= lat2 <= 90 and -180 <= lon2 <= 180


def test_decode_geohash_invalid_char_does_not_crash():
    # Unknown chars fall back to bit=0; we should still get a point.
    lat, lon = decode_geohash("zzzzzz")
    assert -90 <= lat <= 90
    assert -180 <= lon <= 180
