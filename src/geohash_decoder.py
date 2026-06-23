"""
Geohash decoder — converts geohash strings to (latitude, longitude) pairs.
"""

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_BASE32_MAP = {c: i for i, c in enumerate(_BASE32)}


def decode_geohash(geohash_str: str) -> tuple[float, float]:
    """Decode a geohash string to its approximate (latitude, longitude) centre.

    Parameters
    ----------
    geohash_str : str
        A geohash-encoded location string (e.g. ``"qp02z1"``).

    Returns
    -------
    tuple[float, float]
        ``(latitude, longitude)`` of the geohash cell centre.
    """
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    is_lon = True

    for ch in geohash_str:
        bits = _BASE32_MAP.get(ch, 0)
        for i in range(4, -1, -1):
            bit = (bits >> i) & 1
            if is_lon:
                mid = (lon_interval[0] + lon_interval[1]) / 2
                if bit:
                    lon_interval[0] = mid
                else:
                    lon_interval[1] = mid
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2
                if bit:
                    lat_interval[0] = mid
                else:
                    lat_interval[1] = mid
            is_lon = not is_lon

    lat = (lat_interval[0] + lat_interval[1]) / 2
    lon = (lon_interval[0] + lon_interval[1]) / 2
    return lat, lon
