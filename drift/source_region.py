"""
Source-region estimation from backward drift trajectories.
"""

from math import radians, sin, cos, sqrt, atan2


EARTH_RADIUS_KM = 6371.0


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Calculate distance between two geographic coordinates."""

    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)

    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)

    a = (
        sin(delta_lat / 2) ** 2
        + cos(lat1_rad)
        * cos(lat2_rad)
        * sin(delta_lon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return EARTH_RADIUS_KM * c


def estimate_source_region(positions):
    """
    Estimate the center, spread, and confidence of a probable
    source region.

    Parameters
    ----------
    positions : list of tuple
        List of (latitude, longitude) coordinates.

    Returns
    -------
    dict
        Estimated source center, radius, particle count,
        and confidence.
    """

    if not positions:
        raise ValueError("At least one position is required.")

    latitudes = [position[0] for position in positions]
    longitudes = [position[1] for position in positions]

    center_lat = sum(latitudes) / len(latitudes)
    center_lon = sum(longitudes) / len(longitudes)

    distances = [
        haversine_distance_km(
            center_lat,
            center_lon,
            lat,
            lon,
        )
        for lat, lon in positions
    ]

    radius_km = max(distances) if distances else 0.0

    confidence = 1.0 / (1.0 + radius_km / 10.0)

    return {
        "center": {
            "latitude": center_lat,
            "longitude": center_lon,
        },
        "radius_km": radius_km,
        "particle_count": len(positions),
        "confidence": round(confidence, 3),
    }
