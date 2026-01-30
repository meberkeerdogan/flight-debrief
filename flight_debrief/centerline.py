from __future__ import annotations
import numpy as np

EARTH_R_M = 6371000.0

def latlon_to_local_xy_m(lat_deg: np.ndarray, lon_deg: np.ndarray, lat0_deg: float, lon0_deg: float):
    """
    Small-area approximation: converts lat/lon to local meters around (lat0, lon0).
    Good enough for runway + final approach distances.
    """
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    lat0 = np.deg2rad(lat0_deg)
    lon0 = np.deg2rad(lon0_deg)

    dlat = lat - lat0
    dlon = lon - lon0

    x = EARTH_R_M * np.cos(lat0) * dlon
    y = EARTH_R_M * dlat
    return x, y

def cross_track_error_m(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    a_lat: float, a_lon: float,
    b_lat: float, b_lon: float,
):
    """
    Per-point signed distance to line AB (runway centerline), in meters.
    Positive/negative indicates side of centerline (useful for debugging).
    """
    # Reference origin at runway threshold A
    px, py = latlon_to_local_xy_m(lat_deg, lon_deg, a_lat, a_lon)
    ax, ay = 0.0, 0.0
    bx, by = latlon_to_local_xy_m(np.array([b_lat]), np.array([b_lon]), a_lat, a_lon)
    bx, by = float(bx[0]), float(by[0])

    ab = np.array([bx - ax, by - ay], dtype=float)
    ab2 = float(np.dot(ab, ab))
    if ab2 < 1e-9:
        return np.full_like(px, np.nan, dtype=float)

    # Projection factor onto AB
    t = (px * ab[0] + py * ab[1]) / ab2
    # Closest point on infinite line
    cx = t * ab[0]
    cy = t * ab[1]

    # Perp distance magnitude
    dx = px - cx
    dy = py - cy
    dist = np.sqrt(dx * dx + dy * dy)

    # Signed side using 2D cross product sign
    sign = np.sign(ab[0] * py - ab[1] * px)
    return dist * sign
