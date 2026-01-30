import numpy as np

EARTH_R = 6371000.0  # meters

def _deg2rad(x: np.ndarray) -> np.ndarray:
    return np.deg2rad(x)

def _enu_from_latlon(lat_deg, lon_deg, lat0_deg, lon0_deg):
    """
    Convert lat/lon to local ENU meters relative to (lat0, lon0).
    Flat-earth approximation good for small areas (final approach).
    """
    lat = _deg2rad(np.asarray(lat_deg))
    lon = _deg2rad(np.asarray(lon_deg))
    lat0 = np.deg2rad(lat0_deg)
    lon0 = np.deg2rad(lon0_deg)

    dlat = lat - lat0
    dlon = lon - lon0

    east  = EARTH_R * dlon * np.cos(lat0)
    north = EARTH_R * dlat
    return east, north

def cross_track_ft(lat_deg, lon_deg, runway_thr_lat, runway_thr_lon, runway_true_brg_deg):
    """
    Returns signed cross-track distance to runway centerline in feet.
    + = right of centerline, - = left (depending on axis convention).
    """
    e, n = _enu_from_latlon(lat_deg, lon_deg, runway_thr_lat, runway_thr_lon)

    brg = np.deg2rad(runway_true_brg_deg)
    # runway forward unit vector in ENU
    u_e = np.sin(brg)
    u_n = np.cos(brg)

    # perpendicular unit vector (left/right)
    v_e = -u_n
    v_n =  u_e

    xtrack_m = e * v_e + n * v_n
    return xtrack_m * 3.28084
