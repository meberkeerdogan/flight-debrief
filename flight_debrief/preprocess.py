"""Data preprocessing and signal processing for flight telemetry."""

from __future__ import annotations
from pathlib import Path
from typing import Union, IO

import numpy as np
import pandas as pd

from .domain import AircraftProfile


# Type alias for CSV input sources
CSVSource = Union[str, Path, IO[bytes], IO[str]]


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """
    Compute moving average with edge padding.

    Args:
        x: Input signal array
        win: Window size in samples

    Returns:
        Smoothed array with same length as input
    """
    if win <= 1:
        return x.copy()

    kernel = np.ones(win, dtype=float) / win
    left = win // 2
    right = win - 1 - left
    xpad = np.pad(x, (left, right), mode="edge")

    return np.convolve(xpad, kernel, mode="valid")


def rolling_std(x: np.ndarray, win: int) -> np.ndarray:
    """
    Compute rolling standard deviation.

    Used to detect pitch oscillations ("pitch chasing") where the variability
    of pitch indicates unstable control inputs rather than the absolute value.

    Args:
        x: Input signal array
        win: Window size in samples

    Returns:
        Rolling std array with same length as input
    """
    if win <= 1:
        return np.zeros_like(x)
    return (
        pd.Series(x)
        .rolling(win, center=True, min_periods=1)
        .std()
        .fillna(0.0)
        .to_numpy()
    )


def load_and_preprocess(
    csv_source: CSVSource,
    runway_elev_m: float,
    profile: AircraftProfile,
    runway=None,
) -> pd.DataFrame:
    """
    Load CSV telemetry and compute derived signals.

    Handles multiple CSV formats:
    - Time column: 't' or 'Time'
    - Altitude: 'alt_msl_m' (meters) or 'alt_msl_ft' (feet)
    - Vertical speed: 'vs_fpm' or 'vs_fps'

    Args:
        csv_source: Path or file-like object containing CSV data
        runway_elev_m: Runway elevation in meters MSL
        profile: Aircraft profile with processing parameters
        runway: Optional runway object (for future use)

    Returns:
        DataFrame with original and derived columns:
        - agl_ft: Altitude above ground level
        - ias_s, vs_s, pitch_s, roll_s, throttle_s: Smoothed signals
        - pitch_std: Rolling standard deviation of pitch

    Raises:
        ValueError: If required columns are missing or data is invalid
    """
    df = pd.read_csv(csv_source)

    # Normalize time column
    if "t" in df.columns:
        df["t"] = df["t"].astype(float)
    elif "Time" in df.columns:
        df["t"] = df["Time"].astype(float)
    else:
        raise ValueError(
            f"No time column found. Expected 't' or 'Time'. Found: {list(df.columns)}"
        )

    df["t"] = pd.to_numeric(df["t"], errors="coerce").astype(float)

    # Remove duplicates and sort
    dedupe_cols = ["t", "alt_msl_m", "ias_kt", "vs_fpm", "pitch_deg", "roll_deg", "throttle"]
    dedupe_cols = [c for c in dedupe_cols if c in df.columns]
    df = df.drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)
    df = df.sort_values("t").reset_index(drop=True)
    df = df.loc[df["t"].diff().fillna(1.0) > 0].reset_index(drop=True)

    # Normalize altitude column
    if "alt_msl_m" in df.columns:
        df["alt_msl_m"] = df["alt_msl_m"].astype(float)
    elif "alt_msl_ft" in df.columns:
        df["alt_msl_m"] = df["alt_msl_ft"].astype(float) * 0.3048
    else:
        raise ValueError(
            f"No altitude column found. Expected 'alt_msl_m' or 'alt_msl_ft'. "
            f"Found: {list(df.columns)}"
        )

    df["alt_msl_ft"] = df["alt_msl_m"] * 3.28084

    # Normalize vertical speed column
    if "vs_fpm" in df.columns:
        df["vs_fpm"] = df["vs_fpm"].astype(float)
    elif "vs_fps" in df.columns:
        df["vs_fpm"] = df["vs_fps"].astype(float) * 60.0
    else:
        raise ValueError(
            f"No vertical speed column found. Expected 'vs_fpm' or 'vs_fps'. "
            f"Found: {list(df.columns)}"
        )

    # Validate required columns
    required = ["t", "alt_msl_m", "ias_kt", "vs_fpm", "pitch_deg", "roll_deg", "throttle"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Found columns: {list(df.columns)}"
        )

    # Compute AGL
    df["agl_ft"] = (df["alt_msl_m"] - runway_elev_m) * 3.28084

    # Estimate sampling interval
    t = df["t"].to_numpy(dtype=float)
    dts = np.diff(t)
    dts = dts[dts > 0]
    if len(dts) == 0:
        raise ValueError("Time column 't' must contain increasing values to estimate dt.")
    dt = float(np.median(dts))

    # Apply smoothing
    smooth_win = max(1, int(round(profile.smooth_window_s / dt)))
    df["ias_s"] = moving_average(df["ias_kt"].to_numpy(float), smooth_win)
    df["vs_s"] = moving_average(df["vs_fpm"].to_numpy(float), smooth_win)
    df["pitch_s"] = moving_average(df["pitch_deg"].to_numpy(float), smooth_win)
    df["roll_s"] = moving_average(df["roll_deg"].to_numpy(float), smooth_win)
    df["throttle_s"] = moving_average(df["throttle"].to_numpy(float), smooth_win)

    # Compute pitch variability
    pitch_std_win = max(1, int(round(profile.pitch_std_window_s / dt)))
    df["pitch_std"] = rolling_std(df["pitch_s"].to_numpy(float), pitch_std_win)

    # Store sampling interval in DataFrame metadata
    df.attrs["dt"] = dt

    return df
