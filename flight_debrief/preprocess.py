from __future__ import annotations
from pathlib import Path
from typing import Union, IO
import numpy as np
import pandas as pd

from .domain import AircraftProfile

# -----------------------------
# Helpers
# -----------------------------

def _normalize_col(c: str) -> str:
    return c.strip().lower().replace(" ", "").replace("_", "")

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    # match by normalized name
    norm_map = {_normalize_col(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize_col(cand)
        if key in norm_map:
            return norm_map[key]
    return None



def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """Simple moving average guaranteed to return same length as x."""
    if win <= 1:    # If the window size is 1 or smaller: No smoothing needed / return a copy of the input array
        return x.copy()

    kernel = np.ones(win, dtype=float) / win

    left = win // 2
    right = win - 1 - left  # makes total pad = win-1
    xpad = np.pad(x, (left, right), mode="edge")

    return np.convolve(xpad, kernel, mode="valid")  # length == len(x)




def rolling_std(x: np.ndarray, win: int) -> np.ndarray:
    """Rolling standard deviation (same length, center-aligned)."""
    if win <= 1:
        return np.zeros_like(x)
    # Use pandas rolling for vectorized performance (much faster than Python loop)
    return pd.Series(x).rolling(win, center=True, min_periods=1).std().fillna(0.0).to_numpy()

"""
rolling_std computes the local variability of a signal.
Not:

“Is pitch high?”

“Is pitch low?”

But:

“Is pitch moving around a lot right now?”

That's a completely different question.

////////////

Imagine two approaches:

Case A — perfectly stable
Pitch: 3.0°, 3.1°, 3.0°, 3.1°, 3.0°

Case B — pilot chasing the glideslope
Pitch: 2.0°, 4.5°, 1.8°, 4.8°, 2.1°


In both cases:

Pitch never exceeds, say, ±6°

Absolute pitch limits would NOT trigger

But only Case B is unstable.

The instability is in the oscillation, not the magnitude.

////////////

Standard deviation answers one question:

“How spread out are these values?”

Rolling standard deviation answers:

“How spread out are the values in this time window?”

So:

Low std → smooth, controlled motion

High std → oscillation, over-control, chasing

This is exactly what pilots call “pitch chasing”.

"""

CSVSource = Union[str, Path, IO[bytes], IO[str]]

def _get_runway_field(runway, *names):
    """
    Try to read a field from runway either as attribute or dict key.
    Example: _get_runway_field(runway, "thr_lat_deg", "thr_lat")
    """
    if runway is None:
        return None
    for n in names:
        # attribute style
        if hasattr(runway, n):
            return getattr(runway, n)
        # dict style
        if isinstance(runway, dict) and n in runway:
            return runway[n]
    return None



# -----------------------------
# Core pipeline
# -----------------------------
def load_and_preprocess(csv_source: CSVSource, runway_elev_m: float, profile: AircraftProfile, runway=None) -> pd.DataFrame:
    """
    Loads CSV and adds derived/smoothed signals.

    Expected CSV columns:
      t, alt_msl_m, ias_kt, vs_fpm, pitch_deg, roll_deg, throttle (hdg_deg optional)

    Returns a DataFrame with extra columns:
      agl_ft, ias_s, vs_s, pitch_s, roll_s, throttle_s, pitch_std
    """
    df = pd.read_csv(csv_source)    # Read the CSV (a DataFrame is like a table: columns + rows)

    
    # time can be "t" (new logs) or "Time" (older logs)
    if "t" in df.columns:
        df["t"] = df["t"].astype(float)
    elif "Time" in df.columns:
        df["t"] = df["Time"].astype(float)
    else:
        raise ValueError(f"No time column found. Expected 't' or 'Time'. Found: {list(df.columns)}")

    df["t"] = pd.to_numeric(df["t"], errors="coerce").astype(float)


    # Drop only *exact* duplicate rows (same time + same telemetry)
    dedupe_cols = ["t", "alt_msl_m", "ias_kt", "vs_fpm", "pitch_deg", "roll_deg", "throttle"]
    dedupe_cols = [c for c in dedupe_cols if c in df.columns]  # safety
    df = df.drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)

    # Sort by time
    df = df.sort_values("t").reset_index(drop=True)

    # Now enforce strictly increasing time by removing remaining duplicate timestamps
    # (If any remain, they must differ in values; you said they usually don't, but this makes dt safe.)
    df = df.loc[df["t"].diff().fillna(1.0) > 0].reset_index(drop=True)


    # altitude can be already meters (alt_msl_m) or feet (alt_msl_ft)
    if "alt_msl_m" in df.columns:
        df["alt_msl_m"] = df["alt_msl_m"].astype(float)
    elif "alt_msl_ft" in df.columns:
        df["alt_msl_m"] = df["alt_msl_ft"].astype(float) * 0.3048
    else:
        raise ValueError(
            f"No altitude column found. Expected 'alt_msl_m' or 'alt_msl_ft'. Found: {list(df.columns)}"
        )
    
    df["alt_msl_ft"] = df["alt_msl_m"] * 3.28084


    # vertical speed can be already fpm (vs_fpm) or fps (vs_fps)
    if "vs_fpm" in df.columns:
        df["vs_fpm"] = df["vs_fpm"].astype(float)
    elif "vs_fps" in df.columns:
        df["vs_fpm"] = df["vs_fps"].astype(float) * 60.0
    else:
        raise ValueError(
            f"No vertical speed column found. Expected 'vs_fpm' or 'vs_fps'. Found: {list(df.columns)}"
        )



    required = ["t", "alt_msl_m", "ias_kt", "vs_fpm", "pitch_deg", "roll_deg", "throttle"]    # Needed Column names
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    # Compute AGL in feet
    df["agl_ft"] = (df["alt_msl_m"] - runway_elev_m) * 3.28084
    # All stability rules are framed in feet AGL (500 ft gate, 1500–20 ft window, etc.)

    # Estimate dt from median positive spacing
    t = df["t"].to_numpy(dtype=float)

    dts = np.diff(t)
    dts = dts[dts > 0]  # ignore non-positive (shouldn't exist after cleanup)
    if len(dts) == 0:
        raise ValueError("Time column 't' must contain increasing values to estimate dt.")

    dt = float(np.median(dts))

    # Optional sanity check - large dt may indicate low-rate data
    # (Warning handled silently; UI can check df.attrs["dt"] if needed)


    """
    Let`s break it:

    np.diff(t) computes differences between consecutive times:

    If t = [0.0, 0.2, 0.4, 0.6]
    then np.diff(t) = [0.2, 0.2, 0.2]

    np.median(...) finds the median spacing.
    Using median instead of mean makes it robust to one weird glitch.

    float(...) converts NumPy`s scalar type to a normal Python float.

    So dt becomes something like 0.2.
    """



    # Compute smoothing window in samples
    smooth_win = max(1, int(round(profile.smooth_window_s / dt))) # your data can come at different rates (0.1s, 0.2s, 1.0s). but you want “smooth for ~1 second” regardless.

    # Create smoothed columns
    # Each of these takes a raw column, converts it to NumPy float array, then smooths it using moving average.
    df["ias_s"] = moving_average(df["ias_kt"].to_numpy(float), smooth_win)
    df["vs_s"] = moving_average(df["vs_fpm"].to_numpy(float), smooth_win)
    df["pitch_s"] = moving_average(df["pitch_deg"].to_numpy(float), smooth_win)
    df["roll_s"] = moving_average(df["roll_deg"].to_numpy(float), smooth_win)
    df["throttle_s"] = moving_average(df["throttle"].to_numpy(float), smooth_win)

    # Pitch variability over rolling window
    pitch_std_win = max(1, int(round(profile.pitch_std_window_s / dt)))
    df["pitch_std"] = rolling_std(df["pitch_s"].to_numpy(float), pitch_std_win)

    df.attrs["dt"] = dt    # This puts dt into the DataFrame's attribute dictionary. / Stores dt in DataFrame metadata (attrs). / That is a nice trick: downstream functions like detect_unstable can retrieve dt without recomputing it.

    return df
    # Now the caller receives a DataFrame with original and derived columns
"""
What this function accomplishes (in plain terms)

It turns raw CSV telemetry into analysis-ready signals by:

validating the input format

converting altitude into AGL feet

inferring sampling rate dt

smoothing noisy signals

computing a “pilot chasing” indicator (pitch_std)

returning everything in one DataFrame for downstream steps

This is basically the “data cleaning + feature engineering” step of your pipeline.
"""


# Supports both CLI and Streamlit upload
# Keeps all signals processing in one place