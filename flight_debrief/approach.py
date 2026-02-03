"""Approach window extraction and target speed estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd


# Approach band altitude limits (feet AGL)
APPROACH_BAND_UPPER_FT = 1500
APPROACH_BAND_LOWER_FT = -20

# Target speed estimation band (feet AGL)
TARGET_SPEED_BAND_UPPER_FT = 700
TARGET_SPEED_BAND_LOWER_FT = 500
TARGET_SPEED_MIN_SAMPLES = 10


def extract_approach_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the final approach segment from flight data.

    Finds continuous segments within the approach altitude band (1500 to -20 ft AGL)
    and selects the one that reaches closest to the ground.

    Args:
        df: Flight data DataFrame with 'agl_ft' column

    Returns:
        DataFrame containing only the approach segment, or empty DataFrame
        if no valid approach segment exists
    """
    in_band = (df["agl_ft"] <= APPROACH_BAND_UPPER_FT) & (df["agl_ft"] >= APPROACH_BAND_LOWER_FT)

    idx = np.where(in_band.to_numpy())[0]
    if len(idx) == 0:
        return df.iloc[0:0].copy()

    # Split into continuous index blocks
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    segments = np.split(idx, breaks)

    # Choose segment that gets closest to ground (lowest AGL, then longest)
    def seg_score(seg):
        agl_min = float(df.iloc[seg]["agl_ft"].min())
        length = len(seg)
        return (agl_min, -length)

    best_seg = min(segments, key=seg_score)

    approach = df.iloc[best_seg].copy()
    approach.reset_index(drop=True, inplace=True)
    return approach


def estimate_target_speed(approach: pd.DataFrame) -> float:
    """
    Estimate the target approach speed from stabilized flight segment.

    Uses median airspeed in the 700-500 ft AGL band where the approach
    should be stabilized. Falls back to overall median if insufficient
    data in the target band.

    Args:
        approach: Approach segment DataFrame with 'agl_ft' and 'ias_s' columns

    Returns:
        Estimated target speed in knots, or NaN if no valid data
    """
    band = approach[
        (approach["agl_ft"] <= TARGET_SPEED_BAND_UPPER_FT) &
        (approach["agl_ft"] >= TARGET_SPEED_BAND_LOWER_FT)
    ]

    if len(band) >= TARGET_SPEED_MIN_SAMPLES:
        return float(np.median(band["ias_s"]))

    if len(approach) > 0:
        return float(np.median(approach["ias_s"]))

    return float("nan")
