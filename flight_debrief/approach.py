from __future__ import annotations
import numpy as np
import pandas as pd


def extract_approach_window(df: pd.DataFrame) -> pd.DataFrame:    # Takes a pandas DataFrame containing the whole flight / Returns another DataFrame containing only the approach segment
    """
    Extract the final continuous segment where the aircraft is in the
    approach altitude band (1500..20 ft AGL). Don't require vs_s < -100,
    because flare/float would get cut out.
    """
    in_band = (df["agl_ft"] <= 1500) & (df["agl_ft"] >= -20)

    idx = np.where(in_band.to_numpy())[0]    # np.where(condition) returns the indices where condition is True. / But the return type is important: It returns a tuple of arrays, one array per dimension. / For a 1D condition array, it returns a tuple with one element: (np.array([2,3,4,...]),) / So np.where(...)[0] extracts that first (and only) array from the tuple.
    if len(idx) == 0:
        return df.iloc[0:0].copy()  # empty

    # Split into continuous index blocks
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    segments = np.split(idx, breaks)

    # Choose the segment that gets closest to the ground (lowest AGL)
    def seg_score(seg):
        agl_min = float(df.iloc[seg]["agl_ft"].min())
        length = len(seg)
        # prioritize lowest AGL, then longer segment
        return (agl_min, -length)

    best_seg = min(segments, key=seg_score)

    approach = df.iloc[best_seg].copy()
    approach.reset_index(drop=True, inplace=True)
    return approach




def estimate_target_speed(approach: pd.DataFrame) -> float:
    """
    Estimate target approach speed from the segment between 700 and 500 ft AGL.
    Fallback: overall median in approach window.
    """
    band = approach[(approach["agl_ft"] <= 700) & (approach["agl_ft"] >= 500)]    # So band contains only rows where altitude is between 700 and 500 ft AGL
    if len(band) >= 10:    # If there are at least 10 rows in that band / With too few samples, a median is unreliable
        return float(np.median(band["ias_s"]))
    return float(np.median(approach["ias_s"])) if len(approach) else float("nan")
"""
This function estimates the intended approach speed by taking the median smoothed 
airspeed in a stable altitude band (700-500 ft AGL), falling back to the whole approach if 
needed, and returning NaN only when no valid data exists.
"""

# This is "domain pipeline" and should be reusable outside any UI.