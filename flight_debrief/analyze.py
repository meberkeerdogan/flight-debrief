"""Pipeline orchestration for flight approach analysis."""

from __future__ import annotations
from typing import Optional

import pandas as pd

from .domain import AircraftProfile, Event
from .preprocess import load_and_preprocess
from .approach import extract_approach_window
from .detect import detect_unstable
from .airports import Runway


# Type alias for analysis result
AnalysisResult = tuple[pd.DataFrame, str, float, list[Event], dict[str, float]]


def analyze(
    csv_source,
    runway_elev_m: float,
    profile: AircraftProfile,
    runway: Optional[Runway] = None,
) -> tuple[Optional[AnalysisResult], Optional[str]]:
    """
    Run complete approach stability analysis pipeline.

    Orchestrates the full analysis workflow:
    1. Load and preprocess CSV telemetry
    2. Extract approach window
    3. Detect stability violations
    4. Compute severity scores

    Args:
        csv_source: Path or file-like object containing flight CSV
        runway_elev_m: Runway elevation in meters MSL
        profile: Aircraft profile with stability thresholds
        runway: Optional runway object

    Returns:
        Tuple of (result, error):
        - On success: ((approach_df, label, target, events, metrics), None)
        - On failure: (None, error_message)
    """
    try:
        df = load_and_preprocess(csv_source, runway_elev_m=runway_elev_m, profile=profile)
        approach = extract_approach_window(df)

        if len(approach) == 0:
            return None, "No approach segment found in 1500..-20 ft AGL band."

        label, target, events, metrics = detect_unstable(approach, profile, runway=runway)
        return (approach, label, target, events, metrics), None

    except Exception as e:
        return None, str(e)
