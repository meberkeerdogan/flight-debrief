from __future__ import annotations
from typing import Optional, Tuple, Any

import pandas as pd

from .domain import AircraftProfile
from .preprocess import load_and_preprocess
from .approach import extract_approach_window
from .detect import detect_unstable
from .airports import Runway


def analyze(
    csv_source,
    runway_elev_m: float,
    profile: AircraftProfile,
    runway: Optional[Runway] = None,
) -> Tuple[Optional[tuple], Optional[str]]:
    """
    Orchestrates the pipeline.

    Returns:
      (approach_df, label, target, events, metrics), None
    or:
      None, "error message"
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

# Your UI now has exactly one job: call analyze() and display results.