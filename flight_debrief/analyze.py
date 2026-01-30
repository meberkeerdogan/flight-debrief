from __future__ import annotations
from typing import IO, Dict, List, Tuple
import pandas as pd

from .domain import AircraftProfile, Event
from .preprocess import load_and_preprocess
from .approach import extract_approach_window
from .detect import detect_unstable

def analyze(csv_file: IO[bytes], runway_elev_m: float, profile: AircraftProfile):
    csv_file.seek(0)
    df = load_and_preprocess(csv_file, runway_elev_m=runway_elev_m, profile=profile)
    approach = extract_approach_window(df)
    if len(approach) == 0:
        return None, "No approach segment found in 1500..-20 ft AGL band."
    label, target, events, metrics = detect_unstable(approach, profile)
    return (approach, label, target, events, metrics), None

# Your UI now has exactly one job: call analyze() and display results.