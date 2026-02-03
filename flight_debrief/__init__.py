"""
Flight Debrief - Approach Stability Analyzer

A toolkit for detecting unstable approaches in flight telemetry data.
Analyzes CSV logs from FlightGear to identify stability violations
such as excessive sink rate, speed deviations, bank angles, and
pitch oscillations.
"""

from .domain import AircraftProfile, Event
from .preprocess import load_and_preprocess, moving_average, rolling_std
from .approach import extract_approach_window, estimate_target_speed
from .detect import detect_unstable, find_continuous_events, compute_event_severity
from .analyze import analyze
from .render import make_plot_figure

__all__ = [
    # Domain models
    "AircraftProfile",
    "Event",
    # Preprocessing
    "load_and_preprocess",
    "moving_average",
    "rolling_std",
    # Approach extraction
    "extract_approach_window",
    "estimate_target_speed",
    # Detection
    "detect_unstable",
    "find_continuous_events",
    "compute_event_severity",
    # Pipeline
    "analyze",
    # Visualization
    "make_plot_figure",
]

__version__ = "0.1.0"
