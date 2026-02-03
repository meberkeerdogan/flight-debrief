"""Domain models for flight approach stability analysis."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class AircraftProfile:
    """
    Aircraft-specific stability thresholds and detection parameters.

    Attributes:
        name: Profile identifier
        gate_ft: Altitude AGL below which stability is evaluated
        speed_tol_kt: Allowed speed deviation from target (±)
        sink_rate_limit_fpm: Maximum descent rate (negative value)
        bank_limit_deg: Maximum bank angle (±)
        pitch_std_window_s: Window for pitch variability calculation
        pitch_std_limit_deg: Maximum allowed pitch standard deviation
        smooth_window_s: Signal smoothing window duration
        min_violation_duration_s: Minimum duration to count as violation
    """
    name: str = "C172 (MVP)"
    gate_ft: float = 500.0
    speed_tol_kt: float = 10.0
    sink_rate_limit_fpm: float = -1000.0
    bank_limit_deg: float = 15.0
    pitch_std_window_s: float = 5.0
    pitch_std_limit_deg: float = 2.0
    smooth_window_s: float = 1.0
    min_violation_duration_s: float = 2.0


@dataclass
class Event:
    """
    A detected stability violation during approach.

    Attributes:
        rule: Which stability rule was violated
        t_start: Event start time (seconds)
        t_end: Event end time (seconds)
        agl_start_ft: Altitude AGL at event start
        agl_end_ft: Altitude AGL at event end
        worst_value: Most extreme value during violation
        severity: Computed severity score (0-100)
    """
    rule: str
    t_start: float
    t_end: float
    agl_start_ft: float
    agl_end_ft: float
    worst_value: float
    severity: float = 0.0