"""Stability violation detection and severity scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .domain import AircraftProfile, Event
from .approach import estimate_target_speed
from .airports import Runway


# Severity calculation constants
SINK_RATE_EXCEED_FULL_SCORE_FPM = 500.0  # fpm over limit for full magnitude
DURATION_SATURATION_S = 5.0  # seconds at which duration factor saturates


def find_continuous_events(
    t: np.ndarray,
    agl_ft: np.ndarray,
    mask: np.ndarray,
    min_dur_s: float,
    dt: float,
    rule_name: str,
    worst_signal: np.ndarray,
    worst_mode: str = "min",
) -> list[Event]:
    """
    Convert a boolean violation mask into discrete events.

    Only violations lasting at least min_dur_s are counted as events.

    Args:
        t: Time stamps array (seconds)
        agl_ft: Altitude AGL array (feet)
        mask: Boolean array where True indicates violation
        min_dur_s: Minimum duration to count as event
        dt: Time step between samples
        rule_name: Name of the violated rule
        worst_signal: Signal array to compute worst value from
        worst_mode: How to compute worst value:
            - "min": minimum value (for sink rate)
            - "max": maximum value (for pitch std)
            - "absmax": maximum absolute value (for bank, speed error)

    Returns:
        List of Event objects for violations meeting duration threshold
    """
    events: list[Event] = []
    min_len = int(np.ceil(min_dur_s / dt))

    i = 0
    n = len(mask)

    while i < n:
        if not mask[i]:
            i += 1
            continue

        # Find end of continuous violation
        j = i
        while j < n and mask[j]:
            j += 1

        # Check duration requirement
        if (j - i) >= min_len:
            seg_t = t[i:j]
            seg_agl = agl_ft[i:j]
            seg_w = worst_signal[i:j]

            if worst_mode == "min":
                worst_val = float(np.min(seg_w))
            elif worst_mode == "max":
                worst_val = float(np.max(seg_w))
            elif worst_mode == "absmax":
                worst_val = float(np.max(np.abs(seg_w)))
            else:
                worst_val = float(seg_w[-1])

            events.append(
                Event(
                    rule=rule_name,
                    t_start=float(seg_t[0]),
                    t_end=float(seg_t[-1] + dt),
                    agl_start_ft=float(seg_agl[0]),
                    agl_end_ft=float(seg_agl[-1]),
                    worst_value=worst_val,
                )
            )

        i = j

    return events


def compute_event_severity(e: Event, profile: AircraftProfile) -> float:
    """
    Compute severity score (0-100) for an event.

    Severity is computed from three factors:
    - Magnitude: how far beyond the threshold
    - Duration: longer violations are worse (saturates at 5s)
    - Altitude: violations closer to ground are worse

    Args:
        e: The event to score
        profile: Aircraft profile with thresholds

    Returns:
        Severity score from 0 to 100
    """
    duration_s = max(0.0, e.t_end - e.t_start)
    agl_min = min(e.agl_start_ft, e.agl_end_ft)

    # Magnitude factor (0..1)
    if e.rule == "High sink rate":
        exceed = max(0.0, abs(e.worst_value) - abs(profile.sink_rate_limit_fpm))
        mag = min(1.0, exceed / SINK_RATE_EXCEED_FULL_SCORE_FPM)
    elif e.rule == "Speed out of band":
        exceed = max(0.0, e.worst_value - profile.speed_tol_kt)
        mag = min(1.0, exceed / profile.speed_tol_kt)
    elif e.rule == "Excessive bank":
        exceed = max(0.0, e.worst_value - profile.bank_limit_deg)
        mag = min(1.0, exceed / profile.bank_limit_deg)
    elif e.rule == "Pitch chasing":
        exceed = max(0.0, e.worst_value - profile.pitch_std_limit_deg)
        mag = min(1.0, exceed / profile.pitch_std_limit_deg)
    else:
        mag = 0.0

    # Duration factor (0.5..1.0) - even brief violations matter
    dur = 0.5 + 0.5 * min(1.0, duration_s / DURATION_SATURATION_S)

    # Altitude factor (0.5..1.0) - lower is worse
    alt = 0.5 + 0.5 * min(1.0, max(0.0, (profile.gate_ft - agl_min) / profile.gate_ft))

    return float(100.0 * mag * dur * alt)


def detect_unstable(
    approach: pd.DataFrame,
    profile: AircraftProfile,
    runway: Runway | None = None,
) -> tuple[str, float, list[Event], dict[str, float]]:
    """
    Run rule-based stability detection on approach data.

    Evaluates four stability rules below the gate altitude:
    - Speed: deviation from target speed
    - Sink rate: excessive descent rate
    - Bank: excessive roll angle
    - Pitch chasing: high pitch variability

    Args:
        approach: Preprocessed approach DataFrame
        profile: Aircraft profile with thresholds
        runway: Optional runway (for future use)

    Returns:
        Tuple of (label, target_speed, events, metrics):
        - label: "Stable", "Unstable", or "Insufficient data"
        - target_speed: Estimated target approach speed (kt)
        - events: List of detected violation events
        - metrics: Summary statistics dictionary
    """
    if len(approach) < 5:
        return "Insufficient data", float("nan"), [], {}

    dt = float(approach.attrs.get("dt", 0.2))

    t = approach["t"].to_numpy(float)
    agl = approach["agl_ft"].to_numpy(float)
    target = estimate_target_speed(approach)

    # Mask for data below gate altitude
    below_gate = agl <= profile.gate_ft

    # Compute violation signals
    speed_err = approach["ias_s"].to_numpy(float) - target
    vs = approach["vs_s"].to_numpy(float)
    roll = approach["roll_s"].to_numpy(float)
    pitch_std = approach["pitch_std"].to_numpy(float)

    # Rule masks (only apply below gate)
    speed_bad = below_gate & (np.abs(speed_err) > profile.speed_tol_kt)
    sink_bad = below_gate & (vs < profile.sink_rate_limit_fpm)
    bank_bad = below_gate & (np.abs(roll) > profile.bank_limit_deg)
    pitch_bad = below_gate & (pitch_std > profile.pitch_std_limit_deg)

    # Find events for each rule
    events: list[Event] = []
    events += find_continuous_events(
        t, agl, speed_bad, profile.min_violation_duration_s, dt,
        "Speed out of band", speed_err, worst_mode="absmax"
    )
    events += find_continuous_events(
        t, agl, sink_bad, profile.min_violation_duration_s, dt,
        "High sink rate", vs, worst_mode="min"
    )
    events += find_continuous_events(
        t, agl, bank_bad, profile.min_violation_duration_s, dt,
        "Excessive bank", roll, worst_mode="absmax"
    )
    events += find_continuous_events(
        t, agl, pitch_bad, profile.min_violation_duration_s, dt,
        "Pitch chasing", pitch_std, worst_mode="max"
    )
    events = sorted(events, key=lambda e: e.t_start)

    # Compute severity per event
    for e in events:
        e.severity = compute_event_severity(e, profile)

    # Combine into overall severity (probabilistic combination)
    if events:
        overall_severity = 100.0 * (
            1.0 - float(np.prod([1.0 - (e.severity / 100.0) for e in events]))
        )
    else:
        overall_severity = 0.0

    label = "Stable" if len(events) == 0 else "Unstable"

    # Compute summary metrics
    if np.any(below_gate):
        metrics = {
            "target_speed_kt": target,
            "pct_speed_bad": float(100 * np.mean(speed_bad[below_gate])),
            "pct_sink_bad": float(100 * np.mean(sink_bad[below_gate])),
            "pct_bank_bad": float(100 * np.mean(bank_bad[below_gate])),
            "pct_pitch_bad": float(100 * np.mean(pitch_bad[below_gate])),
            "overall_severity": float(overall_severity),
        }
    else:
        metrics = {
            "target_speed_kt": target,
            "overall_severity": float(overall_severity),
        }

    return label, target, events, metrics
