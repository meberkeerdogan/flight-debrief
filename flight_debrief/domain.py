from __future__ import annotations
from dataclasses import dataclass


# -----------------------------
# Configuration / "Aircraft Profile"
# -----------------------------
@dataclass(frozen=True)     # immutable dataclass (frozen=True means immutable)
class AircraftProfile:
    name: str = "C172 (MVP)"
    gate_ft: float = 500.0  # below this altitude, you start judging stability (500 ft AGL)

    speed_tol_kt: float = 10.0 # Speed error band (± 10 kt)
    sink_rate_limit_fpm: float = -1000.0    # too-fast descent threshold (-1000 fpm) 
    bank_limit_deg: float = 15.0    # bank limit (±15°)

    pitch_std_window_s: float = 5.0     # “pitch chasing” proxy: rolling standard deviation of pitch over 5 seconds; if > 2°, call it unstable / not "pitch too high" but "pitch is oscillating a lot", which indicates over-control and poor energy management
    pitch_std_limit_deg: float = 2.0

    smooth_window_s: float = 1.0    # smoothing length (1 sec moving average)
    min_violation_duration_s: float = 2.0   # rule must persist for at least 2 seconds to become an event

@dataclass
class Event:    # This is a “detected violation segment”: name, time window, altitude window, and the worst observed value during that interval.
    rule: str   # Which rule triggered this event (e.g., "Speed out of band", "High sink rate", "Pitch chasing" etc.)
    t_start: float # The time (seconds) when the event started
    t_end: float # The time (seconds) when the event ended
    agl_start_ft: float # Altitude AGL (feet) at start of event
    agl_end_ft: float # Altitude AGL (feet) at end of event
    worst_value: float # How bad did it get during this event?
    severity: float = 0.0  # severity score 
# A dataclass, but not frozen, because you later do: e.severity = compute_event_severity(e, profile)

"""
For the worst_value field, its meaning depends on the rule:
| Rule              | worst_value meaning               |
| ----------------- | --------------------------------- |
| High sink rate    | most negative VS (e.g. -1279 fpm) |
| Speed out of band | largest speed error magnitude     |
| Excessive bank    | largest bank angle                |
| Pitch chasing     | highest pitch std                 |

During the approach, you don't just want to know “something went wrong”.

You want to know:

what went wrong (speed? sink rate? bank?)

when it happened

for how long

at what altitude

how bad it got

An Event is a structured record of one unstable episode.

Think of it as a flight incident log entry, not a single data point.

This a object that only stores data, not behavior (methods).
A dataclass is basically a nicer dictionary with types.

"""

# These are your "types", used everywhere. Centralizing them prevents circular imports and chaos.