"""
File: mvp_detect_unstable_approach.py

What this script does:
- Loads a flight time-series CSV (synthetic or simulator log)
- Computes AGL (Altitude Above Ground Level) using runway elevation
- Extracts the approach window (1500 ft -> 20 ft AGL, descending)
- Estimates target approach speed from data (median IAS between 700-500 ft AGL)
- Detects unstable approach events below 500 ft AGL using simple rules:
    - speed deviation > 10 kt for >= 2 seconds
    - sink rate < -1000 fpm for >= 2 seconds
    - optional: bank > 15 deg for >= 2 seconds
    - pitch "chasing": high pitch variability (std dev) over 5 seconds
- Prints a short report
- Saves a plot highlighting unstable intervals

Where it should live:
- In your project root, e.g. flight-mvp/mvp_detect_unstable_approach.py

How to run:
python mvp_detect_unstable_approach.py --input data/samples/c172_approach_unstable.csv --out outputs/report_unstable

Or for the stable file:
python mvp_detect_unstable_approach.py --input data/samples/c172_approach_stable.csv --out outputs/report_stable
"""

from __future__ import annotations

import argparse     # parse CLI flags like --input ...
from dataclasses import dataclass   # lightweight “structs” with typed fields
from pathlib import Path    # sane file paths (better than string paths)
from typing import Dict, List, Tuple    # typing helpers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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



# -----------------------------
# Helpers
# -----------------------------

def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """Simple moving average guaranteed to return same length as x."""
    if win <= 1:    # If the window size is 1 or smaller: No smoothing needed / return a copy of the input array
        return x.copy()

    kernel = np.ones(win, dtype=float) / win

    left = win // 2
    right = win - 1 - left  # makes total pad = win-1
    xpad = np.pad(x, (left, right), mode="edge")

    y = np.convolve(xpad, kernel, mode="valid")  # length == len(x)
    return y



def rolling_std(x: np.ndarray, win: int) -> np.ndarray:
    """Rolling standard deviation (same length, edge-padded)."""
    if win <= 1:
        return np.zeros_like(x)
    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    out = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        w = xpad[i : i + win]
        out[i] = float(np.std(w))
    return out

"""
rolling_std computes the local variability of a signal.
Not:

“Is pitch high?”

“Is pitch low?”

But:

“Is pitch moving around a lot right now?”

That’s a completely different question.

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



def find_continuous_events(
    t: np.ndarray,     # numpy array of time stamps (seconds). Example: [0.0, 0.2, 0.4, ...]
    agl_ft: np.ndarray,     # numpy array of altitude above ground in feet, same length as t
    mask: np.ndarray,      # numpy array of booleans, same length. True means violation at that index.
    min_dur_s: float, # minimum duration in seconds required to count as an event (e.g., 2.0)
    dt: float, # time step between samples (e.g., 0.2 seconds)
    rule_name: str, # string describing which rule this mask belongs to (“High sink rate” etc.)
    worst_signal: np.ndarray,   # numeric array (same length) used to compute “how bad it got”
    worst_mode: str = "min",    # how to compute “worst”. Default "min".
) -> List[Event]: # Returns: a Python list of Event objects.
    """
    Convert a boolean mask into continuous events, requiring persistence for min_dur_s.

    worst_mode:
      - "min": worst_value is min(worst_signal) in event (good for sink rate)
      - "max": worst_value is max(worst_signal) in event
      - "absmax": worst_value is max(abs(worst_signal)) in event
    """

    """
    3 The mental model (THIS is the key)

    Imagine you slide along the mask like this:

    i →
    False False True True True True False False
                ↑           ↑
            start         end


    Whenever you see:

    False → True → start of an event

    True → False → end of an event

    This function finds those regions and asks:

    Did it last long enough?

    If yes, summarize it into ONE Event
    """


    events: List[Event] = [] # creates an empty list named events / This will be filled with Event(...) objects
    min_len = int(np.ceil(min_dur_s / dt)) # min_dur_s / dt converts seconds → number of samples. / np.ceil(...) rounds up (so you don’t accidentally accept shorter-than-required events).
    """
    Example:

    min_dur_s = 2.0

    dt = 0.2

    2.0 / 0.2 = 10

    ceil(10) = 10

    min_len = 10

    Meaning:

    We need at least 10 consecutive True samples to count as an event.
    """

    i = 0 # i is the current index we are examining in the mask
    n = len(mask) # n is the number of samples, we will scan indices 0 through n-1
    while i < n: # This loop continues until i reaches the end. / Inside this loop we do; If mask is False -> move forward by 1,, If mask is True -> find the entire continuous True block
        if not mask[i]: # If current sample is NOT a violation, skip it(True / False). 
            i += 1
            continue # skip the rest of the loop and start next iteration
            
            # If we are here, mask[i] is True (violation detected at index i)
        
        j = i # Goal: move j forward until violation stops
        while j < n and mask[j]: # j is still inside the array bounds AND mask[j] is True keep increasing j
            j += 1

        # So the continuous violation segment is from i (inclusive) to j (exclusive) [i, j)
        if (j - i) >= min_len: # Check of this segment is long enough to count
            # Extract the segment’s data (time, altitude, signal)
            seg_t = t[i:j] # all times during the event
            seg_agl = agl_ft[i:j] # all AGL values during the event
            seg_w = worst_signal[i:j] # all values of the signal you want to measure severity from

            if worst_mode == "min":    # Take the minimum value in seg_w. / Good for sink rate because more negative is worse.
                worst_val = float(np.min(seg_w))
            elif worst_mode == "max":    # Take maximum value. / Good for pitch std etc.
                worst_val = float(np.max(seg_w))
            elif worst_mode == "absmax":    # Take absolute value first, then max. / Good for “magnitude matters” like bank angle or speed error.
                worst_val = float(np.max(np.abs(seg_w)))
            else:
                worst_val = float(seg_w[-1])

            events.append(    # Create an Event object and store it 
                Event(
                    rule=rule_name,    # label this event with the rule’s name.
                    t_start=float(seg_t[0]),    # start time is first time in the segment.
                    t_end=float(seg_t[-1] + dt),    # end time is last time in the segment.
                    agl_start_ft=float(seg_agl[0]),    # AGL at start.
                    agl_end_ft=float(seg_agl[-1]),   # AGL at end.
                    worst_value=worst_val,    # the severity number we computed.
                )
            )
        # Then events.append(...) adds it to the list.
        i = j    # Move i to j to continue searching for next event.

    return events


def compute_event_severity(e: Event, profile: AircraftProfile) -> float:
    """
    Returns a 0-100 severity score for one event.

    Components:
    - magnitude factor: how far beyond the threshold
    - duration factor: longer violations are worse
    - altitude factor: low altitude violations are worse
    """
    duration_s = max(0.0, e.t_end - e.t_start)
    agl_min = min(e.agl_start_ft, e.agl_end_ft)  # descending so end is usually lower

    # --- Magnitude factor (0..1) ---
    # We convert "how far beyond limit" into a normalized ratio.
    # Then clamp to 1.0 so it cannot blow up.
    if e.rule == "High sink rate":
        # sink limit is negative: e.g. -1000 fpm.
        # worst_value is also negative: e.g. -1279 fpm.
        exceed = max(0.0, abs(e.worst_value) - abs(profile.sink_rate_limit_fpm))  # 279
        # Normalize: 500 fpm exceed -> full magnitude score (tunable)
        mag = min(1.0, exceed / 500.0)

    elif e.rule == "Speed out of band":
        exceed = max(0.0, e.worst_value - profile.speed_tol_kt)  # worst_value is abs(speed_err)
        mag = min(1.0, exceed / profile.speed_tol_kt)  # exceed by tol -> full

    elif e.rule == "Excessive bank":
        exceed = max(0.0, e.worst_value - profile.bank_limit_deg)  # worst_value is abs(roll)
        mag = min(1.0, exceed / profile.bank_limit_deg)

    elif e.rule == "Pitch chasing":
        exceed = max(0.0, e.worst_value - profile.pitch_std_limit_deg)  # worst_value is pitch_std
        mag = min(1.0, exceed / profile.pitch_std_limit_deg)

    else:
        mag = 0.0

    # --- Duration factor (0.5..1.0) ---
    # We don’t want duration to zero out severity; even a brief violation matters.
    # At 5 seconds or more, duration factor saturates.
    dur = 0.5 + 0.5 * min(1.0, duration_s / 5.0)

    # --- Altitude factor (0.5..1.0) ---
    # Below 500 ft gets progressively more weight as you get closer to the ground.
    # At 500 ft: factor ~0.5, at 0 ft: factor ~1.0.
    alt = 0.5 + 0.5 * min(1.0, max(0.0, (profile.gate_ft - agl_min) / profile.gate_ft))

    # Combine
    score = 100.0 * mag * dur * alt
    return float(score)


# -----------------------------
# Core pipeline
# -----------------------------
def load_and_preprocess(csv_path: Path, runway_elev_m: float, profile: AircraftProfile) -> pd.DataFrame:
    """
    Loads CSV and adds derived/smoothed signals.

    Expected CSV columns:
      t, alt_msl_m, ias_kt, vs_fpm, pitch_deg, roll_deg, throttle (hdg_deg optional)

    Returns a DataFrame with extra columns:
      agl_ft, ias_s, vs_s, pitch_s, roll_s, throttle_s, pitch_std
    """
    df = pd.read_csv(csv_path)    # Read the CSV (a DataFrame is like a table: columns + rows)

    
    # Convert logger time to seconds (already seconds in your CSV)
    df["t"] = df["Time"].astype(float)

    # Drop only *exact* duplicate rows (same time + same telemetry)
    dedupe_cols = ["t", "alt_msl_ft", "ias_kt", "vs_fps", "pitch_deg", "roll_deg", "throttle"]
    dedupe_cols = [c for c in dedupe_cols if c in df.columns]  # safety
    df = df.drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)

    # Sort by time
    df = df.sort_values("t").reset_index(drop=True)

    # Now enforce strictly increasing time by removing remaining duplicate timestamps
    # (If any remain, they must differ in values; you said they usually don't, but this makes dt safe.)
    df = df.loc[df["t"].diff().fillna(1.0) > 0].reset_index(drop=True)


    # Convert altitude feet -> meters
    df["alt_msl_m"] = df["alt_msl_ft"] * 0.3048

    # Convert vertical speed feet/sec -> feet/min
    df["vs_fpm"] = df["vs_fps"] * 60.0


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

    # Optional sanity check
    if dt > 0.5:
        print(f"Warning: large dt={dt:.3f}s detected; data may be low-rate or timestamps quantized.")


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

    df.attrs["dt"] = dt    # This puts dt into the DataFrame’s attribute dictionary. / Stores dt in DataFrame metadata (attrs). / That is a nice trick: downstream functions like detect_unstable can retrieve dt without recomputing it.

    print("Rows after cleanup:", len(df))
    print("dt:", df.attrs["dt"])
    print("t min/max:", df["t"].iloc[0], df["t"].iloc[-1])


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


def detect_unstable(approach: pd.DataFrame, profile: AircraftProfile) -> Tuple[str, float, List[Event], Dict[str, float]]:
    """
    Runs rule-based detection below profile.gate_ft.

    Returns:
      label: "Stable" or "Unstable"
      target_speed_kt
      events: list of Event
      metrics: summary numbers for report (dict)
    """
    if len(approach) < 5:
        return "Insufficient data", float("nan"), [], {}

    dt = float(approach.attrs.get("dt", 0.2))    # If dt exists, use it / otherwise use default 0.2
    # Convert DataFrame columns into NumPy arrays for efficient math.
    t = approach["t"].to_numpy(float)
    agl = approach["agl_ft"].to_numpy(float)

    target = estimate_target_speed(approach)    # Your speed rule measures deviation from "what the pilot was trying to fly", not from a fixed hardcoded number.

    below_gate = agl <= profile.gate_ft    # profile.gate_ft is 500.0 by default / Example: if agl = [600, 520, 480, 300] below_gate = [False, False, True, True]

    speed_err = approach["ias_s"].to_numpy(float) - target    # speed error relative to target
    vs = approach["vs_s"].to_numpy(float)    # smoothed vertical speed fom
    roll = approach["roll_s"].to_numpy(float)    # smoothed roll angle deg
    pitch_std = approach["pitch_std"].to_numpy(float)    # rolling variability of pitch (deg)
    # All are NumPy arrays

    # Rule masks (only apply below gate)
    speed_bad = below_gate & (np.abs(speed_err) > profile.speed_tol_kt)
    sink_bad = below_gate & (vs < profile.sink_rate_limit_fpm)
    bank_bad = below_gate & (np.abs(roll) > profile.bank_limit_deg)
    pitch_bad = below_gate & (pitch_std > profile.pitch_std_limit_deg)

    events: List[Event] = []    # Creates an empty list of Event objects
    events += find_continuous_events(t, agl, speed_bad, profile.min_violation_duration_s, dt,
                                     "Speed out of band", speed_err, worst_mode="absmax")
    events += find_continuous_events(t, agl, sink_bad, profile.min_violation_duration_s, dt,
                                     "High sink rate", vs, worst_mode="min")
    events += find_continuous_events(t, agl, bank_bad, profile.min_violation_duration_s, dt,
                                     "Excessive bank", roll, worst_mode="absmax")
    events += find_continuous_events(t, agl, pitch_bad, profile.min_violation_duration_s, dt,
                                     "Pitch chasing", pitch_std, worst_mode="max")


    events = sorted(events, key=lambda e: e.t_start)

    # Compute severity per event
    for e in events:
        e.severity = compute_event_severity(e, profile)

    # Combine into an overall severity (0..100)
    # We combine probabilistically so multiple events increase severity but never exceed 100.
    overall_severity = 100.0 * (1.0 - float(np.prod([1.0 - (e.severity / 100.0) for e in events]))) if events else 0.0


    # Basic label
    label = "Stable" if len(events) == 0 else "Unstable"    # If no events detected -> stable / If at least one event -> unstable


    """
    metrics = {"target_speed_kt": target, "overall_severity": float(overall_severity)}

    if np.any(below_gate):
        metrics.update({
            "pct_speed_bad": float(100 * np.mean(speed_bad[below_gate])),
            "pct_sink_bad": float(100 * np.mean(sink_bad[below_gate])),
            "pct_bank_bad": float(100 * np.mean(bank_bad[below_gate])),
            "pct_pitch_bad": float(100 * np.mean(pitch_bad[below_gate])),
        })

    """



    # Summary metrics (below gate)
    # Compute summary metrics below the gate
    # Percent of time below gate spent violating each rule
    if np.any(below_gate):    # np.any(...) returns True if at least one sample is below gate.
        metrics = {    # speed_bad[below_gate] => This selects only the values of speed_bad where below_gate is True. So You are looking ONLY at samples below gate
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


def write_report(out_base: Path, label: str, target: float, events: List[Event], metrics: Dict[str, float], profile: AircraftProfile) -> Path:
    """
    Writes a short text report to <out_base>.txt
    """
    out_path = out_base.with_suffix(".txt")    # Build the output file path
    lines = []    # Create a list to hold report lines
    lines.append(f"Approach Stability Report (MVP) — Aircraft profile: {profile.name}")
    lines.append(f"Gate: {profile.gate_ft:.0f} ft AGL")
    lines.append(f"Result: {label}")
    if np.isfinite(target):
        lines.append(f"Estimated target approach speed: {target:.1f} kt")
    lines.append("")

    if metrics:    # If metrics dictionary is not empty
        lines.append("Summary metrics below gate:")
        for k, v in metrics.items():
            if k == "target_speed_kt":
                continue
            if k == "overall_severity":
                lines.append(f"  - {k}: {v:.1f} (0-100)")
            else:
                lines.append(f"  - {k}: {v:.1f}%")
        lines.append("")

    if not events:
        lines.append("No unstable events detected below gate.")
    else:
        lines.append("Detected unstable events:")
        for e in events:
            lines.append(
                f"  - {e.rule}: t={e.t_start:.1f}s→{e.t_end:.1f}s, "
                f"AGL={e.agl_start_ft:.0f}→{e.agl_end_ft:.0f} ft, worst={e.worst_value:.2f}"
            )
        lines.append("")

    
    if "overall_severity" in metrics:
        lines.append(f"Overall severity (0-100): {metrics['overall_severity']:.1f}")
        lines.append("")

    if events:
        lines.append("Detected unstable events:")
        for e in events:
            lines.append(
                f"  - {e.rule}: t={e.t_start:.1f}s→{e.t_end:.1f}s, "
                f"AGL={e.agl_start_ft:.0f}→{e.agl_end_ft:.0f} ft, "
                f"worst={e.worst_value:.2f}, severity={e.severity:.1f}"
            )



    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path



def save_plot(out_base: Path, approach: pd.DataFrame, events: List[Event],
              profile: AircraftProfile, target_speed: float) -> Path:
    """
    Saves a multi-panel matplotlib plot to <out_base>.png highlighting unstable intervals.
    Panels: IAS, VS, Roll, Pitch std, and AGL.
    """
    out_path = out_base.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t = approach["t"].to_numpy(float)
    agl = approach["agl_ft"].to_numpy(float)
    ias = approach["ias_s"].to_numpy(float)
    vs  = approach["vs_s"].to_numpy(float)
    roll = approach["roll_s"].to_numpy(float)
    pstd = approach["pitch_std"].to_numpy(float)

    # 5 stacked panels, shared x
    fig, axes = plt.subplots(
        nrows=5, ncols=1, figsize=(14, 11), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.2, 1.0, 1.0, 1.0]}
    )
    ax_ias, ax_vs, ax_roll, ax_pstd, ax_agl = axes

    # --- Panel 1: IAS ---
    ax_ias.plot(t, ias, linewidth=2.0, label="IAS (kt)")
    if np.isfinite(target_speed):
        ax_ias.axhline(target_speed, linestyle="--", linewidth=1.5, label="Target IAS")
        ax_ias.axhline(target_speed + profile.speed_tol_kt, linestyle=":", linewidth=1.2, label="IAS band")
        ax_ias.axhline(target_speed - profile.speed_tol_kt, linestyle=":", linewidth=1.2)
    ax_ias.set_ylabel("IAS (kt)")
    ax_ias.grid(True, alpha=0.2)
    ax_ias.legend(loc="upper right")

    # --- Panel 2: Vertical speed ---
    ax_vs.plot(t, vs, linewidth=2.0, label="VS (fpm)")
    ax_vs.axhline(profile.sink_rate_limit_fpm, linestyle=":", linewidth=1.5, label="Sink limit")
    ax_vs.set_ylabel("VS (fpm)")
    ax_vs.grid(True, alpha=0.2)
    ax_vs.legend(loc="upper right")

    # --- Panel 3: Roll ---
    ax_roll.plot(t, roll, linewidth=2.0, label="Roll (deg)")
    ax_roll.axhline(profile.bank_limit_deg, linestyle=":", linewidth=1.5, label="Bank limit")
    ax_roll.axhline(-profile.bank_limit_deg, linestyle=":", linewidth=1.5)
    ax_roll.set_ylabel("Roll (deg)")
    ax_roll.grid(True, alpha=0.2)
    ax_roll.legend(loc="upper right")

    # --- Panel 4: Pitch std ---
    ax_pstd.plot(t, pstd, linewidth=2.0, linestyle="-.", label="Pitch std (deg)")
    ax_pstd.axhline(profile.pitch_std_limit_deg, linestyle=":", linewidth=1.5, label="Pitch std limit")
    ax_pstd.set_ylabel("Pitch std (deg)")
    ax_pstd.grid(True, alpha=0.2)
    ax_pstd.legend(loc="upper right")

    # --- Panel 5: AGL ---
    ax_agl.plot(t, agl, linewidth=2.0, linestyle="--", label="AGL (ft)")
    ax_agl.axhline(profile.gate_ft, linestyle="--", linewidth=1.5, label="Gate AGL")
    ax_agl.set_ylabel("AGL (ft)")
    ax_agl.set_xlabel("Time (s)")
    ax_agl.grid(True, alpha=0.2)
    ax_agl.legend(loc="upper right")

    # Shade unstable intervals across ALL panels
    for e in events:
        for ax in axes:
            ax.axvspan(e.t_start, e.t_end, alpha=0.18, color="grey")

    fig.suptitle(f"Approach Signals — Gate {profile.gate_ft:.0f} ft AGL (events shaded)", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path





# -----------------------------
# CLI entry point
# -----------------------------
def main():
    parser = argparse.ArgumentParser()    # creates an ArgumentParser object / argparse is Python’s standard library for command-line arguments.
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--out", type=str, required=True, help="Output base path (no extension)")
    parser.add_argument("--runway_elev_m", type=float, default=450.0, help="Runway elevation MSL in meters") # Default can be changed
    args = parser.parse_args()
    """
    Parses the actual command-line arguments and returns an object (a Namespace).
    After this, you can access:

    args.input

    args.out

    args.runway_elev_m
    Why: This is the moment the CLI becomes real values your program can use. If required args are missing or types are invalid, argparse prints a nice error and exits.
    """


    profile = AircraftProfile()    # Instantiates the AircraftProfile dataclass

    csv_path = Path(args.input)    # args.input and args.out are strings / Convert them into pathlib.Path objects
    out_base = Path(args.out)

    # Run preprocessing
    df = load_and_preprocess(csv_path, runway_elev_m=args.runway_elev_m, profile=profile)
    """
    This step:

    reads the CSV

    validates columns

    computes AGL

    computes dt

    smooths signals

    computes pitch variability

    Returns a fully prepared DataFrame.
    """
    

    # Extract approach window
    approach = extract_approach_window(df)
    
    # For debugging
    if len(approach) == 0:
        print("No approach segment found in 1500-20 ft AGL band.")
        return
    else:
        print("Approach t min/max:", approach["t"].iloc[0], approach["t"].iloc[-1])
        print("Approach AGL min/max:", approach["agl_ft"].min(), approach["agl_ft"].max())
        print("Approach rows:", len(approach))


    """
    This filters:

    only 1500 - 20 ft AGL

    only descending

    Everything else is discarded.
    """

    # Detect unstable approach
    label, target, events, metrics = detect_unstable(approach, profile)
    """
    This is the core analysis.

    It returns:

    label → Stable / Unstable

    target → estimated approach speed

    events → list of Event objects

    metrics → summary percentages

    Nothing is saved yet; this is pure computation.
    """


    # Write text report
    report_path = write_report(out_base, label, target, events, metrics, profile)    # Creates <out_base>.txt / Returns the file path
    plot_path = save_plot(out_base, approach, events, profile, target)    # Creates <out_base>.png / Visual explanation of the approach 

    print(f"Done.")
    print(f"Report: {report_path}")
    print(f"Plot:   {plot_path}")


if __name__ == "__main__":
    main()

    """
    What it means

    If this file is run directly:

    python mvp_detect_unstable_approach.py


    → main() runs

    If this file is imported:

    import mvp_detect_unstable_approach


    → main() does not run automatically

    This allows:

    reuse as a library

    unit testing

    safe imports
    """