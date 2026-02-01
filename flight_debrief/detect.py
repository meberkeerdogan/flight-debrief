from . domain import AircraftProfile, Event
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from .approach import estimate_target_speed
from .airports import Runway

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
    min_len = int(np.ceil(min_dur_s / dt)) # min_dur_s / dt converts seconds → number of samples. / np.ceil(...) rounds up if dt doesn't divide evenly (so you don’t accidentally accept shorter-than-required events).
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



def detect_unstable(approach: pd.DataFrame, profile: AircraftProfile, runway: Optional[Runway] = None) -> Tuple[str, float, List[Event], Dict[str, float]]:
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


# Now your "brains" are cleanly isolated