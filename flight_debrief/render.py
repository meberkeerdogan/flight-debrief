from __future__ import annotations
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .domain import AircraftProfile, Event


def make_plot_figure(
    approach: pd.DataFrame,
    events: List[Event],
    profile: AircraftProfile,
    target_speed: float,
):
    t = approach["t"].to_numpy(float)
    agl = approach["agl_ft"].to_numpy(float)
    ias = approach["ias_s"].to_numpy(float)
    vs = approach["vs_s"].to_numpy(float)
    roll = approach["roll_s"].to_numpy(float)
    pstd = approach["pitch_std"].to_numpy(float)

    has_xtrack = "xtrack_ft" in approach.columns
    xtrack = approach["xtrack_ft"].to_numpy(float) if has_xtrack else None

    # --- Build layout ---
    if has_xtrack:
        nrows = 6
        height_ratios = [1.2, 1.2, 1.0, 1.0, 1.0, 1.0]
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=1,
            figsize=(14, 12),
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        ax_ias, ax_vs, ax_roll, ax_pstd, ax_xtrk, ax_agl = axes
    else:
        nrows = 5
        height_ratios = [1.2, 1.2, 1.0, 1.0, 1.0]
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=1,
            figsize=(14, 11),
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        ax_ias, ax_vs, ax_roll, ax_pstd, ax_agl = axes
        ax_xtrk = None

    # --- IAS ---
    ax_ias.plot(t, ias, linewidth=2.0, label="IAS (kt)")
    if np.isfinite(target_speed):
        ax_ias.axhline(target_speed, linestyle="--", linewidth=1.5, label="Target IAS")
        ax_ias.axhline(target_speed + profile.speed_tol_kt, linestyle=":", linewidth=1.2, label="IAS band")
        ax_ias.axhline(target_speed - profile.speed_tol_kt, linestyle=":", linewidth=1.2)
    ax_ias.set_ylabel("IAS (kt)")
    ax_ias.grid(True, alpha=0.2)
    ax_ias.legend(loc="upper right")

    # --- VS ---
    ax_vs.plot(t, vs, linewidth=2.0, label="VS (fpm)")
    ax_vs.axhline(profile.sink_rate_limit_fpm, linestyle=":", linewidth=1.5, label="Sink limit")
    ax_vs.set_ylabel("VS (fpm)")
    ax_vs.grid(True, alpha=0.2)
    ax_vs.legend(loc="upper right")

    # --- Roll ---
    ax_roll.plot(t, roll, linewidth=2.0, label="Roll (deg)")
    ax_roll.axhline(profile.bank_limit_deg, linestyle=":", linewidth=1.5, label="Bank limit")
    ax_roll.axhline(-profile.bank_limit_deg, linestyle=":", linewidth=1.5)
    ax_roll.set_ylabel("Roll (deg)")
    ax_roll.grid(True, alpha=0.2)
    ax_roll.legend(loc="upper right")

    # --- Pitch std ---
    ax_pstd.plot(t, pstd, linewidth=2.0, linestyle="-.", label="Pitch std (deg)")
    ax_pstd.axhline(profile.pitch_std_limit_deg, linestyle=":", linewidth=1.5, label="Pitch std limit")
    ax_pstd.set_ylabel("Pitch std (deg)")
    ax_pstd.grid(True, alpha=0.2)
    ax_pstd.legend(loc="upper right")

    # --- XTrack / Centerline deviation ---
    if has_xtrack and ax_xtrk is not None and xtrack is not None:
        ax_xtrk.plot(t, xtrack, linewidth=2.0, label="XTrack (ft)")
        ax_xtrk.axhline(0.0, linestyle="--", linewidth=1.5, label="Centerline")
        ax_xtrk.set_ylabel("XTrack (ft)")
        ax_xtrk.grid(True, alpha=0.2)
        ax_xtrk.legend(loc="upper right")

    # --- AGL ---
    ax_agl.plot(t, agl, linewidth=2.0, linestyle="--", label="AGL (ft)")
    ax_agl.axhline(profile.gate_ft, linestyle="--", linewidth=1.5, label="Gate AGL")
    ax_agl.set_ylabel("AGL (ft)")
    ax_agl.set_xlabel("Time (s)")
    ax_agl.grid(True, alpha=0.2)
    ax_agl.legend(loc="upper right")

    # --- Per-rule shading (THIS is the key fix) ---
    # Only shade the subplot that corresponds to the rule
    rule_to_ax = {
        "Speed out of band": ax_ias,
        "High sink rate": ax_vs,
        "Excessive bank": ax_roll,
        "Pitch chasing": ax_pstd,
        # Future: "Off centerline": ax_xtrk,
    }
    if ax_xtrk is not None:
        rule_to_ax["Off centerline"] = ax_xtrk  # if you add this rule later

    span_alpha = 0.18

    for e in events:
        ax = rule_to_ax.get(e.rule, None)
        if ax is not None:
            ax.axvspan(e.t_start, e.t_end, alpha=span_alpha, color="grey")

        # Optional: thin start/end markers on AGL only (doesn't grey out)
        ax_agl.axvline(e.t_start, alpha=0.12, color="grey", linewidth=1.0)
        ax_agl.axvline(e.t_end, alpha=0.12, color="grey", linewidth=1.0)

    fig.suptitle(f"Approach Signals — Gate {profile.gate_ft:.0f} ft AGL", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    return fig
