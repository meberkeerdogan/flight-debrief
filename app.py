import streamlit as st
import pandas as pd

from flight_debrief.domain import AircraftProfile
from flight_debrief.analyze import analyze
from flight_debrief.render import make_plot_figure
from flight_debrief.airports import AIRPORTS

# -----------------------------
# Preset profiles
# -----------------------------
PRESET_PROFILES: dict[str, AircraftProfile] = {
    "C172 (MVP)": AircraftProfile(
        name="C172 (MVP)",
        gate_ft=500.0,
        speed_tol_kt=10.0,
        sink_rate_limit_fpm=-1000.0,
        bank_limit_deg=15.0,
        pitch_std_window_s=5.0,
        pitch_std_limit_deg=2.0,
        smooth_window_s=1.0,
        min_violation_duration_s=2.0,
    ),
    "Generic GA (strict)": AircraftProfile(
        name="Generic GA (strict)",
        gate_ft=500.0,
        speed_tol_kt=8.0,
        sink_rate_limit_fpm=-900.0,
        bank_limit_deg=12.0,
        pitch_std_window_s=5.0,
        pitch_std_limit_deg=1.8,
        smooth_window_s=1.0,
        min_violation_duration_s=2.0,
    ),
    "Turboprop-ish (looser)": AircraftProfile(
        name="Turboprop-ish (looser)",
        gate_ft=500.0,
        speed_tol_kt=12.0,
        sink_rate_limit_fpm=-1200.0,
        bank_limit_deg=20.0,
        pitch_std_window_s=5.0,
        pitch_std_limit_deg=2.5,
        smooth_window_s=1.0,
        min_violation_duration_s=2.0,
    ),
}


# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="Flight Debrief MVP", layout="wide")
st.title("✈️ Flight Debrief MVP — Unstable Approach Detector")
st.write("Upload a flight CSV, set runway elevation, and get an approach stability debrief.")


# -----------------------------
# Session-state helper
# -----------------------------
def _load_profile_into_state(p: AircraftProfile) -> None:
    st.session_state["p_name"] = p.name
    st.session_state["p_gate_ft"] = float(p.gate_ft)
    st.session_state["p_speed_tol_kt"] = float(p.speed_tol_kt)
    st.session_state["p_sink_rate_limit_fpm"] = float(p.sink_rate_limit_fpm)
    st.session_state["p_bank_limit_deg"] = float(p.bank_limit_deg)
    st.session_state["p_pitch_std_window_s"] = float(p.pitch_std_window_s)
    st.session_state["p_pitch_std_limit_deg"] = float(p.pitch_std_limit_deg)
    st.session_state["p_smooth_window_s"] = float(p.smooth_window_s)
    st.session_state["p_min_violation_duration_s"] = float(p.min_violation_duration_s)


# -----------------------------
# Sidebar: runway + profile selection/edit
# -----------------------------
with st.sidebar:
    st.header("Settings")
    
    st.subheader("Airport / Runway")
    airport_icao = st.selectbox("Airport", options=list(AIRPORTS.keys()), index=0)
    airport = AIRPORTS[airport_icao]

    runway_id = st.selectbox("Landing runway", options=list(airport.runways.keys()), index=0)
    runway = airport.runways[runway_id]

    runway_elev_m = runway.thr_elev_m
    runway_elev_m_default = runway.thr_elev_m
    st.caption(f"Default RWY {runway_id} threshold elev: {runway.thr_elev_ft:.1f} ft ({runway_elev_m_default:.1f} m)")

    override = st.checkbox("Override runway elevation", value=False)
    if override:
        runway_elev_m = st.number_input("Runway elevation (MSL) in meters", value=float(runway_elev_m_default), step=1.0)
    else:
        runway_elev_m = runway_elev_m_default


    st.divider()
    st.header("Aircraft profile")

    profile_name = st.selectbox("Preset", options=list(PRESET_PROFILES.keys()), index=0)
    preset = PRESET_PROFILES[profile_name]

    # Initialize state on first run or when preset changes
    if "selected_profile_name" not in st.session_state:
        st.session_state["selected_profile_name"] = profile_name
        _load_profile_into_state(preset)

    if st.session_state["selected_profile_name"] != profile_name:
        st.session_state["selected_profile_name"] = profile_name
        _load_profile_into_state(preset)

    edit = st.checkbox("Edit profile", value=False)
    reset = st.button("Reset to preset")

    if reset:
        _load_profile_into_state(preset)

    if edit:
        st.subheader("Edit values")
        st.text_input("Profile name", key="p_name")
        st.number_input("Gate altitude AGL (ft)", step=10.0, key="p_gate_ft")
        st.number_input("Speed tolerance (kt)", step=1.0, key="p_speed_tol_kt")
        st.number_input("Sink rate limit (fpm)", step=50.0, key="p_sink_rate_limit_fpm")
        st.number_input("Bank limit (deg)", step=1.0, key="p_bank_limit_deg")
        st.number_input("Pitch std window (s)", step=0.5, key="p_pitch_std_window_s")
        st.number_input("Pitch std limit (deg)", step=0.1, key="p_pitch_std_limit_deg")
        st.number_input("Smoothing window (s)", step=0.1, key="p_smooth_window_s")
        st.number_input("Min violation duration (s)", step=0.5, key="p_min_violation_duration_s")
    else:
        st.subheader("Preset values (read-only)")
        st.write(
            {
                "name": preset.name,
                "gate_ft": preset.gate_ft,
                "speed_tol_kt": preset.speed_tol_kt,
                "sink_rate_limit_fpm": preset.sink_rate_limit_fpm,
                "bank_limit_deg": preset.bank_limit_deg,
                "pitch_std_window_s": preset.pitch_std_window_s,
                "pitch_std_limit_deg": preset.pitch_std_limit_deg,
                "smooth_window_s": preset.smooth_window_s,
                "min_violation_duration_s": preset.min_violation_duration_s,
            }
        )


# Build the profile object AFTER sidebar widgets exist
if edit:
    profile = AircraftProfile(
        name=st.session_state["p_name"],
        gate_ft=float(st.session_state["p_gate_ft"]),
        speed_tol_kt=float(st.session_state["p_speed_tol_kt"]),
        sink_rate_limit_fpm=float(st.session_state["p_sink_rate_limit_fpm"]),
        bank_limit_deg=float(st.session_state["p_bank_limit_deg"]),
        pitch_std_window_s=float(st.session_state["p_pitch_std_window_s"]),
        pitch_std_limit_deg=float(st.session_state["p_pitch_std_limit_deg"]),
        smooth_window_s=float(st.session_state["p_smooth_window_s"]),
        min_violation_duration_s=float(st.session_state["p_min_violation_duration_s"]),
    )
else:
    profile = preset

st.caption(f"Active profile: **{profile.name}** (gate {profile.gate_ft:.0f} ft AGL)")


# -----------------------------
# Upload + preview
# -----------------------------
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to begin. Try one of your samples in data/samples/.")
    st.stop()

uploaded.seek(0)
df_preview = pd.read_csv(uploaded, nrows=20)

st.subheader("Raw preview (as uploaded)")

# Format only columns that exist (supports both schemas)
fmt = {}
for k, v in {
    "pitch_deg": "{:.3f}",
    "roll_deg": "{:.3f}",
    "vs_fps": "{:.2f}",
    "vs_fpm": "{:.1f}",
    "ias_kt": "{:.2f}",
    "alt_msl_ft": "{:.2f}",
    "alt_msl_m": "{:.2f}",
}.items():
    if k in df_preview.columns:
        fmt[k] = v

st.dataframe(df_preview.style.format(fmt), use_container_width=True)
uploaded.seek(0)


# -----------------------------
# Run analysis
# -----------------------------
try:
    result, err = analyze(uploaded, runway_elev_m=runway_elev_m, profile=profile, runway=runway)
except Exception as e:
    st.error(f"Error while analyzing file: {e}")
    st.stop()

if err or result is None:
    st.error(err or "Analysis failed (no result returned).")
    st.stop()

approach, label, target, events, metrics = result


# -----------------------------
# Display results
# -----------------------------
col1, col2, col3 = st.columns([1, 1, 1])
col1.metric("Result", label)
col2.metric("Target IAS (kt)", f"{target:.1f}" if pd.notna(target) else "N/A")
col3.metric("Overall severity", f"{metrics.get('overall_severity', 0.0):.1f}")

st.subheader("Summary metrics (below gate)")
st.json(metrics)

st.subheader("Detected events")
if len(events) == 0:
    st.success("No unstable events detected below the gate.")
else:
    df_events = pd.DataFrame(
        [
            {
                "rule": e.rule,
                "t_start": e.t_start,
                "t_end": e.t_end,
                "agl_start_ft": e.agl_start_ft,
                "agl_end_ft": e.agl_end_ft,
                "worst_value": e.worst_value,
                "severity": e.severity,
            }
            for e in events
        ]
    )
    st.dataframe(df_events, use_container_width=True)

st.subheader("Approach plot")
fig = make_plot_figure(approach, events, profile, target)
st.pyplot(fig, clear_figure=True)
