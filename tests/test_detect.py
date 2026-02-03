"""Tests for event detection and severity scoring in detect.py"""

import numpy as np
import pandas as pd
import pytest

from flight_debrief.detect import find_continuous_events, compute_event_severity, detect_unstable
from flight_debrief.domain import AircraftProfile, Event


class TestFindContinuousEvents:
    """Tests for the find_continuous_events function."""

    def test_no_violations_returns_empty(self):
        """No violations should return empty list."""
        t = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
        agl = np.array([400.0, 380.0, 360.0, 340.0, 320.0])
        mask = np.array([False, False, False, False, False])
        signal = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        events = find_continuous_events(
            t, agl, mask, min_dur_s=0.4, dt=0.2, rule_name="Test", worst_signal=signal
        )
        assert events == []

    def test_short_violation_filtered_out(self):
        """Violation shorter than min_dur_s should be filtered."""
        t = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
        agl = np.array([400.0, 380.0, 360.0, 340.0, 320.0])
        mask = np.array([False, True, False, False, False])  # Only 0.2s violation
        signal = np.array([0.0, -1200.0, 0.0, 0.0, 0.0])

        events = find_continuous_events(
            t, agl, mask, min_dur_s=0.4, dt=0.2, rule_name="Test", worst_signal=signal
        )
        assert events == []

    def test_long_violation_detected(self):
        """Violation longer than min_dur_s should be detected."""
        t = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        agl = np.array([400.0, 380.0, 360.0, 340.0, 320.0, 300.0])
        mask = np.array([False, True, True, True, True, False])  # 0.8s violation
        signal = np.array([0.0, -1100.0, -1200.0, -1150.0, -1100.0, 0.0])

        events = find_continuous_events(
            t, agl, mask, min_dur_s=0.4, dt=0.2, rule_name="High sink rate", worst_signal=signal
        )
        assert len(events) == 1
        assert events[0].rule == "High sink rate"
        assert events[0].t_start == pytest.approx(0.2)
        assert events[0].t_end == pytest.approx(1.0)  # t_end = last_t + dt

    def test_multiple_events_detected(self):
        """Multiple separate violations should create multiple events."""
        t = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
        agl = np.array([400.0, 380.0, 360.0, 340.0, 320.0, 300.0, 280.0, 260.0, 240.0, 220.0])
        # Two separate violations
        mask = np.array([True, True, True, False, False, False, True, True, True, False])
        signal = np.array([-1100.0, -1200.0, -1150.0, 0.0, 0.0, 0.0, -1300.0, -1250.0, -1280.0, 0.0])

        events = find_continuous_events(
            t, agl, mask, min_dur_s=0.4, dt=0.2, rule_name="High sink rate", worst_signal=signal
        )
        assert len(events) == 2

    def test_worst_mode_min(self):
        """worst_mode='min' should return minimum value."""
        t = np.array([0.0, 0.2, 0.4, 0.6])
        agl = np.array([400.0, 380.0, 360.0, 340.0])
        mask = np.array([True, True, True, True])
        signal = np.array([-1100.0, -1300.0, -1200.0, -1150.0])  # min is -1300

        events = find_continuous_events(
            t, agl, mask, min_dur_s=0.4, dt=0.2, rule_name="Test", worst_signal=signal, worst_mode="min"
        )
        assert events[0].worst_value == pytest.approx(-1300.0)

    def test_worst_mode_max(self):
        """worst_mode='max' should return maximum value."""
        t = np.array([0.0, 0.2, 0.4, 0.6])
        agl = np.array([400.0, 380.0, 360.0, 340.0])
        mask = np.array([True, True, True, True])
        signal = np.array([1.5, 2.8, 2.2, 1.9])  # max is 2.8

        events = find_continuous_events(
            t, agl, mask, min_dur_s=0.4, dt=0.2, rule_name="Test", worst_signal=signal, worst_mode="max"
        )
        assert events[0].worst_value == pytest.approx(2.8)

    def test_worst_mode_absmax(self):
        """worst_mode='absmax' should return max of absolute values."""
        t = np.array([0.0, 0.2, 0.4, 0.6])
        agl = np.array([400.0, 380.0, 360.0, 340.0])
        mask = np.array([True, True, True, True])
        signal = np.array([-15.0, 12.0, -18.0, 10.0])  # absmax is 18

        events = find_continuous_events(
            t, agl, mask, min_dur_s=0.4, dt=0.2, rule_name="Test", worst_signal=signal, worst_mode="absmax"
        )
        assert events[0].worst_value == pytest.approx(18.0)

    def test_event_agl_values(self):
        """Event should capture AGL at start and end."""
        t = np.array([0.0, 0.2, 0.4, 0.6])
        agl = np.array([450.0, 420.0, 390.0, 360.0])
        mask = np.array([True, True, True, True])
        signal = np.array([1.0, 1.0, 1.0, 1.0])

        events = find_continuous_events(
            t, agl, mask, min_dur_s=0.4, dt=0.2, rule_name="Test", worst_signal=signal
        )
        assert events[0].agl_start_ft == pytest.approx(450.0)
        assert events[0].agl_end_ft == pytest.approx(360.0)


class TestComputeEventSeverity:
    """Tests for the compute_event_severity function."""

    @pytest.fixture
    def profile(self):
        """Default aircraft profile."""
        return AircraftProfile()

    def test_sink_rate_severity(self, profile):
        """High sink rate event should have non-zero severity."""
        event = Event(
            rule="High sink rate",
            t_start=0.0,
            t_end=3.0,  # 3 second duration
            agl_start_ft=400.0,
            agl_end_ft=300.0,
            worst_value=-1300.0,  # 300 fpm over limit
        )
        severity = compute_event_severity(event, profile)
        assert 0 < severity <= 100

    def test_speed_severity(self, profile):
        """Speed violation should have non-zero severity."""
        event = Event(
            rule="Speed out of band",
            t_start=0.0,
            t_end=2.0,
            agl_start_ft=400.0,
            agl_end_ft=350.0,
            worst_value=15.0,  # 15 kt error (5 over tolerance)
        )
        severity = compute_event_severity(event, profile)
        assert 0 < severity <= 100

    def test_bank_severity(self, profile):
        """Excessive bank should have non-zero severity."""
        event = Event(
            rule="Excessive bank",
            t_start=0.0,
            t_end=2.5,
            agl_start_ft=350.0,
            agl_end_ft=300.0,
            worst_value=22.0,  # 22 degrees (7 over limit)
        )
        severity = compute_event_severity(event, profile)
        assert 0 < severity <= 100

    def test_pitch_chasing_severity(self, profile):
        """Pitch chasing should have non-zero severity."""
        event = Event(
            rule="Pitch chasing",
            t_start=0.0,
            t_end=4.0,
            agl_start_ft=450.0,
            agl_end_ft=350.0,
            worst_value=3.5,  # 3.5 deg std (1.5 over limit)
        )
        severity = compute_event_severity(event, profile)
        assert 0 < severity <= 100

    def test_longer_duration_higher_severity(self, profile):
        """Longer violations should have higher severity (all else equal)."""
        short_event = Event(
            rule="High sink rate",
            t_start=0.0,
            t_end=1.0,  # 1 second
            agl_start_ft=400.0,
            agl_end_ft=380.0,
            worst_value=-1300.0,
        )
        long_event = Event(
            rule="High sink rate",
            t_start=0.0,
            t_end=5.0,  # 5 seconds
            agl_start_ft=400.0,
            agl_end_ft=380.0,
            worst_value=-1300.0,
        )
        short_severity = compute_event_severity(short_event, profile)
        long_severity = compute_event_severity(long_event, profile)
        assert long_severity > short_severity

    def test_lower_altitude_higher_severity(self, profile):
        """Violations at lower altitude should have higher severity."""
        high_event = Event(
            rule="High sink rate",
            t_start=0.0,
            t_end=3.0,
            agl_start_ft=480.0,
            agl_end_ft=450.0,  # Higher altitude
            worst_value=-1300.0,
        )
        low_event = Event(
            rule="High sink rate",
            t_start=0.0,
            t_end=3.0,
            agl_start_ft=150.0,
            agl_end_ft=100.0,  # Lower altitude
            worst_value=-1300.0,
        )
        high_severity = compute_event_severity(high_event, profile)
        low_severity = compute_event_severity(low_event, profile)
        assert low_severity > high_severity

    def test_severity_bounded_0_to_100(self, profile):
        """Severity should always be between 0 and 100."""
        # Extreme event
        extreme_event = Event(
            rule="High sink rate",
            t_start=0.0,
            t_end=10.0,
            agl_start_ft=100.0,
            agl_end_ft=20.0,
            worst_value=-2500.0,  # Very bad
        )
        severity = compute_event_severity(extreme_event, profile)
        assert 0 <= severity <= 100


class TestDetectUnstable:
    """Tests for the detect_unstable function."""

    @pytest.fixture
    def profile(self):
        """Default aircraft profile."""
        return AircraftProfile()

    @pytest.fixture
    def stable_approach(self):
        """Create a stable approach DataFrame."""
        n = 50
        t = np.arange(0, n * 0.2, 0.2)
        df = pd.DataFrame({
            "t": t,
            "agl_ft": np.linspace(600, 100, n),
            "ias_s": np.full(n, 75.0),  # Steady speed
            "vs_s": np.full(n, -700.0),  # Normal descent
            "roll_s": np.full(n, 0.0),  # Wings level
            "pitch_std": np.full(n, 0.5),  # Low variability
        })
        df.attrs["dt"] = 0.2
        return df

    @pytest.fixture
    def unstable_approach(self):
        """Create an unstable approach DataFrame with violations."""
        n = 50
        t = np.arange(0, n * 0.2, 0.2)
        agl = np.linspace(600, 100, n)

        # Create violations below 500 ft gate
        vs = np.full(n, -700.0)
        vs[agl < 400] = -1200.0  # High sink rate below 400 ft

        df = pd.DataFrame({
            "t": t,
            "agl_ft": agl,
            "ias_s": np.full(n, 75.0),
            "vs_s": vs,
            "roll_s": np.full(n, 0.0),
            "pitch_std": np.full(n, 0.5),
        })
        df.attrs["dt"] = 0.2
        return df

    def test_stable_approach_returns_stable(self, stable_approach, profile):
        """Stable approach should return 'Stable' label."""
        label, target, events, metrics = detect_unstable(stable_approach, profile)
        assert label == "Stable"
        assert len(events) == 0

    def test_unstable_approach_returns_unstable(self, unstable_approach, profile):
        """Unstable approach should return 'Unstable' label."""
        label, target, events, metrics = detect_unstable(unstable_approach, profile)
        assert label == "Unstable"
        assert len(events) > 0

    def test_returns_target_speed(self, stable_approach, profile):
        """Should return estimated target speed."""
        label, target, events, metrics = detect_unstable(stable_approach, profile)
        assert target == pytest.approx(75.0, rel=0.1)

    def test_returns_metrics_dict(self, stable_approach, profile):
        """Should return metrics dictionary."""
        label, target, events, metrics = detect_unstable(stable_approach, profile)
        assert isinstance(metrics, dict)
        assert "target_speed_kt" in metrics
        assert "overall_severity" in metrics

    def test_insufficient_data(self, profile):
        """Should handle insufficient data gracefully."""
        tiny_df = pd.DataFrame({
            "t": [0.0, 0.2],
            "agl_ft": [500.0, 480.0],
            "ias_s": [75.0, 75.0],
            "vs_s": [-700.0, -700.0],
            "roll_s": [0.0, 0.0],
            "pitch_std": [0.5, 0.5],
        })
        tiny_df.attrs["dt"] = 0.2
        label, target, events, metrics = detect_unstable(tiny_df, profile)
        assert label == "Insufficient data"

    def test_events_sorted_by_time(self, profile):
        """Events should be sorted by start time."""
        n = 100
        t = np.arange(0, n * 0.2, 0.2)
        agl = np.linspace(600, 50, n)

        # Create multiple violations at different times
        roll = np.zeros(n)
        roll[(agl < 450) & (agl > 350)] = 20.0  # Bank violation
        roll[(agl < 200)] = 25.0  # Another bank violation

        df = pd.DataFrame({
            "t": t,
            "agl_ft": agl,
            "ias_s": np.full(n, 75.0),
            "vs_s": np.full(n, -700.0),
            "roll_s": roll,
            "pitch_std": np.full(n, 0.5),
        })
        df.attrs["dt"] = 0.2

        label, target, events, metrics = detect_unstable(df, profile)

        # Verify events are sorted
        for i in range(1, len(events)):
            assert events[i].t_start >= events[i - 1].t_start

    def test_overall_severity_computation(self, unstable_approach, profile):
        """Overall severity should be computed from event severities."""
        label, target, events, metrics = detect_unstable(unstable_approach, profile)
        assert "overall_severity" in metrics
        assert 0 <= metrics["overall_severity"] <= 100

        # If there are events, severity should be > 0
        if len(events) > 0:
            assert metrics["overall_severity"] > 0
