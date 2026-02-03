"""Tests for signal processing functions in preprocess.py"""

import numpy as np
import pandas as pd
import pytest
from io import StringIO

from flight_debrief.preprocess import moving_average, rolling_std, load_and_preprocess
from flight_debrief.domain import AircraftProfile


class TestMovingAverage:
    """Tests for the moving_average function."""

    def test_returns_same_length(self):
        """Output array should have same length as input."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        for win in [1, 2, 3, 5]:
            result = moving_average(x, win)
            assert len(result) == len(x)

    def test_window_one_returns_copy(self):
        """Window of 1 should return a copy of the input."""
        x = np.array([1.0, 2.0, 3.0])
        result = moving_average(x, 1)
        np.testing.assert_array_equal(result, x)
        # Verify it's a copy, not the same object
        assert result is not x

    def test_window_zero_returns_copy(self):
        """Window of 0 should return a copy of the input."""
        x = np.array([1.0, 2.0, 3.0])
        result = moving_average(x, 0)
        np.testing.assert_array_equal(result, x)

    def test_constant_signal_unchanged(self):
        """Constant signal should remain constant after smoothing."""
        x = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = moving_average(x, 3)
        np.testing.assert_array_almost_equal(result, x)

    def test_smooths_noisy_signal(self):
        """Moving average should reduce variance of noisy signal."""
        np.random.seed(42)
        x = np.random.randn(100) + 10  # noisy signal around 10
        result = moving_average(x, 5)
        # Smoothed signal should have lower variance
        assert np.var(result) < np.var(x)

    def test_known_values(self):
        """Test with known input/output values."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = moving_average(x, 3)
        # Center values should be exact averages
        assert result[1] == pytest.approx(2.0, rel=1e-10)  # (1+2+3)/3
        assert result[2] == pytest.approx(3.0, rel=1e-10)  # (2+3+4)/3
        assert result[3] == pytest.approx(4.0, rel=1e-10)  # (3+4+5)/3


class TestRollingStd:
    """Tests for the rolling_std function."""

    def test_returns_same_length(self):
        """Output array should have same length as input."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        for win in [1, 2, 3, 5]:
            result = rolling_std(x, win)
            assert len(result) == len(x)

    def test_window_one_returns_zeros(self):
        """Window of 1 should return all zeros (no variance possible)."""
        x = np.array([1.0, 2.0, 3.0])
        result = rolling_std(x, 1)
        np.testing.assert_array_equal(result, np.zeros_like(x))

    def test_constant_signal_zero_std(self):
        """Constant signal should have zero standard deviation."""
        x = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = rolling_std(x, 3)
        np.testing.assert_array_almost_equal(result, np.zeros_like(x))

    def test_oscillating_signal_high_std(self):
        """Oscillating signal should have higher std than stable signal."""
        stable = np.array([3.0, 3.1, 3.0, 3.1, 3.0])
        oscillating = np.array([2.0, 4.5, 1.8, 4.8, 2.1])

        stable_std = rolling_std(stable, 3)
        oscillating_std = rolling_std(oscillating, 3)

        # Oscillating signal should have higher mean std
        assert np.mean(oscillating_std) > np.mean(stable_std)

    def test_non_negative_output(self):
        """Standard deviation should always be non-negative."""
        x = np.array([-5.0, 10.0, -3.0, 8.0, -1.0])
        result = rolling_std(x, 3)
        assert np.all(result >= 0)


class TestLoadAndPreprocess:
    """Tests for the load_and_preprocess function."""

    @pytest.fixture
    def sample_csv(self):
        """Create a minimal valid CSV for testing."""
        return StringIO("""t,alt_msl_m,ias_kt,vs_fpm,pitch_deg,roll_deg,throttle
0.0,100.0,80.0,-500.0,3.0,0.0,0.5
0.2,99.0,81.0,-510.0,3.1,-1.0,0.5
0.4,98.0,80.0,-490.0,2.9,1.0,0.5
0.6,97.0,82.0,-520.0,3.2,0.0,0.5
0.8,96.0,79.0,-480.0,2.8,-0.5,0.5
1.0,95.0,80.0,-500.0,3.0,0.5,0.5
""")

    @pytest.fixture
    def profile(self):
        """Default aircraft profile for testing."""
        return AircraftProfile()

    def test_returns_dataframe(self, sample_csv, profile):
        """Should return a pandas DataFrame."""
        result = load_and_preprocess(sample_csv, runway_elev_m=50.0, profile=profile)
        assert isinstance(result, pd.DataFrame)

    def test_adds_agl_column(self, sample_csv, profile):
        """Should add agl_ft column."""
        result = load_and_preprocess(sample_csv, runway_elev_m=50.0, profile=profile)
        assert "agl_ft" in result.columns

    def test_agl_calculation(self, sample_csv, profile):
        """AGL should be MSL altitude minus runway elevation."""
        runway_elev_m = 50.0
        result = load_and_preprocess(sample_csv, runway_elev_m=runway_elev_m, profile=profile)
        # First row: alt_msl_m = 100.0, runway = 50.0, so AGL = 50m = ~164 ft
        expected_agl_ft = (100.0 - 50.0) * 3.28084
        assert result["agl_ft"].iloc[0] == pytest.approx(expected_agl_ft, rel=1e-5)

    def test_adds_smoothed_columns(self, sample_csv, profile):
        """Should add smoothed signal columns."""
        result = load_and_preprocess(sample_csv, runway_elev_m=50.0, profile=profile)
        expected_cols = ["ias_s", "vs_s", "pitch_s", "roll_s", "throttle_s", "pitch_std"]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_stores_dt_in_attrs(self, sample_csv, profile):
        """Should store estimated dt in DataFrame attrs."""
        result = load_and_preprocess(sample_csv, runway_elev_m=50.0, profile=profile)
        assert "dt" in result.attrs
        assert result.attrs["dt"] == pytest.approx(0.2, rel=1e-5)

    def test_handles_time_column_alias(self, profile):
        """Should accept 'Time' as alternative to 't'."""
        csv_with_Time = StringIO("""Time,alt_msl_m,ias_kt,vs_fpm,pitch_deg,roll_deg,throttle
0.0,100.0,80.0,-500.0,3.0,0.0,0.5
0.2,99.0,81.0,-510.0,3.1,-1.0,0.5
""")
        result = load_and_preprocess(csv_with_Time, runway_elev_m=50.0, profile=profile)
        assert "t" in result.columns

    def test_handles_altitude_in_feet(self, profile):
        """Should accept alt_msl_ft instead of alt_msl_m."""
        csv_with_feet = StringIO("""t,alt_msl_ft,ias_kt,vs_fpm,pitch_deg,roll_deg,throttle
0.0,328.0,80.0,-500.0,3.0,0.0,0.5
0.2,325.0,81.0,-510.0,3.1,-1.0,0.5
""")
        result = load_and_preprocess(csv_with_feet, runway_elev_m=50.0, profile=profile)
        assert "alt_msl_m" in result.columns

    def test_handles_vs_in_fps(self, profile):
        """Should accept vs_fps instead of vs_fpm."""
        csv_with_fps = StringIO("""t,alt_msl_m,ias_kt,vs_fps,pitch_deg,roll_deg,throttle
0.0,100.0,80.0,-8.33,3.0,0.0,0.5
0.2,99.0,81.0,-8.5,3.1,-1.0,0.5
""")
        result = load_and_preprocess(csv_with_fps, runway_elev_m=50.0, profile=profile)
        assert "vs_fpm" in result.columns
        # -8.33 fps * 60 = -500 fpm
        assert result["vs_fpm"].iloc[0] == pytest.approx(-8.33 * 60, rel=1e-2)

    def test_raises_on_missing_time(self, profile):
        """Should raise ValueError if no time column found."""
        bad_csv = StringIO("""alt_msl_m,ias_kt,vs_fpm,pitch_deg,roll_deg,throttle
100.0,80.0,-500.0,3.0,0.0,0.5
""")
        with pytest.raises(ValueError, match="No time column found"):
            load_and_preprocess(bad_csv, runway_elev_m=50.0, profile=profile)

    def test_raises_on_missing_altitude(self, profile):
        """Should raise ValueError if no altitude column found."""
        bad_csv = StringIO("""t,ias_kt,vs_fpm,pitch_deg,roll_deg,throttle
0.0,80.0,-500.0,3.0,0.0,0.5
""")
        with pytest.raises(ValueError, match="No altitude column found"):
            load_and_preprocess(bad_csv, runway_elev_m=50.0, profile=profile)

    def test_removes_duplicate_timestamps(self, profile):
        """Should remove duplicate timestamps."""
        csv_with_dupes = StringIO("""t,alt_msl_m,ias_kt,vs_fpm,pitch_deg,roll_deg,throttle
0.0,100.0,80.0,-500.0,3.0,0.0,0.5
0.0,100.0,80.0,-500.0,3.0,0.0,0.5
0.2,99.0,81.0,-510.0,3.1,-1.0,0.5
""")
        result = load_and_preprocess(csv_with_dupes, runway_elev_m=50.0, profile=profile)
        assert len(result) == 2  # Duplicate removed
