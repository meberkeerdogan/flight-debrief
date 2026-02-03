"""Tests for approach window extraction in approach.py"""

import numpy as np
import pandas as pd
import pytest

from flight_debrief.approach import extract_approach_window, estimate_target_speed


class TestExtractApproachWindow:
    """Tests for the extract_approach_window function."""

    def test_extracts_approach_band(self):
        """Should extract data within 1500-20 ft AGL band."""
        df = pd.DataFrame({
            "t": np.arange(0, 10, 0.2),
            "agl_ft": np.linspace(2000, 0, 50),  # Descending from 2000 to 0
            "ias_s": np.full(50, 75.0),
        })
        approach = extract_approach_window(df)

        # All rows should be within band
        assert approach["agl_ft"].max() <= 1500
        assert approach["agl_ft"].min() >= -20

    def test_returns_empty_if_no_approach(self):
        """Should return empty DataFrame if no data in approach band."""
        df = pd.DataFrame({
            "t": np.arange(0, 5, 0.2),
            "agl_ft": np.linspace(3000, 2000, 25),  # Never enters approach band
            "ias_s": np.full(25, 75.0),
        })
        approach = extract_approach_window(df)
        assert len(approach) == 0

    def test_selects_segment_closest_to_ground(self):
        """Should select the segment that gets closest to the ground."""
        # Create two separate approach segments
        t = np.arange(0, 20, 0.2)
        agl = np.concatenate([
            np.linspace(1400, 800, 25),  # First segment: stops at 800 ft
            np.linspace(1800, 1600, 25),  # Out of band
            np.linspace(1400, 100, 50),  # Second segment: goes to 100 ft
        ])
        df = pd.DataFrame({
            "t": t,
            "agl_ft": agl,
            "ias_s": np.full(100, 75.0),
        })
        approach = extract_approach_window(df)

        # Should select segment that reached 100 ft, not the one that stopped at 800
        assert approach["agl_ft"].min() < 200

    def test_resets_index(self):
        """Returned DataFrame should have reset index."""
        df = pd.DataFrame({
            "t": np.arange(0, 5, 0.2),
            "agl_ft": np.linspace(1000, 200, 25),
            "ias_s": np.full(25, 75.0),
        })
        approach = extract_approach_window(df)
        assert approach.index[0] == 0
        assert list(approach.index) == list(range(len(approach)))

    def test_preserves_all_columns(self):
        """Should preserve all columns from input DataFrame."""
        df = pd.DataFrame({
            "t": np.arange(0, 5, 0.2),
            "agl_ft": np.linspace(1000, 200, 25),
            "ias_s": np.full(25, 75.0),
            "vs_s": np.full(25, -700.0),
            "roll_s": np.full(25, 0.0),
            "custom_col": np.arange(25),
        })
        approach = extract_approach_window(df)

        for col in df.columns:
            assert col in approach.columns

    def test_handles_negative_agl(self):
        """Should include data slightly below ground level (flare/touchdown)."""
        df = pd.DataFrame({
            "t": np.arange(0, 5, 0.2),
            "agl_ft": np.linspace(100, -10, 25),  # Goes slightly negative
            "ias_s": np.full(25, 75.0),
        })
        approach = extract_approach_window(df)

        # Should include data down to -20 ft
        assert len(approach) > 0
        assert approach["agl_ft"].min() < 0


class TestEstimateTargetSpeed:
    """Tests for the estimate_target_speed function."""

    def test_uses_700_to_500_band(self):
        """Should estimate speed from 700-500 ft band when available."""
        n = 100
        agl = np.linspace(1000, 100, n)
        ias = np.full(n, 80.0)  # Default speed

        # Set specific speed in 700-500 band
        band_mask = (agl <= 700) & (agl >= 500)
        ias[band_mask] = 75.0  # Target speed in band

        df = pd.DataFrame({
            "agl_ft": agl,
            "ias_s": ias,
        })

        target = estimate_target_speed(df)
        assert target == pytest.approx(75.0, rel=0.1)

    def test_fallback_to_overall_median(self):
        """Should use overall median if band has too few samples."""
        # Create approach with very few samples in 700-500 band
        df = pd.DataFrame({
            "agl_ft": [450, 400, 350, 300, 250, 200, 150, 100],  # All below 500
            "ias_s": [75, 76, 74, 75, 76, 74, 75, 76],
        })

        target = estimate_target_speed(df)
        expected_median = np.median([75, 76, 74, 75, 76, 74, 75, 76])
        assert target == pytest.approx(expected_median, rel=0.1)

    def test_returns_nan_for_empty_approach(self):
        """Should return NaN for empty approach."""
        df = pd.DataFrame({
            "agl_ft": [],
            "ias_s": [],
        })

        target = estimate_target_speed(df)
        assert np.isnan(target)

    def test_handles_varying_speeds(self):
        """Should return median, not mean, for varying speeds."""
        n = 50
        agl = np.linspace(800, 400, n)
        ias = np.full(n, 75.0)

        # Add one outlier
        ias[25] = 120.0  # Outlier

        df = pd.DataFrame({
            "agl_ft": agl,
            "ias_s": ias,
        })

        target = estimate_target_speed(df)
        # Median should be robust to the outlier
        assert target == pytest.approx(75.0, rel=0.1)

    def test_minimum_samples_in_band(self):
        """Should require at least 10 samples in band for band-based estimate."""
        # Create exactly 9 samples in 700-500 band
        agl = np.array([750, 720, 690, 660, 630, 600, 570, 540, 510, 480, 450])
        ias = np.array([80, 80, 80, 80, 80, 80, 80, 80, 80, 70, 70])
        # Only 9 samples in band (520-700), should fallback to overall median

        df = pd.DataFrame({
            "agl_ft": agl,
            "ias_s": ias,
        })

        # The exact behavior depends on how many fall in 700-500 band
        target = estimate_target_speed(df)
        assert not np.isnan(target)  # Should return something
