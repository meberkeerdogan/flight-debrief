# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Flight Debrief is a flight approach stability analyzer that detects unstable approaches from FlightGear flight simulator telemetry. It processes CSV logs to identify stability violations (excessive sink rate, speed deviations, bank angles, pitch oscillations) and computes severity scores.

## Commands

```bash
# Activate virtual environment (required for all commands)
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Run the Streamlit web app
streamlit run app.py

# Run all tests
pytest

# Run tests with coverage
pytest --cov=flight_debrief --cov-report=term-missing

# Run a single test file
pytest tests/test_detect.py -v

# Run a specific test
pytest tests/test_detect.py::TestFindContinuousEvents::test_long_violation_detected -v

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # includes pytest
```

## Architecture

### Data Pipeline

```
CSV Input → preprocess.py → approach.py → detect.py → render.py → Streamlit UI
```

1. **preprocess.py**: Loads CSV, normalizes columns (handles multiple formats), computes AGL altitude, applies signal smoothing, calculates pitch variability (rolling std)

2. **approach.py**: Extracts the approach segment (1500 to -20 ft AGL band), estimates target approach speed from 700-500 ft band

3. **detect.py**: Applies four stability rules below gate altitude (500 ft), finds continuous violations, computes per-event and overall severity scores

4. **analyze.py**: Orchestrates the pipeline, returns `(result, error)` tuple pattern

5. **render.py**: Creates matplotlib figure with 5 subplots, shades violation events on relevant panels

### Domain Models (domain.py)

- `AircraftProfile`: Immutable dataclass with all detection thresholds (gate altitude, speed tolerance, sink rate limit, bank limit, pitch std parameters)
- `Event`: Detected violation with rule name, time/altitude window, worst value, and severity score

### Key Design Decisions

- **Smoothed signals**: All rules use `_s` suffixed columns (e.g., `ias_s`, `vs_s`) which are moving-averaged
- **Pitch chasing detection**: Uses rolling standard deviation of pitch, not absolute pitch values
- **Severity scoring**: Multi-factor (magnitude × duration × altitude), combined probabilistically across events
- **DataFrame attrs**: Sampling interval `dt` stored in `df.attrs["dt"]` for downstream use

### CSV Format Flexibility

The preprocessor handles multiple column naming conventions:
- Time: `t` or `Time`
- Altitude: `alt_msl_m` (meters) or `alt_msl_ft` (feet)
- Vertical speed: `vs_fpm` or `vs_fps`
