# Flight Debrief

A flight approach stability analyzer that detects unstable approaches from FlightGear telemetry data.

## Features

- **Stability Detection**: Identifies four types of approach instability:
  - Excessive sink rate (> -1000 fpm)
  - Speed deviation from target (> ±10 kt)
  - Excessive bank angle (> ±15°)
  - Pitch oscillations ("pitch chasing")

- **Severity Scoring**: Multi-factor severity calculation based on:
  - Magnitude of violation
  - Duration of violation
  - Altitude (lower = more severe)

- **Interactive Dashboard**: Streamlit-based UI for analysis and visualization

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/flight-debrief.git
cd flight-debrief

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Usage

### Web Interface

```bash
streamlit run app.py
```

Then upload a FlightGear CSV log file and configure the aircraft profile.

### Programmatic Usage

```python
from flight_debrief import analyze, AircraftProfile

profile = AircraftProfile(
    name="C172",
    gate_ft=500.0,
    speed_tol_kt=10.0,
    sink_rate_limit_fpm=-1000.0,
)

result, error = analyze(
    "flight_log.csv",
    runway_elev_m=50.0,
    profile=profile,
)

if result:
    approach, label, target, events, metrics = result
    print(f"Result: {label}")
    print(f"Severity: {metrics['overall_severity']:.1f}")
```

## CSV Format

Expected columns (FlightGear property log format):

| Column | Description | Unit |
|--------|-------------|------|
| `t` or `Time` | Timestamp | seconds |
| `alt_msl_m` or `alt_msl_ft` | Altitude MSL | meters or feet |
| `ias_kt` | Indicated airspeed | knots |
| `vs_fpm` or `vs_fps` | Vertical speed | fpm or fps |
| `pitch_deg` | Pitch angle | degrees |
| `roll_deg` | Roll angle | degrees |
| `throttle` | Throttle position | 0-1 |

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run tests with coverage
pytest --cov=flight_debrief --cov-report=term-missing
```

## Project Structure

```
flight-debrief/
├── app.py                 # Streamlit web application
├── flight_debrief/
│   ├── __init__.py        # Package exports
│   ├── domain.py          # Data models (AircraftProfile, Event)
│   ├── preprocess.py      # CSV loading and signal processing
│   ├── approach.py        # Approach window extraction
│   ├── detect.py          # Stability detection and scoring
│   ├── analyze.py         # Pipeline orchestration
│   ├── render.py          # Matplotlib visualization
│   └── airports.py        # Airport/runway data
├── tests/                 # Unit tests
├── configs/               # FlightGear logging config
└── data/samples/          # Sample flight data
```

## License

MIT
