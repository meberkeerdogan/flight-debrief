"""Airport and runway data for approach analysis."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Runway:
    """
    Runway physical characteristics.

    Attributes:
        ident: Runway identifier (e.g., "01", "28L")
        thr_lat_deg: Threshold latitude in decimal degrees
        thr_lon_deg: Threshold longitude in decimal degrees
        end_lat_deg: Runway end latitude in decimal degrees
        end_lon_deg: Runway end longitude in decimal degrees
        true_brg_deg: True bearing of runway heading
        thr_elev_ft: Threshold elevation in feet MSL
    """
    ident: str
    thr_lat_deg: float
    thr_lon_deg: float
    end_lat_deg: float
    end_lon_deg: float
    true_brg_deg: float
    thr_elev_ft: float

    @property
    def thr_elev_m(self) -> float:
        """Threshold elevation in meters MSL."""
        return self.thr_elev_ft * 0.3048


@dataclass(frozen=True)
class Airport:
    """
    Airport with associated runways.

    Attributes:
        icao: ICAO airport code
        name: Human-readable airport name
        runways: Dictionary of runway identifier to Runway object
    """
    icao: str
    name: str
    runways: dict[str, Runway]


# Keflavík International Airport (BIKF)
# Source: Iceland eAIP "BIKF AD 2.12 RUNWAY PHYSICAL CHARACTERISTICS"
BIKF = Airport(
    icao="BIKF",
    name="Keflavík International (KEF)",
    runways={
        "01": Runway(
            ident="01",
            thr_lat_deg=63.9644778,
            thr_lon_deg=-22.6054528,
            end_lat_deg=63.9918778,
            end_lon_deg=-22.6054333,
            true_brg_deg=0.02,
            thr_elev_ft=135.5,
        ),
        "19": Runway(
            ident="19",
            thr_lat_deg=63.9918778,
            thr_lon_deg=-22.6054333,
            end_lat_deg=63.9644778,
            end_lon_deg=-22.6054528,
            true_brg_deg=180.02,
            thr_elev_ft=161.4,
        ),
        "10": Runway(
            ident="10",
            thr_lat_deg=63.9850389,
            thr_lon_deg=-22.6550056,
            end_lat_deg=63.9850417,
            end_lon_deg=-22.5923917,
            true_brg_deg=89.97,
            thr_elev_ft=109.1,
        ),
        "28": Runway(
            ident="28",
            thr_lat_deg=63.9850417,
            thr_lon_deg=-22.5923917,
            end_lat_deg=63.9850389,
            end_lon_deg=-22.6550056,
            true_brg_deg=270.02,
            thr_elev_ft=169.2,
        ),
    },
)

# Registry of available airports
AIRPORTS: dict[str, Airport] = {
    "BIKF": BIKF,
}
