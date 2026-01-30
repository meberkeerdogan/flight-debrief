from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class Runway:
    ident: str
    thr_lat_deg: float
    thr_lon_deg: float
    end_lat_deg: float
    end_lon_deg: float
    true_brg_deg: float
    thr_elev_ft: float

    @property
    def thr_elev_m(self) -> float:
        return self.thr_elev_ft * 0.3048


@dataclass(frozen=True)
class Airport:
    icao: str
    name: str
    runways: Dict[str, Runway]


# BIKF values from Iceland eAIP "BIKF AD 2.12 RUNWAY PHYSICAL CHARACTERISTICS"
# THR coords are in DMS-ish format (DDMMSS.ssN / DDDMMSS.ssW) in the table.
# Here we hardcode decimal degrees equivalents for MVP simplicity.
# Source: eAIP Iceland (Isavia) table lines for RWY 01/19/10/28. :contentReference[oaicite:1]{index=1}

# Helper conversions (precomputed for brevity):
# 635752.12N -> 63 + 57/60 + 52.12/3600 = 63.9644778
# 0223619.63W -> -(22 + 36/60 + 19.63/3600) = -22.6054528
# 635930.76N -> 63.9918778
# 0223619.56W -> -22.6054333
# 635906.14N -> 63.9850389
# 0223918.02W -> -22.6550056
# 0223532.61W -> -22.5923917

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

AIRPORTS: Dict[str, Airport] = {
    "BIKF": BIKF,
}
