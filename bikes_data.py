"""Lakeshore Bike Share: one row per station per hour. Given, never edited.
Run `python bikes_data.py` to see the first rows. Every call with the same
seed returns identical rows, so everyone works from the same table.
"""
import numpy as np
import pandas as pd
STATION_TYPES = ["downtown", "residential", "park"]
TARGET = "rentals"
def _bump(hour, centre, width):
    return np.exp(-0.5 * ((hour - centre) / width) ** 2)
def make_rentals(n: int = 6000, seed: int = 2026) -> pd.DataFrame:
    """Return n invented station-hours with the number of bikes rented."""
    rng = np.random.default_rng(seed)
    hour = rng.integers(0, 24, n)
    weekend = rng.random(n) < 2 / 7
    station = rng.choice(STATION_TYPES, n, p=[0.40, 0.35, 0.25])
    temperature = np.round(rng.normal(16, 9, n).clip(-15, 35), 1)
    wet = rng.random(n) < 0.30
    spell = rng.exponential(3.0, n)
    rain = np.round(np.where(wet, spell, 0.0).clip(0, 15), 1)
    wind = np.round(rng.gamma(4.0, 4.0, n).clip(0, 60), 1)
    commute = _bump(hour, 8, 1.3) + 0.9 * _bump(hour, 17.5, 1.6)
    midday = _bump(hour, 13, 3.2)
    downtown = station == "downtown"
    residential = station == "residential"
    park = station == "park"
    scale_wd = np.select([downtown, residential], [80.0, 55.0], 15.0)
    scale_we = np.select([downtown, residential], [35.0, 30.0], 70.0)
    weekday_shape = np.where(park, 0.5 * midday, commute)
    shape = np.where(
    weekend,
    scale_we * (0.1 + midday),
    scale_wd * (0.12 + weekday_shape),
)
    comfort = 0.05 + 0.95 * np.exp(-0.5 * ((temperature - 21) / 8.5) ** 2)
    dry = 1.0 / (1.0 + 0.45 * rain)
    calm = np.exp(-0.025 * np.clip(wind - 14, 0, None))
    rentals = rng.poisson(shape * comfort * dry * calm)
    return pd.DataFrame(
    {
    "hour": hour,
    "temperature_c": temperature,
    "rain_mm": rain,
    "wind_kph": wind,
    "is_weekend": weekend.astype(int),
    "station_type": station,
    "rentals": rentals,
    }
)
if __name__ == "__main__":
    frame = make_rentals()

    print(frame.head())
    print(frame["rentals"].describe().round(2))