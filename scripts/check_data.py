"""Look at the satellite readings before training: coverage, seasonality, year-to-year drift, outliers.

    python scripts/check_data.py

Read-only. Prints plain-English flags at the end.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from algaebro import config

pd.set_option("display.width", 140)

if __name__ == "__main__":
    df = pd.read_csv(config.NDCI_CSV, parse_dates=["date"])
    if df.empty:
        sys.exit("ndci.csv is empty. Run scripts/02_fetch_satellite.py first.")
    df["year"], df["month"] = df["date"].dt.year, df["date"].dt.month
    n_dates = df["date"].nunique()
    print(f"DATA SOURCE: {config.data_source().upper()}")
    print(f"{len(df)} readings, {n_dates} dates, {df['date'].min().date()} to {df['date'].max().date()}\n")

    flags = []

    print("PER ZONE (share of dates with a reading, and typical value)")
    z = df.groupby("zone_id").agg(readings=("ndci", "size"), median=("ndci", "median"), spread_p10=("ndci", lambda s: s.quantile(.1)), spread_p90=("ndci", lambda s: s.quantile(.9)))
    z["coverage"] = (z["readings"] / n_dates).round(2)
    print(z.round(3).to_string())
    for zid, row in z.iterrows():
        if row["coverage"] < 0.6:
            flags.append(f"{zid} has a reading on only {row['coverage']:.0%} of dates: forecasts there will often say 'unknown'.")

    print("\nBY MONTH (a bloom season should show higher values in summer)")
    m = df.groupby("month")["ndci"].agg(["size", "median", lambda s: s.quantile(.9)]).round(3)
    m.columns = ["readings", "median", "p90"]
    print(m.to_string())
    summer, shoulder = df[df["month"].isin([7, 8, 9])]["ndci"].median(), df[df["month"].isin([5, 10])]["ndci"].median()
    if summer - shoulder < 0.01:
        flags.append(f"Summer median ({summer:+.3f}) is barely above May/October ({shoulder:+.3f}): little seasonal signal, so forecasts may not beat persistence.")

    print("\nBY YEAR (watch for a sudden jump at 2022, when the satellite offset changed)")
    y = df.groupby("year")["ndci"].agg(["size", "median", lambda s: s.quantile(.9)]).round(3)
    y.columns = ["readings", "median", "p90"]
    print(y.to_string())
    med = y["median"]
    if 2021 in med.index and 2022 in med.index and abs(med[2022] - med[2021]) > 0.05:
        flags.append(f"Median jumps {med[2021]:+.3f} to {med[2022]:+.3f} between 2021 and 2022: the offset correction may be wrong, or it is real. Tell Claude.")
    if med.max() - med.min() > 0.08:
        flags.append(f"Yearly medians range from {med.min():+.3f} to {med.max():+.3f}: large year-to-year differences.")

    print("\nTOP 8 HIGHEST READINGS (basin-wide on that date = likely real; one zone alone = possible artifact)")
    day_med = df.groupby("date")["ndci"].transform("median")
    df["date_median"], df["above_date_median"] = day_med, df["ndci"] - day_med
    top = df.sort_values("ndci", ascending=False).head(8)[["date", "zone_id", "ndci", "date_median", "above_date_median", "n_valid"]].copy()
    shown = top.assign(date=top["date"].dt.strftime("%Y-%m-%d")).round({"ndci": 3, "date_median": 3, "above_date_median": 3})
    print(shown.to_string(index=False))
    lone = top[top["above_date_median"] > 0.12]
    if len(lone):
        flags.append(f"{len(lone)} of the top 8 readings are 0.12+ above every other zone that day: likely a local artifact (glare, shadow, boat). Consider capping outliers.")

    print("\nFLAGS")
    print("\n".join(f"  - {f}" for f in flags) if flags else "  none: nothing looks off. Go ahead and train.")
