"""STEP 3. Download daily weather history (Open-Meteo, free, no key). Writes data/weather.csv."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algaebro import weather

if __name__ == "__main__":
    df = weather.fetch_all()
    print(df.tail(5).to_string(index=False))
