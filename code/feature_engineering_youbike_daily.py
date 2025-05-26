# feature_engineering_youbike_daily.py  (minimal calendar features)
"""
Extract features for single‑station daily YouBike transfers.
Calendar features retained: **is_weekend, day_of_year** only.
Generates: 2023_youbike_daily_features.csv
"""
import pandas as pd
from pathlib import Path

# ----------------------------------------------------------------------
# 1. Load daily series
df = pd.read_csv("2023_youbike_daily_summary.csv", parse_dates=["rent_date"]).sort_values("rent_date")

# ----------------------------------------------------------------------
# 2. Minimal calendar features
cal = df["rent_date"].dt

df["day_of_year"] = cal.dayofyear

df["is_weekend"]  = (cal.dayofweek >= 5).astype(int)

# Optional holiday flag (keep if file exists) ---------------------------
holiday_path = Path("tw_holidays_2023.csv")
if holiday_path.exists():
    hol_dates = pd.read_csv(holiday_path, parse_dates=["date"])["date"].dt.normalize()
    df["is_holiday"] = df["rent_date"].dt.normalize().isin(hol_dates).astype(int)
else:
    df["is_holiday"] = 0

# ----------------------------------------------------------------------
# 3. Lag features
for lag in [1, 7, 14]:
    df[f"lag_{lag}"] = df["daily_rent_count"].shift(lag)

# ----------------------------------------------------------------------
# 4. Rolling mean / std
for w in [7, 14, 30]:
    df[f"roll_mean_{w}"] = df["daily_rent_count"].rolling(w).mean()
    df[f"roll_std_{w}"]  = df["daily_rent_count"].rolling(w).std()

# ----------------------------------------------------------------------
# 5. Fill NaN (front/back fill)
na_cols = df.columns[df.isna().any()]
df[na_cols] = df[na_cols].fillna(method="bfill").fillna(method="ffill")

# ----------------------------------------------------------------------
# 6. Save
OUT = "2023_youbike_daily_features.csv"
df.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"✅ Features saved to {OUT}. Columns: {list(df.columns)}")
