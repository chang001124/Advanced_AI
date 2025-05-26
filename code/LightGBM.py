# lgbm_model_youbike_daily.py
"""
LightGBM regression for single‑station daily YouBike transfers
=============================================================
Assumptions
-----------
* Feature file   : 2023_youbike_daily_features.csv (output of feature_engineering_Youbike_daily)
* Holiday file   : tw_holidays_2023.csv (optional, column `date`)
* Weather file   : tpe_weather_2023.csv (optional, columns `date`, `rain_mm`, `max_temp`)

Steps
-----
1. Merge external features (holiday, weather) if provided
2. Train/valid split by time (80 % / 20 %)
3. LightGBM model with early stopping
4. Print MAE + RMSE and save model + feature list
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import joblib

# ---------------------------------------------------------------------
# 1. 讀取主要特徵
feat_path = Path("2023_youbike_daily_features.csv")
assert feat_path.exists(), "請先執行特徵腳本"
df = pd.read_csv(feat_path, parse_dates=["rent_date"])

# ---------------------------------------------------------------------
# 2. 合併假日
hol_path = Path("tw_holidays_2023.csv")
if hol_path.exists():
    hol = pd.read_csv(hol_path, parse_dates=["date"])
    df["is_holiday"] = df["rent_date"].isin(hol["date"]).astype(int)
else:
    df["is_holiday"] = 0

# 3. 合併天氣（臺北站）
weather_path = Path("tpe_weather_2023.csv")
if weather_path.exists():
    weather = pd.read_csv(weather_path, parse_dates=["date"])
    df = df.merge(weather, left_on="rent_date", right_on="date", how="left")
    df.drop("date", axis=1, inplace=True)
else:
    df["rain_mm"]  = 0.0
    df["max_temp"] = df["roll_mean_7"].pct_change().fillna(0)  # dummy temperate proxy

# ---------------------------------------------------------------------
# 4. 準備特徵矩陣與標籤
y = df["daily_rent_count"].values
X = df.drop(["rent_date", "daily_rent_count"], axis=1)
feature_names = X.columns.tolist()
X = X.values

# 時序 80/20 切分
tsplit = int(len(df)*0.8)
X_train, X_test = X[:tsplit], X[tsplit:]
y_train, y_test = y[:tsplit], y[tsplit:]

# ---------------------------------------------------------------------
# 5. 建立 LightGBM Dataset
train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)

params = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42
}

model = lgb.train(
    params,
    train_set,
    num_boost_round=2000,
    valid_sets=[train_set, valid_set],
    valid_names=["train", "valid"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=200)
    ]
)

# ---------------------------------------------------------------------
# 6. 評估
pred = model.predict(X_test, num_iteration=model.best_iteration)
mae  = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print(f"\nTest MAE  = {mae:.2f}")
print(f"Test RMSE = {rmse:.2f}")

# ---------------------------------------------------------------------
# 7. 儲存模型與特徵
model.save_model("youbike_lgbm.txt")
joblib.dump(feature_names, "youbike_feature_list.save")
print("✅ LightGBM model saved (youbike_lgbm.txt)")
