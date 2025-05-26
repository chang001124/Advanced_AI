import pandas as pd

# 讀取原始全年檔案
file_path = "2023_youbike_transfer_all.csv"
df = pd.read_csv(file_path)
print(f"✅ 原始資料筆數：{len(df)}")

# 顯示欄位名稱（實際為轉乘資料）
print("\n📌 原始欄位：", df.columns.tolist())

# 統一欄位名稱（原始為中文）
rename_map = {
    "借車日期": "rent_date",
    "借車時間": "start_time",
    "還車時間": "end_time",
    "借車站": "start_station",
    "還車站": "end_station",
    "租借時數": "duration_hour",
    "資料月份": "month"
}
df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

# 檢查缺值情況
print("\n📊 各欄位缺值比例：")
print(df.isna().mean().round(4) * 100)

# 處理時間欄位為 datetime 格式（如有）
if "rent_date" in df.columns:
    df["rent_date"] = pd.to_datetime(df["rent_date"], errors="coerce")
    df = df[df["rent_date"].notna()]

# 移除明顯非必要欄位
drop_cols = [col for col in df.columns if col.startswith("_") or "import" in col]
df.drop(columns=drop_cols, inplace=True, errors="ignore")

# 統一欄位順序與預覽
columns = ["rent_date", "start_station", "end_station", "duration_hour", "month"]
df = df[[col for col in columns if col in df.columns]].sort_values("rent_date")
print("\n✅ 清洗後欄位：", df.columns.tolist())

# 儲存清洗後資料
df.to_csv("2023_youbike_rental_cleaned.csv", index=False, encoding="utf-8-sig")

# 額外：建立每日租借次數統計
daily_stat = df.groupby("rent_date").size().reset_index(name="daily_rent_count")
daily_stat.to_csv("2023_youbike_daily_summary.csv", index=False, encoding="utf-8-sig")
print(f"\n✅ 每日統計完成，總天數：{len(daily_stat)}，儲存為 2023_youbike_daily_summary.csv")