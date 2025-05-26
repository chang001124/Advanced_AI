import requests
import pandas as pd

# 2023 年每月對應的 dataset API URL（完整 12 月）
resources = {
    "2023-01": "https://data.taipei/api/v1/dataset/d49d31c6-f53c-448d-a782-f5d82d18669f?scope=resourceAquire",
    "2023-02": "https://data.taipei/api/v1/dataset/abe3381b-88c5-4601-b6da-bf466e655ffc?scope=resourceAquire",
    "2023-03": "https://data.taipei/api/v1/dataset/65ee5383-cf71-4dc3-aeff-655b6f9c8356?scope=resourceAquire",
    "2023-04": "https://data.taipei/api/v1/dataset/acf1e719-d5e0-471e-a261-5989d34c8ddb?scope=resourceAquire",
    "2023-05": "https://data.taipei/api/v1/dataset/7deecfb2-28b2-4647-9002-42227314e17c?scope=resourceAquire",
    "2023-06": "https://data.taipei/api/v1/dataset/fbc4205e-1aa3-4674-8e3a-4ff62960bf30?scope=resourceAquire",
    "2023-07": "https://data.taipei/api/v1/dataset/ac971551-d594-4e84-895f-41a624ed7679?scope=resourceAquire",
    "2023-08": "https://data.taipei/api/v1/dataset/58defa98-c041-4988-97ee-fe3798177e67?scope=resourceAquire",
    "2023-09": "https://data.taipei/api/v1/dataset/f678018a-339f-40d0-a0f8-c544a8911b55?scope=resourceAquire",
    "2023-10": "https://data.taipei/api/v1/dataset/c0d56ec4-d176-47c1-897f-b747be1db0c3?scope=resourceAquire",
    "2023-11": "https://data.taipei/api/v1/dataset/be19e3ae-6c6e-480a-ace7-529bd84c1c3e?scope=resourceAquire",
    "2023-12": "https://data.taipei/api/v1/dataset/8ddb45ba-a0f5-41d6-85a0-1691096fdcf8?scope=resourceAquire"
}

def fetch_all_rows(base_url, limit=10000):
    all_rows = []
    offset = 0
    while True:
        url = f"{base_url}&limit={limit}&offset={offset}"
        try:
            res = requests.get(url, timeout=30).json()
            rows = res.get("result", {}).get("results", [])
            if not rows:
                break
            all_rows.extend(rows)
            offset += limit
        except Exception as e:
            print(f"❌ 抓取錯誤：{e}")
            break
    return pd.DataFrame(all_rows)

all_months = []
for month, base_url in resources.items():
    print(f"📥 下載 {month} 資料...")
    df = fetch_all_rows(base_url)
    print(f" → 筆數：{len(df)}")
    if not df.empty:
        df["資料月份"] = month
        all_months.append(df)

# 儲存合併結果
df_all = pd.concat(all_months, ignore_index=True)
df_all.to_csv("2023_youbike_transfer_all.csv", index=False, encoding="utf-8-sig")
print(f"✅ 已完成全年下載與儲存，共 {len(df_all)} 筆，檔案為 2023_youbike_transfer_all.csv")
