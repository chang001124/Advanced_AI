# visual_analysis_zh.py
# -- 2023 年 YouBike 轉乘資料視覺化（中文版本）------------------------
"""
執行步驟：
1. 將 `NotoSansTC-Regular.otf`（或任一支援繁體中文的字型）放在與本檔同一資料夾。
   下載來源： https://github.com/adobe-fonts/source-han-sans/releases  (OTF Traditional Chinese)
2. 執行腳本，將產生 5 張中文標題與軸標的 PNG 圖。
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm
from pathlib import Path
plt.style.use("ggplot")

# -------------------------------------------------------------
# 0. 動態註冊中文字型
FONT_PATH = Path(__file__).parent / "NotoSansTC-Regular.otf"
if FONT_PATH.exists():
    fm.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = "Noto Sans TC"
    print("✔ 已載入字型 → Noto Sans TC")
else:
    print("⚠️ 找不到字型檔，將以系統預設字型繪圖（中文可能為豆腐字）")

# -------------------------------------------------------------
# 1. 讀取資料
clean_path = "2023_youbike_rental_cleaned.csv"
daily_path = "2023_youbike_daily_summary.csv"

df    = pd.read_csv(clean_path, parse_dates=["rent_date"])
daily = pd.read_csv(daily_path,  parse_dates=["rent_date"])

# -------------------------------------------------------------
# 2. 欄位缺值比例條形圖
na_ratios = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
plt.figure(figsize=(8,4))
sns.barplot(x=na_ratios.values, y=na_ratios.index, color="#5B8FF9")
plt.title("各欄位缺值比例 (%)")
plt.xlabel("缺值百分比 (%)")
plt.tight_layout()
plt.savefig("fig_missing_ratio_zh.png")
plt.close()

# -------------------------------------------------------------
# 3. 每日轉乘量折線圖
plt.figure(figsize=(10,4))
plt.plot(daily["rent_date"], daily["daily_rent_count"], lw=1.2)
plt.title("每日轉乘租借次數 (2023)")
plt.ylabel("租借次數")
plt.xlabel("日期")
plt.tight_layout()
plt.savefig("fig_daily_timeseries_zh.png")
plt.close()

# -------------------------------------------------------------
# 4. 月份箱型圖（季節性）
daily["month"] = daily["rent_date"].dt.month
plt.figure(figsize=(8,4))
sns.boxplot(x="month", y="daily_rent_count", data=daily, palette="pastel")
plt.title("各月份每日轉乘次數分佈")
plt.xlabel("月份")
plt.ylabel("每日租借次數")
plt.tight_layout()
plt.savefig("fig_monthly_boxplot_zh.png")
plt.close()

# -------------------------------------------------------------
# 5. 租借量前十站 (出發站) 長條圖
top10 = (
    df.groupby("start_station").size()
      .sort_values(ascending=False)
      .head(10)
      .sort_values()
)
plt.figure(figsize=(7,5))
plt.barh(top10.index, top10.values, color="#F6BD16")
plt.title("租借量前十站（出發站）")
plt.xlabel("租借次數")
plt.tight_layout()
plt.savefig("fig_top10_stations_zh.png")
plt.close()

# -------------------------------------------------------------
# 6. 星期別箱型圖
week_map = {0:"星期一",1:"星期二",2:"星期三",3:"星期四",4:"星期五",5:"星期六",6:"星期日"}
daily["weekday_zh"] = daily["rent_date"].dt.dayofweek.map(week_map)
weekday_order = ["星期一","星期二","星期三","星期四","星期五","星期六","星期日"]
plt.figure(figsize=(8,4))
sns.boxplot(x="weekday_zh", y="daily_rent_count", data=daily, order=weekday_order, palette="Set2")
plt.title("星期別每日轉乘租借分佈")
plt.xlabel("星期")
plt.ylabel("每日租借次數")
plt.tight_layout()
plt.savefig("fig_weekday_boxplot_zh.png")
plt.close()

print("✅ 圖表繪製完成，已輸出中文版本 PNG 圖檔。")