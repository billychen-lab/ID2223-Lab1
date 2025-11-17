# 03_predict_and_plot.py  —— 预测图从“今天”算起，严格 7 天；回测只到昨天；支持 AQI 色带风格
import os
import joblib
import pandas as pd
import numpy as np
import hopsworks as hs
import matplotlib.pyplot as plt
#import matplotlib.dates as mdates

# ========= 配置 =========
CITY = "HongKong"
STATION_ID = "hk-tung-chung"
#STATION_ID = ["hk-tuen-mun", "hk-yuen-long", hk-tsuen-wan, hk-Kwai-Chung, hk-tung-chung]
MODEL_PATH = f"models/{STATION_ID}_rf.joblib"
VERSION = 2

# 过去用于对比的天数（回测图横向显示多少天）
BACK_DAYS = 14
# 未来要展示的天数（严格 7 天）
FORECAST_DAYS = 7

# 图像风格：True=带 AQI 色带 + 对数坐标；False=普通线性坐标（类似作业参考图）
USE_AQI_BANDS = True

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# ========= 登录 Hopsworks =========
project = hs.login(
    api_key_value=os.environ["HOPSWORKS_API_KEY"],
    project=os.getenv("HOPSWORKS_PROJECT", None),
)
fs = project.get_feature_store()

fg_w = fs.get_feature_group("weather_daily_forecast", version=VERSION)
fg_aq = fs.get_feature_group("air_quality_daily", version=VERSION)

# ========= 读取天气（过去 + 未来） =========
w_df = fg_w.read()
# 统一成 tz-naive（去掉 UTC），方便和 pandas 比较
w_df["date"] = pd.to_datetime(w_df["date"], utc=True).dt.tz_localize(None)
w_df = w_df[w_df["station_id"] == STATION_ID].copy()

today = pd.Timestamp.today(tz=None).normalize()
start_date = today - pd.Timedelta(days=BACK_DAYS)
# 多给两天冗余，后面再精确截 7 天
end_date = today + pd.Timedelta(days=FORECAST_DAYS + 2)

w_df = (
    w_df[(w_df["date"] >= start_date) & (w_df["date"] <= end_date)]
      .sort_values("date")
      .drop_duplicates(["station_id", "date"])
      .reset_index(drop=True)
)
if w_df.empty:
    raise SystemExit(
        f"[error] weather window 为空：{start_date.date()} ~ {end_date.date()}。\n"
        f"请先用 01 扩大 PAST_DAYS/FORECAST_DAYS 后写入 v2。"
    )

# ========= 读取标签（仅用于回测对比与 MAE） =========
aq_df = fg_aq.read()
aq_df["date"] = pd.to_datetime(aq_df["date"], utc=True).dt.tz_localize(None)
aq_df = aq_df[(aq_df["station_id"] == STATION_ID) & (aq_df["city"] == CITY)]
# 标签严格到昨天（< today），与回测一致
aq_df = aq_df[(aq_df["date"] >= start_date) & (aq_df["date"] <= today-pd.Timedelta(days=1))]
aq_df = (
    aq_df[["date", "pm2_5"]]
      .drop_duplicates("date")
      .rename(columns={"pm2_5": "pm2_5_true"})
)

# ========= 加载模型并预测 =========
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feat_cols = bundle["features"]

# 只保留当前窗口可用的特征
exist_feats = [c for c in feat_cols if c in w_df.columns]
if len(exist_feats) == 0:
    raise SystemExit(f"[error] 预测特征在 weather 表中一个都找不到：{feat_cols}")
if len(exist_feats) < len(feat_cols):
    miss = sorted(list(set(feat_cols) - set(exist_feats)))
    print(f"[warn] 当前窗口缺少特征：{miss}（将忽略）")

X = w_df[exist_feats]
pred = model.predict(X)

# 合并预测与真值
res = (
    pd.DataFrame({"date": w_df["date"].values, "pm2_5_pred": pred})
      .merge(aq_df, on="date", how="left")
      .sort_values("date")
      .reset_index(drop=True)
)
res["city"] = CITY
res["station_id"] = STATION_ID

# —— 切分 —— #
# 回测（hindcast）：到昨天为止
hind = res[res["date"] <= today- pd.Timedelta(days=1)].copy()
# 未来（forecast）：从今天开始，严格取 7 天
future = (
    res[res["date"] >= today]
      .sort_values("date")
      .drop_duplicates("date")
      .head(FORECAST_DAYS)
      .copy()
)
if len(future) < FORECAST_DAYS:
    print(f"[warn] 天气特征里只有 {len(future)} 天可用（少于 {FORECAST_DAYS} 天）。")

# 计算 MAE（仅用有真值的回测区间）
hind_with_truth = hind.dropna(subset=["pm2_5_true"])
if not hind_with_truth.empty:
    mae = float(np.mean(np.abs(hind_with_truth["pm2_5_true"].to_numpy()
                               - hind_with_truth["pm2_5_pred"].to_numpy())))
else:
    mae = np.nan

print(f"[info] rows: hind={len(hind)} (truth={len(hind_with_truth)}), "
      f"future={len(future)}, MAE={mae:.2f}")

# ========= 导出 CSV =========
csv_path = os.path.join(OUTDIR, f"{STATION_ID}_predictions.csv")
res.to_csv(csv_path, index=False)
print(f"[ok] saved CSV -> {csv_path}")

# ========= 工具：添加 AQI 色带（可选） =========
def add_aqi_bands(ax):
    """在 ax 上加 AQI 色带，并切换对数 y 轴（更好看）"""
    bands = [
        (0,   50,  "#c4e6c3", "Good: 0-49"),
        (50, 100,  "#f6f3a6", "Moderate: 50-99"),
        (100,150,  "#f9d39b", "Unhealthy for Some: 100-149"),
        (150,200,  "#f6a6a6", "Unhealthy: 150-199"),
        (200,300,  "#e2c4f6", "Very Unhealthy: 200-299"),
        (300,500,  "#e6d9d6", "Hazardous: 300-500"),
    ]
    for low, high, color, lab in bands:
        ax.axhspan(low, high, color=color, alpha=0.35, label=lab)
    ax.set_yscale("log")
    ax.set_ylim(0, 500)  # 让色带完整露出

# ========= 绘制 hindcast（逐日横坐标 + 斜体标签）=========
hind_fig = os.path.join(OUTDIR, f"{STATION_ID}_hindcast.png")
plt.figure(figsize=(10, 8))
ax = plt.gca()

if USE_AQI_BANDS:
    add_aqi_bands(ax)  # 会切到对数y轴
    aqi_ticks = [50, 100, 150, 200, 300, 500]
    ax.set_yticks(aqi_ticks)
    ax.set_yticklabels([str(v) for v in aqi_ticks])
    ax.set_ylim(10, 500)

# 只画最近 BACK_DAYS 天
hind_plot = hind.tail(BACK_DAYS).copy()

# 有真值 -> 两条线；否则只有预测
if not hind_with_truth.empty:
    ax.plot(hind_plot["date"], hind_plot["pm2_5_true"],
            label="True (pm2.5)", color="#1f77b4", linewidth=1.2, marker="o", markersize=4)
ax.plot(hind_plot["date"], hind_plot["pm2_5_pred"],
        label="Pred", color="#ff7f0e", linewidth=1.2, marker="o", markersize=4)

# ---- 逐日刻度 + 倾斜标签 ----
import matplotlib.dates as mdates
xmin = hind_plot["date"].min().normalize()
xmax = hind_plot["date"].max().normalize()
day_ticks = pd.date_range(xmin, xmax, freq="D")          # 每天一个刻度
ax.set_xticks(day_ticks)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")  # 斜体显示

# 其它装饰
title = (f"{CITY} / {STATION_ID} - Hindcast (MAE={mae:.2f})"
         if not np.isnan(mae) else f"{CITY} / {STATION_ID} - Hindcast")
ax.set_title(title)
#ax.axvline(today, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Date"); ax.set_ylabel("PM2.5")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(hind_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"[ok] saved hindcast plot -> {hind_fig}")


# ========= 绘制 Forecast（从今天起严格 7 天）=========
forecast_fig = os.path.join(OUTDIR, f"{STATION_ID}_forecast.png")
plt.figure(figsize=(10, 8))
ax = plt.gca()

if USE_AQI_BANDS:
    add_aqi_bands(ax)
    aqi_ticks = [50, 100, 150, 200, 300, 500]
    ax.set_yticks(aqi_ticks)
    ax.set_yticklabels([str(v) for v in aqi_ticks])
    ax.set_ylim(10, 500)

ax.plot(future["date"], future["pm2_5_pred"], label="Forecast Pred",
        color="#d62728", marker="o", markersize=4, linewidth=1.2)
#ax.axvline(today, color="gray", linestyle="--", linewidth=1)
ax.set_title(f"{CITY}, {STATION_ID} - Next {FORECAST_DAYS} Days Forecast")
ax.set_xlabel("Date"); ax.set_ylabel("PM2.5")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(forecast_fig, dpi=150)
plt.close()
print(f"[ok] saved forecast plot -> {forecast_fig}")
