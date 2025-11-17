# 04_train_and_feature_view_multi.py
# 使用现有 baseline 模型（models/{station_id}_rf.joblib），
# 再训练 lag(+weather) 模型；同时输出：
#   1) 时间序 80/20 校验集 MAE 对比
#   2) 最近 14 天 hindcast MAE 对比
# 结果写到 outputs/lag_report.csv

import os
import joblib
import numpy as np
import pandas as pd
import hopsworks as hs
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 只做这些站点
STATION_WHITELIST = {"hk-tuen-mun"}

MIN_TRAIN_ROWS = 12
VERSION = 2
MODELS_DIR = "models"
OUT_DIR = "outputs"
HINDCAST_DAYS = 14  # 最近多少天

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 1) 登录 Hopsworks ----------
project = hs.login(
    api_key_value=os.environ["HOPSWORKS_API_KEY"],
    project=os.getenv("HOPSWORKS_PROJECT", None),
)
fs = project.get_feature_store()

# ---------- 2) 取 v2 的 Feature Groups ----------
fg_aq = fs.get_feature_group("air_quality_daily", version=VERSION)
fg_w  = fs.get_feature_group("weather_daily_forecast", version=VERSION)

# ---------- 3) 读取并预处理 ----------
aq_df = fg_aq.read()
w_df  = fg_w.read()

aq_df["date"] = pd.to_datetime(aq_df["date"], utc=True).dt.tz_localize(None)
w_df["date"]  = pd.to_datetime(w_df["date"],  utc=True).dt.tz_localize(None)

if STATION_WHITELIST:
    aq_df = aq_df[aq_df["station_id"].isin(STATION_WHITELIST)]
    w_df  = w_df[w_df["station_id"].isin(STATION_WHITELIST)]

# ---------- 4) 合并 ----------
keys = ["city", "station_id", "date"]
have = [k for k in keys if (k in aq_df.columns and k in w_df.columns)]
if not {"station_id", "date"}.issubset(have):
    raise SystemExit("[error] weather/label 缺少 station_id 或 date，无法 join。")
print(f"[info] join on keys: {have}")

df = aq_df.merge(w_df, on=have, how="inner", suffixes=("", "_wx"))
if "city" not in df.columns and "city" in aq_df.columns:
    df = df.merge(
        aq_df[["station_id", "date", "city"]].drop_duplicates(),
        on=["station_id", "date"], how="left"
    )

df = (
    df.dropna()
      .drop_duplicates(["city", "station_id", "date"])
      .sort_values(["station_id", "date"])
      .reset_index(drop=True)
)

if len(df) == 0:
    raise SystemExit("[warn] 合并后为空，检查 01/02 的数据写入。")

# ---------- 5) 构造 lag 特征 ----------
def add_lags(g):
    g = g.sort_values("date").copy()
    g["pm2_5_lag1"] = g["pm2_5"].shift(1)
    g["pm2_5_lag2"] = g["pm2_5"].shift(2)
    g["pm2_5_lag3"] = g["pm2_5"].shift(3)
    return g

df_lag = df.groupby("station_id", group_keys=False).apply(add_lags)
df_lag_clean = df_lag.dropna(subset=["pm2_5_lag1", "pm2_5_lag2", "pm2_5_lag3"]).copy()

# ---------- 6) 小工具 ----------
def intersect_existing(frame, cols):
    """确保模型所需特征在当前 frame 中都存在；取交集。"""
    return [c for c in cols if c in frame.columns]

def weather_plus_lag_features(frame, weather_cols):
    feats = list(weather_cols)
    for c in ["pm2_5_lag1", "pm2_5_lag2", "pm2_5_lag3"]:
        if c in frame.columns:
            feats.append(c)
    return feats

def last_n_dates(frame, n):
    """取该站点最后 n 天的日期集合（按出现顺序去重）"""
    return frame["date"].drop_duplicates().sort_values().tail(n).tolist()

# ---------- 7) 主循环 ----------
report_rows = []

for st_id, g in df.groupby("station_id"):
    g = g.sort_values("date").copy()

    # === 7.1 baseline：加载你已有的 *_rf.joblib === #
    base_path = os.path.join(MODELS_DIR, f"{st_id}_rf.joblib")
    if not os.path.isfile(base_path):
        print(f"[skip] {st_id}: 未找到 baseline 模型文件 {base_path}")
        continue

    base_bundle = joblib.load(base_path)
    base_model = base_bundle["model"]
    base_feats_saved = base_bundle.get("features", [])
    base_feats = intersect_existing(g, base_feats_saved)
    if len(base_feats) == 0:
        print(f"[skip] {st_id}: baseline 特征与当前数据不匹配（无交集）。")
        continue

    # ---- 80/20 校验（baseline 使用现成模型）----
    split = int(len(g) * 0.8)
    tr, te = g.iloc[:split], g.iloc[split:]
    if len(tr) < MIN_TRAIN_ROWS or len(te) == 0:
        print(f"[skip] {st_id}: baseline 数据不足 (train={len(tr)}, test={len(te)})")
        continue

    X_te_base = te[base_feats]
    y_te_base = te["pm2_5"]
    mae80_base = float(mean_absolute_error(y_te_base, base_model.predict(X_te_base)))
    print(f"[BASE 80/20] {st_id}: rows={len(g)}, feats={len(base_feats)}, MAE={mae80_base:.2f}")

    # ---- 最近 14 天 hindcast（baseline）----
    dates14 = last_n_dates(g, HINDCAST_DAYS)
    hind_base = g[g["date"].isin(dates14)].copy()
    mae14_base = np.nan
    if len(hind_base) > 0:
        X_hb = hind_base[base_feats]
        y_hb = hind_base["pm2_5"]
        mae14_base = float(mean_absolute_error(y_hb, base_model.predict(X_hb)))
        print(f"[BASE 14d ] {st_id}: rows14={len(hind_base)}, MAE14={mae14_base:.2f}")
    else:
        print(f"[BASE 14d ] {st_id}: 没有可用的 14 天样本")

    # === 7.2 lag(+weather)：新训练 === #
    g2 = df_lag_clean[df_lag_clean["station_id"] == st_id].sort_values("date").copy()
    if len(g2) < MIN_TRAIN_ROWS:
        print(f"[skip] {st_id}: lag 数据不足 ({len(g2)})")
        report_rows.append([
            st_id,
            len(g), len(base_feats), mae80_base,
            0, np.nan, np.nan,
            len(hind_base), mae14_base, np.nan
        ])
        continue

    split2 = int(len(g2) * 0.8)
    tr2, te2 = g2.iloc[:split2], g2.iloc[split2:]
    lag_feats = weather_plus_lag_features(g2, base_feats)

    X_tr2, y_tr2 = tr2[lag_feats], tr2["pm2_5"]
    X_te2, y_te2 = te2[lag_feats], te2["pm2_5"]

    m_lag = RandomForestRegressor(n_estimators=400, random_state=42).fit(X_tr2, y_tr2)
    mae80_lag = float(mean_absolute_error(y_te2, m_lag.predict(X_te2)))

    # 保存 lag 模型
    lag_path = os.path.join(MODELS_DIR, f"{st_id}_rf_lag123.joblib")
    joblib.dump({"model": m_lag, "features": lag_feats, "uses_lag": True}, lag_path)
    print(f"[LAG  80/20] {st_id}: rows={len(g2)}, feats={len(lag_feats)}, MAE={mae80_lag:.2f} -> saved {lag_path}")

    # ---- 最近 14 天 hindcast（lag）----
    hind_lag = g2[g2["date"].isin(dates14)].copy()  # 与 baseline 对齐日期
    mae14_lag = np.nan
    if len(hind_lag) > 0:
        X_hl = hind_lag[lag_feats]
        y_hl = hind_lag["pm2_5"]
        mae14_lag = float(mean_absolute_error(y_hl, m_lag.predict(X_hl)))
        print(f"[LAG  14d ] {st_id}: rows14={len(hind_lag)}, MAE14={mae14_lag:.2f}")
    else:
        print(f"[LAG  14d ] {st_id}: 没有可用的 14 天样本（因滞后丢行或日期缺失）")

    # 记录对比
    report_rows.append([
        st_id,
        len(g), len(base_feats), mae80_base,          # 80/20 baseline
        len(lag_feats), mae80_lag, mae80_lag - mae80_base,   # 80/20 lag
        len(hind_base), mae14_base, mae14_lag, (mae14_lag - mae14_base) if not np.isnan(mae14_lag) and not np.isnan(mae14_base) else np.nan
    ])

# ---------- 8) 输出对比报告 ----------
if report_rows:
    rep = pd.DataFrame(
        report_rows,
        columns=[
            "station_id",
            "rows_all", "feats_baseline", "MAE80_baseline",
            "feats_lag123", "MAE80_lag123", "delta80(MAE_lag-base)",
            "rows14", "MAE14_baseline", "MAE14_lag123", "delta14(MAE_lag-base)"
        ],
    )
    rep_path = os.path.join(OUT_DIR, "lag_report.csv")
    rep.to_csv(rep_path, index=False)
    print("\n=== Lag vs Baseline Report (80/20 & last-14d) ===")
    print(rep)
    print(f"[ok] report saved -> {rep_path}")
else:
    print("[warn] 没有任何站点完成 lag 对比。")
