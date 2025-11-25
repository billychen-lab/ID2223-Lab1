# 02_train_feature_view_multi.py
# 使用 Feature View 自动 join 天气与 PM2.5，按站点训练模型

import os
import joblib
import numpy as np
import pandas as pd
import hopsworks as hs
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ------------ 配置 ------------
STATION_WHITELIST = {
    "hk-tuen-mun",
    "hk-yuen-long",
    "hk-tsuen-wan",
    "hk-Kwai-Chung",
    "hk-tung-chung"
}
MIN_TRAIN_ROWS = 10

# ------------ 登录 Hopsworks ------------
project = hs.login(
    api_key_value=os.environ["HOPSWORKS_API_KEY"],
    project=os.getenv("HOPSWORKS_PROJECT", None),
)
fs = project.get_feature_store()

# ------------ 1) 获取 Feature Groups ------------
fg_aq = fs.get_feature_group("air_quality_daily", version=2)
fg_w  = fs.get_feature_group("weather_daily_forecast", version=2)

# ------------ 2) 定义 Feature View 查询 ------------
query = (
    fg_aq.select(["pm2_5", "city", "station_id", "date"])
         .join(fg_w.select_all(), on=["city", "station_id", "date"])
)

# ------------ 3) 创建 Feature View ------------
fv = fs.get_or_create_feature_view(
    name="air_quality_fv_multi",
    version=1,
    labels=["pm2_5"],
    query=query,
    description="PM2.5 labels joined with Open-Meteo features"
)

# ------------ 4) 自动生成训练数据 ------------
df = fv.get_training_data()

df["date"] = pd.to_datetime(df["date"])
df = df.dropna().sort_values("date")

if STATION_WHITELIST:
    df = df[df["station_id"].isin(STATION_WHITELIST)]

print("\n[info] Feature View rows:", len(df))
print(df.groupby("station_id")["date"].agg(["min", "max", "count"]))

if len(df) == 0:
    raise SystemExit("[error] FV 生成的数据为空，请检查 FG 数据时间重叠。")

# ------------ 5) 逐站点训练模型 ------------
os.makedirs("models", exist_ok=True)
results = []

DROP_COLS = ["pm2_5", "city", "station_id", "date"]

for st_id, g in df.groupby("station_id"):

    g = g.sort_values("date")
    split = int(len(g) * 0.8)
    tr, te = g.iloc[:split], g.iloc[split:]

    num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c not in DROP_COLS]

    if len(tr) < MIN_TRAIN_ROWS or len(feat_cols) == 0:
        print(f"[skip] {st_id}: 数据不足（rows={len(tr)}, feats={len(feat_cols)}）")
        continue

    X_tr, y_tr = tr[feat_cols], tr["pm2_5"]
    X_te, y_te = te[feat_cols], te["pm2_5"]

    model = RandomForestRegressor(n_estimators=400, random_state=42)
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    mae = float(mean_absolute_error(y_te, pred))

    joblib.dump({"model": model, "features": feat_cols}, f"models/{st_id}_rf.joblib")

    print(f"[ok] {st_id}: rows={len(g)}, feats={len(feat_cols)}, MAE={mae:.2f}")
    results.append((st_id, len(g), len(feat_cols), mae))

# ------------ Summary ------------
print("\n=== Summary ===")
for st_id, n, k, mae in results:
    print(f"{st_id}: rows={n}, feats={k}, MAE={mae:.2f}")

print("\n[done] 使用 Feature View 的多站点训练完成。")
