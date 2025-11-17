# 02_train_and_feature_view_multi.py
# 读取 v2 的标签/天气表；自动选择 join 键；做逐站点重叠诊断；每站点训练随机森林

import os
import joblib
import numpy as np
import pandas as pd
import hopsworks as hs
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ============ 配置：可选，仅训练这些站点(留 None 表示全部) ============
STATION_WHITELIST = {
    # 只训屯门：
    "hk-tuen-mun",
    "hk-yuen-long",
    "hk-tsuen-wan",
    "hk-Kwai-Chung",
    "hk-tung-chung"
    # 如果也想训瑞典站，取消下面注释（但瑞典站目前没标签会被自动跳过）：
    # "se-0001",
}
MIN_TRAIN_ROWS = 10  # 单站最小训练样本行数

# ---------- 1) 登录 ----------
project = hs.login(
    api_key_value=os.environ["HOPSWORKS_API_KEY"],
    project=os.getenv("HOPSWORKS_PROJECT", None)
)
fs = project.get_feature_store()

# ---------- 2) 取 v2 的 Feature Groups ----------
fg_aq = fs.get_feature_group("air_quality_daily", version=2)        # PK=["city","station_id"], event_time="date"
fg_w  = fs.get_feature_group("weather_daily_forecast", version=2)   # PK=["city","station_id"], event_time="date"

# ---------- 3) 读 FG ----------
aq_df = fg_aq.read()     # 标签
w_df  = fg_w.read()      # 天气特征

# 转时间类型
aq_df["date"] = pd.to_datetime(aq_df["date"])
w_df["date"]  = pd.to_datetime(w_df["date"])

# 可选白名单过滤（只保留关注的站点）
if STATION_WHITELIST:
    aq_df = aq_df[aq_df["station_id"].isin(STATION_WHITELIST)]
    w_df  = w_df[w_df["station_id"].isin(STATION_WHITELIST)]

# 诊断信息（整体）
print("\n[label] per-station date range:")
print(aq_df.groupby("station_id")["date"].agg(["min", "max", "count"]))
print("\n[weather] per-station date range:")
print(w_df.groupby("station_id")["date"].agg(["min", "max", "count"]))
print("\n[diag] aq_df cols:", list(aq_df.columns))
print("[diag] w_df  cols:", list(w_df.columns))

# ---------- 4) 动态选择 join 键并合并 ----------
aq_cols = set(aq_df.columns)
w_cols  = set(w_df.columns)

candidate_keys = ["city", "station_id", "date"]
join_keys = [k for k in candidate_keys if (k in aq_cols and k in w_cols)]

must_have = {"station_id", "date"}
if not must_have.issubset(set(join_keys)):
    missing = must_have - set(join_keys)
    raise SystemExit(f"[error] 右表缺少必须连接键：{missing}。"
                     f"当前 join_keys={join_keys}；请检查天气表是否包含 station_id 与 date。")

print(f"[info] join on keys: {join_keys}")

# 合并
df = aq_df.merge(
    w_df,
    on=join_keys,
    how="inner",
    suffixes=("", "_wx")
)

# 若合并后没有 city，则从标签补回
if "city" not in df.columns and "city" in aq_df.columns:
    df = df.merge(
        aq_df[["station_id", "date", "city"]].drop_duplicates(),
        on=["station_id", "date"],
        how="left"
    )

# 清洗
dedup_keys = [k for k in ["city", "station_id", "date"] if k in df.columns]
df = (
    df.dropna()
      .drop_duplicates(dedup_keys)
      .sort_values("date")
)

# 逐站点重叠诊断（看每站最终可训练行数，以及合并前后时间交集）
print("\n[overlap] per-station rows after merge:")
if len(df):
    print(df.groupby("station_id")["date"].agg(["min", "max", "count"]))
else:
    print("(empty)")

print("[info] total training rows:", len(df))
if len(df) == 0:
    raise SystemExit(
        "[warn] 合并后没有可训练的数据。\n"
        "通常是标签与天气的日期区间没有重叠：\n"
        "请在 01 中加大 PAST_DAYS（或用我给你的 01 自动按标签最新日期扩窗），\n"
        "或者更新 CSV 到最近日期，然后重跑 01 与本脚本。"
    )

# ---------- 5) 训练：每站一个模型 ----------
os.makedirs("models", exist_ok=True)
results = []

# 标签/标识列需要排除
DROP_COLS = [c for c in ["pm2_5", "city", "station_id", "date"] if c in df.columns]

for st_id, g in df.groupby("station_id"):
    # 时间顺序切分：80% 训练，20% 验证
    g = g.sort_values("date")
    split = int(len(g) * 0.8)
    tr, te = g.iloc[:split], g.iloc[split:]

    num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c not in DROP_COLS]

    if len(tr) < MIN_TRAIN_ROWS or len(feat_cols) == 0:
        print(f"[skip] {st_id} 样本不足或无有效特征（rows={len(tr)}, feats={len(feat_cols)}）")
        continue

    X_tr, y_tr = tr[feat_cols], tr["pm2_5"]
    X_te, y_te = te[feat_cols], te["pm2_5"]

    model = RandomForestRegressor(n_estimators=400, random_state=42)
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    mae = float(mean_absolute_error(y_te, pred))

    joblib.dump({"model": model, "features": feat_cols}, f"models/{st_id}_rf.joblib")
    results.append((st_id, len(g), len(feat_cols), mae))
    print(f"[ok] {st_id}: rows={len(g)}, feats={len(feat_cols)}, MAE={mae:.2f} -> models/{st_id}_rf.joblib")

if not results:
    print("[warn] 没有任何站点完成训练。")
else:
    print("\n=== Summary ===")
    for st_id, n, k, mae in results:
        print(f"{st_id}: rows={n}, feats={k}, MAE={mae:.2f}")
