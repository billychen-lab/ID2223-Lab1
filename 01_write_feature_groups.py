import os
import copy
import requests
import pandas as pd
import hopsworks as hs

# ===================== 站点清单 =====================
# 可保留瑞典站 se-0001（无标签），并新增香港屯门站（有 CSV 标签）
stations = [
    {
        "city": "SE_City_1",
        "station_id": "se-0001",
        "lat": 62.99,
        "lon": 17.64,
        "timezone": "Europe/Stockholm",
        "sensor_csv": None,  # 无标签
    },
    {
        "city": "HongKong",
        "station_id": "hk-tuen-mun",
        "lat": 22.394984,
        "lon": 113.973140,
        "timezone": "Asia/Hong_Kong",
        "sensor_csv": r"D:\ID2223\tuen-mun-air-quality.csv",
    },
    {
        "city": "HongKong",
        "station_id": "hk-yuen-long",
        "lat": 22.446221,          # 新站：元朗
        "lon": 114.035288,
        "timezone": "Asia/Hong_Kong",
        "sensor_csv": r"D:\ID2223\yuen-long-air-quality.csv",  # 你的新CSV
    },
    {
        "city": "HongKong",
        "station_id": "hk-tsuen-wan",
        "lat": 22.37167,
        "lon": 114.11347,
        "timezone": "Asia/Hong_Kong",
        "sensor_csv": r"D:\ID2223\tsuen-wan-air-quality.csv",  # 你的新CSV
    },
    {
        "city": "HongKong",
        "station_id": "hk-Kwai-Chung",
        "lat": 22.35104,
        "lon": 114.13080,
        "timezone": "Asia/Hong_Kong",
        "sensor_csv": r"D:\ID2223\kwai-chung-air-quality.csv",  # 你的新CSV
    },
    {
        "city": "HongKong",
        "station_id": "hk-tung-chung",
        "lat": 22.28924,
        "lon": 113.94137,
        "timezone": "Asia/Hong_Kong",
        "sensor_csv": r"D:\ID2223\tung-chung-air-quality.csv",  # 你的新CSV
    },
]

# 回填/预测窗口（可用环境变量覆盖）
DEFAULT_PAST_DAYS = int(os.getenv("PAST_DAYS", "14"))
DEFAULT_FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", "7"))  # 建议 6

# Open-Meteo 变量
aq_vars = "pm2_5,pm10,ozone,nitrogen_dioxide,carbon_monoxide,sulphur_dioxide,us_aqi"
wx_vars = "temperature_2m,relative_humidity_2m,dew_point_2m,wind_speed_10m,wind_direction_10m,precipitation,pressure_msl,visibility"


def _fetch_openmeteo(url, params, name):
    """取 JSON；若没有 hourly，自动降级参数重试，仍失败就抛出带 reason 的错误"""
    r = requests.get(url, params=params, timeout=60)
    j = r.json()
    if "hourly" in j:
        return j

    p = copy.deepcopy(params)
    p.pop("past_days", None)
    r = requests.get(url, params=p, timeout=60)
    j = r.json()
    if "hourly" in j:
        return j

    p2 = copy.deepcopy(p)
    p2["hourly"] = "pm2_5" if name == "air" else "temperature_2m"
    r = requests.get(url, params=p2, timeout=60)
    j = r.json()
    if "hourly" in j:
        return j

    raise RuntimeError(f"{name} api returned no 'hourly'. payload={j}")


def _hourly_to_df(payload, cols):
    df = pd.DataFrame({"time": pd.to_datetime(payload["hourly"]["time"])})
    for c in cols.split(","):
        df[c] = payload["hourly"].get(c)
    return df


def fetch_openmeteo_daily(lat, lon, tz, past_days=14, forecast_days=7):
    aq = _fetch_openmeteo(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        {
            "latitude": lat, "longitude": lon, "timezone": tz,
            "hourly": aq_vars, "past_days": past_days, "forecast_days": forecast_days,
        },
        name="air",
    )
    wx = _fetch_openmeteo(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": lat, "longitude": lon, "timezone": tz,
            "hourly": wx_vars, "past_days": past_days, "forecast_days": forecast_days,
        },
        name="wx",
    )
    hourly = _hourly_to_df(aq, aq_vars).merge(
        _hourly_to_df(wx, wx_vars), on="time", how="inner"
    )
    hourly["date"] = pd.to_datetime(hourly["time"]).dt.tz_localize(None).dt.normalize()
    return hourly


def read_sensor_daily(csv_path, city, station_id):
    raw = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="ignore")
    raw.columns = [str(c).strip() for c in raw.columns]
    lower_map = {c.lower(): c for c in raw.columns}

    time_keys = [k for k in lower_map if k in ["time", "timestamp", "datetime", "date", "日期", "时间"]]
    pm_keys = [k for k in lower_map if k in ["pm2.5", "pm2_5", "pm25", "pm 2.5", "pm-2.5", "pm₂.₅", "pm₂.5"]]
    if not time_keys or not pm_keys:
        raise ValueError(f"[{station_id}] 找不到时间/PM2.5 列。列名(规范化后)={list(lower_map.keys())}")

    tcol = lower_map[time_keys[0]]
    pcol = lower_map[pm_keys[0]]

    df = raw[[tcol, pcol]].copy()
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df[pcol] = pd.to_numeric(df[pcol], errors="coerce")
    df = df.dropna(subset=[tcol, pcol])

    df["date"] = df[tcol].dt.tz_localize(None).dt.normalize()
    out = (
        df.groupby("date", as_index=False)[pcol]
          .mean()
          .rename(columns={pcol: "pm2_5"})
    )
    out["city"] = city
    out["station_id"] = station_id
    return out[["city", "station_id", "date", "pm2_5"]]


def build_weather_features_for_station(st):
    """根据标签最新日期，自动放大 past_days；确保天气覆盖标签"""
    want_past = DEFAULT_PAST_DAYS
    if st.get("sensor_csv"):
        try:
            lbl = read_sensor_daily(st["sensor_csv"], st["city"], st["station_id"])
            if not lbl.empty:
                max_label_date = pd.to_datetime(lbl["date"]).max()
                today = pd.Timestamp.now(tz=st["timezone"]).normalize().tz_localize(None)
                need_days = max(0, int((today - max_label_date).days) + 1)
                want_past = max(DEFAULT_PAST_DAYS, min(360, need_days))
                print(f"[info] {st['station_id']} label max={max_label_date.date()}, past_days -> {want_past}")
        except Exception as e:
            print(f"[warn] read labels failed for {st['station_id']}: {e}")

    hourly = fetch_openmeteo_daily(
        lat=st["lat"], lon=st["lon"], tz=st["timezone"],
        past_days=want_past, forecast_days=DEFAULT_FORECAST_DAYS,
    )
    hourly["city"] = st["city"]
    hourly["station_id"] = st["station_id"]

    daily = hourly.groupby(["city", "station_id", "date"], as_index=False).agg(
        pm2_5_mean=("pm2_5", "mean"),
        pm2_5_max=("pm2_5", "max"),
        pm10_mean=("pm10", "mean"),
        ozone_mean=("ozone", "mean"),
        nitrogen_dioxide_mean=("nitrogen_dioxide", "mean"),
        carbon_monoxide_mean=("carbon_monoxide", "mean"),
        sulphur_dioxide_mean=("sulphur_dioxide", "mean"),
        us_aqi_mean=("us_aqi", "mean"),
        temperature_2m_mean=("temperature_2m", "mean"),
        relative_humidity_2m_mean=("relative_humidity_2m", "mean"),
        dew_point_2m_mean=("dew_point_2m", "mean"),
        wind_speed_10m_mean=("wind_speed_10m", "mean"),
        wind_direction_10m_mean=("wind_direction_10m", "mean"),
        precipitation_sum=("precipitation", "sum"),
        pressure_msl_mean=("pressure_msl", "mean"),
        visibility_mean=("visibility", "mean"),
    )
    return daily


def main():
    weather_all, labels_all = [], []

    for st in stations:
        print(f"[features] {st['city']} / {st['station_id']} @ ({st['lat']}, {st['lon']})")
        weather_all.append(build_weather_features_for_station(st))

        if st.get("sensor_csv"):
            print(f"[labels]   from {st['sensor_csv']}")
            try:
                labels_all.append(read_sensor_daily(st["sensor_csv"], st["city"], st["station_id"]))
            except Exception as e:
                print(f"[warn] label ingestion failed for {st['station_id']}: {e}")
        else:
            print(f"[skip]     no sensor_csv for {st['station_id']}")

    weather_df = pd.concat(weather_all, ignore_index=True)
    sensor_df = pd.concat(labels_all, ignore_index=True) if labels_all else pd.DataFrame()

    weather_df["date"] = pd.to_datetime(weather_df["date"])
    if not sensor_df.empty:
        sensor_df["date"] = pd.to_datetime(sensor_df["date"])

    project = hs.login(api_key_value=os.environ["HOPSWORKS_API_KEY"],
                       project=os.getenv("HOPSWORKS_PROJECT", None))
    fs = project.get_feature_store()

    weather_fg = fs.get_or_create_feature_group(
        name="weather_daily_forecast",
        version=2,
        description="Open-Meteo daily features (multi-station)",
        primary_key=["city", "station_id"],
        event_time="date",
        online_enabled=False,
    )
    aq_fg = fs.get_or_create_feature_group(
        name="air_quality_daily",
        version=2,
        description="Daily PM2.5 label (multi-station)",
        primary_key=["city", "station_id"],
        event_time="date",
        online_enabled=False,
    )

    weather_fg.insert(weather_df.drop_duplicates(["city", "station_id", "date"]),
                      write_options={"wait_for_job": True})
    print("[ok] inserted weather rows:", len(weather_df))

    if not sensor_df.empty:
        aq_fg.insert(sensor_df.drop_duplicates(["city", "station_id", "date"]),
                     write_options={"wait_for_job": True})
        print("[ok] inserted label rows:", len(sensor_df))
    else:
        print("[info] no labels inserted (only features)")

    print("[done] multi-station backfill finished.")


if __name__ == "__main__":
    main()
