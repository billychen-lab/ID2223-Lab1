import os
import requests
import pandas as pd
import datetime
import hopsworks

# --------------------------
# 
# --------------------------
STATIONS = [
    {"station_id": "hk-tuen-mun",    "api_id": 3928, "lat": 22.394984, "lon": 113.973140},
    {"station_id": "hk-yuen-long",   "api_id": 2569, "lat": 22.446221, "lon": 114.035288},
    {"station_id": "hk-tsuen-wan",   "api_id": 2567, "lat": 22.371670, "lon": 114.113470},
    {"station_id": "hk-Kwai-Chung",  "api_id": 2561, "lat": 22.351040, "lon": 114.130800},
    {"station_id": "hk-tung-chung",  "api_id": 2568, "lat": 22.289240, "lon": 113.941370},
]

# --------------------------
#  AQICN Token
# --------------------------
AQICN_TOKEN = os.environ["AQICN_API_KEY"]  # GitHub Secrets 或 Hopsworks Secrets 导入
HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]

FORECAST_DAYS = 7  # 未来 7 天天气预报


# --------------------------
#  PM2.5（
# --------------------------
def get_pm25_today(api_id):
    url = f"https://api.waqi.info/feed/@{api_id}/?token={AQICN_TOKEN}"
    r = requests.get(url, timeout=30).json()

    if r["status"] != "ok":
        raise RuntimeError(f"API error: {r}")

    data = r["data"]
    pm25 = data["iaqi"]["pm25"]["v"]
    time_str = data["time"]["s"]  # e.g., "2025-01-12 15:00:00"
    date = pd.to_datetime(time_str).normalize()

    df = pd.DataFrame([{
        "station_id": api_id,
        "date": date,
        "pm2_5": pm25
    }])

    return df


# --------------------------
# weather
# --------------------------
def get_weather(lat, lon):
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "Asia/Hong_Kong",
        "start_date": yesterday,
        "end_date": today + datetime.timedelta(days=FORECAST_DAYS),
        "daily": "temperature_2m_mean,precipitation_sum,wind_speed_10m_max,wind_direction_10m_dominant",
    }

    r = requests.get(url, params=params).json()
    daily = r["daily"]

    df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "temperature_2m_mean": daily["temperature_2m_mean"],
        "precipitation_sum": daily["precipitation_sum"],
        "wind_speed_10m_max": daily["wind_speed_10m_max"],
        "wind_direction_10m_dominant": daily["wind_direction_10m_dominant"],
    })

    return df


# --------------------------
# main
# --------------------------
def main():
    print("  Logging in to Hopsworks ...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    weather_rows = []
    pm_rows = []

    for st in STATIONS:
        print(f"Fetching: {st['station_id']}")

        # ============= 1. 今日 PM2.5 =============
        pm_df = get_pm25_today(st["api_id"])
        pm_df["station_id"] = st["station_id"]
        pm_rows.append(pm_df)

        # ============= 2. 天气（昨日 + 明天 + 未来7天） =============
        wx_df = get_weather(st["lat"], st["lon"])
        wx_df["station_id"] = st["station_id"]
        weather_rows.append(wx_df)

    pm_df_all = pd.concat(pm_rows)
    weather_df_all = pd.concat(weather_rows)

    # -------------------------
    # 写入 Feature Store
    # -------------------------
    print(" Uploading to Hopsworks ...")

    weather_fg = fs.get_or_create_feature_group(
        name="weather_daily_forecast",
        version=1,
        primary_key=["station_id"],
       # event_time="date",
        description="Daily weather + future forecasts",
        online_enabled=False
    )

    aq_fg = fs.get_or_create_feature_group(
        name="air_quality_daily",
        version=1,
        primary_key=["station_id"],
      #  event_time="date",
        description="Daily PM2.5 observations",
        online_enabled=False
    )

    weather_fg.insert(weather_df_all, write_options={"wait_for_job": True})
    aq_fg.insert(pm_df_all, write_options={"wait_for_job": True})

    print(" Done! Daily pipeline completed.")


if __name__ == "__main__":
    main()
