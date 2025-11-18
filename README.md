# ID2223 – Lab 1: Air Quality Prediction Service

## Project Info

- **Course**: ID2223 – Scalable Machine Learning and Deep Learning   
- **Lab**: Lab 1 – Air Quality Prediction Service  
- **Student**: Yunquan Chen(yunquan@kth.se),  Sibo Zhang(siboz@kth.se) 
- **GitHub repository**: https://github.com/billychen-lab/ID2223-Lab1/tree/main  
- **Dashboard URL**: https://teal-custard-82b5f9.netlify.app

---

## 1. Overview

This project implements an end-to-end serverless machine-learning system that predicts daily PM2.5 for several air-quality sensors in a region of **Hong Kong** for the next 7 days.

The system is built around the **Hopsworks Feature Store** and consists of:

1. A **multi-station feature & label pipeline** that ingests Open-Meteo weather/air-quality data and historical PM2.5 CSVs.
2. A **training pipeline** that joins features and labels and trains a separate Random Forest model for each station.
3. A **batch inference & evaluation pipeline** that produces:
   - 14-day **hindcast** plots (prediction vs. ground truth); MAE values in the plots mean the average prediction error in the area of this sensor with the unit µg/m³ in these 14 days. The results of MAE show that our models' prediction accuracy is good.
   - 7-day **forecast** plots starting from “today”.
   - CSV files with predictions and (when available) true values, the lines 2-15 of the predictions csv files are predictions and real values of PM2.5 of the sensor in last 14 days, the lines 16-22 are predictions of the sensor in the next 7 days, which are also shown in the plots. 
4. A **lag-feature experiment** comparing the baseline weather-only model against weather+lagged PM2.5.
5. A static **multi-page dashboard** that visualizes all Hong Kong station sensors we have selected.

The implementation covers Lab 1 tasks for **grades E, C and A**. (covers the requirements for grade A)

---

## 2. Repository Structure

```text
.
├── .github/
│   └── workflows/           # GitHub Actions workflow for the daily pipeline
├── outputs/                 # Generated plots and prediction CSVs (one set per station)
│   ├── hk-Kwai-Chung_forecast.png
│   ├── hk-Kwai-Chung_hindcast.png
│   ├── hk-Kwai-Chung_predictions.csv
│   ├── hk-tsuen-wan_forecast.png
│   ├── hk-tsuen-wan_hindcast.png
│   ├── hk-tsuen-wan_predictions.csv
│   ├── hk-tuen-mun_forecast.png
│   ├── hk-tuen-mun_hindcast.png
│   ├── hk-tuen-mun_predictions.csv
│   ├── hk-tung-chung_forecast.png
│   ├── hk-tung-chung_hindcast.png
│   ├── hk-tung-chung_predictions.csv
│   ├── hk-yuen-long_forecast.png
│   ├── hk-yuen-long_hindcast.png
│   └── hk-yuen-long_predictions.csv
├── 01_write_feature_groups.py     # Multi-station feature & label pipeline (backfill + daily)
├── 02_train_and_feature_view_multi.py  # Join features + labels, train per-station models
├── 03_predict_and_plot.py         # Batch inference, hindcast & forecast plots for one station
├── 04_lag_vs_baseline.py          # Lag features vs. baseline comparison (hk-tuen-mun)
├── build_dashboard.py             # Static dashboard generator (uses files in outputs/)
├── kwai-chung-air-quality.csv     # Historical PM2.5 CSVs (labels) for each Hong Kong station
├── tsuen-wan-air-quality.csv
├── tuen-mun-air-quality.csv
├── tung-chung-air-quality.csv
├── yuen-long-air-quality.csv
├── lag_report.csv                 # Result table from lag vs baseline experiment
├── requirement.txt                # Python dependencies
└── README.md
