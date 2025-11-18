# ID2223 – Lab 1: Air Quality Prediction Service

## Project Info

- **Course**: ID2223 – Scalable Machine Learning and Deep Learning Systems  
- **Lab**: Lab 1 – Air Quality Prediction Service  
- **Student**: TODO: Your name, KTH ID  
- **GitHub repository**: TODO: link to this repo  
- **Dashboard URL**: TODO: public URL of the `site/` folder (e.g. GitHub Pages)

---

## 1. Overview

This project implements an end-to-end serverless machine-learning system that predicts daily PM2.5 for several air-quality stations in **Hong Kong** for the next 7 days.

The system is built around the **Hopsworks Feature Store** and consists of:

1. A **multi-station feature & label pipeline** that ingests Open-Meteo weather/air-quality data and historical PM2.5 CSVs.
2. A **training pipeline** that joins features and labels and trains a separate Random Forest model for each station.
3. A **batch inference & evaluation pipeline** that produces:
   - 14-day **hindcast** plots (prediction vs. ground truth)
   - 7-day **forecast** plots starting from “today”
   - CSV files with predictions and (when available) true values
4. A **lag-feature experiment** comparing the baseline weather-only model against weather+lagged PM2.5.
5. A static **multi-page dashboard** that visualizes all Hong Kong stations.

The implementation covers Lab 1 tasks for **grades E, C and A**.

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
