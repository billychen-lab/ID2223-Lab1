import os
import glob
import shutil
from datetime import datetime, timezone
import pandas as pd

OUTPUT_DIR = "outputs"
SITE_DIR = "site"
ASSETS_DIR = os.path.join(SITE_DIR, "assets")

STATIONS = [
    ("hk-kwai-chung", "Kwai Chung"),
    ("hk-tsuen-wan", "Tsuen Wan"),
    ("hk-tuen-mun", "Tuen Mun"),
    ("hk-tung-chung", "Tung Chung"),
    ("hk-yuen-long", "Yuen Long"),
]

def ensure_dirs():
    os.makedirs(SITE_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

def copy_asset(src_path: str) -> str:
    if not src_path or not os.path.isfile(src_path):
        return ""
    dst_path = os.path.join(ASSETS_DIR, os.path.basename(src_path))
    shutil.copy2(src_path, dst_path)
    return os.path.relpath(dst_path, SITE_DIR).replace("\\", "/")

def find_outputs_for(station_id: str):
    hind = glob.glob(os.path.join(OUTPUT_DIR, f"{station_id}_hindcast.png"))
    fore = glob.glob(os.path.join(OUTPUT_DIR, f"{station_id}_forecast.png"))
    csv = glob.glob(os.path.join(OUTPUT_DIR, f"{station_id}_predictions.csv"))
    return (hind[0] if hind else ""), (fore[0] if fore else ""), (csv[0] if csv else "")

def summarize_csv(csv_path: str):
    if not csv_path or not os.path.isfile(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if "date" not in df.columns or "pm2_5_pred" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    mae = None
    if "pm2_5_true" in df.columns:
        hind = df.dropna(subset=["pm2_5_true"]).copy()
        if not hind.empty:
            mae = float((hind["pm2_5_true"] - hind["pm2_5_pred"]).abs().mean())
    today = pd.Timestamp.today().normalize()
    future = df[df["date"] >= today].copy().sort_values("date")
    next7 = future.head(7)
    next7_mean = float(next7["pm2_5_pred"].mean()) if len(next7) > 0 else None
    tmr_date = today + pd.Timedelta(days=1)
    tmr_row = future[future["date"] == tmr_date]
    tmr = float(tmr_row["pm2_5_pred"].iloc[0]) if len(tmr_row) else None
    return {"mae": mae, "next7_mean": next7_mean, "tmr": tmr}

def render_detail_page(station_id, friendly, hind_rel, fore_rel, csv_rel, metrics):
    """生成单个传感器详情页 HTML"""
    mae = f"{metrics['mae']:.2f}" if metrics and metrics.get("mae") is not None else "–"
    now_str = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{friendly} Air Quality Detail</title>
<style>
body {{
  font-family: system-ui, sans-serif; background: #f7f7f7; margin: 0; padding: 0; color: #222;
}}
.header {{
  background: #0b7; color: #fff; padding: 16px 24px;
}}
.container {{
  max-width: 900px; margin: 20px auto; background: #fff; border-radius: 12px; padding: 20px;
  box-shadow: 0 4px 14px rgba(0,0,0,0.06);
}}
img {{
  width: 100%; height: auto; border-radius: 10px; border: 1px solid #ddd; margin-bottom: 20px;
}}
.btn {{
  display: inline-block; background: #0b7; color: #fff; padding: 10px 16px; border-radius: 8px;
  text-decoration: none;
}}
.note {{ color: #777; }}
</style>
</head>
<body>
  <div class="header">
    <h1>{friendly} ({station_id})</h1>
  </div>
  <div class="container">
    <p class="note">Last updated: {now_str}</p>
    <p><b>Hindcast MAE:</b> {mae}</p>
    <img src="{hind_rel}" alt="Hindcast plot"/>
    <img src="{fore_rel}" alt="Forecast plot"/>
    <p><a class="btn" href="{csv_rel}" download>Download CSV</a>
       <a class="btn" href="index.html">← Back to all stations</a></p>
  </div>
</body>
</html>"""
    page_path = os.path.join(SITE_DIR, f"{station_id}.html")
    with open(page_path, "w", encoding="utf-8") as f:
        f.write(html)

def main():
    ensure_dirs()
    cards = []
    for sid, friendly in STATIONS:
        hind, fore, csvp = find_outputs_for(sid)
        hind_rel = copy_asset(hind)
        fore_rel = copy_asset(fore)
        csv_rel = copy_asset(csvp)
        metrics = summarize_csv(csvp)
        # 生成详情页
        render_detail_page(sid, friendly, hind_rel, fore_rel, csv_rel, metrics)
        # 主页卡片
        cards.append(f"""
        <div class="card">
          <h2><a href="{sid}.html">{friendly}</a> <span class="small">({sid})</span></h2>
          <div class="row">
            <div><img src="{hind_rel}" alt="hindcast"></div>
            <div><img src="{fore_rel}" alt="forecast"></div>
          </div>
          <p><a class="btn" href="{csv_rel}" download>Download CSV</a></p>
        </div>
        """)
    now_str = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Hong Kong Air Quality Dashboard</title>
<style>
body {{
  font-family: system-ui, sans-serif; margin:0; background:#f7f7f7; color:#222;
}}
.header {{ background:#0b7; color:#fff; padding:16px 24px; }}
.container {{ max-width:1200px; margin:24px auto; padding:0 16px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(350px,1fr)); gap:16px; }}
.card {{ background:#fff; padding:16px; border-radius:12px; box-shadow:0 4px 14px rgba(0,0,0,0.06); }}
.card h2 a {{ color:#0b7; text-decoration:none; }}
.card img {{ width:100%; border-radius:10px; border:1px solid #eaeaea; }}
.btn {{ display:inline-block; padding:8px 12px; background:#0b7; color:#fff; border-radius:8px; text-decoration:none; }}
.small {{ color:#777; font-size:13px; }}
</style>
</head>
<body>
  <div class="header">
    <h1>Hong Kong Air Quality Dashboard</h1>
  </div>
  <div class="container">
    <p class="small">Last updated: {now_str}</p>
    <div class="grid">
      {''.join(cards)}
    </div>
  </div>
</body>
</html>"""
    with open(os.path.join(SITE_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print("[ok] generated multi-page dashboard in site/")

if __name__ == "__main__":
    main()
