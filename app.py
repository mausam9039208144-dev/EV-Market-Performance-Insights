from functools import lru_cache
from pathlib import Path
import os

import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

app = Flask(__name__)

# Only the columns the site actually uses (keeps memory low on free hosts).
USE_COLUMNS = [
    "vehicle_id", "model_name", "manufacturer", "model_year",
    "manufacturing_country", "market_region", "category",
    "battery_capacity_kwh", "battery_type", "battery_cycle_life",
    "annual_sales_units", "customer_rating", "range_km",
]
NUMERIC = ["model_year", "battery_capacity_kwh", "battery_cycle_life",
           "annual_sales_units", "customer_rating", "range_km"]


@lru_cache(maxsize=1)
def load_data():
    """Read every .csv / .gz file in /data once, combine them, and cache the result."""
    files = sorted(list(DATA_DIR.glob("*.gz")) + list(DATA_DIR.glob("*.csv")))
    if not files:
        raise FileNotFoundError(f"No data files found in {DATA_DIR}")
    frames = [pd.read_csv(f, usecols=USE_COLUMNS) for f in files]
    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(subset="vehicle_id")
    for col in NUMERIC:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def options_for(data, column):
    return sorted(data[column].dropna().astype(str).unique().tolist())


def grouped(df, column, metric, top=None, ascending=False):
    """Total sales (millions of units) or record count per group."""
    if metric == "records":
        s = df.groupby(column)["vehicle_id"].count()
    else:
        s = df.groupby(column)["annual_sales_units"].sum() / 1e6
    s = s.sort_values(ascending=ascending)
    if top:
        s = s.head(top)
    return {"x": s.index.astype(str).tolist(), "y": [round(float(v), 2) for v in s.values]}


@app.route("/")
def index():
    data = load_data()

    year = request.args.get("year", "All")
    region = request.args.get("region", "All")
    manufacturer = request.args.get("manufacturer", "All")
    category = request.args.get("category", "All")
    battery_type = request.args.get("battery_type", "All")
    metric = request.args.get("metric", "sales")
    if metric not in ("sales", "records"):
        metric = "sales"

    filtered = data
    if year != "All":
        filtered = filtered[filtered["model_year"] == pd.to_numeric(year, errors="coerce")]
    if region != "All":
        filtered = filtered[filtered["market_region"] == region]
    if manufacturer != "All":
        filtered = filtered[filtered["manufacturer"] == manufacturer]
    if category != "All":
        filtered = filtered[filtered["category"] == category]
    if battery_type != "All":
        filtered = filtered[filtered["battery_type"] == battery_type]

    def num(value, digits):
        return round(float(value), digits) if pd.notna(value) else None

    summary = {
        "total_records": int(len(filtered)),
        "unique_models": int(filtered["model_name"].nunique()),
        "total_sales_m": num(filtered["annual_sales_units"].sum() / 1e6, 1) or 0,
        "avg_rating": num(filtered["customer_rating"].mean(), 2),
        "avg_range": num(filtered["range_km"].mean(), 0),
        "avg_capacity": num(filtered["battery_capacity_kwh"].mean(), 2),
        "avg_cycle_life": num(filtered["battery_cycle_life"].mean(), 1),
        "cycle_missing": int(filtered["battery_cycle_life"].isna().sum()),
    }

    by_year = grouped(filtered.dropna(subset=["model_year"]).assign(
        model_year=lambda d: d["model_year"].astype(int)), "model_year", metric, ascending=True)

    battery = (
        filtered.groupby("battery_type")
        .agg(avg_capacity=("battery_capacity_kwh", "mean"),
             avg_cycle_life=("battery_cycle_life", "mean"))
        .fillna(0).round(1)
    )

    chart_data = {
        "metric_label": "Sales (millions of units)" if metric == "sales" else "Records",
        "year": by_year,
        "manufacturer": grouped(filtered, "manufacturer", metric, top=12, ascending=False),
        "category": grouped(filtered, "category", metric),
        "region": grouped(filtered, "market_region", metric),
        "battery": grouped(filtered, "battery_type", metric),
        "country": grouped(filtered, "manufacturing_country", metric, top=15),
        "battery_quality": {
            "x": battery.index.astype(str).tolist(),
            "avg_capacity": battery["avg_capacity"].tolist(),
            "avg_cycle_life": battery["avg_cycle_life"].tolist(),
        },
    }

    return render_template(
        "index.html",
        summary=summary,
        chart_data=chart_data,
        filters={"year": year, "region": region, "manufacturer": manufacturer,
                 "category": category, "battery_type": battery_type, "metric": metric},
        options={
            "years": sorted(data["model_year"].dropna().astype(int).astype(str).unique().tolist()),
            "regions": options_for(data, "market_region"),
            "manufacturers": options_for(data, "manufacturer"),
            "categories": options_for(data, "category"),
            "battery_types": options_for(data, "battery_type"),
        },
    )


@app.route("/tableau")
def tableau():
    # Set TABLEAU_DASHBOARD_URL to your published Tableau Public URL.
    tableau_url = os.environ.get("TABLEAU_DASHBOARD_URL", "").strip()
    return render_template("tableau.html", tableau_url=tableau_url)


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    # Local development only. In production use: gunicorn app:app
    app.run(debug=True)
