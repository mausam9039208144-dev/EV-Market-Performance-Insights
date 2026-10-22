from pathlib import Path
import os
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ev_market_data.csv"

app = Flask(__name__)

EXPECTED_COLUMNS = [
    "vehicle_id", "model_name", "manufacturer", "model_year",
    "manufacturing_country", "market_region", "category",
    "battery_capacity_kwh", "battery_type", "battery_cycle_life"
]

def load_data():
    data = pd.read_csv(DATA_PATH)
    missing = [column for column in EXPECTED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError("Dataset is missing expected columns: " + ", ".join(missing))
    data["model_year"] = pd.to_numeric(data["model_year"], errors="coerce")
    data["battery_capacity_kwh"] = pd.to_numeric(data["battery_capacity_kwh"], errors="coerce")
    data["battery_cycle_life"] = pd.to_numeric(data["battery_cycle_life"], errors="coerce")
    return data

def options_for(data, column):
    return sorted(data[column].dropna().astype(str).unique().tolist())

@app.route("/")
def index():
    data = load_data()

    year = request.args.get("year", "All")
    region = request.args.get("region", "All")
    manufacturer = request.args.get("manufacturer", "All")
    category = request.args.get("category", "All")
    battery_type = request.args.get("battery_type", "All")

    filtered = data.copy()
    if year != "All":
        filtered = filtered[filtered["model_year"].astype("Int64").astype(str) == year]
    if region != "All":
        filtered = filtered[filtered["market_region"].astype(str) == region]
    if manufacturer != "All":
        filtered = filtered[filtered["manufacturer"].astype(str) == manufacturer]
    if category != "All":
        filtered = filtered[filtered["category"].astype(str) == category]
    if battery_type != "All":
        filtered = filtered[filtered["battery_type"].astype(str) == battery_type]

    total_records = int(len(filtered))
    unique_models = int(filtered["model_name"].nunique())
    avg_capacity = filtered["battery_capacity_kwh"].mean()
    avg_cycle_life = filtered["battery_cycle_life"].mean()
    cycle_missing = int(filtered["battery_cycle_life"].isna().sum())

    # Aggregated chart data are passed as plain JSON-compatible dictionaries.
    year_data = (
        filtered.dropna(subset=["model_year"])
        .groupby("model_year", as_index=False)
        .agg(records=("vehicle_id", "count"), avg_capacity=("battery_capacity_kwh", "mean"))
        .sort_values("model_year")
    )
    manufacturer_data = (
        filtered.groupby("manufacturer", as_index=False)
        .agg(records=("vehicle_id", "count"), avg_capacity=("battery_capacity_kwh", "mean"))
        .sort_values("records", ascending=False)
        .head(12)
        .sort_values("records")
    )
    category_data = (
        filtered.groupby("category", as_index=False)
        .agg(records=("vehicle_id", "count"))
        .sort_values("records", ascending=False)
    )
    region_data = (
        filtered.groupby("market_region", as_index=False)
        .agg(records=("vehicle_id", "count"))
        .sort_values("records", ascending=False)
    )
    battery_data = (
        filtered.groupby("battery_type", as_index=False)
        .agg(avg_capacity=("battery_capacity_kwh", "mean"),
             avg_cycle_life=("battery_cycle_life", "mean"),
             records=("vehicle_id", "count"))
        .sort_values("records", ascending=False)
    )
    country_data = (
        filtered.groupby("manufacturing_country", as_index=False)
        .agg(records=("vehicle_id", "count"))
        .sort_values("records", ascending=False)
        .head(15)
    )

    chart_data = {
        "year": {
            "x": year_data["model_year"].fillna(0).astype(int).tolist(),
            "records": year_data["records"].astype(int).tolist(),
            "avg_capacity": year_data["avg_capacity"].round(2).fillna(0).tolist()
        },
        "manufacturer": {
            "x": manufacturer_data["manufacturer"].astype(str).tolist(),
            "records": manufacturer_data["records"].astype(int).tolist()
        },
        "category": {
            "x": category_data["category"].astype(str).tolist(),
            "records": category_data["records"].astype(int).tolist()
        },
        "region": {
            "x": region_data["market_region"].astype(str).tolist(),
            "records": region_data["records"].astype(int).tolist()
        },
        "battery": {
            "x": battery_data["battery_type"].astype(str).tolist(),
            "avg_capacity": battery_data["avg_capacity"].round(2).fillna(0).tolist(),
            "avg_cycle_life": battery_data["avg_cycle_life"].round(1).fillna(0).tolist(),
            "records": battery_data["records"].astype(int).tolist()
        },
        "country": {
            "x": country_data["manufacturing_country"].astype(str).tolist(),
            "records": country_data["records"].astype(int).tolist()
        }
    }

    summary = {
        "total_records": total_records,
        "unique_models": unique_models,
        "avg_capacity": round(float(avg_capacity), 2) if pd.notna(avg_capacity) else None,
        "avg_cycle_life": round(float(avg_cycle_life), 1) if pd.notna(avg_cycle_life) else None,
        "cycle_missing": cycle_missing,
    }

    return render_template(
        "index.html",
        summary=summary,
        chart_data=chart_data,
        filters={
            "year": year, "region": region, "manufacturer": manufacturer,
            "category": category, "battery_type": battery_type
        },
        options={
            "years": sorted(data["model_year"].dropna().astype(int).astype(str).unique().tolist()),
            "regions": options_for(data, "market_region"),
            "manufacturers": options_for(data, "manufacturer"),
            "categories": options_for(data, "category"),
            "battery_types": options_for(data, "battery_type"),
        }
    )

@app.route("/tableau")
def tableau():
    # Set TABLEAU_DASHBOARD_URL to your published Tableau Public embed URL.
    tableau_url = os.environ.get("TABLEAU_DASHBOARD_URL", "").strip()
    return render_template("tableau.html", tableau_url=tableau_url)

@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    # Local development only. For deployment, use a production WSGI server.
    app.run(debug=True)
