"""EV Market Performance & Insights - Flask companion to the Tableau dashboard and story.

Mirrors the Tableau workbook "EV_Market_Performance_Market_Story":
  * EV Executive Dashboard : KPI strip, 7 charts, manufacturer / battery filters
  * EV Market Journey      : story (8 points) + Executive Overview (1 point)

Run:  python app.py      then open http://127.0.0.1:5000
"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request

BASE = Path(__file__).resolve().parent

# The Tableau KPIs (200 manufacturers, 17,980 EVs, 1.55B sales ...) match
# compressed_data-2.csv.gz exactly, so that is the default source.
# Override with:  EV_DATA_FILE=/path/to/file.csv.gz python app.py
DATA_CANDIDATES = [
    os.environ.get("EV_DATA_FILE"),
    BASE / "data" / "compressed_data-2.csv.gz",
    BASE / "compressed_data-2.csv.gz",
    BASE.parent / "compressed_data-2.csv.gz",
]

USE_COLS = [
    "manufacturer", "model_name", "model_year", "manufacturing_country",
    "market_region", "category", "battery_type", "battery_capacity_kwh",
    "range_km", "annual_sales_units", "customer_rating",
]
CAT_COLS = ["manufacturer", "manufacturing_country", "market_region", "category", "battery_type"]

# Query-string name -> dataframe column
FILTER_PARAMS = {
    "manufacturer": "manufacturer",
    "battery_type": "battery_type",
    "region": "market_region",
    "category": "category",
}

# Representative lon/lat for the bubble on the manufacturing map
COUNTRY_XY = {
    "India": (78.9, 22.5), "China": (104.0, 35.0), "Vietnam": (106.3, 16.0),
    "UK": (-2.5, 54.0), "USA": (-98.0, 39.0), "South Korea": (127.8, 36.4),
    "France": (2.5, 46.6), "Germany": (10.4, 51.1), "Sweden": (15.0, 62.0),
    "Japan": (138.5, 36.5),
}

TABLEAU_LINKS = {
    "dashboard": "https://public.tableau.com/app/profile/mausam.kumar.pandit/viz/EV_Market_Performance_Market_Story_2026-10-25/EVExecutiveDashboard",
    "story": "https://public.tableau.com/app/profile/mausam.kumar.pandit/viz/EV_Market_Performance_Green_Executive_Refresh/Story2",
    "github": "https://github.com/mausam9039208144-dev/EV-Market-Performance-Insights",
    "kaggle": "https://www.kaggle.com/datasets/tharishreddy22/electric-vehicle-market-classification-dataset",
}


# --------------------------------------------------------------------------- data
def find_data_file() -> Path:
    for c in DATA_CANDIDATES:
        if c and Path(c).is_file():
            return Path(c)
    raise FileNotFoundError(
        "Dataset not found. Put compressed_data-2.csv.gz in ./data/ or set EV_DATA_FILE."
    )


def load_data():
    path = find_data_file()
    raw = pd.read_csv(path)
    n_raw = len(raw)
    df = raw.drop_duplicates()              # source file is never modified
    info = {"file": path.name, "rows_in_file": n_raw, "duplicates_removed": n_raw - len(df)}
    df = df[USE_COLS].copy()
    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    df["model_year"] = df["model_year"].astype(int)
    info["rows"] = len(df)
    return df, info


DF, DATA_INFO = load_data()


# --------------------------------------------------------------------------- helpers
def compact(n: float) -> str:
    a = abs(n)
    if a >= 1e9:
        return f"{n / 1e9:.2f}B"
    if a >= 1e6:
        return f"{n / 1e6:.1f}M"
    if a >= 1e3:
        return f"{n / 1e3:.1f}K"
    return f"{n:,.0f}"


def clean(v):
    """JSON-safe float (NaN -> None)."""
    if v is None:
        return None
    v = float(v)
    return None if np.isnan(v) else v


def rows(series: pd.Series):
    return [{"name": str(k), "value": clean(v)} for k, v in series.items()]


class Filters:
    def __init__(self, args=None):
        args = args or {}
        getlist = args.getlist if hasattr(args, "getlist") else (lambda k: [])
        self.values = {col: getlist(param) for param, col in FILTER_PARAMS.items()}
        get = args.get if hasattr(args, "get") else (lambda k, d=None, type=None: d)
        self.year_min = get("year_min", type=int)
        self.year_max = get("year_max", type=int)

    @property
    def active(self) -> bool:
        return any(self.values.values()) or self.year_min is not None or self.year_max is not None

    def apply(self, df: pd.DataFrame, skip=()) -> pd.DataFrame:
        mask = np.ones(len(df), dtype=bool)
        for col, vals in self.values.items():
            if vals and col not in skip:
                mask &= df[col].isin(vals).to_numpy()
        if self.year_min is not None:
            mask &= (df["model_year"] >= self.year_min).to_numpy()
        if self.year_max is not None:
            mask &= (df["model_year"] <= self.year_max).to_numpy()
        return df[mask]


def sales_by(df, col):
    return df.groupby(col, observed=True)["annual_sales_units"].sum().sort_values(ascending=False)


def kpis(df: pd.DataFrame):
    """The six tiles of the dashboard KPI strip (always computed on all rows)."""
    mfrs = int(df["manufacturer"].nunique())
    models = int(df["model_name"].nunique())
    sales = float(df["annual_sales_units"].sum())
    rating = float(df["customer_rating"].mean())
    rng = float(df["range_km"].mean())
    batt = float(df["battery_capacity_kwh"].mean())
    return [
        {"key": "mfrs", "label": "Manufacturers", "value": mfrs, "display": f"{mfrs:,}"},
        {"key": "evs", "label": "EV models", "value": models, "display": f"{models:,}"},
        {"key": "sales", "label": "Annual sales", "value": sales, "display": compact(sales), "unit": "units"},
        {"key": "rating", "label": "Avg rating", "value": rating, "display": f"{rating:.2f}", "unit": "/ 5"},
        {"key": "range", "label": "Avg range", "value": rng, "display": f"{rng:.1f}", "unit": "km"},
        {"key": "battery", "label": "Avg battery", "value": batt, "display": f"{batt:.1f}", "unit": "kWh"},
    ]


@lru_cache(maxsize=1)
def kpis_all():
    return kpis(DF)


def rating_by_manufacturer(df):
    return df.groupby("manufacturer", observed=True)["customer_rating"].mean().dropna().sort_values(ascending=False)


def range_by_category(df):
    return df.groupby("category", observed=True)["range_km"].mean().dropna().sort_values(ascending=False)


def rating_axis(series: pd.Series):
    """Tableau fixes the rating axis to the data range; do the same so small gaps are visible."""
    if series.empty:
        return 0.0, 5.0
    lo = np.floor((series.min() - 0.01) * 100) / 100
    hi = np.ceil(series.max() * 100) / 100
    return float(lo), float(hi)


# --------------------------------------------------------------------------- dashboard
def build_dashboard(f: Filters, top_n: int):
    sub = f.apply(DF)
    sub_any_mfr = f.apply(DF, skip=("manufacturer",))   # Manufacturer Sales ignores its own selection
    selected = f.values["manufacturer"]

    mfr_sales = sales_by(sub_any_mfr, "manufacturer").head(top_n)

    ratings = rating_by_manufacturer(sub)
    lo, hi = rating_axis(ratings)
    shown = ratings if selected else ratings.head(top_n)

    battery = sales_by(sub, "battery_type")
    battery_total = battery.sum()
    battery_rows = [
        {**r, "share": (r["value"] / battery_total if battery_total else None)} for r in rows(battery)
    ]

    country = sales_by(sub, "manufacturing_country")
    country_rows = [
        {"name": str(k), "value": clean(v), "lon": COUNTRY_XY[str(k)][0], "lat": COUNTRY_XY[str(k)][1]}
        for k, v in country.items() if str(k) in COUNTRY_XY
    ]

    return {
        "scope": {"rows": int(len(sub)), "total": int(len(DF)), "active": f.active},
        "kpis": kpis_all(),
        "manufacturer_sales": rows(mfr_sales),
        "customer_rating": {"rows": rows(shown), "axis_min": lo, "axis_max": hi, "manufacturers": int(len(ratings))},
        "regional_sales": rows(sales_by(sub, "market_region")),
        "battery_sales": battery_rows,
        "category_sales": rows(sales_by(sub, "category")),
        "range_by_category": rows(range_by_category(sub)),
        "manufacturing_map": country_rows,
    }


# --------------------------------------------------------------------------- story
def pct(part, whole):
    return 100.0 * part / whole if whole else 0.0


def sentence_list(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


@lru_cache(maxsize=1)
def build_story():
    """Nine story points: the 8 points of 'EV Market Journey' + the 'Executive Overview' snapshot.
    All text is generated from the data, so it cannot drift from the numbers."""
    k = {x["key"]: x for x in kpis_all()}
    total = float(DF["annual_sales_units"].sum())

    mfr = sales_by(DF, "manufacturer")
    top_mfr, top_mfr_v = str(mfr.index[0]), float(mfr.iloc[0])
    top10_share = pct(mfr.head(10).sum(), total)

    bat = sales_by(DF, "battery_type")
    bat_names = [str(i) for i in bat.index]
    bat_share = [pct(v, total) for v in bat.values]

    reg = sales_by(DF, "market_region")
    reg_gap = pct(reg.iloc[0] - reg.iloc[-1], reg.iloc[0])

    cty = sales_by(DF, "manufacturing_country")

    cat = sales_by(DF, "category")
    bottom3 = [str(i) for i in cat.index[-3:]][::-1]
    bottom3_share = pct(cat.iloc[-3:].sum(), total)

    cat_rating = DF.groupby("category", observed=True)["customer_rating"].mean()
    mfr_rating = rating_by_manufacturer(DF)
    r_lo, r_hi = rating_axis(mfr_rating)

    rng = range_by_category(DF)

    def bars(rows_, fmt, color, **extra):
        return {"type": "bars", "rows": rows_, "format": fmt, "color": color, **extra}

    points = []

    points.append({
        "id": 1, "caption": "Market Snapshot", "sheet": "EV Executive Dashboard",
        "headline": f"{k['mfrs']['display']} manufacturers, {k['evs']['display']} EV models, {k['sales']['display']} units sold",
        "text": (f"The average EV in this dataset has a {k['rating']['display']} out of 5 customer rating, "
                 f"{k['range']['display']} km of range and a {k['battery']['display']} kWh battery. "
                 f"Every chart that follows is built from the same {DATA_INFO['rows']:,} records."),
        "chart": {"type": "kpis", "items": kpis_all()},
    })

    points.append({
        "id": 2, "caption": "Manufacturer Sales", "sheet": "Manufacturer Sales",
        "headline": f"{top_mfr} sells the most, but no manufacturer dominates",
        "text": (f"{top_mfr} leads with {compact(top_mfr_v)} units, {pct(top_mfr_v, total):.1f}% of all sales. "
                 f"The top 10 of {len(mfr)} manufacturers together hold {top10_share:.1f}%, "
                 f"so the market is highly fragmented."),
        "chart": bars(rows(mfr.head(15)), "compact", "teal", note=f"Top 15 of {len(mfr)} manufacturers by annual sales units"),
    })

    points.append({
        "id": 3, "caption": "Battery Technology Mix", "sheet": "EV Sales by Battery type",
        "headline": f"{bat_names[0]} and {bat_names[1]} account for {bat_share[0] + bat_share[1]:.0f}% of sales",
        "text": (f"{bat_names[2]} follows with {bat_share[2]:.1f}%. The two smallest chemistries, "
                 f"{bat_names[-2]} and {bat_names[-1]}, hold {bat_share[-2] + bat_share[-1]:.1f}% combined."),
        "chart": bars([{**r, "share": pct(r["value"], total) / 100} for r in rows(bat)], "compact", "blue", note="Annual sales units by battery type"),
    })

    even = "evenly" if reg_gap < 10 else "unevenly"
    points.append({
        "id": 4, "caption": "Regional Comparison", "sheet": "Regional Sales",
        "headline": (f"Sales are spread {even} across all {len(reg)} regions" if reg_gap < 10
                     else f"{reg.index[0]} leads the {len(reg)} regions"),
        "text": (f"{reg.index[0]} is highest at {compact(reg.iloc[0])} units and {reg.index[-1]} lowest at "
                 f"{compact(reg.iloc[-1])}, a gap of {reg_gap:.1f}% of the leader's volume."),
        "chart": bars(rows(reg), "compact", "green", note="Annual sales units by market region"),
    })

    points.append({
        "id": 5, "caption": "Manufacturing Footprint", "sheet": "Manufacturing Map",
        "headline": f"Vehicles are built in {len(cty)} countries, led by {cty.index[0]} and {cty.index[1]}",
        "text": (f"{cty.index[0]} accounts for {compact(cty.iloc[0])} units ({pct(cty.iloc[0], total):.1f}%) and "
                 f"{cty.index[1]} for {compact(cty.iloc[1])}. {cty.index[-1]} is lowest at {compact(cty.iloc[-1])}. "
                 f"Bubble size on the map is proportional to annual sales units."),
        "chart": {"type": "map", "rows": [
            {"name": str(k_), "value": clean(v), "lon": COUNTRY_XY[str(k_)][0], "lat": COUNTRY_XY[str(k_)][1]}
            for k_, v in cty.items() if str(k_) in COUNTRY_XY]},
    })

    points.append({
        "id": 6, "caption": "Annual Sales by Vehicle Type", "sheet": "Vehicle Sales by Category",
        "headline": f"{cat.index[0]} is the best-selling vehicle type",
        "text": (f"{cat.index[0]} sells {compact(cat.iloc[0])} units ({pct(cat.iloc[0], total):.1f}% of the total). "
                 f"At the other end, {sentence_list(bottom3)} together make up only {bottom3_share:.1f}%."),
        "chart": bars(rows(cat), "compact", "green", note="Annual sales units by vehicle category"),
    })

    spread_cat = float(cat_rating.max() - cat_rating.min())
    points.append({
        "id": 7, "caption": "Customer Ratings", "sheet": "Customer Rating",
        "headline": ("Customers rate every vehicle type almost the same" if spread_cat < 0.1
                     else "Customer ratings differ by vehicle type"),
        "text": (f"Average ratings by category run from {cat_rating.min():.2f} to {cat_rating.max():.2f} out of 5. "
                 f"By manufacturer they range from {mfr_rating.min():.2f} ({mfr_rating.index[-1]}) to "
                 f"{mfr_rating.max():.2f} ({mfr_rating.index[0]}). The chart axis starts at {r_lo:.2f} so these small gaps are visible."),
        "chart": bars(rows(mfr_rating.head(15)), "rating", "teal", axis_min=r_lo, axis_max=r_hi,
                      note=f"Top 15 of {len(mfr_rating)} manufacturers by average rating. Axis starts at {r_lo:.2f}."),
    })

    points.append({
        "id": 8, "caption": "Driving Range by Category", "sheet": "Range Category",
        "headline": f"{rng.index[0]} goes the furthest at {rng.iloc[0]:.0f} km",
        "text": (f"{rng.index[-1]} has the shortest average range at {rng.iloc[-1]:.0f} km. "
                 f"The longest-range categories are {sentence_list([str(i) for i in rng.index[:3]])}; "
                 f"the shortest are {sentence_list([str(i) for i in rng.index[-3:]][::-1])}."),
        "chart": bars(rows(rng), "km", "blue", note="Average range (km) by vehicle category"),
    })

    points.append({
        "id": 9, "caption": "Executive Summary", "sheet": "Executive Overview",
        "headline": "What the dashboard shows",
        "text": "",
        "chart": {"type": "summary", "items": [
            f"The market is fragmented: the leading manufacturer holds {pct(top_mfr_v, total):.1f}% of sales and the top 10 hold {top10_share:.1f}%.",
            f"{bat_names[0]} and {bat_names[1]} batteries make up {bat_share[0] + bat_share[1]:.0f}% of sales.",
            f"Regions are close: only {reg_gap:.1f}% separates {reg.index[0]} from {reg.index[-1]}.",
            f"{cat.index[0]} is the best-selling vehicle type; {rng.index[0]} has the longest average range ({rng.iloc[0]:.0f} km).",
            f"Customer ratings are high and uniform ({k['rating']['display']} out of 5 on average).",
        ], "caveat": (
            "Sales and CO₂ figures are totals of dataset fields. Confirm what each row represents "
            "before presenting them as market-wide numbers."
        )},
    })
    return points


# --------------------------------------------------------------------------- app
app = Flask(__name__)
app.json.sort_keys = False


@lru_cache(maxsize=1)
def filter_options():
    return {
        "manufacturers": sorted(str(x) for x in DF["manufacturer"].cat.categories),
        "battery_types": sorted(str(x) for x in DF["battery_type"].cat.categories),
        "regions": sorted(str(x) for x in DF["market_region"].cat.categories),
        "categories": sorted(str(x) for x in DF["category"].cat.categories),
        "year_min": int(DF["model_year"].min()),
        "year_max": int(DF["model_year"].max()),
    }


@app.context_processor
def inject_globals():
    return {"links": TABLEAU_LINKS, "data_info": DATA_INFO}


@app.route("/")
def dashboard():
    return render_template("dashboard.html", page="dashboard")


@app.route("/story")
def story():
    return render_template("story.html", page="story")


@app.route("/about")
def about():
    return render_template("about.html", page="about", opts=filter_options(), kpis=kpis_all())


@app.route("/api/meta")
def api_meta():
    return jsonify(filter_options())


@app.route("/api/dashboard")
def api_dashboard():
    top_n = request.args.get("top_n", default=15, type=int)
    if top_n not in (10, 15, 25, 50):
        top_n = 15
    f = Filters(request.args)
    if f.year_min is not None and f.year_max is not None and f.year_min > f.year_max:
        abort(400, "year_min must not be greater than year_max")
    return jsonify(build_dashboard(f, top_n))


@app.route("/api/story")
def api_story():
    return jsonify(build_story())


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, **DATA_INFO})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)