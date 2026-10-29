"""EV Market Performance & Insights - single-file Flask website (code, pages, CSS and JS in this one file).

Companion to the Tableau dashboard and story. Needs: pip install flask pandas
Put compressed_data-2.csv.gz next to this file (or set EV_DATA_FILE).

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
from flask import Flask, Response, abort, jsonify, render_template, request
from jinja2 import DictLoader

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
        if args is None:
            args = {}
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
# --------------------------------------------------------------------------- embedded templates
TEMPLATES = {
    "base.html": r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{% block title %}EV Market Performance{% endblock %}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700;800&family=Source+Sans+3:wght@400;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body data-page="{{ page }}">
  <a class="skip" href="#main">Skip to content</a>
  <header class="top">
    <a class="brand" href="{{ url_for('dashboard') }}">EV Market Performance</a>
    <nav aria-label="Main">
      <a href="{{ url_for('dashboard') }}" {% if page == 'dashboard' %}aria-current="page"{% endif %}>Dashboard</a>
      <a href="{{ url_for('story') }}" {% if page == 'story' %}aria-current="page"{% endif %}>Market story</a>
      <a href="{{ url_for('about') }}" {% if page == 'about' %}aria-current="page"{% endif %}>About the data</a>
      <a href="{{ links.dashboard }}" target="_blank" rel="noopener">Tableau Public</a>
    </nav>
  </header>
  <main id="main">{% block content %}{% endblock %}</main>
  <footer class="foot">
    <span>{{ '{:,}'.format(data_info.rows) }} records from {{ data_info.file }}</span>
    <a href="{{ links.github }}" target="_blank" rel="noopener">Source on GitHub</a>
  </footer>
  <script src="{{ url_for('static', filename='js/main.js') }}" defer></script>
</body>
</html>
""",
    "dashboard.html": r"""{% extends "base.html" %}
{% block title %}Dashboard | EV Market Performance{% endblock %}
{% block content %}
<section class="hero">
  <h1>Who sells the most electric vehicles, and where?</h1>
  <div id="kpis" class="kpis" aria-live="polite"></div>
</section>
<div class="layout">
  <form id="filters" class="filters" aria-label="Filters">
    <h2>Filters</h2>
    <label>Manufacturer<select name="manufacturer" multiple size="6"></select></label>
    <label>Battery type<select name="battery_type" multiple size="4"></select></label>
    <label>Region<select name="region" multiple size="4"></select></label>
    <label>Vehicle category<select name="category" multiple size="5"></select></label>
    <div class="pair">
      <label>From year<select name="year_min"></select></label>
      <label>To year<select name="year_max"></select></label>
    </div>
    <label>Show top<select name="top_n"><option>10</option><option selected>15</option><option>25</option><option>50</option></select></label>
    <button type="reset" id="reset">Clear filters</button>
    <p class="hint">Hold Ctrl (Cmd on Mac) to pick several values.</p>
  </form>
  <div class="grid">
    <p id="scope" class="scope" aria-live="polite"></p>
    <article class="panel wide"><h2>Manufacturer sales</h2><div id="c-mfr" class="bars"></div></article>
    <article class="panel"><h2>Sales by battery type</h2><div id="c-battery" class="bars"></div></article>
    <article class="panel"><h2>Sales by region</h2><div id="c-region" class="bars"></div></article>
    <article class="panel"><h2>Sales by vehicle category</h2><div id="c-category" class="bars"></div></article>
    <article class="panel"><h2>Average range by category</h2><div id="c-range" class="bars"></div></article>
    <article class="panel wide"><h2>Customer rating by manufacturer</h2><div id="c-rating" class="bars"></div></article>
    <article class="panel wide"><h2>Where vehicles are made</h2><div id="c-map" class="map"></div></article>
  </div>
</div>
{% endblock %}
""",
    "story.html": r"""{% extends "base.html" %}
{% block title %}Market story | EV Market Performance{% endblock %}
{% block content %}
<section class="story">
  <ol id="steps" class="steps" aria-label="Story points"></ol>
  <article class="stage" aria-live="polite">
    <h1 id="s-head"></h1>
    <p id="s-text" class="lead"></p>
    <div id="s-chart"></div>
    <p id="s-note" class="hint"></p>
    <div class="nav">
      <button id="prev" type="button">Previous</button>
      <span id="s-count"></span>
      <button id="next" type="button">Next</button>
    </div>
  </article>
</section>
{% endblock %}
""",
    "about.html": r"""{% extends "base.html" %}
{% block title %}About the data | EV Market Performance{% endblock %}
{% block content %}
<section class="doc">
  <h1>About the data</h1>
  <p>The analysis uses the <a href="{{ links.kaggle }}" target="_blank" rel="noopener">Electric Vehicle Market Classification dataset</a> from Kaggle. This app reads <strong>{{ data_info.file }}</strong>: {{ '{:,}'.format(data_info.rows_in_file) }} rows, of which {{ '{:,}'.format(data_info.duplicates_removed) }} duplicate{{ '' if data_info.duplicates_removed == 1 else 's' }} {{ 'is' if data_info.duplicates_removed == 1 else 'are' }} removed in memory. The file itself is never changed.</p>
  <h2>Headline figures</h2>
  <dl class="facts">
    {% for k in kpis %}<div><dt>{{ k.label }}</dt><dd>{{ k.display }} {{ k.unit or '' }}</dd></div>{% endfor %}
  </dl>
  <h2>How to read the numbers</h2>
  <ul>
    <li>Missing values are skipped in averages and sums, not counted as zero.</li>
    <li>Sales are totals of the dataset's annual sales field. Confirm what each row represents before quoting them as market-wide sales.</li>
    <li>The six headline figures always cover the full dataset. The charts respond to the filters.</li>
    <li>Model years run from {{ opts.year_min }} to {{ opts.year_max }}.</li>
  </ul>
  <h2>Published work</h2>
  <ul>
    <li><a href="{{ links.dashboard }}" target="_blank" rel="noopener">Tableau Public dashboard</a></li>
    <li><a href="{{ links.story }}" target="_blank" rel="noopener">Tableau Public story</a></li>
    <li><a href="{{ links.github }}" target="_blank" rel="noopener">Source code</a></li>
  </ul>
</section>
{% endblock %}
""",
}

# --------------------------------------------------------------------------- embedded static files
ASSETS = {
    "css/style.css": (r""":root{--ink:#10231d;--forest:#0e3b2e;--mint:#2fbf8a;--sun:#f0b429;--paper:#f3f7f5;--card:#fff;--line:#d3dfd9;--muted:#52665e;--teal:#1a8f7a;--blue:#3a6ea5;--green:#5aa02c;--r:6px}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:400 1rem/1.55 "Source Sans 3",system-ui,sans-serif}
h1,h2,.brand,.kpi strong{font-family:"Bricolage Grotesque","Source Sans 3",sans-serif;line-height:1.15}
a{color:var(--teal)}
:focus-visible{outline:3px solid var(--sun);outline-offset:2px}
.skip{position:absolute;left:-999px}.skip:focus{left:8px;top:8px;background:#fff;padding:.5rem;z-index:9}
.top{display:flex;flex-wrap:wrap;gap:.75rem 2rem;align-items:center;justify-content:space-between;padding:.9rem clamp(1rem,4vw,3rem);background:var(--forest)}
.brand{color:#fff;font-weight:800;font-size:1.15rem;text-decoration:none}
.top nav{display:flex;flex-wrap:wrap;gap:1.25rem}
.top nav a{color:#cfe7dc;text-decoration:none;padding:.2rem 0;border-bottom:2px solid transparent}
.top nav a:hover,.top nav a[aria-current]{color:#fff;border-color:var(--mint)}
.hero{background:var(--forest);color:#fff;padding:1.5rem clamp(1rem,4vw,3rem) 2.25rem}
.hero h1{margin:0 0 1.5rem;font-size:clamp(1.7rem,4vw,2.8rem);font-weight:800;max-width:24ch}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:#2b5a4b;border-radius:var(--r);overflow:hidden}
.kpi{background:var(--forest);padding:1rem 1.1rem;display:flex;flex-direction:column;gap:.15rem}
.kpi strong{font-size:clamp(1.6rem,3vw,2.3rem);font-weight:800;color:var(--mint)}
.kpi span{font-size:.9rem;color:#cfe7dc}
.kpis.light{background:var(--line)}.kpis.light .kpi{background:var(--card)}.kpis.light .kpi strong{color:var(--teal)}.kpis.light .kpi span{color:var(--muted)}
.layout{display:grid;grid-template-columns:260px 1fr;gap:1.5rem;padding:1.5rem clamp(1rem,4vw,3rem);align-items:start}
.filters{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:1rem;position:sticky;top:1rem;display:grid;gap:.8rem}
.filters h2,.panel h2{margin:0;font-size:1.05rem}
label{display:grid;gap:.25rem;font-weight:600;font-size:.92rem}
select{font:inherit;padding:.3rem;border:1px solid var(--line);border-radius:4px;background:#fff;width:100%}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:.5rem}
button{font:inherit;font-weight:600;padding:.55rem 1rem;border-radius:var(--r);border:1px solid var(--forest);background:var(--forest);color:#fff;cursor:pointer}
button[type=reset]{background:#fff;color:var(--forest)}
button:disabled{opacity:.4;cursor:not-allowed}
.hint{color:var(--muted);font-size:.85rem;margin:.5rem 0 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:1rem}
.scope{grid-column:1/-1;margin:0;color:var(--muted)}
.panel{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:1.1rem;min-width:0}
.panel.wide{grid-column:1/-1}
.panel h2{margin-bottom:.8rem}
.bar-row{display:grid;grid-template-columns:minmax(90px,28%) 1fr 4.5rem;gap:.6rem;align-items:center;padding:.18rem 0;font-size:.92rem}
.bar-label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar-track{background:#e8efeb;height:.8rem;border-radius:3px;overflow:hidden}
.bar-fill{display:block;height:100%;border-radius:3px;transition:width .3s}
.bar-fill.teal{background:var(--teal)}.bar-fill.blue{background:var(--blue)}.bar-fill.green{background:var(--green)}
.bar-val{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
.empty{color:var(--muted);margin:0}
.map svg{width:100%;height:auto;background:#eaf2ee;border-radius:var(--r)}
.graticule line{stroke:#c7d6ce;stroke-width:.4}
.map circle{fill:var(--teal);fill-opacity:.55;stroke:var(--forest);stroke-width:.6}
.map text{font-size:5px;fill:var(--ink)}
.story{display:grid;grid-template-columns:240px 1fr;gap:1.5rem;padding:1.5rem clamp(1rem,4vw,3rem)}
.steps{list-style:none;margin:0;padding:0;display:grid;gap:.35rem;align-content:start}
.steps button{width:100%;text-align:left;background:#fff;color:var(--ink);border-color:var(--line)}
.steps button[aria-current]{background:var(--forest);color:#fff;border-color:var(--forest)}
.stage{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:clamp(1rem,3vw,2rem);min-width:0}
.stage h1{margin:0 0 .6rem;font-size:clamp(1.5rem,3vw,2.2rem)}
.lead{max-width:70ch;font-size:1.08rem}
.nav{display:flex;align-items:center;justify-content:space-between;margin-top:1.25rem}
.summary{padding-left:1.1rem;max-width:70ch}.summary li{margin:.4rem 0}
.doc{max-width:760px;margin:0 auto;padding:2rem 1rem 3rem}
.doc h1{font-size:2.2rem;margin-top:0}.doc h2{margin-top:2rem}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:.75rem}
.facts div{background:#fff;border:1px solid var(--line);border-radius:var(--r);padding:.7rem 1rem}
.facts dt{color:var(--muted);font-size:.88rem}.facts dd{margin:0;font-weight:700;font-size:1.2rem}
.foot{display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;padding:1.2rem clamp(1rem,4vw,3rem);border-top:1px solid var(--line);color:var(--muted);font-size:.9rem}
@media(max-width:860px){.layout,.story{grid-template-columns:1fr}.filters{position:static}.steps{grid-auto-flow:column;overflow-x:auto;grid-auto-columns:max-content}}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
""", "text/css"),
    "js/main.js": (r""""use strict";
const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = {
  compact: (n) => (n == null ? "n/a" : new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 2 }).format(n)),
  rating: (n) => (n == null ? "n/a" : n.toFixed(2)),
  km: (n) => (n == null ? "n/a" : `${Math.round(n)} km`),
};
const EMPTY = '<p class="empty">No records match these filters. Clear a filter to see data.</p>';

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} returned ${res.status}`);
  return res.json();
}

function renderBars(el, rows, { format = "compact", color = "teal", axisMin = 0, axisMax, note } = {}) {
  if (!rows.length) { el.innerHTML = EMPTY; return; }
  const max = axisMax ?? Math.max(...rows.map((r) => r.value || 0), 1);
  const span = max - axisMin || 1;
  el.innerHTML = rows.map((r) => {
    const w = r.value == null ? 0 : Math.max(1, ((r.value - axisMin) / span) * 100);
    return `<div class="bar-row"><span class="bar-label" title="${esc(r.name)}">${esc(r.name)}</span>` +
      `<span class="bar-track"><span class="bar-fill ${color}" style="width:${w}%"></span></span>` +
      `<span class="bar-val">${fmt[format](r.value)}</span></div>`;
  }).join("") + (note ? `<p class="hint">${esc(note)}</p>` : "");
}

function renderMap(el, rows) {
  if (!rows.length) { el.innerHTML = EMPTY; return; }
  const max = Math.max(...rows.map((r) => r.value || 0), 1);
  const grid = [-120, -60, 0, 60, 120].map((x) => `<line x1="${x + 180}" y1="0" x2="${x + 180}" y2="180"/>`).join("") +
    [-60, -30, 0, 30, 60].map((y) => `<line x1="0" y1="${90 - y}" x2="360" y2="${90 - y}"/>`).join("");
  const dots = rows.map((r) => {
    const rad = 3 + 13 * Math.sqrt((r.value || 0) / max);
    return `<g><circle cx="${r.lon + 180}" cy="${90 - r.lat}" r="${rad.toFixed(1)}"><title>${esc(r.name)}: ${fmt.compact(r.value)} units</title></circle>` +
      `<text x="${r.lon + 180}" y="${90 - r.lat + rad + 7}" text-anchor="middle">${esc(r.name)}</text></g>`;
  }).join("");
  el.innerHTML = `<svg viewBox="0 0 360 180" role="img" aria-label="Bubble map of annual sales by manufacturing country"><g class="graticule">${grid}</g>${dots}</svg>` +
    '<p class="hint">Bubble area is proportional to annual sales units.</p>';
}

/* ------------------------------------------------------------ dashboard */
function fillSelect(sel, values, selected) {
  sel.innerHTML = values.map((v) => `<option${String(v) === String(selected) ? " selected" : ""}>${esc(v)}</option>`).join("");
}

function queryString(form) {
  const p = new URLSearchParams();
  for (const sel of form.querySelectorAll("select")) {
    for (const o of sel.selectedOptions) if (o.value !== "") p.append(sel.name, o.value);
  }
  return p.toString();
}

async function initDashboard() {
  const form = $("#filters");
  const meta = await getJSON("/api/meta");
  fillSelect(form.manufacturer, meta.manufacturers);
  fillSelect(form.battery_type, meta.battery_types);
  fillSelect(form.region, meta.regions);
  fillSelect(form.category, meta.categories);
  const years = Array.from({ length: meta.year_max - meta.year_min + 1 }, (_, i) => meta.year_min + i);
  const setYears = () => { fillSelect(form.year_min, years, meta.year_min); fillSelect(form.year_max, years, meta.year_max); };
  setYears();

  let timer;
  const refresh = async () => {
    try {
      const d = await getJSON(`/api/dashboard?${queryString(form)}`);
      $("#kpis").innerHTML = d.kpis.map((k) =>
        `<div class="kpi"><strong>${esc(k.display)}</strong><span>${esc(k.label)}${k.unit ? ` (${esc(k.unit)})` : ""}</span></div>`).join("");
      $("#scope").textContent = `Showing ${d.scope.rows.toLocaleString()} of ${d.scope.total.toLocaleString()} records.`;
      renderBars($("#c-mfr"), d.manufacturer_sales, { color: "teal" });
      renderBars($("#c-battery"), d.battery_sales, { color: "blue" });
      renderBars($("#c-region"), d.regional_sales, { color: "green" });
      renderBars($("#c-category"), d.category_sales, { color: "green" });
      renderBars($("#c-range"), d.range_by_category, { format: "km", color: "blue" });
      renderBars($("#c-rating"), d.customer_rating.rows, {
        format: "rating", color: "teal", axisMin: d.customer_rating.axis_min, axisMax: d.customer_rating.axis_max,
        note: `Axis starts at ${d.customer_rating.axis_min.toFixed(2)} so small gaps are visible.`,
      });
      renderMap($("#c-map"), d.manufacturing_map);
    } catch (err) {
      $("#scope").textContent = "The data could not be loaded. Check that the server is running, then reload.";
      console.error(err);
    }
  };
  form.addEventListener("change", () => { clearTimeout(timer); timer = setTimeout(refresh, 150); });
  form.addEventListener("reset", () => setTimeout(() => { setYears(); refresh(); }, 0));
  refresh();
}

/* ---------------------------------------------------------------- story */
function renderStoryChart(el, c) {
  if (c.type === "kpis") {
    el.innerHTML = `<div class="kpis light">${c.items.map((k) => `<div class="kpi"><strong>${esc(k.display)}</strong><span>${esc(k.label)}</span></div>`).join("")}</div>`;
  } else if (c.type === "map") {
    renderMap(el, c.rows);
  } else if (c.type === "summary") {
    el.innerHTML = `<ul class="summary">${c.items.map((t) => `<li>${esc(t)}</li>`).join("")}</ul><p class="hint">${esc(c.caveat)}</p>`;
  } else {
    el.innerHTML = '<div class="bars"></div>';
    renderBars($(".bars", el), c.rows, { format: c.format, color: c.color, axisMin: c.axis_min ?? 0, axisMax: c.axis_max, note: c.note });
  }
}

async function initStory() {
  const points = await getJSON("/api/story");
  let i = 0;
  const steps = $("#steps");
  steps.innerHTML = points.map((p, n) => `<li><button type="button" data-i="${n}">${esc(p.caption)}</button></li>`).join("");
  const show = (n) => {
    i = Math.min(Math.max(n, 0), points.length - 1);
    const p = points[i];
    $("#s-head").textContent = p.headline;
    $("#s-text").textContent = p.text;
    $("#s-note").textContent = `Tableau sheet: ${p.sheet}`;
    $("#s-count").textContent = `${i + 1} of ${points.length}`;
    $("#prev").disabled = i === 0;
    $("#next").disabled = i === points.length - 1;
    steps.querySelectorAll("button").forEach((b, k) => b.toggleAttribute("aria-current", k === i));
    renderStoryChart($("#s-chart"), p.chart);
  };
  steps.addEventListener("click", (e) => { const b = e.target.closest("button[data-i]"); if (b) show(+b.dataset.i); });
  $("#prev").addEventListener("click", () => show(i - 1));
  $("#next").addEventListener("click", () => show(i + 1));
  document.addEventListener("keydown", (e) => { if (e.key === "ArrowLeft") show(i - 1); if (e.key === "ArrowRight") show(i + 1); });
  show(0);
}

const page = document.body.dataset.page;
if (page === "dashboard") initDashboard();
if (page === "story") initStory();
""", "text/javascript"),
}

app = Flask(__name__, static_folder=None)
app.jinja_loader = DictLoader(TEMPLATES)
app.json.sort_keys = False


@app.route("/static/<path:filename>", endpoint="static")
def static_files(filename):
    if filename not in ASSETS:
        abort(404)
    body, mime = ASSETS[filename]
    return Response(body, mimetype=mime)


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