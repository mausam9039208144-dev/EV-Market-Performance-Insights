# Electric Vehicle Market Performance & Insights

An interactive EV market analysis built with Tableau Public, a Python Flask app and a static project website. The dashboard turns the EV market dataset into KPIs, charts and a guided story, so manufacturers, regions and battery types can be compared quickly. The original dataset is never modified.

## Project links

- [Tableau Public Dashboard](https://public.tableau.com/app/profile/mausam.kumar.pandit/viz/EV_Market_Performance_Market_Story_2026-10-25/EVExecutiveDashboard)
- [Tableau Public Story](https://public.tableau.com/app/profile/mausam.kumar.pandit/viz/EV_Market_Performance_Green_Executive_Refresh/Story2)
- [Project Website](https://mausam9039208144-dev.github.io/EV-Market-Performance-Insights/)
- Dataset: [paste dataset link here]

## What this project covers

- KPI cards for sales, customer rating, range, battery capacity and CO₂ savings. [edit to match your dashboard]
- Interactive charts comparing manufacturers, battery types, regions, vehicle categories and manufacturing countries. [edit to match your dashboard]
- Filters for manufacturer, battery type, region, category and model year, with tooltips, labels and legends.
- A Tableau Story that walks through the main findings step by step.
- A Flask app (`app.py`) that processes the data behind the dashboard.
- A static project website (`index.html`) that presents the dashboard and story online.
- A compressed copy of the dataset, stored as `.csv.gz` files, so it fits within GitHub's upload limits.

## Data source

The analysis uses an electric vehicle market dataset, listed in the project links above. For this repository the data is stored in two compressed files:

- `compressed_data.csv.gz`: dataset part 1 ([add rows] rows, [add columns] columns)
- `compressed_data-2.csv.gz`: dataset part 2 ([add rows] rows, [add columns] columns)

To use them, extract the files first:

```
gunzip -k compressed_data.csv.gz
gunzip -k compressed_data-2.csv.gz
```

Duplicate rows are removed while the data is processed, and the source files are left unchanged. Missing values are skipped in sums and averages instead of being counted as zero. Sales and CO₂ figures are totals of the dataset fields, so confirm what each row represents before presenting them as market-wide numbers. Check the dataset's license terms before sharing the data publicly.

## Run locally

Use Python 3.11 or later. From the project folder, open a terminal:

```
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install flask pandas
python app.py
```

Open `http://127.0.0.1:5000` in your browser. Keep the terminal open while you use the app and press `Ctrl+C` to stop it. Chart libraries and fonts load from CDNs, so an internet connection is needed for the visuals.

## Tableau

The packaged workbook `EV_Market_Performance_Market_Story.twbx` holds the worksheets, executive dashboard, filters and story. Open it in Tableau Desktop, or view the published versions on Tableau Public:

- [Executive Dashboard](https://public.tableau.com/app/profile/mausam.kumar.pandit/viz/EV_Market_Performance_Market_Story_2026-10-25/EVExecutiveDashboard)
- [Green Executive Refresh, Story 2](https://public.tableau.com/app/profile/mausam.kumar.pandit/viz/EV_Market_Performance_Green_Executive_Refresh/Story2)

The extract inside a Tableau Public workbook is publicly visible. Flask is a separate companion app and does not replace the Tableau Public deliverable.

## Project website

`index.html` is a static page that embeds the published Tableau dashboard and story. To publish it with GitHub Pages, go to Settings, then Pages, then Build and deployment, choose Deploy from a branch, and select `main` and the root folder. GitHub Pages hosts only static files, so the Flask app still has to be run locally or on a Python-capable host.

## Repository map

```
EV_Market_Performance_Market_Story.twbx   Tableau workbook (dashboard, filters, story)
app.py                                    Flask routes and data processing
index.html                                Static project website
Website for EV Market Page                Website files for the EV market page
compressed_data.csv.gz                    Compressed dataset, part 1
compressed_data-2.csv.gz                  Compressed dataset, part 2
README.md                                 Project documentation
```

## Author

GitHub: [mausam9039208144-dev](https://github.com/mausam9039208144-dev)
