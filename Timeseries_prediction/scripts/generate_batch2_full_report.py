#!/usr/bin/env python3

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "Data_Batch_2"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def newest_parent_containing(root: Path, filename: str) -> Path:
    """Find the newest directory containing a required result file."""
    candidates = list(root.rglob(filename))
    if not candidates:
        raise FileNotFoundError(f"Cannot find {filename} under {root}")

    newest_file = max(candidates, key=lambda p: p.stat().st_mtime)
    return newest_file.parent


def find_latest_error_metric_run(root: Path) -> Path | None:
    base = root / "error_metric"
    if not base.exists():
        return None

    runs = [
        p for p in base.iterdir()
        if p.is_dir() and list(p.glob("metrics/*/metrics.json"))
    ]
    if not runs:
        return None

    return max(runs, key=lambda p: p.stat().st_mtime)


def flatten_json(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested JSON into key-value pairs."""
    result: dict[str, Any] = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_json(value, new_key))
    elif isinstance(obj, list):
        if all(not isinstance(x, (dict, list)) for x in obj):
            result[prefix] = ", ".join(map(str, obj))
        else:
            result[prefix] = json.dumps(obj, ensure_ascii=False)
    else:
        result[prefix] = obj

    return result


def prettify_column(name: str) -> str:
    replacements = {
        "model_name": "Model",
        "case_id": "Case",
        "n_cases": "No. Cases",
        "MAE_V": "Voltage MAE",
        "RMSE_V": "Voltage RMSE",
        "R2_V": "Voltage R²",
        "MaxError_V": "Voltage Max Error",
        "MAE_T": "Temperature MAE",
        "RMSE_T": "Temperature RMSE",
        "R2_T": "Temperature R²",
        "MaxError_T": "Temperature Max Error",
        "voltage_end_mae": "End-Voltage MAE",
        "T_peak_MAE": "Peak-Temperature MAE",
    }
    return replacements.get(name, name.replace("_", " ").title())


def format_value(value: Any) -> str:
    if pd.isna(value):
        return "—"

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, int):
        return f"{value:,}"

    if isinstance(value, float):
        abs_value = abs(value)

        if abs_value == 0:
            return "0"
        if abs_value < 1e-4:
            return f"{value:.3e}"
        if abs_value < 1:
            return f"{value:.6f}"
        return f"{value:.4f}"

    return str(value)


def formatted_df(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output.columns = [prettify_column(str(c)) for c in output.columns]

    for col in output.columns:
        output[col] = output[col].map(format_value)

    return output


def df_to_markdown(df: pd.DataFrame) -> str:
    """Markdown conversion without requiring tabulate."""
    if df.empty:
        return "_No data available._"

    df = formatted_df(df)
    headers = [str(c).replace("|", r"\|") for c in df.columns]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in df.iterrows():
        values = [
            str(v).replace("|", r"\|").replace("\n", " ")
            for v in row.tolist()
        ]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def df_to_html(df: pd.DataFrame, table_class: str = "report-table") -> str:
    if df.empty:
        return "<p><em>No data available.</em></p>"

    return formatted_df(df).to_html(
        index=False,
        escape=True,
        classes=table_class,
        border=0,
    )


def metric_direction(column: str) -> str:
    name = column.lower()

    maximize_terms = ["r2", "r²", "accuracy", "auc", "f1"]
    if any(term in name for term in maximize_terms):
        return "max"

    return "min"


def build_best_models(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    excluded = {
        "model_name",
        "n_cases",
        "case_id",
        "n_samples",
        "seed",
        "fold",
    }

    for column in df.columns:
        if column in excluded:
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")
        if not numeric.notna().any():
            continue

        direction = metric_direction(column)
        idx = numeric.idxmax() if direction == "max" else numeric.idxmin()

        rows.append(
            {
                "Metric": prettify_column(column),
                "Best Model": df.loc[idx, "model_name"]
                if "model_name" in df.columns else "—",
                "Best Value": numeric.loc[idx],
                "Selection": "Higher is better"
                if direction == "max" else "Lower is better",
            }
        )

    return pd.DataFrame(rows)


def build_ranking(
    df: pd.DataFrame,
    metric: str,
    label: str,
) -> pd.DataFrame:
    if metric not in df.columns or "model_name" not in df.columns:
        return pd.DataFrame()

    work = df[["model_name", metric]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna()

    ascending = metric_direction(metric) == "min"
    work = work.sort_values(metric, ascending=ascending).reset_index(drop=True)
    work.insert(0, "Rank", range(1, len(work) + 1))
    work = work.rename(
        columns={
            "model_name": "Model",
            metric: label,
        }
    )
    return work


def read_error_metric_results(run_dir: Path | None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if run_dir is None:
        return pd.DataFrame(), {}

    summary_rows = []
    detailed_tables: dict[str, pd.DataFrame] = {}

    for metrics_file in sorted(run_dir.glob("metrics/*/metrics.json")):
        model_name = metrics_file.parent.name

        with metrics_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        flat = flatten_json(payload)
        details = pd.DataFrame(
            {
                "Metric": list(flat.keys()),
                "Value": list(flat.values()),
            }
        )
        detailed_tables[model_name] = details

        row: dict[str, Any] = {"model_name": model_name}

        for key, value in flat.items():
            lower = key.lower()

            if not isinstance(value, (int, float)):
                continue

            if "overall" in lower and "rmse" in lower:
                row["Overall RMSE"] = value
            elif (
                ("overall" in lower and "r2" in lower)
                or ("overall" in lower and "r²" in lower)
            ):
                row["Overall R²"] = value
            elif lower.endswith("reload_ok"):
                row["Reload OK"] = value
            elif "rmse_voltage" in lower:
                row["Voltage RMSE"] = value
            elif "rmse_temperature" in lower:
                row["Temperature RMSE"] = value

        summary_rows.append(row)

    return pd.DataFrame(summary_rows), detailed_tables


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ---------------------------------------------------------------------
# Locate latest completed runs
# ---------------------------------------------------------------------

ts_run = newest_parent_containing(
    OUTPUT_ROOT,
    "metrics_by_model.csv",
)
em_run = find_latest_error_metric_run(OUTPUT_ROOT)

metrics_by_model = pd.read_csv(ts_run / "metrics_by_model.csv")
metrics_summary = read_optional_csv(ts_run / "metrics_summary.csv")
metrics_by_target = read_optional_csv(ts_run / "metrics_by_target.csv")

best_models = build_best_models(metrics_by_model)

rankings: dict[str, pd.DataFrame] = {}

ranking_specs = [
    ("MAE_V", "Voltage MAE Ranking"),
    ("RMSE_V", "Voltage RMSE Ranking"),
    ("R2_V", "Voltage R² Ranking"),
    ("voltage_end_mae", "End-Voltage MAE Ranking"),
    ("MAE_T", "Temperature MAE Ranking"),
    ("RMSE_T", "Temperature RMSE Ranking"),
    ("R2_T", "Temperature R² Ranking"),
    ("MaxError_T", "Temperature Max-Error Ranking"),
]

for metric, title in ranking_specs:
    ranking = build_ranking(
        metrics_by_model,
        metric,
        prettify_column(metric),
    )
    if not ranking.empty:
        rankings[title] = ranking

error_summary, error_details = read_error_metric_results(em_run)

existing_summary_md = ""
summary_path = ts_run / "experiment_summary.md"
if summary_path.exists():
    existing_summary_md = summary_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
safe_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

md_path = REPORT_DIR / f"Batch2_full_report_{safe_timestamp}.md"
html_path = REPORT_DIR / f"Batch2_full_report_{safe_timestamp}.html"

latest_md = REPORT_DIR / "Batch2_full_report_latest.md"
latest_html = REPORT_DIR / "Batch2_full_report_latest.html"


# ---------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------

md: list[str] = []

md.append("# Data Batch 2 — Full Experiment Report")
md.append("")
md.append(f"**Generated:** {timestamp}")
md.append("")
md.append(f"**Time-series run:** `{ts_run.relative_to(ROOT)}`")
md.append("")
md.append(
    f"**Error-metric run:** "
    f"`{em_run.relative_to(ROOT) if em_run else 'Not found'}`"
)
md.append("")

md.append("## 1. Experiment Overview")
md.append("")
md.append(
    f"- Time-series models: **{metrics_by_model['model_name'].nunique()}**"
    if "model_name" in metrics_by_model.columns
    else "- Time-series models: unavailable"
)
md.append(
    f"- Time-series cases: **{int(metrics_by_model['n_cases'].max())}**"
    if "n_cases" in metrics_by_model.columns
    else "- Time-series cases: unavailable"
)
md.append(f"- Detailed time-series runs: **{len(metrics_summary)}**")
md.append(f"- Error-metric models: **{len(error_summary)}**")
md.append("")

md.append("## 2. Best Time-Series Model by Metric")
md.append("")
md.append(df_to_markdown(best_models))
md.append("")

md.append("## 3. Average Time-Series Results Across All Cases")
md.append("")
md.append(df_to_markdown(metrics_by_model))
md.append("")

md.append("## 4. Model Rankings")
md.append("")

for title, table in rankings.items():
    md.append(f"### {title}")
    md.append("")
    md.append(df_to_markdown(table))
    md.append("")

if not metrics_by_target.empty:
    md.append("## 5. Results by Prediction Target")
    md.append("")
    md.append(df_to_markdown(metrics_by_target))
    md.append("")

md.append("## 6. Detailed Results for All Case–Model Runs")
md.append("")
md.append(
    f"This table contains **{len(metrics_summary)}** evaluated runs."
)
md.append("")
md.append(df_to_markdown(metrics_summary))
md.append("")

md.append("## 7. Error-Metric Prediction Results")
md.append("")

if not error_summary.empty:
    md.append(df_to_markdown(error_summary))
else:
    md.append("_No Error-Metric summary was found._")

md.append("")

for model_name, table in error_details.items():
    md.append(f"### Error-Metric Details — {model_name}")
    md.append("")
    md.append(df_to_markdown(table))
    md.append("")

if existing_summary_md:
    md.append("## 8. Original Pipeline Summary")
    md.append("")
    md.append(existing_summary_md)
    md.append("")

md.append("## 9. Source Artifacts")
md.append("")
md.append(f"- `{ts_run.relative_to(ROOT) / 'metrics_by_model.csv'}`")
md.append(f"- `{ts_run.relative_to(ROOT) / 'metrics_summary.csv'}`")

if (ts_run / "metrics_by_target.csv").exists():
    md.append(
        f"- `{ts_run.relative_to(ROOT) / 'metrics_by_target.csv'}`"
    )

if em_run:
    for f in sorted(em_run.glob("metrics/*/metrics.json")):
        md.append(f"- `{f.relative_to(ROOT)}`")

md.append("")

md_content = "\n".join(md)
md_path.write_text(md_content, encoding="utf-8")
latest_md.write_text(md_content, encoding="utf-8")


# ---------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------

html_sections: list[str] = []

html_sections.append("""
<section class="hero">
  <p class="eyebrow">VINFAST BATTERY PREDICTION</p>
  <h1>Data Batch 2 — Full Experiment Report</h1>
  <p class="subtitle">
    Consolidated results for time-series and error-metric prediction tasks
  </p>
</section>
""")

html_sections.append(f"""
<section class="metadata">
  <div><strong>Generated</strong><br>{html.escape(timestamp)}</div>
  <div><strong>Time-Series Run</strong><br><code>{html.escape(str(ts_run.relative_to(ROOT)))}</code></div>
  <div><strong>Error-Metric Run</strong><br><code>{html.escape(str(em_run.relative_to(ROOT)) if em_run else "Not found")}</code></div>
</section>
""")

overview_cards = [
    (
        "Time-Series Models",
        str(metrics_by_model["model_name"].nunique())
        if "model_name" in metrics_by_model.columns else "—",
    ),
    (
        "Cases per Model",
        str(int(metrics_by_model["n_cases"].max()))
        if "n_cases" in metrics_by_model.columns else "—",
    ),
    ("Case–Model Runs", str(len(metrics_summary))),
    ("Error-Metric Models", str(len(error_summary))),
]

html_sections.append(
    '<section><h2>1. Experiment Overview</h2><div class="cards">'
    + "".join(
        f'<div class="card"><span>{html.escape(label)}</span>'
        f'<strong>{html.escape(value)}</strong></div>'
        for label, value in overview_cards
    )
    + "</div></section>"
)

html_sections.append(
    "<section><h2>2. Best Time-Series Model by Metric</h2>"
    + df_to_html(best_models)
    + "</section>"
)

html_sections.append(
    "<section><h2>3. Average Time-Series Results Across All Cases</h2>"
    + '<div class="table-wrap">'
    + df_to_html(metrics_by_model)
    + "</div></section>"
)

ranking_html = [
    "<section><h2>4. Model Rankings</h2>"
]

for title, table in rankings.items():
    ranking_html.append(
        f"<h3>{html.escape(title)}</h3>"
        f'<div class="table-wrap compact">'
        f"{df_to_html(table)}</div>"
    )

ranking_html.append("</section>")
html_sections.append("".join(ranking_html))

section_number = 5

if not metrics_by_target.empty:
    html_sections.append(
        f"<section><h2>{section_number}. Results by Prediction Target</h2>"
        '<div class="table-wrap">'
        + df_to_html(metrics_by_target)
        + "</div></section>"
    )
    section_number += 1

html_sections.append(
    f"<section><h2>{section_number}. Detailed Results for All "
    f"{len(metrics_summary)} Case–Model Runs</h2>"
    '<div class="table-wrap large-table">'
    + df_to_html(metrics_summary)
    + "</div></section>"
)
section_number += 1

error_html = [
    f"<section><h2>{section_number}. Error-Metric Prediction Results</h2>"
]

if not error_summary.empty:
    error_html.append(
        '<div class="table-wrap">'
        + df_to_html(error_summary)
        + "</div>"
    )
else:
    error_html.append("<p><em>No Error-Metric summary was found.</em></p>")

for model_name, table in error_details.items():
    error_html.append(
        f"<h3>{html.escape(model_name)}</h3>"
        '<div class="table-wrap compact">'
        + df_to_html(table)
        + "</div>"
    )

error_html.append("</section>")
html_sections.append("".join(error_html))

html_document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Data Batch 2 — Full Experiment Report</title>
<style>
  :root {{
    --bg: #f4f6f8;
    --surface: #ffffff;
    --text: #17202a;
    --muted: #667085;
    --border: #d9dee7;
    --header: #16263d;
    --accent: #2f6fed;
    --row-alt: #f8fafc;
  }}

  * {{
    box-sizing: border-box;
  }}

  body {{
    margin: 0;
    padding: 36px;
    background: var(--bg);
    color: var(--text);
    font-family: Inter, Arial, Helvetica, sans-serif;
    line-height: 1.5;
  }}

  main {{
    max-width: 1500px;
    margin: 0 auto;
  }}

  section {{
    margin: 22px 0;
    padding: 28px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    box-shadow: 0 5px 18px rgba(16, 24, 40, 0.05);
  }}

  .hero {{
    padding: 38px;
    background: var(--header);
    color: white;
  }}

  .hero h1 {{
    margin: 6px 0;
    font-size: 34px;
  }}

  .eyebrow {{
    margin: 0;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.14em;
    opacity: 0.75;
  }}

  .subtitle {{
    margin-bottom: 0;
    opacity: 0.82;
  }}

  h2 {{
    margin-top: 0;
    padding-bottom: 10px;
    border-bottom: 2px solid #e9edf3;
    font-size: 23px;
  }}

  h3 {{
    margin-top: 28px;
    color: #344054;
  }}

  .metadata {{
    display: grid;
    grid-template-columns: 0.8fr 2fr 2fr;
    gap: 16px;
  }}

  .metadata div {{
    overflow-wrap: anywhere;
  }}

  code {{
    padding: 2px 5px;
    background: #eef2f6;
    border-radius: 4px;
    font-size: 12px;
  }}

  .cards {{
    display: grid;
    grid-template-columns: repeat(4, minmax(150px, 1fr));
    gap: 16px;
  }}

  .card {{
    padding: 20px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: #f9fafb;
  }}

  .card span {{
    display: block;
    color: var(--muted);
    font-size: 13px;
  }}

  .card strong {{
    display: block;
    margin-top: 6px;
    font-size: 28px;
    color: var(--accent);
  }}

  .table-wrap {{
    width: 100%;
    overflow-x: auto;
  }}

  .report-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    white-space: nowrap;
  }}

  .report-table th {{
    position: sticky;
    top: 0;
    padding: 11px 12px;
    background: var(--header);
    color: white;
    text-align: left;
    border: 1px solid #33445d;
  }}

  .report-table td {{
    padding: 9px 12px;
    border: 1px solid var(--border);
  }}

  .report-table tbody tr:nth-child(even) {{
    background: var(--row-alt);
  }}

  .report-table tbody tr:hover {{
    background: #eaf1ff;
  }}

  .compact {{
    max-width: 850px;
  }}

  .large-table {{
    max-height: 850px;
    overflow: auto;
    border: 1px solid var(--border);
  }}

  @media print {{
    body {{
      padding: 0;
      background: white;
    }}

    section {{
      box-shadow: none;
      break-inside: avoid;
    }}

    .large-table {{
      max-height: none;
      overflow: visible;
    }}
  }}

  @media (max-width: 900px) {{
    body {{
      padding: 12px;
    }}

    .metadata,
    .cards {{
      grid-template-columns: 1fr;
    }}
  }}
</style>
</head>
<body>
<main>
{''.join(html_sections)}
</main>
</body>
</html>
"""

html_path.write_text(html_document, encoding="utf-8")
latest_html.write_text(html_document, encoding="utf-8")

print("=" * 72)
print("BATCH 2 FULL REPORT GENERATED")
print("=" * 72)
print(f"Time-series run : {ts_run}")
print(f"Error-metric run: {em_run or 'NOT FOUND'}")
print(f"Markdown report : {md_path}")
print(f"HTML report     : {html_path}")
print(f"Latest Markdown : {latest_md}")
print(f"Latest HTML     : {latest_html}")
print("=" * 72)
