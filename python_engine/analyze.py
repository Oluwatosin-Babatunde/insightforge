import sys
import json
import io
import base64
import pathlib
import re
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------
# Helpers
# -----------------------

def fig_to_md_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"![chart](data:image/png;base64,{b64})"


def safe_read_table(file_path: str, max_rows: int = 200000):
    p = pathlib.Path(file_path)
    ext = p.suffix.lower()
    sampled = False

    if ext == ".csv":
        try:
            df = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")
        except Exception:
            df = pd.read_csv(file_path, encoding="latin1", on_bad_lines="skip")
        source_type = "csv"

    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
        source_type = "excel"

    elif ext == ".json":
        try:
            df = pd.read_json(file_path)
            if isinstance(df, pd.Series):
                df = df.to_frame().T
        except Exception:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            df = pd.json_normalize(raw)
        source_type = "json"

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if len(df) > max_rows:
        df = df.head(max_rows)
        sampled = True

    return df, sampled, source_type


def to_json_safe(obj):
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    return str(obj)


def df_preview_json_safe(df, n=5):
    preview = df.head(n).to_dict(orient="records")
    safe = []
    for row in preview:
        safe.append({k: to_json_safe(v) for k, v in row.items()})
    return safe


def md_table(rows):
    lines = ["| Metric | Value |", "|---|---|"]
    for k, v in rows:
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def clean_labels(values, max_len=18):
    out = []
    for v in values:
        s = str(v)
        out.append(s if len(s) <= max_len else s[: max_len - 1] + "…")
    return out


# -----------------------
# Column detection
# -----------------------

def detect_date_columns(df, threshold=0.6):
    date_cols = []
    for c in df.columns:
        if df[c].dtype == "object":
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() >= threshold:
                date_cols.append(c)
    return date_cols


def pick_numeric_focus_column(df):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return None

    prefs = ["amount", "total", "cost", "price", "revenue", "sales", "salary", "adr", "earn", "value"]
    for pref in prefs:
        for c in num_cols:
            if pref in c.lower():
                return c

    variances = df[num_cols].var(numeric_only=True).sort_values(ascending=False)
    return variances.index[0] if len(variances) else num_cols[0]


def pick_categorical_column(df):
    preferred = ["status", "dept", "department", "hotel", "category", "type", "region", "rateapplied"]
    for pref in preferred:
        for c in df.columns:
            if c.lower() == pref.lower():
                return c

    for c in df.columns:
        if df[c].dtype == "object":
            n = df[c].nunique(dropna=True)
            if 2 <= n <= 30:
                return c
    return None


def extract_column_mentions(df, intent: str):
    if not intent:
        return []
    i = intent.lower()
    hits = []
    for col in df.columns:
        if col.lower() in i:
            hits.append(col)
    return hits


def infer_group_column(df, intent: str):
    cols = extract_column_mentions(df, intent)
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    for c in cols:
        if c in cat_cols:
            return c
    return pick_categorical_column(df)


def infer_value_column(df, intent: str):
    cols = extract_column_mentions(df, intent)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    for c in cols:
        if c in num_cols:
            return c
    return pick_numeric_focus_column(df)


def infer_date_column(df, intent: str):
    date_cols = detect_date_columns(df)
    cols = extract_column_mentions(df, intent)
    for c in cols:
        if c in date_cols:
            return c
    return date_cols[0] if date_cols else None


# -----------------------
# Dataset Story
# -----------------------

def build_dataset_story(df):
    cols = [c.lower() for c in df.columns]
    story = []
    story.append("## 🧠 Dataset Story\n")

    if any("employee" in c for c in cols) and any("trip" in c for c in cols):
        story.append(
            "This dataset looks like a **corporate travel workflow log** tracking employee trips, approvals, and completion steps. "
            "It is useful for identifying **approval bottlenecks**, **processing delays**, and **workflow compliance issues**."
        )
    elif any("hotel" in c for c in cols) and any("adr" in c for c in cols):
        story.append(
            "This dataset resembles a **hotel booking + demand dataset**, which is ideal for exploring **cancellations**, **seasonality**, "
            "**lead time patterns**, and **revenue drivers**."
        )
    elif any("earn" in c for c in cols) or any("salary" in c for c in cols):
        story.append(
            "This dataset appears related to **earnings or financial transactions**, which makes it useful for **outlier detection**, "
            "**payment trends**, and **distribution analysis**."
        )
    else:
        story.append(
            "This dataset appears to be a general structured table. InsightForge will treat it as a universal dataset "
            "and focus on profiling, segmentation, trend detection, and anomaly signals."
        )

    story.append("")
    story.append("✅ **Best use cases for this dataset:**")
    story.append("- KPI monitoring and trend tracking")
    story.append("- Segment breakdowns (by categories, status, department, etc.)")
    story.append("- Detecting duplicates, missing-value risks, and suspicious anomalies")
    story.append("")
    return "\n".join(story)


# -----------------------
# Data Quality Score
# -----------------------

def compute_quality_score(df):
    rows, cols = df.shape
    if rows == 0 or cols == 0:
        return 0, ["Dataset is empty."]

    issues = []
    missing_cells = int(df.isna().sum().sum())
    missing_rate = missing_cells / (rows * cols)

    dupes = int(df.duplicated().sum())
    dup_rate = dupes / rows if rows else 0.0

    score = 100
    score -= min(55, missing_rate * 100 * 2.2)
    score -= min(30, dup_rate * 100 * 1.5)
    score = max(0, int(round(score)))

    if missing_rate > 0.05:
        issues.append(f"High missing rate detected ({missing_rate:.1%} of all cells).")
    if dup_rate > 0.05:
        issues.append(f"High duplicate rate detected ({dup_rate:.1%} of all rows).")
    if not issues:
        issues.append("No major data quality issues detected.")

    return score, issues


# -----------------------
# Executive Summary
# -----------------------

def build_executive_summary(df, intent, source_type, sampled):
    rows, cols = df.shape
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]

    missing_total = int(df.isna().sum().sum())
    dupes = int(df.duplicated().sum())

    text = (
        f"This dataset was loaded from a **{source_type.upper()}** source containing **{rows:,} rows** and **{cols:,} columns**. "
        f"It includes **{len(numeric_cols)} numeric fields** and **{len(cat_cols)} categorical/text fields**. "
        f"We detected **{missing_total:,} missing cells** and **{dupes:,} duplicate rows**."
    )

    if intent:
        text += f" The analysis was guided by your intent: **{intent}**."

    out = ["## Executive Summary\n", text]
    if sampled:
        out.append("\n\n⚠️ **Large dataset** — analysis uses the first 200,000 rows for speed.")
    out.append("")
    return "\n".join(out)


# -----------------------
# KPI Snapshot
# -----------------------

def build_kpi_snapshot(df):
    rows, cols = df.shape
    numeric = df.select_dtypes(include="number")
    cat = [c for c in df.columns if df[c].dtype == "object"]

    missing_cells = int(df.isna().sum().sum())
    missing_rate = (missing_cells / (rows * cols)) if rows and cols else 0.0
    dupes = int(df.duplicated().sum())

    kpis = [
        ("Rows × Columns", f"{rows:,} × {cols:,}"),
        ("Missing cells", f"{missing_cells:,} ({missing_rate:.1%})"),
        ("Duplicate rows", f"{dupes:,}"),
        ("Numeric fields", f"{numeric.shape[1]}"),
        ("Categorical/Text fields", f"{len(cat)}"),
    ]

    dcol = infer_date_column(df, "")
    if dcol:
        parsed = pd.to_datetime(df[dcol], errors="coerce")
        if parsed.notna().any():
            kpis.append((f"Date range ({dcol})", f"{parsed.min().date()} → {parsed.max().date()}"))

    return "## KPI Snapshot\n\n" + md_table(kpis) + "\n"


# -----------------------
# Quality + Duplicate section
# -----------------------

def build_quality_and_duplicates(df):
    score, issues = compute_quality_score(df)
    dupes = int(df.duplicated().sum())
    rows = len(df)
    rate = dupes / rows if rows else 0.0

    out = []
    out.append("## 🧪 Data Quality Score\n")
    out.append(f"**Score:** **{score}/100**\n")
    out.append("### Issues detected")
    for i in issues:
        out.append(f"- {i}")
    out.append("")

    out.append("## 🔁 Duplicate Analysis\n")
    out.append(f"- **Duplicate rows:** {dupes:,} ({rate:.1%} of dataset)\n")
    if dupes > 0:
        out.append("**Impact:** duplicates can inflate counts, distort KPIs, and bias trend/correlation results.\n")
        out.append("**Recommended actions:**")
        out.append("- Remove exact duplicates for reporting")
        out.append("- Deduplicate using a key set (ID + date + category)")
        out.append("- Confirm whether duplicates are repeated events or ingestion errors\n")
    else:
        out.append("✅ No duplicate rows detected.\n")

    return "\n".join(out)


# -----------------------
# Profiling tables
# -----------------------

def build_profiling_tables(df):
    out = []

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        out.append("### Missing Values\n")
        out.append(missing.to_frame("missing_count").to_markdown())
        out.append("")
    else:
        out.append("### Missing Values\n\n✅ No missing values found.\n")

    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] > 0:
        out.append("### Numeric Summary Statistics\n")
        out.append(numeric.describe().T.to_markdown())
        out.append("")
    else:
        out.append("### Numeric Summary Statistics\n\n_This dataset contains **no numeric columns**, so numeric summary and correlation are not available._\n")

    return "\n".join(out)


# -----------------------
# Smart Suggestions
# -----------------------

def build_smart_suggestions(df):
    cat_col = pick_categorical_column(df)
    num_col = pick_numeric_focus_column(df)
    date_col = infer_date_column(df, "")

    suggestions = []
    suggestions.append("duplicates check")
    suggestions.append("missing values breakdown")

    if cat_col:
        suggestions.append(f"distribution by {cat_col}")

    if date_col:
        suggestions.append(f"trend over time using {date_col}")

    if cat_col and num_col:
        suggestions.append(f"sum {num_col} by {cat_col}")
        suggestions.append(f"average {num_col} by {cat_col}")
        suggestions.append(f"highest {num_col} by {cat_col}")

    out = []
    out.append("## 💡 Smart Suggestions\n")
    out.append("Try one of these intents:\n")
    for s in suggestions[:10]:
        out.append(f"- {s}")
    out.append("")
    return "\n".join(out)


# -----------------------
# Intent Engine
# -----------------------

def classify_intent(intent: str):
    if not intent:
        return "profiling"
    i = intent.lower()

    if "duplicate" in i:
        return "duplicates"
    if "missing" in i or "null" in i:
        return "missing"
    if any(x in i for x in ["trend", "over time", "monthly", "daily", "growth"]):
        return "trend"
    if any(x in i for x in ["distribution", "frequency", "breakdown"]):
        return "distribution"
    if any(x in i for x in ["group by", "sum by", "average by", "avg by", "mean by", "total by", "compare"]):
        return "group_agg"
    if any(x in i for x in ["top", "highest", "max", "largest"]):
        return "top"
    if any(x in i for x in ["lowest", "min", "smallest"]):
        return "lowest"
    if any(x in i for x in ["outlier", "anomaly"]):
        return "outliers"

    if " by " in i:
        return "group_agg"

    return "profiling"


def run_intent_query(df, intent):
    intent_type = classify_intent(intent)

    section = []
    section.append("## 🎯 Answer to Your Intent\n")
    section.append(f"You asked: **{intent or '(none)'}**\n")

    group_col = infer_group_column(df, intent)
    value_col = infer_value_column(df, intent)
    date_col = infer_date_column(df, intent)

    section.append("### 🧠 Interpretation + Query Plan\n")
    section.append(f"Detected intent type: **{intent_type}**\n")

    # --------------- Results
    if intent_type == "trend" and date_col:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        temp = df.copy()
        temp["_parsed_date"] = parsed
        monthly = temp.dropna(subset=["_parsed_date"]).groupby(temp["_parsed_date"].dt.to_period("M")).size()

        section.append(f"### Trend over time (**{date_col}**)\n")
        if len(monthly) > 1:
            fig = plt.figure(figsize=(9, 4.5))
            monthly.sort_index().plot()
            plt.title(f"Monthly Trend ({date_col})")
            plt.xlabel("Month")
            plt.ylabel("Count")
            plt.tight_layout()
            section.append(fig_to_md_image(fig))
        else:
            section.append("⚠️ Not enough variation to compute a trend.\n")

        section.append("")
        return "\n".join(section)

    section.append("✅ No specialized intent matched — showing general KPI profiling.\n")
    section.append("")
    return "\n".join(section)


# -----------------------
# Chart Engine (FORCED MIN 3 charts)
# -----------------------

def build_charts(df):
    out = []
    out.append("## 📊 Charts & Visual Signals\n")

    charts_added = 0

    # Chart 1: Missing values bar (always possible)
    try:
        missing = df.isna().sum().sort_values(ascending=False).head(10)
        if missing.sum() > 0:
            fig = plt.figure(figsize=(9, 4.5))
            missing.plot(kind="bar")
            plt.title("Top Missing Value Columns")
            plt.ylabel("Missing Count")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            out.append(fig_to_md_image(fig))
            out.append("")
            charts_added += 1
    except:
        pass

    # Chart 2: Category distribution
    try:
        focus_cat = pick_categorical_column(df)
        if focus_cat:
            vc = df[focus_cat].astype(str).value_counts().head(10)
            fig = plt.figure(figsize=(9, 4.5))
            vc.index = clean_labels(vc.index.tolist(), 18)
            vc.plot(kind="bar")
            plt.title(f"Top Categories in {focus_cat}")
            plt.ylabel("Count")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            out.append(fig_to_md_image(fig))
            out.append("")
            charts_added += 1
    except:
        pass

    # Chart 3: Monthly trend
    try:
        focus_date = infer_date_column(df, "")
        if focus_date:
            parsed = pd.to_datetime(df[focus_date], errors="coerce")
            temp = df.copy()
            temp["_date"] = parsed
            monthly = temp.dropna(subset=["_date"]).groupby(temp["_date"].dt.to_period("M")).size()
            if len(monthly) > 1:
                fig = plt.figure(figsize=(9, 4.5))
                monthly.sort_index().plot()
                plt.title(f"Monthly Activity Trend ({focus_date})")
                plt.xlabel("Month")
                plt.ylabel("Count")
                plt.tight_layout()
                out.append(fig_to_md_image(fig))
                out.append("")
                charts_added += 1
    except:
        pass

    # Chart 4: Correlation heatmap (only if numeric>=2)
    try:
        numeric = df.select_dtypes(include="number")
        if numeric.shape[1] >= 2:
            corr = numeric.corr(numeric_only=True)
            fig = plt.figure(figsize=(9, 7))
            plt.imshow(corr, aspect="auto")
            plt.colorbar()
            plt.title("Correlation Heatmap (Numeric Columns)")
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
            plt.yticks(range(len(corr.index)), corr.index, fontsize=7)
            plt.tight_layout()
            out.append(fig_to_md_image(fig))
            out.append("")
            charts_added += 1
    except:
        pass

    if charts_added == 0:
        out.append("_No charts could be generated._\n")

    return "\n".join(out)


# -----------------------
# Final Take
# -----------------------

def build_final_take(df):
    rows, cols = df.shape
    dupes = int(df.duplicated().sum())
    missing = int(df.isna().sum().sum())
    num = df.select_dtypes(include="number").shape[1]

    points = []
    points.append("This report is a first-pass profile. Improve accuracy by refining the intent to target your business question.")
    if missing > 0:
        points.append("Handle missing values before using results for policy or forecasting.")
    else:
        points.append("Completeness is strong; you can move to segmentation and dashboarding confidently.")
    if dupes > 0:
        points.append("Deduplicate before KPI reporting to avoid inflated totals.")
    else:
        points.append("Uniqueness looks strong; counts and trend totals should be reliable.")
    if num == 0:
        points.append("With no numeric fields, strongest insights come from distribution and time-based volume trends.")
    else:
        points.append("With numeric fields available, prioritize group+aggregate queries to find the biggest drivers.")

    out = []
    out.append("## ✅ Final Take\n")
    out.append("Here’s the practical takeaway from this dataset:\n")
    for p in points[:6]:
        out.append(f"- {p}")
    out.append("")
    return "\n".join(out)


# -----------------------
# MAIN
# -----------------------

def main(file_path, intent):
    df, sampled, source_type = safe_read_table(file_path)
    rows, cols = df.shape

    report = []
    report.append(build_executive_summary(df, intent, source_type, sampled))
    report.append(build_dataset_story(df))
    report.append(build_kpi_snapshot(df))
    report.append(build_quality_and_duplicates(df))
    report.append(build_profiling_tables(df))
    report.append(build_smart_suggestions(df))
    report.append(run_intent_query(df, intent))
    report.append(build_charts(df))
    report.append(build_final_take(df))

    metadata = {
        "source_type": source_type,
        "rows": rows,
        "cols": cols,
        "sampled": sampled,
        "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
        "preview": df_preview_json_safe(df, n=5),
    }

    return {
        "analysis": "\n".join(report),
        "metadata": json.dumps(metadata, indent=2, default=to_json_safe),
        "code": "InsightForge v1.0 (dataset story + profiling + intent answers + charts + final take)",
    }


if __name__ == "__main__":
    file_path = sys.argv[1]
    intent = sys.argv[2] if len(sys.argv) > 2 else ""
    result = main(file_path, intent)
    print(json.dumps(result))