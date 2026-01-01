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


def format_number(n):
    try:
        if isinstance(n, float):
            return f"{n:,.2f}"
        return f"{int(n):,}"
    except Exception:
        return str(n)


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

    prefs = ["amount", "total", "cost", "price", "revenue", "sales", "salary", "earn", "value"]
    for pref in prefs:
        for c in num_cols:
            if pref in c.lower():
                return c

    variances = df[num_cols].var(numeric_only=True).sort_values(ascending=False)
    return variances.index[0] if len(variances) else num_cols[0]


def pick_categorical_column(df):
    preferred = ["status", "dept", "department", "hotel", "category", "type", "region", "rateapplied", "workclass"]
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
# Dataset Story (Business narrative)
# -----------------------

def build_dataset_story(df, source_type):
    cols = [c.lower() for c in df.columns]
    story = []
    story.append("## 🧠 Dataset Story\n")

    if any("employee" in c for c in cols) and any("trip" in c for c in cols):
        story.append(
            "This dataset looks like a **corporate workflow log** — tracking employee trips, approvals, and processing dates. "
            "It is useful for finding **bottlenecks**, **high volume periods**, and **policy compliance trends**."
        )
    elif any("hotel" in c for c in cols) and any("adr" in c for c in cols):
        story.append(
            "This dataset resembles a **hotel demand and pricing dataset**. "
            "It supports analysis around **seasonality**, **cancellations**, **lead time behavior**, and **revenue drivers**."
        )
    elif any("earn" in c for c in cols) or any("salary" in c for c in cols):
        story.append(
            "This dataset appears related to **earnings / payroll / financial records**. "
            "It can help analyze **distribution patterns**, **high earners/outliers**, and **fairness/variance** across groups."
        )
    elif any("order" in c for c in cols) and any("customer" in c for c in cols):
        story.append(
            "This dataset looks like a **transaction or customer order log**, suitable for analyzing "
            "**spending patterns**, **high-value segments**, and **retention drivers**."
        )
    else:
        story.append(
            "This dataset appears to be a general structured table. InsightForge treats it as a universal dataset "
            "and focuses on profiling, segmentation, trend detection, and anomaly signals."
        )

    story.append("")
    story.append("✅ **Best use cases for this dataset:**")
    story.append("- KPI monitoring and trend tracking")
    story.append("- Segment breakdowns (by categories / departments / status)")
    story.append("- Detecting anomalies, duplicates, missing-value risks")
    story.append("- Finding drivers and correlations (if numeric columns exist)")
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
    missing_rate = missing_cells / (rows * cols) if rows and cols else 0.0

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
# Executive summary + KPI
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

    explanation = (
        "\n**What this means:**\n"
        "- This snapshot summarizes the dataset size and shape.\n"
        "- Missing cells and duplicates directly affect reporting accuracy.\n"
        "- Numeric fields enable deeper analysis like correlations and outliers.\n"
    )

    return "## KPI Snapshot\n\n" + md_table(kpis) + explanation + "\n"


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
# Tables + explanations
# -----------------------

def build_profiling_tables(df):
    out = []

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    out.append("## 📌 Profiling Tables\n")

    if len(missing) > 0:
        out.append("### Missing Values\n")
        out.append(missing.to_frame("missing_count").to_markdown())
        out.append("")
        out.append("**What this means:** Columns with missing values require cleaning before analysis. "
                   "High missing count can distort trends and totals.\n")
    else:
        out.append("### Missing Values\n\n✅ No missing values found.\n")
        out.append("**What this means:** Data completeness is strong. Reports and KPIs are less likely to be biased.\n")

    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] > 0:
        out.append("### Numeric Summary Statistics\n")
        out.append(numeric.describe().T.to_markdown())
        out.append("")
        out.append("**What this means:** This table shows the distribution of numeric fields (min, max, average, etc.). "
                   "Large spreads or extreme max values can indicate outliers.\n")

        if numeric.shape[1] >= 2:
            corr = numeric.corr(numeric_only=True).abs()
            np.fill_diagonal(corr.values, 0)
            i, j = np.unravel_index(np.argmax(corr.values), corr.shape)
            best = float(corr.values[i, j])
            c1 = corr.index[i]
            c2 = corr.columns[j]
            out.append("### Strongest Relationship\n")
            out.append(f"- Highest correlation: **{c1} ↔ {c2} = {best:.2f}**\n")
            out.append("**What this means:** These two numeric columns move together strongly. If one rises, the other tends to rise too.\n")
    else:
        out.append("### Numeric Summary Statistics\n\n_This dataset contains **no numeric columns**, so numeric summary and correlation are not available._\n")
        out.append("**What this means:** The strongest insights will come from category distributions and trends over time.\n")

    return "\n".join(out)


# -----------------------
# Smart suggestions
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
# NEW FEATURE #1: Key Insights (Auto Findings)
# -----------------------

def build_key_insights(df):
    rows, cols = df.shape
    out = []
    out.append("## 🔍 Key Insights (Auto Findings)\n")

    # Dominant category
    cat_col = pick_categorical_column(df)
    if cat_col:
        vc = df[cat_col].astype(str).value_counts()
        top = vc.index[0]
        pct = vc.iloc[0] / rows if rows else 0
        out.append(f"- **Dominant segment:** `{cat_col}` is mostly **{top}** ({pct:.1%} of rows).")

    # Date activity peak
    dcol = infer_date_column(df, "")
    if dcol:
        parsed = pd.to_datetime(df[dcol], errors="coerce")
        if parsed.notna().any():
            temp = parsed.dropna()
            month_counts = temp.dt.to_period("M").value_counts().sort_index()
            peak_month = month_counts.idxmax()
            peak_count = month_counts.max()
            out.append(f"- **Peak activity month:** **{peak_month}** with **{peak_count:,} records**.")

    # Numeric outlier signals
    num_col = pick_numeric_focus_column(df)
    if num_col:
        s = df[num_col].dropna()
        if len(s) > 10:
            p99 = s.quantile(0.99)
            p50 = s.quantile(0.50)
            if p99 > p50 * 3:
                out.append(f"- **Outlier signal:** `{num_col}` has extreme high values (99th percentile = {format_number(p99)}).")

    # Duplicates signal
    dupes = int(df.duplicated().sum())
    if dupes > 0:
        out.append(f"- **Duplicates exist:** {dupes:,} duplicate rows could inflate reporting totals.")

    # Missing signal
    missing_cells = int(df.isna().sum().sum())
    if missing_cells == 0:
        out.append("- **Data completeness:** No missing values detected — KPI reporting is more reliable.")

    out.append("\n✅ These are automatic highlights. For deeper accuracy, refine the analysis intent (e.g., 'trend over time using X').\n")
    return "\n".join(out)


# -----------------------
# NEW FEATURE #2: Anomaly Watchlist
# -----------------------

def build_anomaly_watchlist(df):
    out = []
    out.append("## 🚨 Anomaly Watchlist (Suspicious Patterns)\n")

    warnings = []

    rows = len(df)

    for c in df.columns:
        nunique = df[c].nunique(dropna=True)

        # Suspicious "ID-like" columns
        if nunique > rows * 0.8 and df[c].dtype == "object":
            warnings.append(f"- `{c}` has extremely high unique values ({nunique:,}). It may be an ID column, not a category.")

        # Dominant category columns
        if df[c].dtype == "object":
            vc = df[c].astype(str).value_counts()
            if len(vc) > 1 and (vc.iloc[0] / rows) > 0.9:
                warnings.append(f"- `{c}` is dominated by one value (**{vc.index[0]}**, {vc.iloc[0]/rows:.1%}). May reduce segmentation value.")

        # Mostly zeros
        if df[c].dtype != "object":
            vals = df[c].dropna()
            if len(vals) > 0:
                zero_rate = (vals == 0).mean()
                if zero_rate > 0.95:
                    warnings.append(f"- `{c}` is **{zero_rate:.1%} zeros**. This may indicate missing information stored as 0.")

    if warnings:
        out.append("Potential issues detected:\n")
        out.extend(warnings[:12])
    else:
        out.append("✅ No obvious anomalies detected. Dataset structure looks consistent.\n")

    out.append("")
    return "\n".join(out)


# -----------------------
# NEW FEATURE #3: Segment Driver Explanation
# -----------------------

def build_segment_drivers(df):
    out = []
    out.append("## 📌 Segment Drivers (What’s Driving the Numbers?)\n")

    num_col = pick_numeric_focus_column(df)
    cat_col = pick_categorical_column(df)

    if not num_col or not cat_col:
        out.append("⚠️ Segment driver analysis requires at least one numeric column and one category column.\n")
        return "\n".join(out)

    g_sum = df.groupby(cat_col)[num_col].sum(numeric_only=True).sort_values(ascending=False).head(10)
    g_mean = df.groupby(cat_col)[num_col].mean(numeric_only=True).sort_values(ascending=False).head(10)

    total = g_sum.sum()
    best_seg = g_sum.index[0]
    best_val = g_sum.iloc[0]
    pct = (best_val / total) if total else 0

    out.append(f"✅ **Top contributing segment:** **{best_seg}** contributes **{pct:.1%}** of total `{num_col}`.")
    out.append("")
    out.append(f"### Top segments by total `{num_col}`\n")
    out.append(g_sum.to_frame("total").to_markdown())
    out.append("")
    out.append(f"### Top segments by average `{num_col}`\n")
    out.append(g_mean.to_frame("avg").to_markdown())
    out.append("")
    out.append("**What this means:**\n"
               "- The **sum** view highlights which segments contribute most to total volume.\n"
               "- The **average** view highlights which segments are most intense per record.\n")
    out.append("")
    return "\n".join(out)


# -----------------------
# Intent Engine (Analyst-style plan + computed answers)
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
        if " by " in i:
            return "group_agg"
        return "top"
    if any(x in i for x in ["lowest", "min", "smallest"]):
        if " by " in i:
            return "group_agg"
        return "lowest"
    if any(x in i for x in ["outlier", "anomaly"]):
        return "outliers"
    return "profiling"


def analyst_plan(intent_type, group_col=None, value_col=None, date_col=None):
    plan = []
    plan.append("### 🧠 Interpretation + Query Plan\n")

    if intent_type == "duplicates":
        plan.append("We will check for duplicate rows, measure how many exist, and show sample duplicates if any.")
    elif intent_type == "missing":
        plan.append("We will compute missing counts per column and highlight the most affected fields.")
    elif intent_type == "trend":
        plan.append(f"We will parse the date column **{date_col}**, group activity by month, and visualize the trend.")
    elif intent_type == "distribution":
        plan.append(f"We will summarize frequency counts in **{group_col}**, highlight dominant categories, and chart distribution.")
    elif intent_type == "group_agg":
        plan.append(f"We will group by **{group_col}**, aggregate **{value_col}**, rank segments, and chart top contributors.")
    else:
        plan.append("No specific intent matched. We will perform full profiling and insights.")

    plan.append("")
    return "\n".join(plan)


def run_intent_query(df, intent):
    intent_type = classify_intent(intent)

    section = []
    section.append("## 🎯 Answer to Your Intent\n")
    section.append(f"You asked: **{intent or '(none)'}**\n")
    section.append(f"Detected intent type: **{intent_type}**\n")

    group_col = infer_group_column(df, intent)
    value_col = infer_value_column(df, intent)
    date_col = infer_date_column(df, intent)

    section.append(analyst_plan(intent_type, group_col, value_col, date_col))

    if intent_type == "trend":
        if not date_col:
            section.append("⚠️ No valid date column found for trend analysis.\n")
            return "\n".join(section)

        parsed = pd.to_datetime(df[date_col], errors="coerce")
        temp = df.copy()
        temp["_parsed_date"] = parsed
        monthly = temp.dropna(subset=["_parsed_date"]).groupby(temp["_parsed_date"].dt.to_period("M")).size()

        section.append(f"### Trend over time (**{date_col}**)\n")

        if len(monthly) > 1:
            first, last = int(monthly.iloc[0]), int(monthly.iloc[-1])
            growth = ((last - first) / first) if first else 0
            section.append(f"✅ Activity changed from **{first:,} → {last:,}** (**{growth:+.1%}**) across the dataset time range.\n")

            fig = plt.figure(figsize=(9, 4.5))
            monthly.sort_index().plot()
            plt.title(f"Monthly Trend ({date_col})")
            plt.xlabel("Month")
            plt.ylabel("Count")
            plt.tight_layout()
            section.append(fig_to_md_image(fig))
            section.append("\n**Chart meaning:** This line chart shows how records are distributed across months. Peaks indicate high activity periods.\n")

        else:
            section.append("⚠️ Not enough variation to compute a trend.\n")

        section.append("")
        return "\n".join(section)

    section.append("✅ No specialized intent matched — default profiling used.\n")
    return "\n".join(section)


# -----------------------
# Chart Engine (captions for all charts)
# -----------------------

def build_charts(df):
    out = []
    out.append("## 📊 Charts & Visual Signals\n")
    charts_added = 0

    numeric = df.select_dtypes(include="number")
    focus_num = pick_numeric_focus_column(df)
    focus_cat = pick_categorical_column(df)
    focus_date = infer_date_column(df, "")

    # Histogram
    if focus_num and focus_num in df.columns and df[focus_num].dtype != "object":
        s = df[focus_num].dropna()
        if len(s) > 1:
            fig = plt.figure(figsize=(8.5, 4.5))
            s.hist(bins=30)
            plt.title(f"Distribution of {focus_num}")
            plt.xlabel(focus_num)
            plt.ylabel("Frequency")
            plt.tight_layout()
            out.append(fig_to_md_image(fig))
            out.append(f"**Chart meaning:** This histogram shows how `{focus_num}` values are spread. "
                       f"Skew or spikes can indicate outliers or imbalanced distributions.\n")
            charts_added += 1

    # Category distribution
    if focus_cat and focus_cat in df.columns:
        counts = df[focus_cat].astype(str).value_counts().head(10)
        if len(counts) > 0:
            fig = plt.figure(figsize=(9, 5))
            counts.index = clean_labels(counts.index.tolist(), 18)
            counts.plot(kind="bar")
            plt.title(f"Top Categories in {focus_cat}")
            plt.xlabel(focus_cat)
            plt.ylabel("Count")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            out.append(fig_to_md_image(fig))
            out.append(f"**Chart meaning:** This bar chart shows which `{focus_cat}` categories appear most often. "
                       f"If one dominates strongly, segmentation insights may be limited.\n")
            charts_added += 1

    # Trend line
    if focus_date and focus_date in df.columns:
        parsed = pd.to_datetime(df[focus_date], errors="coerce")
        temp = df.copy()
        temp["_parsed_date"] = parsed
        monthly = temp.dropna(subset=["_parsed_date"]).groupby(temp["_parsed_date"].dt.to_period("M")).size()
        if len(monthly) > 1:
            fig = plt.figure(figsize=(9, 4.5))
            monthly.sort_index().plot()
            plt.title(f"Monthly Trend ({focus_date})")
            plt.xlabel("Month")
            plt.ylabel("Count")
            plt.tight_layout()
            out.append(fig_to_md_image(fig))
            out.append(f"**Chart meaning:** This line chart shows how the number of records changes over time. "
                       f"Rising trends indicate growth; spikes indicate unusual high-activity months.\n")
            charts_added += 1

    # Correlation heatmap
    if numeric.shape[1] >= 2:
        corr = numeric.corr(numeric_only=True)
        fig = plt.figure(figsize=(9.0, 7.0))
        plt.imshow(corr, aspect="auto")
        plt.title("Correlation Heatmap (numeric columns)")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
        plt.yticks(range(len(corr.index)), corr.index, fontsize=7)
        plt.colorbar()
        plt.tight_layout()
        out.append(fig_to_md_image(fig))
        out.append("**Chart meaning:** This heatmap shows how numeric columns relate. "
                   "Values closer to 1.0 mean strong positive relationships.\n")
        charts_added += 1

    if charts_added == 0:
        out.append("_No charts could be generated._\n")

    out.append("")
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
        points.append("Address missing values before decisions based on segmentation or trends.")
    else:
        points.append("Completeness is strong; you can move to segmentation and dashboarding confidently.")

    if dupes > 0:
        points.append("Deduplicate before KPI reporting to avoid inflated totals.")
    else:
        points.append("Row uniqueness is strong; trend counts and totals should be reliable.")

    if num == 0:
        points.append("With no numeric fields, insights are best from distributions and activity trends.")
    else:
        points.append("With numeric fields available, group+aggregate queries will reveal strongest drivers.")

    out = []
    out.append("## ✅ Final Take\n")
    out.append("Here’s the practical takeaway from this dataset:\n")
    for p in points[:6]:
        out.append(f"- {p}")
    out.append("")
    return "\n".join(out)


# -----------------------
# Main
# -----------------------

def main(file_path, intent):
    df, sampled, source_type = safe_read_table(file_path)
    rows, cols = df.shape

    report = []
    report.append(build_executive_summary(df, intent, source_type, sampled))
    report.append(build_dataset_story(df, source_type))
    report.append(build_kpi_snapshot(df))
    report.append(build_quality_and_duplicates(df))
    report.append(build_key_insights(df))
    report.append(build_anomaly_watchlist(df))
    report.append(build_segment_drivers(df))
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
        "code": "InsightForge v1.1 (profiling + story + intent engine + charts captions + key insights + anomaly watchlist + segment drivers)",
    }


if __name__ == "__main__":
    file_path = sys.argv[1]
    intent = sys.argv[2] if len(sys.argv) > 2 else ""
    result = main(file_path, intent)
    print(json.dumps(result))