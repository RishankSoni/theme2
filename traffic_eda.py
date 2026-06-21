"""
Bengaluru Traffic Incident EDA  — with Description NLP
========================================================
Install:
  pip install pandas matplotlib seaborn scipy scikit-learn wordcloud \
              langdetect indic-transliteration --break-system-packages

Run:
  python traffic_eda.py

Expects: traffic_data.csv in the same directory.
Outputs: eda_report/ folder with all plots + eda_summary.txt
"""

import os
import re
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter
from scipy import stats

warnings.filterwarnings("ignore")

OUTPUT_DIR = "eda_report"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────
PALETTE = {
    "blue"  : "#185FA5",
    "teal"  : "#0F6E56",
    "coral" : "#D85A30",
    "amber" : "#BA7517",
    "purple": "#534AB7",
    "green" : "#3B6D11",
    "gray"  : "#5F5E5A",
    "pink"  : "#993556",
}

def save(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


# ─────────────────────────────────────────────
# 1. LOAD & BASIC PARSE
# ─────────────────────────────────────────────

def load_data(filepath="traffic_data.csv"):
    df = pd.read_csv(filepath, low_memory=False)

    dt_cols = ["start_datetime", "end_datetime", "modified_datetime",
               "created_date", "closed_datetime", "resolved_datetime"]
    for col in dt_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    for col in ["latitude", "longitude", "endlatitude", "endlongitude",
                "resolved_at_latitude", "resolved_at_longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace(0, np.nan)

    if "requires_road_closure" in df.columns:
        df["requires_road_closure"] = df["requires_road_closure"].map(
            {"TRUE": True, "FALSE": False, True: True, False: False})

    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df):
    # Resolution time (primary reward proxy)
    df["resolution_minutes"] = (
        df["closed_datetime"] - df["start_datetime"]
    ).dt.total_seconds() / 60

    mask = df["resolution_minutes"].isna() & df["resolved_datetime"].notna()
    df.loc[mask, "resolution_minutes"] = (
        df.loc[mask, "resolved_datetime"] - df.loc[mask, "start_datetime"]
    ).dt.total_seconds() / 60

    df["resolution_minutes"] = df["resolution_minutes"].clip(lower=0, upper=1440)

    df["hour"]        = df["start_datetime"].dt.hour
    df["day_of_week"] = df["start_datetime"].dt.dayofweek
    df["day_name"]    = df["start_datetime"].dt.day_name()
    df["month"]       = df["start_datetime"].dt.month
    df["month_name"]  = df["start_datetime"].dt.month_name()
    df["is_weekend"]  = df["day_of_week"].isin([5, 6])
    df["is_night"]    = df["hour"].between(22, 24) | df["hour"].between(0, 5)

    def hour_bin(h):
        if 7  <= h <= 10: return "Morning peak (7-10)"
        if 17 <= h <= 20: return "Evening peak (17-20)"
        if 11 <= h <= 16: return "Midday (11-16)"
        return "Off-peak"
    df["time_bin"] = df["hour"].apply(hour_bin)

    df["was_resolved"] = df["status"].isin(["closed", "resolved"])

    if "created_date" in df.columns:
        df["log_lag_minutes"] = (
            df["created_date"] - df["start_datetime"]
        ).dt.total_seconds() / 60
        df["log_lag_minutes"] = df["log_lag_minutes"].clip(0, 120)

    return df


# ─────────────────────────────────────────────
# 3. DESCRIPTION NLP PIPELINE
# ─────────────────────────────────────────────

def detect_language(text):
    if not isinstance(text, str) or text.strip() == "":
        return "empty"
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42
        lang = detect(text)
        return lang
    except Exception:
        return "unknown"


def transliterate_kannada(text):
    if not isinstance(text, str):
        return text
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
        return transliterate(text, sanscript.KANNADA, sanscript.ITRANS)
    except Exception:
        return text


def normalize_description(text):
    if not isinstance(text, str) or text.strip() == "":
        return "", "empty"

    lang = detect_language(text)

    if lang == "kn":
        text = transliterate_kannada(text)

    text = re.sub(r"\[LOCATION\]|\[PERSON\]|\[PHONE\]", " ", text)
    text = re.sub(r"http\S+|www\S+|\S+@\S+", " ", text)
    text = text.encode("ascii", errors="ignore").decode()
    text = text.lower()
    text = re.sub(r"[^\w\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text, lang


DOMAIN_KEYWORDS = {
    "breakdown"     : ["breakdown", "broke", "stalled", "starting", "problem",
                       "stopped", "engine", "puncture", "tyre", "flat"],
    "accident"      : ["accident", "collision", "crash", "hit", "injured",
                       "injured", "impact", "overturn"],
    "waterlogging"  : ["water", "waterlogging", "flood", "logging", "rain",
                       "inundated", "submerged", "woter"],
    "tree_fall"     : ["tree", "fall", "fallen", "branch", "uprooted"],
    "road_block"    : ["block", "blocked", "barricade", "closed", "diverted",
                       "diversion", "closure"],
    "construction"  : ["work", "construction", "cement", "digging", "repair",
                       "maintenance", "laying", "pipe"],
    "traffic_slow"  : ["slow", "jam", "heavy", "congestion", "movement",
                       "crawling", "standstill", "traffic"],
    "event"         : ["match", "rally", "event", "festival", "procession",
                       "protest", "stadium", "crowd"],
    "vehicle_type"  : ["lorry", "truck", "bus", "bmtc", "car", "auto",
                       "tanker", "lcv", "heavy"],
    "location_ref"  : ["junction", "circle", "road", "flyover", "underpass",
                       "bridge", "signal", "cross"],
    "severity_high" : ["major", "severe", "serious", "urgent", "critical",
                       "emergency", "blocked completely"],
}

def extract_keyword_flags(text):
    flags = {}
    for category, keywords in DOMAIN_KEYWORDS.items():
        flags[f"desc_{category}"] = int(any(kw in text for kw in keywords))
    return flags


def extract_description_features(df):
    print("  Detecting languages & normalizing descriptions...")

    if "description" not in df.columns:
        print("  WARNING: 'description' column not found — skipping NLP.")
        return df

    df["desc_original"] = df["description"].astype(str)

    df["desc_has_phone"]    = df["desc_original"].str.contains(r"\[PHONE\]",    na=False).astype(int)
    df["desc_has_location"] = df["desc_original"].str.contains(r"\[LOCATION\]", na=False).astype(int)
    df["desc_char_count"]   = df["desc_original"].str.len()

    results = df["desc_original"].apply(normalize_description)
    df["desc_normalized"] = results.apply(lambda x: x[0])
    df["desc_lang"]       = results.apply(lambda x: x[1])
    df["desc_word_count"] = df["desc_normalized"].str.split().str.len().fillna(0).astype(int)

    print("  Extracting domain keyword flags...")
    flag_rows = df["desc_normalized"].apply(extract_keyword_flags)
    flag_df   = pd.DataFrame(list(flag_rows))
    df = pd.concat([df.reset_index(drop=True), flag_df.reset_index(drop=True)], axis=1)

    return df


def cluster_descriptions(df, n_clusters=6):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD

    corpus = df["desc_normalized"].fillna("").tolist()
    non_empty = [t for t in corpus if len(t.strip()) > 2]

    if len(non_empty) < n_clusters:
        print(f"  Not enough descriptions for clustering ({len(non_empty)} rows). Skipping.")
        df["desc_topic"] = -1
        return df, {}

    print(f"  Fitting TF-IDF + KMeans (k={n_clusters}) on {len(non_empty)} descriptions...")

    vec = TfidfVectorizer(
        max_features=300,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english"
    )
    X = vec.fit_transform(corpus)

    svd = TruncatedSVD(n_components=min(20, X.shape[1] - 1), random_state=42)
    X_reduced = svd.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["desc_topic"] = km.fit_predict(X_reduced)

    feature_names = vec.get_feature_names_out()
    topic_labels  = {}
    centers_orig  = km.cluster_centers_ @ svd.components_

    for i, center in enumerate(centers_orig):
        top_idx   = center.argsort()[-5:][::-1]
        top_terms = [feature_names[j] for j in top_idx]
        topic_labels[i] = " | ".join(top_terms)

    return df, topic_labels


# ─────────────────────────────────────────────
# 4. NLP PLOTS
# ─────────────────────────────────────────────

def plot_description_language(df):
    if "desc_lang" not in df.columns:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Description Field — Language & Length", fontsize=14, fontweight="bold")

    lang_map  = {"en": "English", "kn": "Kannada", "empty": "Empty / null",
                 "unknown": "Unknown", "mixed": "Mixed"}
    lang_counts = df["desc_lang"].map(lang_map).fillna("Other").value_counts()
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["gray"],
              PALETTE["amber"], PALETTE["coral"]]
    axes[0].pie(lang_counts.values, labels=lang_counts.index,
                autopct="%1.1f%%", colors=colors[:len(lang_counts)],
                startangle=90)
    axes[0].set_title("Language Detected")

    wc = df[df["desc_word_count"] > 0]["desc_word_count"]
    axes[1].hist(wc, bins=30, color=PALETTE["purple"], edgecolor="white", alpha=0.85)
    axes[1].axvline(wc.median(), color=PALETTE["coral"], linestyle="--",
                    linewidth=2, label=f"Median: {wc.median():.0f} words")
    axes[1].set_title("Description Word Count")
    axes[1].set_xlabel("Words")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    if "resolution_minutes" in df.columns:
        sub = df[df["desc_word_count"] > 0].dropna(subset=["resolution_minutes"])
        axes[2].scatter(sub["desc_word_count"], sub["resolution_minutes"],
                        alpha=0.35, s=18, color=PALETTE["teal"])
        if len(sub) > 5:
            m, b = np.polyfit(sub["desc_word_count"], sub["resolution_minutes"], 1)
            x_line = np.linspace(sub["desc_word_count"].min(), sub["desc_word_count"].max(), 100)
            axes[2].plot(x_line, m * x_line + b, color=PALETTE["coral"],
                         linewidth=2, label=f"Trend (slope={m:.1f})")
            axes[2].legend()
        axes[2].set_title("Description Length vs Resolution Time")
        axes[2].set_xlabel("Word count")
        axes[2].set_ylabel("Resolution (minutes)")

    plt.tight_layout()
    save(fig, "07_description_language.png")


def plot_keyword_flags(df):
    flag_cols = [c for c in df.columns if c.startswith("desc_") and
                 c not in ("desc_normalized", "desc_original", "desc_lang",
                           "desc_word_count", "desc_char_count",
                           "desc_has_phone", "desc_has_location", "desc_topic")]

    if not flag_cols or "event_cause" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Description Keywords — Domain Signal Analysis", fontsize=14, fontweight="bold")

    pivot = df.groupby("event_cause")[flag_cols].mean()
    pivot.columns = [c.replace("desc_", "") for c in pivot.columns]
    sns.heatmap(pivot, ax=axes[0], cmap="YlOrRd", annot=True, fmt=".2f",
                linewidths=0.5, cbar_kws={"shrink": 0.7})
    axes[0].set_title("Keyword Category Rate by Event Cause")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("")

    flag_sums = df[flag_cols].sum().sort_values(ascending=False)
    flag_sums.index = [c.replace("desc_", "") for c in flag_sums.index]
    axes[1].barh(flag_sums.index, flag_sums.values, color=PALETTE["blue"])
    axes[1].set_title("Total Keyword Hits per Category")
    axes[1].set_xlabel("Count")

    plt.tight_layout()
    save(fig, "08_keyword_flags.png")


def plot_keyword_vs_resolution(df):
    flag_cols = [c for c in df.columns if c.startswith("desc_") and
                 c not in ("desc_normalized", "desc_original", "desc_lang",
                           "desc_word_count", "desc_char_count",
                           "desc_has_phone", "desc_has_location", "desc_topic")]

    if not flag_cols or "resolution_minutes" not in df.columns:
        return

    rows = []
    for col in flag_cols:
        grp0 = df[df[col] == 0]["resolution_minutes"].dropna()
        grp1 = df[df[col] == 1]["resolution_minutes"].dropna()
        if len(grp1) < 5:
            continue
        label = col.replace("desc_", "")
        diff  = grp1.median() - grp0.median()
        rows.append({"keyword": label,
                     "median_with"    : grp1.median(),
                     "median_without" : grp0.median(),
                     "delta"          : diff,
                     "n_with"         : len(grp1)})

    if not rows:
        return

    comp = pd.DataFrame(rows).sort_values("delta", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE["coral"] if d > 0 else PALETTE["teal"] for d in comp["delta"]]
    bars = ax.barh(comp["keyword"], comp["delta"], color=colors)
    ax.axvline(0, color=PALETTE["gray"], linewidth=1, linestyle="--")
    ax.set_title("Keyword Impact on Resolution Time\n"
                 "(positive = takes longer when keyword present)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Δ Median resolution time (minutes)")

    for bar, row in zip(bars, comp.itertuples()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"n={row.n_with}", va="center", fontsize=8,
                color=PALETTE["gray"])

    plt.tight_layout()
    save(fig, "09_keyword_resolution_impact.png")


def plot_topic_clusters(df, topic_labels):
    if "desc_topic" not in df.columns or df["desc_topic"].nunique() < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Description Topic Clusters (TF-IDF + KMeans)", fontsize=14, fontweight="bold")

    topic_map = {k: f"T{k}: {v[:40]}" for k, v in topic_labels.items()}
    df["topic_label"] = df["desc_topic"].map(topic_map).fillna("Unknown")

    tc = df["topic_label"].value_counts()
    axes[0].barh(tc.index, tc.values, color=PALETTE["purple"])
    axes[0].set_title("Incidents per Topic Cluster")
    axes[0].set_xlabel("Count")

    if "resolution_minutes" in df.columns:
        topic_order = df.groupby("topic_label")["resolution_minutes"].median().sort_values().index
        data  = [df[df["topic_label"] == t]["resolution_minutes"].dropna().values
                 for t in topic_order]
        labels = [t[:35] for t in topic_order]
        bp = axes[1].boxplot(data, labels=labels, patch_artist=True, vert=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(PALETTE["teal"])
            patch.set_alpha(0.5)
        axes[1].set_title("Resolution Time by Topic Cluster")
        axes[1].set_xlabel("Minutes")
        axes[1].tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    save(fig, "10_topic_clusters.png")


def plot_wordcloud(df):
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("  wordcloud not installed — skipping word cloud.")
        return

    if "desc_normalized" not in df.columns:
        return

    text = " ".join(df["desc_normalized"].dropna().tolist())
    if len(text.strip()) < 50:
        return

    wc = WordCloud(
        width=1000, height=500,
        background_color="white",
        colormap="Blues",
        max_words=150,
        collocations=True,
        stopwords={"the", "a", "an", "is", "in", "at", "on", "of",
                   "and", "to", "for", "that", "this", "it", "be",
                   "with", "from", "are", "was", "were", "has", "have"}
    ).generate(text)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud — All Incident Descriptions\n(Kannada transliterated to Roman)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "11_wordcloud.png")


def plot_description_vs_cause(df):
    if "desc_lang" not in df.columns or "event_cause" not in df.columns:
        return

    lang_map = {"en": "English", "kn": "Kannada", "empty": "Empty",
                "unknown": "Unknown"}
    df["desc_lang_label"] = df["desc_lang"].map(lang_map).fillna("Other")

    pivot = df.groupby(["event_cause", "desc_lang_label"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax,
               color=[PALETTE["blue"], PALETTE["teal"],
                      PALETTE["gray"], PALETTE["amber"]])
    ax.set_title("Description Language per Event Cause", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Language", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    save(fig, "12_desc_language_per_cause.png")


# ─────────────────────────────────────────────
# 5. ORIGINAL EDA PLOTS
# ─────────────────────────────────────────────

def data_quality_report(df, report_lines):
    report_lines += ["=" * 60, "DATA QUALITY SUMMARY", "=" * 60,
                     f"Total records : {len(df)}",
                     f"Total columns : {len(df.columns)}", ""]

    report_lines.append("Null counts (top 20 columns by nulls):")
    null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    for col, pct in null_pct.head(20).items():
        report_lines.append(f"  {col:<35} {pct:>6.1f}% null")

    report_lines.append("")
    for label, col in [("Status", "status"), ("Event type", "event_type"),
                       ("Event cause", "event_cause")]:
        if col in df.columns:
            report_lines.append(f"{label} distribution:")
            for val, cnt in df[col].value_counts().items():
                report_lines.append(f"  {val:<25} {cnt:>5}  ({cnt/len(df)*100:.1f}%)")
            report_lines.append("")

    rt = df["resolution_minutes"].dropna()
    report_lines += ["Resolution time (minutes):",
                     f"  Count  : {len(rt)}",
                     f"  Mean   : {rt.mean():.1f}",
                     f"  Median : {rt.median():.1f}",
                     f"  Std    : {rt.std():.1f}",
                     f"  P25    : {rt.quantile(0.25):.1f}",
                     f"  P75    : {rt.quantile(0.75):.1f}",
                     f"  P90    : {rt.quantile(0.90):.1f}", ""]
    return report_lines


def plot_incident_overview(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Incident Overview", fontsize=14, fontweight="bold")

    cause_counts = df["event_cause"].value_counts()
    axes[0].barh(cause_counts.index, cause_counts.values, color=PALETTE["blue"])
    axes[0].set_title("By Event Cause"); axes[0].set_xlabel("Count")

    if "priority" in df.columns:
        pri = df["priority"].value_counts()
        axes[1].bar(pri.index, pri.values,
                    color=[PALETTE["coral"], PALETTE["amber"], PALETTE["teal"]][:len(pri)])
        axes[1].set_title("By Priority"); axes[1].set_ylabel("Count")

    status = df["status"].value_counts()
    axes[2].pie(status.values, labels=status.index, autopct="%1.1f%%",
                colors=[PALETTE["teal"], PALETTE["blue"], PALETTE["amber"], PALETTE["coral"]])
    axes[2].set_title("Status Breakdown")
    plt.tight_layout(); save(fig, "01_incident_overview.png")


def plot_temporal_patterns(df):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig)
    fig.suptitle("Temporal Patterns", fontsize=14, fontweight="bold")

    ax1 = fig.add_subplot(gs[0, 0])
    hourly = df.groupby("hour").size()
    ax1.bar(hourly.index, hourly.values, color=PALETTE["blue"], alpha=0.85)
    ax1.axvspan(7, 10, alpha=0.12, color=PALETTE["coral"], label="Morning peak")
    ax1.axvspan(17, 20, alpha=0.12, color=PALETTE["amber"], label="Evening peak")
    ax1.set_title("Incidents by Hour"); ax1.set_xlabel("Hour"); ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    day_order  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    day_counts = df.groupby("day_name").size().reindex(day_order)
    ax2.bar(day_order, day_counts.values,
            color=[PALETTE["coral"] if d in ["Saturday","Sunday"] else PALETTE["teal"]
                   for d in day_order])
    ax2.set_title("By Day of Week")
    ax2.set_xticklabels(day_order, rotation=35, ha="right")

    ax3 = fig.add_subplot(gs[1, 0])
    monthly = df.groupby(["month","month_name"]).size().reset_index(name="count").sort_values("month")
    ax3.plot(monthly["month_name"], monthly["count"],
             marker="o", color=PALETTE["purple"], linewidth=2)
    ax3.fill_between(range(len(monthly)), monthly["count"], alpha=0.12, color=PALETTE["purple"])
    ax3.set_title("Monthly Volume")
    ax3.set_xticklabels(monthly["month_name"], rotation=35, ha="right")

    ax4 = fig.add_subplot(gs[1, 1])
    if "event_cause" in df.columns:
        pivot = df.groupby(["time_bin","event_cause"]).size().unstack(fill_value=0)
        time_order = ["Morning peak (7-10)","Midday (11-16)","Evening peak (17-20)","Off-peak"]
        pivot = pivot.reindex([t for t in time_order if t in pivot.index])
        sns.heatmap(pivot, ax=ax4, cmap="YlOrRd", annot=True, fmt="d",
                    linewidths=0.5, cbar_kws={"shrink": 0.7})
        ax4.set_title("Cause × Time of Day")
        ax4.tick_params(axis="x", rotation=30)

    plt.tight_layout(); save(fig, "02_temporal_patterns.png")


def plot_resolution_time(df):
    rt = df[df["resolution_minutes"].notna() & (df["resolution_minutes"] > 0)].copy()
    if rt.empty: return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Resolution Time Analysis (Reward Proxy)", fontsize=14, fontweight="bold")

    axes[0,0].hist(rt["resolution_minutes"], bins=40,
                   color=PALETTE["blue"], edgecolor="white", alpha=0.85)
    axes[0,0].axvline(rt["resolution_minutes"].median(), color=PALETTE["coral"],
                      linestyle="--", linewidth=2,
                      label=f'Median: {rt["resolution_minutes"].median():.0f} min')
    axes[0,0].set_title("Overall Distribution"); axes[0,0].legend()

    if "event_cause" in rt.columns:
        cause_rt = rt.groupby("event_cause")["resolution_minutes"].median().sort_values()
        axes[0,1].barh(cause_rt.index, cause_rt.values, color=PALETTE["teal"])
        axes[0,1].set_title("Median Resolution by Cause"); axes[0,1].set_xlabel("Minutes")

    if "priority" in rt.columns:
        pri_rt = rt.groupby("priority")["resolution_minutes"]
        data   = [g.dropna().values for _, g in pri_rt]
        labels = [k for k, _ in pri_rt]
        axes[1,0].boxplot(data, labels=labels, patch_artist=True,
                          boxprops=dict(facecolor=PALETTE["purple"], alpha=0.5))
        axes[1,0].set_title("Resolution by Priority"); axes[1,0].set_ylabel("Minutes")

    tb_rt = rt.groupby("time_bin")["resolution_minutes"].median()
    time_order = ["Morning peak (7-10)","Midday (11-16)","Evening peak (17-20)","Off-peak"]
    tb_rt = tb_rt.reindex([t for t in time_order if t in tb_rt.index])
    axes[1,1].bar(tb_rt.index, tb_rt.values, color=PALETTE["amber"])
    axes[1,1].set_title("Resolution by Time of Day")
    axes[1,1].set_xticklabels(tb_rt.index, rotation=20, ha="right")

    plt.tight_layout(); save(fig, "03_resolution_time.png")


def plot_spatial_hotspots(df, report_lines):
    report_lines += ["", "=" * 60, "SPATIAL HOTSPOTS", "=" * 60]
    for label, col in [("Top zones", "zone"), ("Top junctions", "junction"),
                       ("Corridors", "corridor")]:
        if col in df.columns:
            top = df[col].value_counts().head(15)
            report_lines.append(f"\n{label}:")
            for v, c in top.items():
                report_lines.append(f"  {str(v):<40} {c:>4}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Spatial Hotspot Analysis", fontsize=14, fontweight="bold")

    if "zone" in df.columns:
        zc = df["zone"].value_counts().head(12).dropna()
        axes[0].barh(zc.index.astype(str), zc.values, color=PALETTE["purple"])
        axes[0].set_title("Top Zones by Volume")

    if "corridor" in df.columns and "event_cause" in df.columns:
        pivot = df.groupby(["corridor","event_cause"]).size().unstack(fill_value=0)
        pivot.plot(kind="bar", stacked=True, ax=axes[1], colormap="tab10", edgecolor="white")
        axes[1].set_title("Cause by Corridor")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].legend(loc="upper right", fontsize=7)

    plt.tight_layout(); save(fig, "04_spatial_hotspots.png")
    return report_lines


def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    if "event_cause" in df.columns:
        for cause in df["event_cause"].dropna().unique():
            numeric_df[f"cause_{cause}"] = (df["event_cause"] == cause).astype(int)
    if "priority" in df.columns:
        numeric_df["priority_num"] = df["priority"].map({"High":3,"Medium":2,"Low":1})
    if "is_weekend" in df.columns:
        numeric_df["is_weekend"] = df["is_weekend"].astype(int)
    if "requires_road_closure" in df.columns:
        numeric_df["requires_road_closure"] = df["requires_road_closure"].astype(float)

    desc_flag_cols = [c for c in df.columns if c.startswith("desc_") and
                      c not in ("desc_normalized","desc_original","desc_lang",
                                "desc_word_count","desc_char_count",
                                "desc_has_phone","desc_has_location","desc_topic")]
    for col in desc_flag_cols:
        if col in df.columns:
            numeric_df[col] = df[col].astype(float)

    cols = ["resolution_minutes","hour","day_of_week","month",
            "priority_num","requires_road_closure","is_weekend",
            "latitude","longitude","log_lag_minutes","desc_word_count"] \
         + [c for c in numeric_df.columns if c.startswith("cause_")] \
         + [c for c in numeric_df.columns if c.startswith("desc_") and
            c in numeric_df.columns and c != "desc_word_count"]

    cols = [c for c in cols if c in numeric_df.columns]
    sub  = numeric_df[cols].dropna(thresh=int(len(df) * 0.3))
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, ax=ax, cmap="RdBu_r", center=0,
                annot=True, fmt=".2f", linewidths=0.5,
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.6})
    ax.set_title("Feature Correlation Matrix (includes description features)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); save(fig, "05_correlation_matrix.png")


def plot_vehicle_analysis(df):
    if "veh_type" not in df.columns: return
    vt = df["veh_type"].dropna()
    if vt.empty: return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Vehicle Type Analysis", fontsize=14, fontweight="bold")

    vc = vt.value_counts()
    axes[0].barh(vc.index, vc.values, color=PALETTE["coral"])
    axes[0].set_title("Breakdowns by Vehicle Type"); axes[0].set_xlabel("Count")

    if "resolution_minutes" in df.columns:
        rt = df.groupby("veh_type")["resolution_minutes"].median().sort_values()
        axes[1].barh(rt.index, rt.values, color=PALETTE["amber"])
        axes[1].set_title("Median Resolution by Vehicle Type"); axes[1].set_xlabel("Minutes")

    plt.tight_layout(); save(fig, "06_vehicle_analysis.png")


# ─────────────────────────────────────────────
# 6. REPORTS
# ─────────────────────────────────────────────

def description_nlp_report(df, topic_labels, report_lines):
    report_lines += ["", "=" * 60, "DESCRIPTION NLP ANALYSIS", "=" * 60]

    if "desc_lang" in df.columns:
        report_lines.append("\nLanguage distribution:")
        for lang, cnt in df["desc_lang"].value_counts().items():
            report_lines.append(f"  {lang:<12} {cnt:>5}  ({cnt/len(df)*100:.1f}%)")

    if "desc_word_count" in df.columns:
        wc = df[df["desc_word_count"] > 0]["desc_word_count"]
        report_lines += ["",
                         f"Word count (non-empty descriptions):",
                         f"  Mean   : {wc.mean():.1f}",
                         f"  Median : {wc.median():.1f}",
                         f"  Max    : {wc.max()}"]

    if "desc_normalized" in df.columns:
        all_words = " ".join(df["desc_normalized"].dropna()).split()
        stop = {"the","a","an","is","in","at","on","of","and","to","for",
                "that","this","it","be","with","from","are","was","were",
                "has","have","not","no",""}
        top_words = Counter(w for w in all_words if w not in stop and len(w) > 2).most_common(20)
        report_lines += ["", "Top 20 words in descriptions (after normalization):"]
        for word, cnt in top_words:
            report_lines.append(f"  {word:<25} {cnt:>5}")

    if topic_labels:
        report_lines += ["", "Topic clusters (TF-IDF + KMeans):"]
        for tid, terms in topic_labels.items():
            n = (df["desc_topic"] == tid).sum()
            report_lines.append(f"  T{tid} ({n} incidents): {terms}")

    flag_cols = [c for c in df.columns if c.startswith("desc_") and
                 c not in ("desc_normalized","desc_original","desc_lang","desc_lang_label",
                           "desc_word_count","desc_char_count",
                           "desc_has_phone","desc_has_location","desc_topic")
                 and pd.api.types.is_numeric_dtype(df[c])]
    if flag_cols:
        report_lines += ["", "Keyword flag hit rates:"]
        for col in flag_cols:
            rate = df[col].mean() * 100
            report_lines.append(f"  {col.replace('desc_',''):<20} {rate:>5.1f}% of incidents")

    return report_lines


def ml_readiness_report(df, report_lines):
    report_lines += ["", "=" * 60, "ML READINESS ASSESSMENT", "=" * 60]

    key_features = {
        "resolution_minutes"   : "Target / reward proxy",
        "event_cause"          : "Feature — incident type",
        "hour"                 : "Feature — time of day",
        "day_of_week"          : "Feature — weekday",
        "latitude"             : "Feature — location",
        "longitude"            : "Feature — location",
        "priority"             : "Feature — severity",
        "corridor"             : "Feature — road importance",
        "zone"                 : "Feature — geography",
        "junction"             : "Feature — geography",
        "requires_road_closure": "Feature — impact flag",
        "veh_type"             : "Feature — vehicle type",
        "desc_word_count"      : "Feature — description length",
        "desc_lang"            : "Feature — language of report",
        "desc_breakdown"       : "Feature — NLP: breakdown mention",
        "desc_traffic_slow"    : "Feature — NLP: congestion mention",
        "desc_waterlogging"    : "Feature — NLP: flooding mention",
        "desc_road_block"      : "Feature — NLP: road block mention",
        "desc_severity_high"   : "Feature — NLP: severity signal",
        "desc_topic"           : "Feature — NLP: topic cluster",
    }

    report_lines.append(f"\n{'Feature':<30} {'Role':<35} {'Fill %':>8}")
    report_lines.append("-" * 78)
    for feat, role in key_features.items():
        if feat in df.columns:
            fill = (1 - df[feat].isna().mean()) * 100
            flag = "✓" if fill > 70 else ("△" if fill > 30 else "✗")
            report_lines.append(f"  {feat:<28} {role:<35} {fill:>6.1f}%  {flag}")
        else:
            report_lines.append(f"  {feat:<28} {role:<35}   NOT FOUND  ✗")

    report_lines += ["",
                     "Missing for RL (action columns):",
                     "  ✗  officers_deployed      — not in dataset",
                     "  ✗  diversion_route_used   — not in dataset",
                     "  ✗  barricades_placed      — not in dataset",
                     "",
                     "Recommendation:",
                     "  1. Use description NLP flags as extra features in a severity predictor.",
                     "  2. Use topic clusters to stratify synthetic data generation.",
                     "  3. Build action space from domain knowledge + synthetic data."]
    return report_lines


def synthetic_data_profile(df, report_lines):
    report_lines += ["", "=" * 60, "SYNTHETIC DATA GENERATION PROFILE", "=" * 60,
                     "Statistical profiles to drive realistic synthetic data:\n"]

    if "event_cause" in df.columns:
        report_lines.append("event_cause probabilities:")
        for v, p in (df["event_cause"].value_counts(normalize=True) * 100).round(1).items():
            report_lines.append(f"  {v:<25} {p}%")

    if "resolution_minutes" in df.columns:
        rt = df["resolution_minutes"].dropna()
        if len(rt) > 5:
            log_rt = np.log(rt[rt > 0])
            mu, sigma = log_rt.mean(), log_rt.std()
            report_lines += ["",
                             "resolution_minutes — Log-normal fit:",
                             f"  log-mean={mu:.3f}, log-std={sigma:.3f}",
                             f"  Sample: np.random.lognormal({mu:.3f}, {sigma:.3f})"]

    if "priority" in df.columns:
        report_lines.append("\npriority probabilities:")
        for v, p in (df["priority"].value_counts(normalize=True) * 100).round(1).items():
            report_lines.append(f"  {v:<10} {p}%")

    if "desc_lang" in df.columns:
        report_lines.append("\ndescription language probabilities:")
        for v, p in (df["desc_lang"].value_counts(normalize=True) * 100).round(1).items():
            report_lines.append(f"  {v:<10} {p}%")

    return report_lines


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def main():
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    filepath = sys.argv[1] if len(sys.argv) > 1 else "traffic_data.csv"

    print(f"Loading data from: {filepath}")
    df = load_data(filepath)
    print(f"  Loaded {len(df)} rows × {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    print("Engineering time features...")
    df = engineer_features(df)

    print("\n── Description NLP pipeline ──")
    df = extract_description_features(df)
    df, topic_labels = cluster_descriptions(df, n_clusters=6)

    report_lines = ["BENGALURU TRAFFIC INCIDENT — EDA REPORT",
                    f"Generated from {len(df)} records", ""]

    print("\n── Generating reports & plots ──")
    report_lines = data_quality_report(df, report_lines)

    print("Plotting incident overview...")
    plot_incident_overview(df)

    print("Plotting temporal patterns...")
    plot_temporal_patterns(df)

    print("Plotting resolution time...")
    plot_resolution_time(df)

    print("Plotting spatial hotspots...")
    report_lines = plot_spatial_hotspots(df, report_lines)

    print("Plotting correlation matrix (with NLP features)...")
    plot_correlation_matrix(df)

    print("Plotting vehicle analysis...")
    plot_vehicle_analysis(df)

    print("Plotting description language analysis...")
    plot_description_language(df)

    print("Plotting keyword flags heatmap...")
    plot_keyword_flags(df)

    print("Plotting keyword vs resolution time...")
    plot_keyword_vs_resolution(df)

    print("Plotting topic clusters...")
    plot_topic_clusters(df, topic_labels)

    print("Generating word cloud...")
    plot_wordcloud(df)

    print("Plotting language per event cause...")
    plot_description_vs_cause(df)

    report_lines = description_nlp_report(df, topic_labels, report_lines)
    report_lines = ml_readiness_report(df, report_lines)
    report_lines = synthetic_data_profile(df, report_lines)

    enriched_path = os.path.join(OUTPUT_DIR, "enriched_data.csv")
    export_cols   = [c for c in df.columns if c != "desc_original"]
    df[export_cols].to_csv(enriched_path, index=False)
    print(f"  Saved enriched dataset: enriched_data.csv")

    report_path = os.path.join(OUTPUT_DIR, "eda_summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # Print the full text report to stdout
    print("\n" + "=" * 60)
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
