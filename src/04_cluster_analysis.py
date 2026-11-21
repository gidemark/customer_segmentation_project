# 04_cluster_analysis.py
# WEEK 4 - Cluster Interpretation & Business Strategy
# Profile each cluster, assign business names, generate visualizations,
# and produce actionable marketing recommendations (High-Value Loyal,
# Churn Risk, etc.).

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spark_env import create_spark_session
from csv_util import write_single_csv
from pyspark.sql.functions import col, avg, count, lit

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE, "data", "processed")
OUTPUTS = os.path.join(BASE, "outputs")
REPORT = os.path.join(BASE, "report")
os.makedirs(OUTPUTS, exist_ok=True)
os.makedirs(REPORT, exist_ok=True)

# Business-friendly colors for the segments
SEGMENT_COLORS = {
    "High-Value Loyal": "#2E86AB",
    "Frequent Low Spender": "#A23B72",
    "Occasional Buyers": "#F18F01",
    "Churn Risk / Inactive": "#C73E1D",
    "At-Risk High Value": "#3B1F2B",
}
DEFAULT_COLOR = "#666666"

spark = create_spark_session("CustomerSegmentation-Week4")
spark.sparkContext.setLogLevel("WARN")

# 1. Load cluster assignments + features
features = spark.read.parquet(os.path.join(PROCESSED, "rfm_features.parquet"))
assignments = spark.read.parquet(os.path.join(PROCESSED, "cluster_assignments.parquet"))
df = features.join(assignments, on="Customer ID", how="inner")

# 2. Profile each cluster (RFM means + size)
feature_cols = ["Recency", "Frequency", "Monetary",
                "AvgBasketSize", "ProductDiversity", "TotalQuantity"]
profile = df.groupBy("prediction").agg(
    *[avg(c).alias(f"avg_{c}") for c in feature_cols],
    count(lit(1)).alias("num_customers")
).orderBy("prediction")

print("=== CLUSTER PROFILE ===")
profile.show(truncate=False)

# 3. Business naming logic
pdf = profile.toPandas()
overall = df.select(
    avg("Recency").alias("r"), avg("Frequency").alias("f"), avg("Monetary").alias("m")
).collect()[0]
ref_r, ref_f, ref_m = overall["r"], overall["f"], overall["m"]
print(f"\nOverall means - Recency: {ref_r:.1f}, Frequency: {ref_f:.1f}, Monetary: {ref_m:.2f}")

def name_cluster(row):
    r = row["avg_Recency"]
    f = row["avg_Frequency"]
    m = row["avg_Monetary"]
    if m > ref_m and r < ref_r and f > ref_f:
        return "High-Value Loyal"
    if m > ref_m and r > ref_r:
        return "At-Risk High Value"
    if m < ref_m and f < ref_f and r > ref_r:
        return "Churn Risk / Inactive"
    if m < ref_m and f > ref_f:
        return "Frequent Low Spender"
    return "Occasional Buyers"

pdf["business_name"] = pdf.apply(name_cluster, axis=1)
pdf["color"] = pdf["business_name"].map(SEGMENT_COLORS).fillna(DEFAULT_COLOR)

print("\n=== BUSINESS SEGMENT NAMES ===")
for _, row in pdf.iterrows():
    print(f"Cluster {int(row['prediction'])}: {row['business_name']} "
          f"(size={int(row['num_customers'])}, M={row['avg_Monetary']:.1f}, "
          f"R={row['avg_Recency']:.1f}, F={row['avg_Frequency']:.1f})")

# 4. Save final segment table + markdown report
# `pdf` is a pandas DataFrame, so write it directly (single flat CSV).
final_csv = os.path.join(OUTPUTS, "final_segments.csv")
pdf.to_csv(final_csv, index=False)

with open(os.path.join(REPORT, "segment_summary.md"), "w") as fh:
    fh.write("# Customer Segmentation - Cluster Summary\n\n")
    fh.write("| Cluster | Business Name | Size | Avg Monetary | Avg Recency (days) | Avg Frequency |\n")
    fh.write("|---|---|---|---|---|---|\n")
    for _, row in pdf.iterrows():
        fh.write(f"| {int(row['prediction'])} | {row['business_name']} | "
                 f"{int(row['num_customers'])} | {row['avg_Monetary']:.1f} | "
                 f"{row['avg_Recency']:.1f} | {row['avg_Frequency']:.1f} |\n")

# ---------------------------------------------------------------------------
# 5. Visualizations (embedded in the notebook for the final report)
# ---------------------------------------------------------------------------
# 5a. Cluster size bar chart
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(pdf["business_name"], pdf["num_customers"],
              color=pdf["color"], edgecolor="white")
for bar, n in zip(bars, pdf["num_customers"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
            f"{n:,}", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Number of customers")
ax.set_title("Customer Count per Segment")
ax.set_ylim(0, pdf["num_customers"].max() * 1.15)
plt.xticks(rotation=20, ha="right")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUTS, "cluster_size.png"), dpi=150)
plt.close(fig)

# 5b. RFM grouped bar chart (normalized per feature for readability)
rfm_cols = ["avg_Recency", "avg_Frequency", "avg_Monetary"]
rfm_labels = ["Recency (days)", "Frequency (orders)", "Monetary ($)"]
data = pdf[rfm_cols].values
data_norm = data / data.max(axis=0)

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(rfm_labels))
width = 0.2
for i, (_, row) in enumerate(pdf.iterrows()):
    ax.bar(x + i * width, data_norm[i], width,
           label=row["business_name"], color=row["color"], edgecolor="white")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(rfm_labels)
ax.set_ylabel("Normalized value (0-1)")
ax.set_title("RFM Profile per Segment (normalized)")
ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.15))
fig.tight_layout()
fig.savefig(os.path.join(OUTPUTS, "rfm_profile.png"), dpi=150)
plt.close(fig)

# 5c. Revenue share pie chart by segment
pdf["revenue"] = pdf["num_customers"] * pdf["avg_Monetary"]
total_rev = pdf["revenue"].sum()

fig, ax = plt.subplots(figsize=(7, 7))
ax.pie(pdf["revenue"], labels=pdf["business_name"], autopct="%1.1f%%",
       colors=pdf["color"], startangle=90, pctdistance=0.75,
       wedgeprops=dict(edgecolor="white"))
ax.set_title(f"Revenue Share by Segment\n(Total: ${total_rev:,.0f})")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUTS, "revenue_share.png"), dpi=150)
plt.close(fig)

print("\nSaved final_segments.csv, report/segment_summary.md, "
      "and outputs/{cluster_size,rfm_profile,revenue_share}.png")
spark.stop()