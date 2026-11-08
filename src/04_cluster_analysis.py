# 04_cluster_analysis.py
# WEEK 4 - Cluster Interpretation & Business Strategy
# Profile each cluster, assign business names, and produce actionable
# marketing recommendations (High-Value Loyal, Churn Risk, etc.).

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spark_env import create_spark_session
from csv_util import write_single_csv
from pyspark.sql.functions import col, avg, count, lit

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE, "data", "processed")
OUTPUTS = os.path.join(BASE, "outputs")
REPORT = os.path.join(BASE, "report")
os.makedirs(OUTPUTS, exist_ok=True)
os.makedirs(REPORT, exist_ok=True)

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

print("\n=== BUSINESS SEGMENT NAMES ===")
for _, row in pdf.iterrows():
    print(f"Cluster {int(row['prediction'])}: {row['business_name']} "
          f"(size={int(row['num_customers'])}, M={row['avg_Monetary']:.1f}, "
          f"R={row['avg_Recency']:.1f}, F={row['avg_Frequency']:.1f})")

# 4. Save final segment table + markdown report
write_single_csv(pdf, os.path.join(OUTPUTS, "final_segments.csv"))

with open(os.path.join(REPORT, "segment_summary.md"), "w") as fh:
    fh.write("# Customer Segmentation - Cluster Summary\n\n")
    fh.write("| Cluster | Business Name | Size | Avg Monetary | Avg Recency (days) | Avg Frequency |\n")
    fh.write("|---|---|---|---|---|---|\n")
    for _, row in pdf.iterrows():
        fh.write(f"| {int(row['prediction'])} | {row['business_name']} | "
                 f"{int(row['num_customers'])} | {row['avg_Monetary']:.1f} | "
                 f"{row['avg_Recency']:.1f} | {row['avg_Frequency']:.1f} |\n")

print("\nSaved final_segments.csv and report/segment_summary.md")
spark.stop()