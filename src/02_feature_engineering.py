# 02_feature_engineering.py
# WEEK 2 - Feature Engineering (RFM + extras)
# Build the customer feature matrix used for clustering.
# Core idea: RFM segmentation
#   R = Recency    (days since last purchase)
#   F = Frequency  (number of distinct invoices)
#   M = Monetary   (total spend)
# Plus: average basket size, average spend per order, product diversity,
#       total quantity purchased.
# Finally: assemble into a feature vector and standardize with StandardScaler.

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spark_env import create_spark_session
from csv_util import write_single_csv
from pyspark.sql.functions import (
    col, sum as spark_sum, countDistinct, max as spark_max,
    datediff, lit, log1p
)
from pyspark.ml.feature import VectorAssembler, StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE, "data", "processed")
OUTPUTS = os.path.join(BASE, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

spark = create_spark_session("CustomerSegmentation-Week2")
spark.sparkContext.setLogLevel("WARN")

# 1. Load cleaned data
df = spark.read.parquet(os.path.join(PROCESSED, "cleaned.parquet"))

# Reference date (max invoice date) so Recency is consistent for all customers
ref_date_row = df.agg(spark_max("InvoiceDateTS")).collect()[0][0]
print(f"Reference date (max invoice): {ref_date_row}")

# 2. RFM feature engineering per customer
rfm = df.groupBy("Customer ID").agg(
    datediff(lit(ref_date_row), spark_max("InvoiceDateTS")).alias("Recency"),
    countDistinct("Invoice").alias("Frequency"),
    spark_sum("TotalAmount").alias("Monetary")
)

# Average basket size (total spend / number of invoices)
rfm = rfm.withColumn("AvgBasketSize", col("Monetary") / col("Frequency"))

# Average spend per order (alias of AvgBasketSize, kept explicit for clarity)
rfm = rfm.withColumn("AvgSpendPerOrder", col("AvgBasketSize"))

# Product diversity = distinct products purchased
diversity = df.groupBy("Customer ID").agg(
    countDistinct("StockCode").alias("ProductDiversity")
)
rfm = rfm.join(diversity, on="Customer ID", how="left")

# Total quantity purchased (proxy for volume)
volume = df.groupBy("Customer ID").agg(
    spark_sum("Quantity").alias("TotalQuantity")
)
rfm = rfm.join(volume, on="Customer ID", how="left")

# Fill any nulls (customers with no diversity/quantity shouldn't exist, but safe)
rfm = rfm.na.fill(0, ["ProductDiversity", "TotalQuantity"])

print("=== RFM FEATURE SAMPLE ===")
rfm.show(10, truncate=False)
print("Feature row count:", rfm.count())

# 3. Log-transform heavy-tailed features
# Monetary, AvgBasketSize and TotalQuantity have long tails (a few customers
# spend orders of magnitude more than the median). log1p compresses the tail
# and makes the distribution more Gaussian, which helps distance-based KMeans
# (outliers otherwise dominate the objective).
rfm = rfm.withColumn("LogMonetary", log1p(col("Monetary")))
rfm = rfm.withColumn("LogAvgBasket", log1p(col("AvgBasketSize")))
rfm = rfm.withColumn("LogTotalQty", log1p(col("TotalQuantity")))

# 4. Normalize features (StandardScaler) and assemble feature vector
# StandardScaler: zero mean, unit variance -> KMeans works best on scaled data
feature_cols = ["Recency", "Frequency", "LogMonetary",
                "LogAvgBasket", "AvgSpendPerOrder", "ProductDiversity", "LogTotalQty"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
rfm_assembled = assembler.transform(rfm)

scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                        withStd=True, withMean=True)
scaler_model = scaler.fit(rfm_assembled)
rfm_scaled = scaler_model.transform(rfm_assembled)

print("=== SCALED FEATURE VECTOR SAMPLE ===")
rfm_scaled.select("Customer ID", "features").show(5, truncate=False)

# 5. Save final feature dataset for clustering (parquet)
# Also save a single CSV copy for inspection / reporting
rfm_scaled.write.mode("overwrite").parquet(os.path.join(PROCESSED, "rfm_features.parquet"))

write_single_csv(
    rfm_scaled.select("Customer ID", "Recency", "Frequency", "Monetary",
                      "AvgBasketSize", "ProductDiversity", "TotalQuantity"),
    os.path.join(OUTPUTS, "rfm_features.csv")
)

print("RFM feature dataset saved to data/processed/rfm_features.parquet")
spark.stop()