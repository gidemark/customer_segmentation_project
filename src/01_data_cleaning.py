# 01_data_cleaning.py
# WEEK 1 - Data Understanding & Exploration
# Load raw data into Spark, inspect schema, handle missing/invalid values,
# and produce exploratory statistics (customers, transactions, revenue).

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spark_env import create_spark_session
from pyspark.sql.functions import col, sum as spark_sum, avg, min, max

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
PROCESSED = os.path.join(BASE, "data", "processed")
os.makedirs(PROCESSED, exist_ok=True)

spark = create_spark_session("CustomerSegmentation-Week1")
spark.sparkContext.setLogLevel("WARN")

# 1. Load dataset into Spark (header + inferSchema)
df = spark.read.option("header", "true") \
    .option("inferSchema", "true") \
    .csv(os.path.join(RAW, "online_retail_II.csv"))

print("=== RAW DATA SCHEMA ===")
df.printSchema()

# 2. Inspect data quality: row count, missing values, duplicate invoices
print("=== ROW COUNT ===")
print(f"Total rows: {df.count()}")

print("=== MISSING VALUES PER COLUMN ===")
for c in df.columns:
    nulls = df.filter(col(c).isNull()).count()
    print(f"  {c}: {nulls} nulls")

print("=== DUPLICATE INVOICES ===")
print(f"Distinct invoices: {df.select('Invoice').distinct().count()}")

# 3. Handle missing / bad data:
#    - drop rows with null CustomerID (cannot segment without a customer)
#    - drop zero/negative quantity (returns/corrections) and zero/negative price
#    - add TotalAmount = Quantity * Price
#    - parse InvoiceDate to timestamp, drop parse failures
#    - drop empty descriptions
df_clean = df.filter(col("Customer ID").isNotNull())
df_clean = df_clean.filter(col("Quantity") > 0)
df_clean = df_clean.filter(col("Price") > 0)
df_clean = df_clean.withColumn(
    "TotalAmount",
    col("Quantity").cast("double") * col("Price").cast("double")
)
df_clean = df_clean.withColumn("InvoiceDateTS", col("InvoiceDate").cast("timestamp"))
df_clean = df_clean.filter(col("InvoiceDateTS").isNotNull())
df_clean = df_clean.filter(col("Description").isNotNull() & (col("Description") != ""))

print("=== CLEANED ROW COUNT ===")
print(f"Cleaned rows: {df_clean.count()}")

# 4. Exploratory statistics
print("=== NUMBER OF CUSTOMERS ===")
print(f"Distinct customers: {df_clean.select('Customer ID').distinct().count()}")

print("=== NUMBER OF TRANSACTIONS ===")
print(f"Distinct transactions (invoices): {df_clean.select('Invoice').distinct().count()}")

print("=== REVENUE DISTRIBUTION ===")
rev = df_clean.agg(
    spark_sum("TotalAmount").alias("total_revenue"),
    avg("TotalAmount").alias("avg_revenue_per_line"),
    min("TotalAmount").alias("min_line_revenue"),
    max("TotalAmount").alias("max_line_revenue")
).collect()[0]
print(f"Total revenue: {rev['total_revenue']:.2f}")
print(f"Avg line revenue: {rev['avg_revenue_per_line']:.2f}")
print(f"Min line revenue: {rev['min_line_revenue']:.2f}")
print(f"Max line revenue: {rev['max_line_revenue']:.2f}")

print("=== TOP 10 COUNTRIES BY REVENUE ===")
df_clean.groupBy("Country") \
    .agg(spark_sum("TotalAmount").alias("revenue")) \
    .orderBy(col("revenue").desc()) \
    .show(10, truncate=False)

print("=== TOP 10 CUSTOMERS BY SPEND ===")
df_clean.groupBy("Customer ID") \
    .agg(spark_sum("TotalAmount").alias("total_spend")) \
    .orderBy(col("total_spend").desc()) \
    .show(10, truncate=False)

print("=== QUANTITY DISTRIBUTION ===")
df_clean.select("Quantity").describe().show()

# 5. Save cleaned data for next week (parquet = fast, columnar, distributed)
df_clean.write.mode("overwrite").parquet(os.path.join(PROCESSED, "cleaned.parquet"))
print("Cleaned data saved to data/processed/cleaned.parquet")

spark.stop()