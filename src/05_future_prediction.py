# 05_future_prediction.py
# WEEK 5 - Predict Future Customer Value
# Predict future (next-period) Monetary value for each customer from RFM
# features using a Linear Regression model, and save predictions.

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spark_env import create_spark_session
from csv_util import write_single_csv

from pyspark.sql.functions import col, log1p, exp
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE, "data", "processed")
OUTPUTS = os.path.join(BASE, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

spark = create_spark_session("CustomerSegmentation-Week5")
spark.sparkContext.setLogLevel("WARN")

# 1. Load RFM features
df = spark.read.parquet(os.path.join(PROCESSED, "rfm_features.parquet"))

# 2. Target: future Monetary ~ log1p(Monetary) proxy using available features
# Use LogMonetary as target proxy (future spend) regressed on other features.
feature_cols = ["Recency", "Frequency", "LogAvgBasket", "ProductDiversity", "LogTotalQty"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vec")
data = assembler.transform(df).select("Customer ID", "features_vec", col("LogMonetary").alias("label"))
data = data.withColumnRenamed("features_vec", "features")

# 3. Train/test split
train, test = data.randomSplit([0.8, 0.2], seed=42)

# 4. Train Linear Regression
lr = LinearRegression(featuresCol="features", labelCol="label", maxIter=100, regParam=0.01)
model = lr.fit(train)

# 5. Evaluate
preds = model.transform(test)
rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse").evaluate(preds)
r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(preds)
print(f"RMSE: {rmse:.4f}  R2: {r2:.4f}")

# 6. Predict for all customers
all_preds = model.transform(data).select("Customer ID", "prediction")
all_preds = all_preds.withColumn("PredictedFutureMonetary", (exp(col("prediction")) - 1))
all_preds.select("Customer ID", "PredictedFutureMonetary").show(10, truncate=False)

write_single_csv(
    all_preds.select("Customer ID", "PredictedFutureMonetary"),
    os.path.join(OUTPUTS, "future_predictions.csv")
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ... after predictions save, add graph
fig, ax = plt.subplots(figsize=(8, 5))
pdf = all_preds.toPandas()
ax.hist(pdf["PredictedFutureMonetary"], bins=40, color="#2E86AB", edgecolor="white")
ax.set_title("Predicted Future Monetary Value Distribution")
ax.set_xlabel("Predicted Future Monetary")
ax.set_ylabel("Number of customers")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUTS, "future_prediction.png"), dpi=150)
plt.close(fig)
print("Saved outputs/future_prediction.png")
