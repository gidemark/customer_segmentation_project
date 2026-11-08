# 03_clustering_model.py
# WEEK 3 - Clustering with MLlib (KMeans)
# Goal: Train KMeans for k=2..8, evaluate with silhouette (Elbow method),
#       choose optimal K, save model, assign cluster labels, compute stats.

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spark_env import create_spark_session
from pyspark.sql.functions import col, avg, count, lit, udf, sum as spark_sum
from pyspark.sql.types import DoubleType
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE, "data", "processed")
MODELS = os.path.join(BASE, "models")
OUTPUTS = os.path.join(BASE, "outputs")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

spark = create_spark_session("CustomerSegmentation-Week3")
spark.sparkContext.setLogLevel("WARN")

# 1. Load feature dataset (scaled RFM vectors)
df = spark.read.parquet(os.path.join(PROCESSED, "rfm_features.parquet"))
print("Feature dataset count:", df.count())


def compute_wss(predictions, model):
    """Compute Within-Cluster Sum of Squares from centroids + predictions."""
    centers = model.clusterCenters()  # list of DenseVectors, one per cluster
    n_clusters = len(centers)
    # Broadcast centroids to workers
    bc_centers = spark.sparkContext.broadcast(centers)

    def sq_dist(vec, cluster_id):
        if vec is None or cluster_id is None:
            return 0.0
        c = bc_centers.value[int(cluster_id)]
        return float(sum((float(x) - float(y)) ** 2 for x, y in zip(vec, c)))

    sq_dist_udf = udf(sq_dist, DoubleType())
    wss = predictions.withColumn("sq_dist", sq_dist_udf(col("features"), col("prediction"))) \
        .agg(spark_sum("sq_dist").alias("wss")) \
        .collect()[0]["wss"]
    bc_centers.unpersist()
    return float(wss) if wss is not None else float("nan")


# 2. Train KMeans for k = 2..8, compute Silhouette score (Elbow method)
# Silhouette measures how similar a point is to its own cluster vs others
# (higher is better, range -1..1). WSS is the within-cluster sum of squares.
results = []
for k in range(2, 9):
    km = KMeans(k=k, seed=1, featuresCol="features")
    model = km.fit(df)
    predictions = model.transform(df)
    silhouette = ClusteringEvaluator(
        predictionCol="prediction", featuresCol="features", metricName="silhouette"
    ).evaluate(predictions)
    wss = compute_wss(predictions, model)
    results.append((k, silhouette, wss))
    print(f"k={k}  silhouette={silhouette:.4f}  WSS={wss:.2f}")

# 3. Choose optimal K
# Primary metric: silhouette (higher = better separation).
# Business constraint: reject solutions where the smallest cluster is < 5%
# of customers (imbalanced splits are not actionable for marketing).
# Among the near-best solutions (silhouette >= 85% of the best), prefer the
# LARGEST K: more segments = more targeted marketing campaigns, provided
# the partition stays balanced and the fit is still good.
MIN_CLUSTER_FRACTION = 0.05
total_customers = df.count()
best_sil = max(sil for _, sil, _ in results)

size_by_k = {}
for k in range(2, 9):
    km = KMeans(k=k, seed=1, featuresCol="features")
    m = km.fit(df)
    sizes = [r["count"] for r in m.transform(df).groupBy("prediction").count().collect()]
    size_by_k[k] = min(sizes) / total_customers

candidates = [(k, sil, wss) for (k, sil, wss) in results
              if sil >= 0.85 * best_sil and size_by_k[k] >= MIN_CLUSTER_FRACTION]
if not candidates:
    candidates = [(k, sil, wss) for (k, sil, wss) in results
                  if size_by_k[k] >= MIN_CLUSTER_FRACTION] or results

# Prefer the largest K among the near-best, balanced candidates
best_k, best_sil, best_wss = max(candidates, key=lambda x: (x[0], x[1]))
print(f"\n=== OPTIMAL K = {best_k} (silhouette={best_sil:.4f}, "
      f"min cluster fraction={size_by_k[best_k]:.3f}) ===")

pd.DataFrame(results, columns=["k", "silhouette", "wss"]).to_csv(
    os.path.join(OUTPUTS, "elbow_results.csv"), index=False
)
final_kmeans = KMeans(k=best_k, seed=1, featuresCol="features")
final_model = final_kmeans.fit(df)
final_model.write().overwrite().save(os.path.join(MODELS, "kmeans_model"))

clustered = final_model.transform(df)

print("=== CLUSTER SIZES ===")
clustered.groupBy("prediction").count().orderBy("prediction").show(truncate=False)

# 5. Compute cluster statistics for interpretation (means of raw features per cluster)
feature_cols = ["Recency", "Frequency", "Monetary",
                "AvgBasketSize", "ProductDiversity", "TotalQuantity"]

stats = clustered.groupBy("prediction").agg(
    *[avg(c).alias(f"avg_{c}") for c in feature_cols],
    count(lit(1)).alias("cluster_size")
).orderBy("prediction")

print("=== CLUSTER MEANS (raw scale) ===")
stats.show(truncate=False)

# Save cluster assignments and cluster summary table
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csv_util import write_single_csv

clustered.select("Customer ID", "prediction").write.mode("overwrite").parquet(
    os.path.join(PROCESSED, "cluster_assignments.parquet")
)

write_single_csv(stats, os.path.join(OUTPUTS, "cluster_summary.csv"))

print("Model saved to models/kmeans_model")
print("Cluster summary saved to outputs/cluster_summary.csv")
spark.stop()