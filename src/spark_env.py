# spark_env.py
# Helper to configure Spark for Windows (Hadoop native IO workaround).
# Import and call setup_spark_env() before creating a SparkSession.
import os

def setup_spark_env():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hadoop_bin = os.path.join(base, "hadoop-bin", "bin")
    if os.path.isdir(hadoop_bin):
        os.environ["HADOOP_HOME"] = os.path.join(base, "hadoop-bin")
        os.environ["PATH"] = os.environ.get("PATH", "") + ";" + hadoop_bin
    os.environ.setdefault("PYSPARK_PYTHON", "python")
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", "python")
    return hadoop_bin

def create_spark_session(app_name, partitions=8):
    from pyspark.sql import SparkSession
    setup_spark_env()
    return SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", str(partitions)) \
        .getOrCreate()