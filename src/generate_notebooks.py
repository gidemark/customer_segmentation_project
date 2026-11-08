"""Generate the 5 Jupyter notebooks for the customer segmentation project."""
import os
import nbformat as nbf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "src")
NOTEBOOKS = os.path.join(BASE, "notebooks")
os.makedirs(NOTEBOOKS, exist_ok=True)

# Common header prepended to every notebook
HEADER = """# Customer Segmentation at Scale Using Spark
# Jupyter notebook wrapper for the weekly pipeline scripts.
# Run cells in order. Each notebook loads the outputs of the previous week.
#
# NOTE: On Windows, Spark needs the Hadoop native IO workaround.
# Run `python src\\setup_env.py` once in the same shell, or execute the
# setup cell below before any Spark operation.
"""

def read_src(name):
    with open(os.path.join(SRC, name), "r", encoding="utf-8") as fh:
        return fh.read()

def setup_cell():
    return nbf.v4.new_code_cell(
        "import os, sys\\n"
        "sys.path.insert(0, os.path.join(os.getcwd(), 'src'))\\n"
        "from spark_env import setup_spark_env\\n"
        "setup_spark_env()\\n"
        "print('Spark Windows environment configured.')"
    )

def make_notebook(title, script_name, extra_intro=""):
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
    }
    nb["metadata"]["language_info"] = {"name": "python"}
    cells = [nbf.v4.new_markdown_cell(HEADER + f"\n## {title}\n{extra_intro}"),
             setup_cell(),
             nbf.v4.new_code_cell(read_src(script_name))]
    nb["cells"] = cells
    out = os.path.join(NOTEBOOKS, script_name.replace(".py", ".ipynb"))
    nbf.write(nb, out)
    print("Wrote", out)

make_notebook("Week 1 - Data Understanding & Exploration",
              "01_data_cleaning.py",
              "Loads the raw Online Retail II dataset into Spark, inspects the "
              "schema, handles missing/invalid values, and produces exploratory "
              "statistics (customers, transactions, revenue).")
make_notebook("Week 2 - Feature Engineering (RFM)",
              "02_feature_engineering.py",
              "Builds the RFM customer feature matrix (Recency, Frequency, Monetary "
              "plus basket size, diversity, volume), log-transforms heavy-tailed "
              "features, and standardizes with StandardScaler.")
make_notebook("Week 3 - Clustering with MLlib (KMeans)",
              "03_clustering_model.py",
              "Trains KMeans for k=2..8, evaluates with silhouette (Elbow method), "
              "selects the optimal K, saves the model, and assigns cluster labels.")
make_notebook("Week 4 - Cluster Interpretation & Business Strategy",
              "04_cluster_analysis.py",
              "Profiles each cluster, assigns business names (High-Value Loyal, "
              "Churn Risk, etc.), and writes the final segment table.")
make_notebook("Week 4 supplement - Visualizations",
              "05_visualization.py",
              "Generates PNG charts (elbow plot, cluster size, RFM profile, revenue "
              "share) for the final report.")
print("Done.")