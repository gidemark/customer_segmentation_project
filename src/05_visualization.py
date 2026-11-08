# 05_visualization.py
# WEEK 4 supplement - Visualizations for the final report.
# Generates PNG charts from the clustering outputs:
#   - Elbow plot (silhouette + WSS vs k)
#   - Cluster size bar chart
#   - RFM radar / grouped bar chart
#   - Revenue pie chart by segment
# Outputs are written to outputs/ and embedded in the PDF report.

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS = os.path.join(BASE, "outputs")
REPORT = os.path.join(BASE, "report")
os.makedirs(OUTPUTS, exist_ok=True)
os.makedirs(REPORT, exist_ok=True)

# Business-friendly colors for the 4 segments
SEGMENT_COLORS = {
    "High-Value Loyal": "#2E86AB",
    "Frequent Low Spender": "#A23B72",
    "Occasional Buyers": "#F18F01",
    "Churn Risk / Inactive": "#C73E1D",
    "At-Risk High Value": "#3B1F2B",
}
DEFAULT_COLOR = "#666666"

# ---------------------------------------------------------------------------
# 1. Elbow plot: silhouette score + WSS vs k
# ---------------------------------------------------------------------------
elbow = pd.read_csv(os.path.join(OUTPUTS, "elbow_results.csv"))

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(elbow["k"], elbow["silhouette"], "o-", color="#2E86AB", label="Silhouette")
ax1.set_xlabel("k (number of clusters)")
ax1.set_ylabel("Silhouette score", color="#2E86AB")
ax1.tick_params(axis="y", labelcolor="#2E86AB")
ax1.set_xticks(elbow["k"])

ax2 = ax1.twinx()
ax2.plot(elbow["k"], elbow["wss"], "s--", color="#C73E1D", label="WSS (inertia)")
ax2.set_ylabel("Within-cluster sum of squares", color="#C73E1D")
ax2.tick_params(axis="y", labelcolor="#C73E1D")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
ax1.set_title("Elbow Method - Silhouette Score and WSS vs k")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUTS, "elbow_plot.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 2. Cluster size bar chart
# ---------------------------------------------------------------------------
segments = pd.read_csv(os.path.join(OUTPUTS, "final_segments.csv"))
segments["color"] = segments["business_name"].map(SEGMENT_COLORS).fillna(DEFAULT_COLOR)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(segments["business_name"], segments["num_customers"],
              color=segments["color"], edgecolor="white")
for bar, n in zip(bars, segments["num_customers"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
            f"{n:,}", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Number of customers")
ax.set_title("Customer Count per Segment")
ax.set_ylim(0, segments["num_customers"].max() * 1.15)
plt.xticks(rotation=20, ha="right")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUTS, "cluster_size.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. RFM grouped bar chart (normalized per feature for readability)
# ---------------------------------------------------------------------------
rfm_cols = ["avg_Recency", "avg_Frequency", "avg_Monetary"]
labels = ["Recency (days)", "Frequency (orders)", "Monetary ($)"]
data = segments[rfm_cols].values
# Normalize each feature to 0-1 for side-by-side comparison
data_norm = data / data.max(axis=0)

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(labels))
width = 0.2
for i, (_, row) in enumerate(segments.iterrows()):
    color = row["color"]
    ax.bar(x + i * width, data_norm[i], width, label=row["business_name"], color=color, edgecolor="white")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(labels)
ax.set_ylabel("Normalized value (0-1)")
ax.set_title("RFM Profile per Segment (normalized)")
ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.15))
fig.tight_layout()
fig.savefig(os.path.join(OUTPUTS, "rfm_profile.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Revenue share pie chart by segment
# ---------------------------------------------------------------------------
segments["revenue"] = segments["num_customers"] * segments["avg_Monetary"]
total_rev = segments["revenue"].sum()

fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    segments["revenue"], labels=segments["business_name"], autopct="%1.1f%%",
    colors=segments["color"], startangle=90, pctdistance=0.75,
    wedgeprops=dict(edgecolor="white")
)
for t in autotexts:
    t.set_fontsize(9)
ax.set_title(f"Revenue Share by Segment\n(Total: ${total_rev:,.0f})")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUTS, "revenue_share.png"), dpi=150)
plt.close(fig)

print("Visualizations saved to outputs/: elbow_plot.png, cluster_size.png, rfm_profile.png, revenue_share.png")