import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "customer_data.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Select behavioral and demographic features for clustering.
features = [
    "age",
    "annual_income_k",
    "spending_score",
    "purchase_frequency_monthly",
    "avg_order_value",
    "tenure_months",
]
X = df[features].copy()

# Standardize features so variables with larger units do not dominate.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means groups customers into four behavior-based segments.
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["segment"] = kmeans.fit_predict(X_scaled)

# Save customer-level segment assignments.
df.to_csv(OUT / "customer_segments.csv", index=False)

# Summarize each segment to support interpretation.
profile = df.groupby("segment").agg(
    customer_count=("customer_id", "count"),
    average_age=("age", "mean"),
    average_annual_income_k=("annual_income_k", "mean"),
    average_spending_score=("spending_score", "mean"),
    average_monthly_purchases=("purchase_frequency_monthly", "mean"),
    average_order_value=("avg_order_value", "mean"),
    average_tenure_months=("tenure_months", "mean"),
).round(2)
profile.to_csv(OUT / "segment_profiles.csv")

# Save overall model quality metric.
silhouette = silhouette_score(X_scaled, df["segment"])
pd.DataFrame(
    {"metric": ["silhouette_score"], "value": [round(silhouette, 4)]}
).to_csv(OUT / "model_evaluation.csv", index=False)

# Chart 1: income versus spending score.
plt.figure(figsize=(9, 6))
for segment_id, group in df.groupby("segment"):
    plt.scatter(
        group["annual_income_k"],
        group["spending_score"],
        alpha=0.7,
        label=f"Segment {segment_id}",
    )
plt.xlabel("Annual Income (thousands)")
plt.ylabel("Spending Score (1–100)")
plt.title("Customer Segmentation: Income vs Spending")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUT / "customer_segments_scatter.png", dpi=160)
plt.close()

# Chart 2: number of customers in each segment.
counts = df["segment"].value_counts().sort_index()
plt.figure(figsize=(8, 5))
counts.plot(kind="bar")
plt.xlabel("Segment")
plt.ylabel("Number of Customers")
plt.title("Customer Count by Segment")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT / "segment_customer_counts.png", dpi=160)
plt.close()

print("Customer segmentation completed.")
print(f"Customers analyzed: {len(df)}")
print(f"Number of segments: {df['segment'].nunique()}")
print(f"Silhouette score: {silhouette:.4f}")
print(profile.to_string())
