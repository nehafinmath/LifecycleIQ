import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("data/processed/customer_predictions.csv")
rfm = df[["recency","frequency","monetary"]].copy()

scaler  = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

df["cluster"] = kmeans.fit_predict(rfm_scaled)
segment_map = {
    0:"At Risk",
    1:"New",
    2:"Loyal",
    3:"VIP"
}
df["customer_segment"] = df["cluster"].map(segment_map)
summary = (df.groupby("customer_segment")[[
    "recency",
    "frequency",
    "monetary",
    "predicted_clv",
    "churn_probability"
]].mean())

print(summary)

df.to_csv("data/processed/customer_segments.csv",index=False)

print("\ncustomer_segments.csv saved successfully")