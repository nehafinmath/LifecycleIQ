import pandas as pd
import joblib

df = pd.read_csv("data/processed/customer_segments.csv")

features = [
    "recency",
    "frequency",
    "monetary",
    "avg_order_value",
    "avg_review_score",
    "avg_delivery_days",
    "total_items",
    "avg_installments"
]

churn_model = joblib.load("models/churn_model.pkl")
clv_model = joblib.load("models/clv_model.pkl")

churn_importance = pd.DataFrame({
    "feature":features,
    "importance":churn_model.feature_importances_,
    "model":"Churn Model"
})
clv_importance = pd.DataFrame({
    "feature":features,
    "importance":clv_model.feature_importances_,
    "model": "CLV Model"
})
importance_df = pd.concat([churn_importance,clv_importance],
                          ignore_index=False)

importance_df.to_csv("data/processed/model_feature_importance.csv", index=False)

print("model_feature_importance.csv saved successfully")
print(importance_df)