import pandas as pd 
import numpy as np 
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, classification_report,
                             mean_absolute_error,
                             mean_squared_error,
                             r2_score)
from xgboost import XGBClassifier, XGBRegressor

df = pd.read_csv("data/processed/customer_features.csv")

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

X = df[features]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

churn_model = XGBClassifier(n_estimators=100,
                            max_depth=4,
                            learning_rate=0.05,
                            eval_metric="logloss",
                            random_state=42)

churn_model.fit(X_train,y_train)

pred_probs = churn_model.predict_proba(X_test)[:,1]
preds = churn_model.predict(X_test)

print("ROC AUC:")
print(roc_auc_score(y_test, pred_probs))

print(classification_report(y_test,preds))

joblib.dump(churn_model, "models/churn_model.pkl")

y_clv = df["monetary"]
X_train, X_test, y_train, y_test = train_test_split(X,y_clv, test_size=0.2,random_state=42)

clv_model = XGBRegressor(n_estimators=100,
                         max_depth=4,
                         learing_rate=0.05,
                         random_state=42)

clv_model.fit(X_train, y_train)
pred_clv = clv_model.predict(X_test)

print("MAE")
print(mean_absolute_error(y_test, pred_clv))

print("RMSE")
print(np.sqrt(mean_squared_error(y_test, pred_clv)))

print("R2")
print(r2_score(y_test, pred_clv))

joblib.dump(clv_model, "models/clv_model.pkl")

df["churn_probability"] = (churn_model.predict_proba(X)[:,1])
df["predicted_clv"] = (clv_model.predict(X))

df.to_csv("data/processed/customer_predictions.csv", index=False)
print("customer_predictions.csv saved")




