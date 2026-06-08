import pandas as pd
import numpy as np


customers = pd.read_csv("data/raw/olist_customers_dataset.csv")
orders = pd.read_csv("data/raw/olist_orders_dataset.csv")
items = pd.read_csv("data/raw/olist_order_items_dataset.csv")
payments = pd.read_csv("data/raw/olist_order_payments_dataset.csv")
reviews = pd.read_csv("data/raw/olist_order_reviews_dataset.csv")

orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])
orders = orders[orders["order_status"] == "delivered"]

items_agg = (items.groupby("order_id").agg(order_value=("price","sum"),
                                          total_items=("order_item_id","count")
                                          ).reset_index())

payments_agg = (payments.groupby("order_id").agg(payment_value=("payment_value","sum"),
                                                 payment_installments=("payment_installments","mean"))
                                                 .reset_index())

reviews_agg = (reviews.groupby("order_id").agg(review_score=("review_score","mean")
                                               ).reset_index())

df = (orders.merge(customers,on="customer_id")
      .merge(items_agg,on="order_id")
      .merge(payments_agg,on="order_id")
      .merge(reviews_agg,on="order_id"))

df["delivery_days"] = (df["order_delivered_customer_date"]-df["order_purchase_timestamp"]).dt.days
snapshot_date = (df["order_purchase_timestamp"].max()+pd.Timedelta(days=1))
customer_features = (df.groupby("customer_unique_id").agg(
    recency = ("order_purchase_timestamp", lambda x: (snapshot_date - x.max()).days),
    frequency = ("order_id", "nunique"),
    monetary = ("payment_value","sum"),
    avg_order_value = ("payment_value","mean"),
    avg_review_score = ("review_score", "mean"),
    avg_delivery_days = ("delivery_days", "mean"),
    total_items = ("total_items","sum"),
    avg_installments = ("payment_installments","mean")
).reset_index())

customer_features["churn"] = (customer_features["recency"]>180).astype(int)
customer_features = customer_features.fillna(0)

customer_features.to_csv("data/processed/customer_features.csv",index=False)
print("customer_features.csv saved successfully")

