import pandas as pd 

df = pd.read_csv("data/processed/customer_predictions.csv")

def assign_action(row):
    if row["churn_probability"]>=0.75 and row["predicted_clv"]>=500:
        return "High-value win-back offer"
    elif row["churn_probability"]>=0.75:
        return "Win-back email campaign"
    elif row["predicted_clv"]>=800:
        return "VIP loyalty campaign"
    elif row["frequency"]>=2:
        return "Cross-sell recommendation"
    else:
        return "Nurture email sequence"
    
df["next_best_action"] = df.apply(assign_action,axis=1)
df.to_csv("data/processed/customer_predictions.csv",index=False)

print("Next best actions added successfully")
print(df["next_best_action"].value_counts())