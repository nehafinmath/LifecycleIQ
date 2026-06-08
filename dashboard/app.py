import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="LifecycleIQ",
    layout="wide"
)

df = pd.read_csv("data/processed/customer_predictions.csv")

st.title("LifecycleIQ: Customer Intelligence Platform")
st.write("Churn prediction, CLV estimation, and next best marketing action recommendations.")

total_customers = df["customer_unique_id"].nunique()
avg_clv = df["predicted_clv"].mean()
avg_churn_prob = df["churn_probability"].mean()
high_risk_customers = (df["churn_probability"] >= 0.75).sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Average Predicted CLV", f"R$ {avg_clv:,.2f}")
col3.metric("Average Churn Risk", f"{avg_churn_prob:.1%}")
col4.metric("High-Risk Customers", f"{high_risk_customers:,}")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Executive Overview",
        "Churn Risk",
        "CLV Analysis",
        "Next Best Action",
        "Campaign Simulator",
        "Customer Table"
    ]
)

if page == "Executive Overview":
    st.header("Executive Overview")

    fig1 = px.histogram(
        df,
        x="churn_probability",
        nbins=30,
        title="Distribution of Churn Probability"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(
        df,
        x="predicted_clv",
        nbins=30,
        title="Distribution of Predicted CLV"
    )
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Churn Risk":
    st.header("Top High-Risk Customers")

    top_risk = df.sort_values(
        "churn_probability",
        ascending=False
    ).head(100)

    st.dataframe(
        top_risk[
            [
                "customer_unique_id",
                "churn_probability",
                "predicted_clv",
                "recency",
                "frequency",
                "monetary",
                "next_best_action"
            ]
        ]
    )

elif page == "CLV Analysis":
    st.header("Top High-Value Customers")

    top_clv = df.sort_values(
        "predicted_clv",
        ascending=False
    ).head(100)

    st.dataframe(
        top_clv[
            [
                "customer_unique_id",
                "predicted_clv",
                "churn_probability",
                "frequency",
                "monetary",
                "avg_order_value",
                "next_best_action"
            ]
        ]
    )

elif page == "Next Best Action":
    st.header("Next Best Marketing Actions")

    action_counts = (
        df["next_best_action"]
        .value_counts()
        .reset_index()
    )

    action_counts.columns = ["Action", "Number of Customers"]

    fig = px.bar(
        action_counts,
        x="Action",
        y="Number of Customers",
        title="Recommended Marketing Actions"
    )

    st.plotly_chart(fig, use_container_width=True)

    selected_action = st.selectbox(
        "Select an action",
        df["next_best_action"].unique()
    )

    st.dataframe(
        df[df["next_best_action"] == selected_action].head(100)
    )

elif page == "Campaign Simulator":
    st.header("Campaign ROI Simulator")

    selected_action = st.selectbox(
        "Choose campaign audience",
        df["next_best_action"].unique()
    )

    expected_uplift = st.slider(
        "Expected retention uplift",
        min_value=0.0,
        max_value=0.30,
        value=0.05,
        step=0.01
    )

    cost_per_customer = st.number_input(
        "Campaign cost per customer",
        min_value=0.0,
        value=5.0
    )

    audience = df[df["next_best_action"] == selected_action]

    audience_size = len(audience)
    avg_audience_clv = audience["predicted_clv"].mean()

    expected_revenue = audience_size * avg_audience_clv * expected_uplift
    campaign_cost = audience_size * cost_per_customer

    if campaign_cost > 0:
        roi = (expected_revenue - campaign_cost) / campaign_cost
    else:
        roi = 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Audience Size", f"{audience_size:,}")
    c2.metric("Avg Audience CLV", f"R$ {avg_audience_clv:,.2f}")
    c3.metric("Expected Revenue", f"R$ {expected_revenue:,.2f}")
    c4.metric("Estimated ROI", f"{roi:.1%}")

elif page == "Customer Table":
    st.header("Customer-Level Predictions")

    st.dataframe(df)