import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu


st.set_page_config(
    page_title="LifecycleIQ",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def load_data():
    return pd.read_csv("data/processed/customer_predictions.csv")


df = load_data()

df["risk_segment"] = pd.cut(
    df["churn_probability"],
    bins=[0, 0.3, 0.7, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"],
    include_lowest=True
)

total_customers = df["customer_unique_id"].nunique()
avg_clv = df["predicted_clv"].mean()
avg_churn = df["churn_probability"].mean()
high_risk = (df["risk_segment"] == "High Risk").sum()


with st.sidebar:
    selected = option_menu(
        "LifecycleIQ",
        [
            "Overview",
            "Churn Risk",
            "CLV Analysis",
            "Next Best Action",
            "Campaign Simulator",
            "Customer Table"
        ],
        icons=[
            "speedometer2",
            "exclamation-triangle",
            "cash-coin",
            "lightning-charge",
            "graph-up-arrow",
            "table"
        ],
        menu_icon="bar-chart-line",
        default_index=0
    )


if selected == "Overview":
    st.title("LifecycleIQ: Customer Intelligence Platform")
    st.caption(
        "Churn prediction, CLV estimation, next-best-action recommendations, and campaign ROI simulation."
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Total Customers", f"{total_customers:,}")
    k2.metric("Average Predicted CLV", f"R$ {avg_clv:,.2f}")
    k3.metric("Average Churn Risk", f"{avg_churn:.1%}")
    k4.metric("High-Risk Customers", f"{high_risk:,}")

    st.divider()

    st.header("Executive Overview")

    left, right = st.columns(2)

    risk_counts = df["risk_segment"].value_counts().reset_index()
    risk_counts.columns = ["Risk Segment", "Customers"]

    with left:
        fig = px.pie(
            risk_counts,
            names="Risk Segment",
            values="Customers",
            hole=0.45,
            title="Customer Churn Risk Segments"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.histogram(
            df,
            x="predicted_clv",
            nbins=40,
            title="Predicted CLV Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Top 10 High-Risk Customers")
        top_risk = df.sort_values("churn_probability", ascending=False).head(10)

        st.dataframe(
            top_risk[
                [
                    "customer_unique_id",
                    "churn_probability",
                    "predicted_clv",
                    "next_best_action"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    with c2:
        st.subheader("Top 10 High-Value Customers")
        top_value = df.sort_values("predicted_clv", ascending=False).head(10)

        st.dataframe(
            top_value[
                [
                    "customer_unique_id",
                    "predicted_clv",
                    "churn_probability",
                    "next_best_action"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


elif selected == "Churn Risk":
    st.title("Churn Risk Explorer")
    st.caption("Identify customers most likely to churn and prioritize retention campaigns.")

    risk_filter = st.multiselect(
        "Filter by risk segment",
        ["Low Risk", "Medium Risk", "High Risk"],
        default=["High Risk"]
    )

    filtered = df[df["risk_segment"].isin(risk_filter)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Filtered Customers", f"{len(filtered):,}")
    c2.metric("Avg Churn Probability", f"{filtered['churn_probability'].mean():.1%}")
    c3.metric("Avg Predicted CLV", f"R$ {filtered['predicted_clv'].mean():,.2f}")

    st.divider()

    fig = px.histogram(
        filtered,
        x="churn_probability",
        nbins=30,
        title="Churn Probability Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("High-Risk Customer List")

    st.dataframe(
        filtered.sort_values("churn_probability", ascending=False)[
            [
                "customer_unique_id",
                "risk_segment",
                "churn_probability",
                "predicted_clv",
                "recency",
                "frequency",
                "next_best_action"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


elif selected == "CLV Analysis":
    st.title("Customer Lifetime Value Analysis")
    st.caption("Explore predicted customer value and identify high-value customer audiences.")

    top_value = df.sort_values("predicted_clv", ascending=False).head(100)

    c1, c2, c3 = st.columns(3)
    c1.metric("Average CLV", f"R$ {df['predicted_clv'].mean():,.2f}")
    c2.metric("Median CLV", f"R$ {df['predicted_clv'].median():,.2f}")
    c3.metric("Top 100 Avg CLV", f"R$ {top_value['predicted_clv'].mean():,.2f}")

    st.divider()

    fig = px.histogram(
        df,
        x="predicted_clv",
        nbins=50,
        title="Predicted Customer Lifetime Value Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top High-Value Customers")

    st.dataframe(
        top_value[
            [
                "customer_unique_id",
                "predicted_clv",
                "churn_probability",
                "frequency",
                "monetary",
                "avg_order_value",
                "next_best_action"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


elif selected == "Next Best Action":
    st.title("Next Best Action Engine")
    st.caption("Recommend customer-level marketing actions using churn risk and predicted CLV.")

    action_counts = df["next_best_action"].value_counts().reset_index()
    action_counts.columns = ["Action", "Customers"]

    fig = px.bar(
        action_counts,
        x="Customers",
        y="Action",
        orientation="h",
        title="Recommended Marketing Actions"
    )
    st.plotly_chart(fig, use_container_width=True)

    selected_action = st.selectbox(
        "Select marketing action",
        df["next_best_action"].unique()
    )

    action_df = df[df["next_best_action"] == selected_action]

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(action_df):,}")
    c2.metric("Avg Churn Risk", f"{action_df['churn_probability'].mean():.1%}")
    c3.metric("Avg Predicted CLV", f"R$ {action_df['predicted_clv'].mean():,.2f}")

    st.divider()

    st.subheader("Recommended Customer Audience")

    st.dataframe(
        action_df[
            [
                "customer_unique_id",
                "churn_probability",
                "predicted_clv",
                "frequency",
                "recency",
                "next_best_action"
            ]
        ].head(100),
        use_container_width=True,
        hide_index=True
    )


elif selected == "Campaign Simulator":
    st.title("Campaign ROI Simulator")
    st.caption("Estimate the potential revenue impact of targeting different customer audiences.")

    selected_action = st.selectbox(
        "Choose campaign audience",
        df["next_best_action"].unique()
    )

    uplift = st.slider(
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
    expected_revenue = audience_size * avg_audience_clv * uplift
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

    st.divider()

    st.subheader("Selected Campaign Audience")

    st.dataframe(
        audience[
            [
                "customer_unique_id",
                "churn_probability",
                "predicted_clv",
                "next_best_action"
            ]
        ].head(100),
        use_container_width=True,
        hide_index=True
    )


elif selected == "Customer Table":
    st.title("Customer-Level Predictions")
    st.caption("Full customer-level dataset with model predictions and recommended actions.")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )