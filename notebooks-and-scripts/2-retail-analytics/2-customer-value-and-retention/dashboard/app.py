"""
Customer Value and Retention — Streamlit analytical demo.

This dashboard explores precomputed outputs from the portfolio notebooks.
It does not retrain models, run machine learning pipelines, or request files
from the user.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from auth_helpers import logout, protect_dashboard
from ui import load_css
from utils.charts import bar_chart, histogram, scatter_chart
from utils.data_loader import load_all_data
from utils.formatters import format_currency, format_number, format_percent, risk_level


st.set_page_config(
    page_title="Customer Value and Retention",
    layout="wide",
)

protect_dashboard()
load_css("assets/styles.css")


# -----------------------------------------------------------------------------
# Generic UI helpers
# -----------------------------------------------------------------------------

def render_header() -> None:
    st.markdown(
        """
        <div class="mic-header">
            <div class="mic-header-bar">
                <div>
                    <div class="mic-header-title">Customer Value and Retention</div>
                    <div class="portfolio-header-subtitle">
                        Segmentation · Churn Risk · Customer Lifetime Value
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(cards: list[tuple[str, str, str]]) -> None:
    html = ['<div class="mic-kpi-grid">']
    for label, value, subtitle in cards:
        html.append(
            f"""
            <div class="mic-kpi-card">
                <div class="mic-kpi-label">{label}</div>
                <div class="mic-kpi-value">{value}</div>
                <div class="mic-kpi-sub">{subtitle}</div>
            </div>
            """
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def optional_multiselect(df: pd.DataFrame, column: str, label: str) -> list[str] | None:
    if column not in df.columns:
        return None
    options = sorted(df[column].dropna().astype(str).unique())
    return st.multiselect(label, options=options, default=options)


def filter_by_selection(df: pd.DataFrame, column: str, selected: list[str] | None) -> pd.DataFrame:
    if selected is None or column not in df.columns:
        return df
    if len(selected) == 0:
        return df.iloc[0:0]
    return df[df[column].astype(str).isin(selected)]


def segment_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("segment", as_index=False)
        .agg(customers=("customer_id", "nunique"))
        .sort_values("customers", ascending=False)
    )


def prepare_churn_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["risk_level"] = out["churn_probability"].apply(risk_level)
    return out


def prepare_clv_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(
        columns={
            "CustomerID": "customer_id",
            "Cluster": "cluster",
            "Segment": "segment",
            "Customer Profit": "customer_profit",
            "Segment Churn Rate": "segment_churn_rate",
            "Segment Discount Rate": "segment_discount_rate",
            "Constructed CLV": "constructed_clv",
            "Predicted CLV": "predicted_clv",
            "Residual": "residual",
        }
    )
    out["risk_adjusted_clv"] = out["predicted_clv"] * (1 - out["segment_churn_rate"])
    return out


def show_dataframe(df: pd.DataFrame, columns: list[str] | None = None) -> None:
    table = df[columns] if columns else df
    st.dataframe(table, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

segmentation_df, churn_raw_df, clv_raw_df = load_all_data()
churn_df = prepare_churn_data(churn_raw_df)
clv_df = prepare_clv_data(clv_raw_df)

render_header()
st.caption(
    "Interactive analytical demo based on precomputed notebook outputs. "
    "The app reads local CSV files from the `data/` folder and does not retrain models."
)

seg_tab, churn_tab, clv_tab = st.tabs(
    ["Customer Segmentation", "Churn Prediction", "Customer Lifetime Value"]
)


# -----------------------------------------------------------------------------
# Tab 1 — Customer Segmentation
# -----------------------------------------------------------------------------

with seg_tab:
    st.subheader("Customer Segmentation")
    st.caption("Explore the RFM-based customer segments and their business profiles.")

    st.markdown("### Filters")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        selected_segments = st.multiselect(
            "Segment",
            options=sorted(segmentation_df["segment"].dropna().unique()),
            default=sorted(segmentation_df["segment"].dropna().unique()),
            key="segmentation_segment_filter",
        )
    with filter_col2:
        selected_territories = optional_multiselect(segmentation_df, "territory", "Territory")
    with filter_col3:
        selected_customer_types = optional_multiselect(segmentation_df, "customer_type", "Customer type")

    seg_filtered = segmentation_df[segmentation_df["segment"].isin(selected_segments)]
    seg_filtered = filter_by_selection(seg_filtered, "territory", selected_territories)
    seg_filtered = filter_by_selection(seg_filtered, "customer_type", selected_customer_types)

    render_kpi_cards(
        [
            ("Customers", format_number(seg_filtered["customer_id"].nunique()), "Filtered customer base"),
            ("Segments", format_number(seg_filtered["segment"].nunique()), "Active RFM groups"),
            ("Avg. Frequency", format_number(seg_filtered["frequency"].mean(), 2), "Orders per customer"),
            ("Avg. Monetary", format_currency(seg_filtered["monetary"].mean(), 0), "Historical spend"),
        ]
    )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        counts = segment_counts(seg_filtered)
        st.plotly_chart(
            bar_chart(
                counts,
                x="segment",
                y="customers",
                title="Customer Distribution by Segment",
                labels={"segment": "Segment", "customers": "Customers"},
            ),
            use_container_width=True,
        )

    with col2:
        rfm_profile = (
            seg_filtered.groupby("segment", as_index=False)
            .agg(
                recency=("recency", "mean"),
                frequency=("frequency", "mean"),
                monetary=("monetary", "mean"),
            )
            .sort_values("monetary", ascending=False)
        )
        rfm_long = rfm_profile.melt(
            id_vars="segment",
            value_vars=["recency", "frequency", "monetary"],
            var_name="rfm_metric",
            value_name="average_value",
        )
        st.plotly_chart(
            px.bar(
                rfm_long,
                x="segment",
                y="average_value",
                color="rfm_metric",
                barmode="group",
                title="Average RFM Profile by Segment",
                labels={"segment": "Segment", "average_value": "Average value", "rfm_metric": "Metric"},
                template="plotly_white",
            ),
            use_container_width=True,
        )

    st.markdown("### RFM Space")
    sample_size = min(4000, len(seg_filtered))
    plot_df = seg_filtered.sample(sample_size, random_state=42) if sample_size else seg_filtered
    fig_rfm = px.scatter_3d(
        plot_df,
        x="recency",
        y="frequency",
        z="monetary",
        color="segment",
        size="rfm_normalized_score",
        hover_data=["customer_id", "rfm_normalized_score"],
        title="RFM Customer Space",
        labels={
            "recency": "Recency",
            "frequency": "Frequency",
            "monetary": "Monetary",
            "rfm_normalized_score": "RFM Score",
        },
        template="plotly_white",
    )
    fig_rfm.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_rfm, use_container_width=True)

    st.markdown("### Filtered Customer Table")
    show_dataframe(
        seg_filtered.sort_values("rfm_normalized_score", ascending=False),
        columns=[
            "customer_id",
            "segment",
            "recency",
            "frequency",
            "monetary",
            "rfm_normalized_score",
            "cluster",
        ],
    )


# -----------------------------------------------------------------------------
# Tab 2 — Churn Prediction
# -----------------------------------------------------------------------------

with churn_tab:
    st.subheader("Churn Prediction")
    st.caption("Explore predicted churn probabilities and high-risk customer groups.")

    st.markdown("### Filters")
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        churn_segments = st.multiselect(
            "Segment",
            options=sorted(churn_df["segment"].dropna().unique()),
            default=sorted(churn_df["segment"].dropna().unique()),
            key="churn_segment_filter",
        )
    with filter_col2:
        risk_levels = st.multiselect(
            "Risk level",
            options=["Low", "Medium", "High"],
            default=["Low", "Medium", "High"],
            key="churn_risk_level_filter",
        )

    churn_filtered = churn_df[
        churn_df["segment"].isin(churn_segments) & churn_df["risk_level"].isin(risk_levels)
    ]

    high_risk_share = (churn_filtered["risk_level"].eq("High").mean() if len(churn_filtered) else np.nan)
    predicted_churn_rate = (churn_filtered["churn_prediction"].mean() if len(churn_filtered) else np.nan)

    render_kpi_cards(
        [
            ("Customers", format_number(churn_filtered["customer_id"].nunique()), "Filtered customer base"),
            ("Avg. Churn Risk", format_percent(churn_filtered["churn_probability"].mean()), "Mean probability"),
            ("Predicted Churn", format_percent(predicted_churn_rate), "Model classification"),
            ("High Risk Share", format_percent(high_risk_share), "Probability ≥ 70%"),
        ]
    )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            histogram(
                churn_filtered,
                x="churn_probability",
                title="Distribution of Churn Probability",
                labels={"churn_probability": "Churn probability"},
                nbins=25,
            ),
            use_container_width=True,
        )

    with col2:
        churn_by_segment = (
            churn_filtered.groupby("segment", as_index=False)
            .agg(avg_churn_probability=("churn_probability", "mean"))
            .sort_values("avg_churn_probability", ascending=False)
        )
        st.plotly_chart(
            bar_chart(
                churn_by_segment,
                x="segment",
                y="avg_churn_probability",
                title="Average Churn Risk by Segment",
                labels={"segment": "Segment", "avg_churn_probability": "Average churn probability"},
            ),
            use_container_width=True,
        )

    st.markdown("### Customers with Highest Churn Risk")
    top_risk = churn_filtered.sort_values("churn_probability", ascending=False).head(25)
    show_dataframe(
        top_risk,
        columns=[
            "customer_id",
            "segment",
            "risk_level",
            "churn_probability",
            "churn_prediction",
            "recency",
            "frequency",
            "monetary",
            "rfm_normalized_score",
        ],
    )

    st.markdown("### Filtered Churn Table")
    show_dataframe(
        churn_filtered.sort_values("churn_probability", ascending=False),
        columns=[
            "customer_id",
            "segment",
            "risk_level",
            "churn_probability",
            "churn_prediction",
            "recency",
            "frequency",
            "monetary",
        ],
    )


# -----------------------------------------------------------------------------
# Tab 3 — Customer Lifetime Value
# -----------------------------------------------------------------------------

with clv_tab:
    st.subheader("Customer Lifetime Value")
    st.caption("Explore predicted CLV, risk-adjusted CLV and customer value rankings.")

    min_clv = float(clv_df["predicted_clv"].min())
    max_clv = float(clv_df["predicted_clv"].max())

    st.markdown("### Filters")
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        clv_segments = st.multiselect(
            "Segment",
            options=sorted(clv_df["segment"].dropna().unique()),
            default=sorted(clv_df["segment"].dropna().unique()),
            key="clv_segment_filter",
        )
    with filter_col2:
        clv_range = st.slider(
            "Predicted CLV range",
            min_value=min_clv,
            max_value=max_clv,
            value=(min_clv, max_clv),
            key="clv_range_filter",
        )

    clv_filtered = clv_df[
        clv_df["segment"].isin(clv_segments)
        & clv_df["predicted_clv"].between(clv_range[0], clv_range[1])
    ]

    render_kpi_cards(
        [
            ("Customers", format_number(clv_filtered["customer_id"].nunique()), "Filtered customer base"),
            ("Avg. CLV", format_currency(clv_filtered["predicted_clv"].mean(), 0), "Predicted CLV"),
            ("Avg. Risk-Adjusted CLV", format_currency(clv_filtered["risk_adjusted_clv"].mean(), 0), "Adjusted for churn"),
            ("Total Predicted CLV", format_currency(clv_filtered["predicted_clv"].sum(), 0), "Portfolio value"),
        ]
    )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        clv_by_segment = (
            clv_filtered.groupby("segment", as_index=False)
            .agg(avg_predicted_clv=("predicted_clv", "mean"))
            .sort_values("avg_predicted_clv", ascending=False)
        )
        st.plotly_chart(
            bar_chart(
                clv_by_segment,
                x="segment",
                y="avg_predicted_clv",
                title="Average Predicted CLV by Segment",
                labels={"segment": "Segment", "avg_predicted_clv": "Average predicted CLV"},
            ),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            scatter_chart(
                clv_filtered,
                x="predicted_clv",
                y="risk_adjusted_clv",
                color="segment",
                title="Predicted CLV vs Risk-Adjusted CLV",
                hover_data=["customer_id", "segment_churn_rate", "customer_profit"],
                labels={
                    "predicted_clv": "Predicted CLV",
                    "risk_adjusted_clv": "Risk-adjusted CLV",
                },
            ),
            use_container_width=True,
        )

    st.markdown("### Customer CLV Ranking")
    ranking = clv_filtered.sort_values("predicted_clv", ascending=False).head(50)
    show_dataframe(
        ranking,
        columns=[
            "customer_id",
            "segment",
            "customer_profit",
            "segment_churn_rate",
            "predicted_clv",
            "risk_adjusted_clv",
            "residual",
        ],
    )

    st.markdown("### Filtered CLV Table")
    show_dataframe(
        clv_filtered.sort_values("predicted_clv", ascending=False),
        columns=[
            "customer_id",
            "segment",
            "customer_profit",
            "segment_churn_rate",
            "segment_discount_rate",
            "constructed_clv",
            "predicted_clv",
            "risk_adjusted_clv",
        ],
    )


st.divider()
with st.container():
    st.markdown('<span id="mic-logout-marker"></span>', unsafe_allow_html=True)
    if st.button("Sign Out", key="mic_logout"):
        logout()
        st.rerun()
