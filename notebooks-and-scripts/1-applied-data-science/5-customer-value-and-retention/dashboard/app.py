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
                        RFM Segmentation · Churn Risk · Customer Lifetime Value
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(cards: list[tuple[str, str, str]]) -> None:
    """Render KPI cards with native Streamlit containers.

    The previous version used one large HTML block. Some Streamlit versions can
    expose part of that block as raw text when the surrounding markdown changes.
    Native containers are more robust and keep the dashboard easier to maintain.
    """
    columns = st.columns(len(cards))
    for column, (label, value, subtitle) in zip(columns, cards):
        with column:
            with st.container(border=True):
                st.caption(label)
                st.markdown(f"### {value}")
                st.caption(subtitle)


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
    out["expected_lost_revenue"] = out["monetary"] * out["churn_probability"]
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
    # Only positive future value should be treated as value at risk.
    # Negative CLV customers should not become artificially "better" after a churn adjustment.
    out["positive_clv_exposure"] = out["predicted_clv"].clip(lower=0)
    out["clv_value_at_risk"] = out["positive_clv_exposure"] * out["segment_churn_rate"].clip(lower=0)
    out["risk_adjusted_clv"] = out["predicted_clv"] - out["clv_value_at_risk"]
    return out


def show_dataframe(df: pd.DataFrame, columns: list[str] | None = None) -> None:
    table = df[columns] if columns else df
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_business_note(title: str, body: str) -> None:
    st.info(f"**{title}** — {body}")



def render_interpretation(title: str, bullets: list[str]) -> None:
    """Render a concise explanation for technical and non-technical reviewers."""
    st.markdown(f"**{title}**")
    for bullet in bullets:
        st.markdown(f"- {bullet}")


def make_segment_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Create a segment-level business profile from RFM output."""
    profile = (
        df.groupby("segment", as_index=False)
        .agg(
            customers=("customer_id", "nunique"),
            recency=("recency", "mean"),
            frequency=("frequency", "mean"),
            monetary=("monetary", "mean"),
            total_monetary=("monetary", "sum"),
            avg_rfm_score=("rfm_normalized_score", "mean"),
        )
    )
    total_customers = profile["customers"].sum()
    total_revenue = profile["total_monetary"].sum()
    profile["customer_share"] = np.where(total_customers > 0, profile["customers"] / total_customers, 0)
    profile["revenue_share"] = np.where(total_revenue > 0, profile["total_monetary"] / total_revenue, 0)
    profile["avg_spend_per_order"] = profile["monetary"] / profile["frequency"].replace(0, np.nan)
    profile["recommended_action"] = profile["segment"].apply(add_strategy_recommendations)
    return profile


def add_churn_deciles(df: pd.DataFrame) -> pd.DataFrame:
    """Build a decile view for explaining model ranking power and business exposure."""
    out = df.copy()
    if len(out) == 0:
        return pd.DataFrame(columns=["risk_decile", "customers", "avg_churn_probability", "actual_churn_rate", "expected_lost_revenue", "monetary"])
    ranked = out["churn_probability"].rank(method="first", ascending=False)
    bins = min(10, len(out))
    out["risk_decile"] = pd.qcut(ranked, q=bins, labels=[f"D{i}" for i in range(1, bins + 1)])
    deciles = (
        out.groupby("risk_decile", observed=True, as_index=False)
        .agg(
            customers=("customer_id", "nunique"),
            avg_churn_probability=("churn_probability", "mean"),
            actual_churn_rate=("churn", "mean"),
            expected_lost_revenue=("expected_lost_revenue", "sum"),
            monetary=("monetary", "sum"),
        )
    )
    deciles["risk_decile"] = deciles["risk_decile"].astype(str)
    return deciles


def add_clv_deciles(df: pd.DataFrame) -> pd.DataFrame:
    """Build CLV deciles to show value concentration."""
    out = df.copy()
    if len(out) == 0:
        return pd.DataFrame(columns=["clv_decile", "customers", "avg_predicted_clv", "total_risk_adjusted_clv", "clv_value_at_risk"])
    ranked = out["predicted_clv"].rank(method="first", ascending=False)
    bins = min(10, len(out))
    out["clv_decile"] = pd.qcut(ranked, q=bins, labels=[f"D{i}" for i in range(1, bins + 1)])
    deciles = (
        out.groupby("clv_decile", observed=True, as_index=False)
        .agg(
            customers=("customer_id", "nunique"),
            avg_predicted_clv=("predicted_clv", "mean"),
            total_risk_adjusted_clv=("risk_adjusted_clv", "sum"),
            clv_value_at_risk=("clv_value_at_risk", "sum"),
        )
    )
    deciles["clv_decile"] = deciles["clv_decile"].astype(str)
    return deciles

def add_strategy_recommendations(seg: pd.Series) -> str:
    """Map segment names to practical commercial recommendations."""
    name = str(seg)
    if name == "Champions":
        return "Protect relationship; use loyalty and premium service"
    if name == "High Value but Cooling":
        return "Win back before inactivity becomes permanent"
    if name == "Frequent Low-Spend Customers":
        return "Grow basket size with bundles or cross-sell"
    if name == "Declining Frequent Buyers":
        return "Check friction; targeted recovery offer"
    if "Recent Low-Value" in name or "Active Minimal" in name:
        return "Nurture through low-cost lifecycle campaigns"
    if "Lost" in name or "Inactive" in name:
        return "Reactivate selectively only when value justifies cost"
    return "Monitor and test lightweight campaigns"


def classify_retention_action(probability: float, exposure: float, exposure_q75: float) -> str:
    """Translate churn probability and value exposure into an action tier."""
    if probability >= 0.70 and exposure >= exposure_q75:
        return "Immediate save"
    if probability >= 0.70:
        return "Scalable retention"
    if probability >= 0.40 and exposure >= exposure_q75:
        return "Value watchlist"
    return "Monitor"


def classify_value_action(clv: float, churn_rate: float, clv_q75: float) -> str:
    """Translate CLV and churn-adjusted exposure into an investment tier."""
    if clv >= clv_q75 and churn_rate >= 0.20:
        return "Defend high value"
    if clv >= clv_q75:
        return "Grow high value"
    if churn_rate >= 0.20:
        return "Automated retention"
    return "Efficient nurture"


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

segmentation_df, churn_raw_df, clv_raw_df = load_all_data()
churn_df = prepare_churn_data(churn_raw_df)
clv_df = prepare_clv_data(clv_raw_df)

render_header()
st.caption(
    "Decision-oriented demo based on precomputed notebook outputs. "
    "The app reads local CSV files and focuses on practical retail actions rather than model retraining."
)

executive_retain_grow = segmentation_df["segment"].isin(
    ["Champions", "High Value but Cooling", "Frequent Low-Spend Customers"]
).mean()
executive_revenue_at_risk = churn_df["expected_lost_revenue"].sum()
executive_risk_adjusted_clv = clv_df["risk_adjusted_clv"].sum()
executive_value_at_risk = clv_df["clv_value_at_risk"].sum()

st.markdown("### Executive Snapshot")
render_kpi_cards(
    [
        ("Customer Base", format_number(segmentation_df["customer_id"].nunique()), "Customers available for action"),
        ("Retain / Grow Pool", format_percent(executive_retain_grow), "Commercially attractive segments"),
        ("Revenue at Risk", format_currency(executive_revenue_at_risk, 0), "Historical value × churn risk"),
        ("Risk-Adjusted CLV", format_currency(executive_risk_adjusted_clv, 0), "Future value after churn exposure"),
    ]
)
render_business_note(
    "Portfolio lens",
    "The dashboard is intentionally compact: segment the base, find where churn threatens value, then rank customers by risk-adjusted future contribution."
)

seg_tab, churn_tab, clv_tab = st.tabs(
    ["Customer Segmentation", "Churn Prediction", "Customer Lifetime Value"]
)


# -----------------------------------------------------------------------------
# Tab 1 — Customer Segmentation
# -----------------------------------------------------------------------------

with seg_tab:
    st.subheader("Customer Segmentation")
    st.caption("RFM segments are used to translate customer behaviour into commercial priorities.")

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

    active_or_high_value = seg_filtered["segment"].isin(
        ["Champions", "High Value but Cooling", "Frequent Low-Spend Customers"]
    ).mean() if len(seg_filtered) else np.nan

    render_kpi_cards(
        [
            ("Customers", format_number(seg_filtered["customer_id"].nunique()), "Filtered customer base"),
            ("Behavioural Segments", format_number(seg_filtered["segment"].nunique()), "Active customer groups"),
            ("Avg. Orders", format_number(seg_filtered["frequency"].mean(), 2), "Orders per customer"),
            ("Retain / Grow Pool", format_percent(active_or_high_value), "Protect, win back or grow"),
        ]
    )

    render_business_note(
        "How to read this view",
        "Use segmentation to separate customer management actions: protect Champions, reactivate High Value but Cooling customers, nurture frequent low-spend buyers, and avoid over-investing in inactive minimal buyers."
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
        rfm_profile = make_segment_profile(seg_filtered).sort_values("monetary", ascending=False)
        st.plotly_chart(
            bar_chart(
                rfm_profile,
                x="segment",
                y="monetary",
                title="Average Historical Spend by Segment",
                labels={"segment": "Segment", "monetary": "Average monetary value"},
            ),
            use_container_width=True,
        )

    st.markdown("### Segment Playbook")
    playbook = rfm_profile.merge(counts, on="segment", suffixes=("", "_count"))
    playbook["priority"] = np.select(
        [
            playbook["segment"].eq("Champions"),
            playbook["segment"].eq("High Value but Cooling"),
            playbook["segment"].str.contains("Lost|Inactive", case=False, regex=True),
        ],
        ["Protect", "Win back", "Selective reactivation"],
        default="Grow efficiently",
    )
    playbook["suggested_action"] = np.select(
        [
            playbook["priority"].eq("Protect"),
            playbook["priority"].eq("Win back"),
            playbook["priority"].eq("Selective reactivation"),
        ],
        [
            "Service quality, loyalty benefits and early access offers",
            "Targeted retention message before the relationship goes cold",
            "Low-cost campaigns only when expected value justifies the spend",
        ],
        default="Cross-sell, bundles and frequency-building campaigns",
    )
    show_dataframe(
        playbook[["segment", "priority", "suggested_action", "customers", "recency", "frequency", "monetary"]]
        .sort_values(["priority", "monetary"], ascending=[True, False])
    )

    st.markdown("### Segment Strategy Matrix")
    st.caption(
        "A simpler alternative to the 3D RFM plot: each point is a segment, positioned by customer freshness and average value, "
        "with bubble size showing how many customers are affected."
    )
    strategy_matrix = rfm_profile.copy()
    strategy_matrix["recommended_action"] = strategy_matrix["segment"].apply(add_strategy_recommendations)
    fig_strategy = px.scatter(
        strategy_matrix,
        x="recency",
        y="monetary",
        size="customers",
        color="segment",
        hover_data=["frequency", "customers", "avg_spend_per_order", "recommended_action"],
        title="Segment Strategy Matrix: Freshness vs Value",
        labels={
            "recency": "Average days since last purchase",
            "monetary": "Average historical spend",
            "customers": "Customers",
            "frequency": "Average orders",
            "avg_spend_per_order": "Average spend per order",
            "recommended_action": "Recommended action",
        },
        template="plotly_white",
    )
    fig_strategy.add_vline(x=strategy_matrix["recency"].median(), line_dash="dash", opacity=0.35)
    fig_strategy.add_hline(y=strategy_matrix["monetary"].median(), line_dash="dash", opacity=0.35)
    fig_strategy.update_layout(title_x=0.02, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_strategy, use_container_width=True)

    render_interpretation(
        "Why this replaces the 3D RFM chart",
        [
            "The old 3D scatter was useful for modelling diagnostics, but it was hard to explain quickly in an interview.",
            "This matrix keeps the core RFM logic while translating it into a commercial question: who is still fresh, who is valuable, and how many customers are affected?",
            "The dashed lines create practical quadrants, making it easier to discuss protect, grow, reactivate and deprioritise decisions.",
        ],
    )

    st.markdown("### Business Composition Views")
    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        revenue_share = rfm_profile.sort_values("revenue_share", ascending=False)
        fig_revenue_share = px.bar(
            revenue_share,
            x="revenue_share",
            y="segment",
            orientation="h",
            title="Revenue Concentration by Segment",
            labels={"revenue_share": "Share of historical revenue", "segment": "Segment"},
            template="plotly_white",
        )
        fig_revenue_share.update_layout(title_x=0.02, margin=dict(l=10, r=10, t=60, b=10), yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_revenue_share, use_container_width=True)

    with comp_col2:
        fig_tree = px.treemap(
            rfm_profile,
            path=["segment"],
            values="customers",
            color="monetary",
            title="Customer Base Size with Average Value",
            labels={"customers": "Customers", "monetary": "Average historical spend"},
            template="plotly_white",
        )
        fig_tree.update_layout(title_x=0.02, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_tree, use_container_width=True)

    score_col1, score_col2 = st.columns(2)
    with score_col1:
        fig_rfm_box = px.box(
            seg_filtered,
            x="segment",
            y="rfm_normalized_score",
            title="RFM Score Spread by Segment",
            labels={"segment": "Segment", "rfm_normalized_score": "Normalised RFM score"},
            template="plotly_white",
        )
        fig_rfm_box.update_layout(title_x=0.02, margin=dict(l=10, r=10, t=60, b=10), xaxis_tickangle=-35)
        st.plotly_chart(fig_rfm_box, use_container_width=True)

    with score_col2:
        fig_freq_value = px.scatter(
            rfm_profile,
            x="frequency",
            y="monetary",
            size="customers",
            color="segment",
            hover_data=["recency", "revenue_share", "recommended_action"],
            title="Frequency vs Value by Segment",
            labels={"frequency": "Average orders", "monetary": "Average historical spend", "customers": "Customers"},
            template="plotly_white",
        )
        fig_freq_value.update_layout(title_x=0.02, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_freq_value, use_container_width=True)

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
    st.caption("Prioritise customers where retention action can protect meaningful value.")

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
            ("Customers", format_number(churn_filtered["customer_id"].nunique()), "Filtered scoring base"),
            ("Avg. Churn Risk", format_percent(churn_filtered["churn_probability"].mean()), "Mean predicted probability"),
            ("Predicted Churn", format_percent(predicted_churn_rate), "Customers classified as churn"),
            ("Revenue at Risk", format_currency(churn_filtered["expected_lost_revenue"].sum(), 0), "Expected commercial exposure"),
        ]
    )

    render_business_note(
        "Retention lens",
        "A high churn probability is not enough on its own. The most useful operating queue combines risk with value, so teams can act first where the customer relationship and commercial exposure are both material."
    )

    st.divider()
    exposure_q75 = churn_filtered["expected_lost_revenue"].quantile(0.75) if len(churn_filtered) else 0
    churn_filtered = churn_filtered.copy()
    churn_filtered["retention_action"] = churn_filtered.apply(
        lambda row: classify_retention_action(
            row["churn_probability"], row["expected_lost_revenue"], exposure_q75
        ),
        axis=1,
    )

    churn_by_segment = (
        churn_filtered.groupby("segment", as_index=False)
        .agg(
            customers=("customer_id", "nunique"),
            avg_churn_probability=("churn_probability", "mean"),
            expected_lost_revenue=("expected_lost_revenue", "sum"),
            avg_monetary=("monetary", "mean"),
        )
        .sort_values("expected_lost_revenue", ascending=False)
    )

    col1, col2 = st.columns(2)

    with col1:
        fig_churn_matrix = px.scatter(
            churn_by_segment,
            x="avg_churn_probability",
            y="expected_lost_revenue",
            size="customers",
            color="segment",
            hover_data=["avg_monetary", "customers"],
            title="Retention Priority Matrix: Risk vs Commercial Exposure",
            labels={
                "avg_churn_probability": "Average churn probability",
                "expected_lost_revenue": "Expected revenue at risk",
                "avg_monetary": "Average historical spend",
                "customers": "Customers",
            },
            template="plotly_white",
        )
        fig_churn_matrix.add_vline(x=churn_by_segment["avg_churn_probability"].median(), line_dash="dash", opacity=0.35)
        fig_churn_matrix.add_hline(y=churn_by_segment["expected_lost_revenue"].median(), line_dash="dash", opacity=0.35)
        fig_churn_matrix.update_layout(title_x=0.02, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_churn_matrix, use_container_width=True)

    with col2:
        action_summary = (
            churn_filtered.groupby("retention_action", as_index=False)
            .agg(
                customers=("customer_id", "nunique"),
                expected_lost_revenue=("expected_lost_revenue", "sum"),
                avg_churn_probability=("churn_probability", "mean"),
            )
            .sort_values("expected_lost_revenue", ascending=False)
        )
        st.plotly_chart(
            bar_chart(
                action_summary,
                x="retention_action",
                y="expected_lost_revenue",
                title="Retention Workload by Action Tier",
                labels={
                    "retention_action": "Action tier",
                    "expected_lost_revenue": "Expected revenue at risk",
                },
            ),
            use_container_width=True,
        )

    st.markdown("### Model-to-Business Diagnostics")
    st.caption("These views keep the ML story visible while explaining why the predictions matter commercially.")
    deciles = add_churn_deciles(churn_filtered)
    diag_col1, diag_col2 = st.columns(2)
    with diag_col1:
        fig_lift = px.line(
            deciles,
            x="risk_decile",
            y=["avg_churn_probability", "actual_churn_rate"],
            markers=True,
            title="Risk Deciles: Predicted vs Observed Churn",
            labels={"risk_decile": "Predicted risk decile", "value": "Churn rate", "variable": "Metric"},
            template="plotly_white",
        )
        fig_lift.update_layout(title_x=0.02, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_lift, use_container_width=True)

    with diag_col2:
        fig_decile_exposure = px.bar(
            deciles,
            x="risk_decile",
            y="expected_lost_revenue",
            title="Revenue at Risk by Risk Decile",
            labels={"risk_decile": "Predicted risk decile", "expected_lost_revenue": "Expected revenue at risk"},
            template="plotly_white",
        )
        fig_decile_exposure.update_layout(title_x=0.02, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_decile_exposure, use_container_width=True)

    render_interpretation(
        "How to explain this to recruiters",
        [
            "The model score is not shown as an abstract accuracy metric only; it is converted into a ranked operating queue.",
            "Risk deciles help a technical reviewer see whether higher predicted probabilities correspond to higher observed churn.",
            "The revenue-at-risk chart helps a business reviewer see where retention effort has the highest potential pay-off.",
        ],
    )

    exposure_col1, exposure_col2 = st.columns(2)
    with exposure_col1:
        top_exposed_segments = churn_by_segment.sort_values("expected_lost_revenue", ascending=True)
        fig_segment_exposure = px.bar(
            top_exposed_segments,
            x="expected_lost_revenue",
            y="segment",
            orientation="h",
            title="Expected Revenue at Risk by Segment",
            labels={"expected_lost_revenue": "Expected revenue at risk", "segment": "Segment"},
            template="plotly_white",
        )
        fig_segment_exposure.update_layout(title_x=0.02, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_segment_exposure, use_container_width=True)

    with exposure_col2:
        fig_action_customers = px.bar(
            action_summary.sort_values("customers", ascending=False),
            x="retention_action",
            y="customers",
            title="Retention Workload by Customer Count",
            labels={"retention_action": "Action tier", "customers": "Customers"},
            template="plotly_white",
        )
        fig_action_customers.update_layout(title_x=0.02, margin=dict(l=10, r=10, t=60, b=10), xaxis_tickangle=-20)
        st.plotly_chart(fig_action_customers, use_container_width=True)

    st.markdown("### Segment Retention Plan")
    segment_retention_plan = churn_by_segment.copy()
    segment_retention_plan["recommended_action"] = np.select(
        [
            segment_retention_plan["expected_lost_revenue"] >= segment_retention_plan["expected_lost_revenue"].quantile(0.75),
            segment_retention_plan["avg_churn_probability"] >= 0.70,
            segment_retention_plan["avg_churn_probability"] >= 0.40,
        ],
        [
            "Prioritise commercially; assign retention owner",
            "Automated save journey with targeted incentives",
            "Monitor and test low-cost nudges",
        ],
        default="Keep in standard lifecycle programme",
    )
    show_dataframe(
        segment_retention_plan[[
            "segment",
            "customers",
            "avg_churn_probability",
            "expected_lost_revenue",
            "avg_monetary",
            "recommended_action",
        ]]
    )

    st.markdown("### Retention Priority Queue")
    priority_queue = churn_filtered.sort_values("expected_lost_revenue", ascending=False).head(25)
    show_dataframe(
        priority_queue,
        columns=[
            "customer_id",
            "segment",
            "risk_level",
            "retention_action",
            "churn_probability",
            "monetary",
            "expected_lost_revenue",
            "recency",
            "frequency",
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
            "retention_action",
            "churn_probability",
            "churn_prediction",
            "recency",
            "frequency",
            "monetary",
            "expected_lost_revenue",
        ],
    )


# -----------------------------------------------------------------------------
# Tab 3 — Customer Lifetime Value
# -----------------------------------------------------------------------------

with clv_tab:
    st.subheader("Customer Lifetime Value")
    st.caption("Compare future value estimates with churn-adjusted value for investment decisions.")

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

    total_value_gap = clv_filtered["clv_value_at_risk"].sum()
    render_kpi_cards(
        [
            ("Customers", format_number(clv_filtered["customer_id"].nunique()), "Filtered value base"),
            ("Avg. CLV", format_currency(clv_filtered["predicted_clv"].mean(), 0), "Predicted future value"),
            ("Avg. Risk-Adjusted CLV", format_currency(clv_filtered["risk_adjusted_clv"].mean(), 0), "Expected value after churn"),
            ("CLV at Risk", format_currency(total_value_gap, 0), "Positive CLV exposed to churn"),
        ]
    )

    render_business_note(
        "Investment lens",
        "CLV is most useful when it guides resource allocation. High-value customers may deserve proactive retention, while low-value customers are better served through scalable, low-cost campaigns."
    )

    st.divider()
    clv_q75 = clv_filtered["predicted_clv"].quantile(0.75) if len(clv_filtered) else 0
    clv_filtered = clv_filtered.copy()
    clv_filtered["value_action"] = clv_filtered.apply(
        lambda row: classify_value_action(row["predicted_clv"], row["segment_churn_rate"], clv_q75),
        axis=1,
    )

    clv_by_segment = (
        clv_filtered.groupby("segment", as_index=False)
        .agg(
            customers=("customer_id", "nunique"),
            avg_predicted_clv=("predicted_clv", "mean"),
            avg_segment_churn_rate=("segment_churn_rate", "mean"),
            total_risk_adjusted_clv=("risk_adjusted_clv", "sum"),
            clv_value_at_risk=("clv_value_at_risk", "sum"),
        )
        .sort_values("total_risk_adjusted_clv", ascending=False)
    )

    col1, col2 = st.columns(2)

    with col1:
        fig_value_matrix = px.scatter(
            clv_by_segment,
            x="avg_segment_churn_rate",
            y="avg_predicted_clv",
            size="customers",
            color="segment",
            hover_data=["total_risk_adjusted_clv", "clv_value_at_risk", "customers"],
            title="Value Investment Matrix: CLV vs Churn Exposure",
            labels={
                "avg_segment_churn_rate": "Average segment churn rate",
                "avg_predicted_clv": "Average predicted CLV",
                "total_risk_adjusted_clv": "Total risk-adjusted CLV",
                "clv_value_at_risk": "CLV at risk",
                "customers": "Customers",
            },
            template="plotly_white",
        )
        fig_value_matrix.add_vline(x=clv_by_segment["avg_segment_churn_rate"].median(), line_dash="dash", opacity=0.35)
        fig_value_matrix.add_hline(y=clv_by_segment["avg_predicted_clv"].median(), line_dash="dash", opacity=0.35)
        fig_value_matrix.update_layout(title_x=0.02, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_value_matrix, use_container_width=True)

    with col2:
        value_action_summary = (
            clv_filtered.groupby("value_action", as_index=False)
            .agg(
                customers=("customer_id", "nunique"),
                total_risk_adjusted_clv=("risk_adjusted_clv", "sum"),
                clv_value_at_risk=("clv_value_at_risk", "sum"),
            )
            .sort_values("total_risk_adjusted_clv", ascending=False)
        )
        st.plotly_chart(
            bar_chart(
                value_action_summary,
                x="value_action",
                y="total_risk_adjusted_clv",
                title="Future Value by Investment Tier",
                labels={
                    "value_action": "Investment tier",
                    "total_risk_adjusted_clv": "Total risk-adjusted CLV",
                },
            ),
            use_container_width=True,
        )

    st.markdown("### Value Concentration and Model Interpretation")
    st.caption("These views explain whether future value is concentrated, exposed to churn, and suitable for targeted investment.")
    clv_deciles = add_clv_deciles(clv_filtered)
    value_col1, value_col2 = st.columns(2)
    with value_col1:
        fig_clv_deciles = px.bar(
            clv_deciles,
            x="clv_decile",
            y="total_risk_adjusted_clv",
            title="Risk-Adjusted CLV by Value Decile",
            labels={"clv_decile": "Predicted CLV decile", "total_risk_adjusted_clv": "Total risk-adjusted CLV"},
            template="plotly_white",
        )
        fig_clv_deciles.update_layout(title_x=0.02, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_clv_deciles, use_container_width=True)

    with value_col2:
        fig_clv_risk = px.bar(
            clv_by_segment.sort_values("clv_value_at_risk", ascending=True),
            x="clv_value_at_risk",
            y="segment",
            orientation="h",
            title="CLV at Risk by Segment",
            labels={"clv_value_at_risk": "CLV at risk", "segment": "Segment"},
            template="plotly_white",
        )
        fig_clv_risk.update_layout(title_x=0.02, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_clv_risk, use_container_width=True)

    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        fig_clv_box = px.box(
            clv_filtered,
            x="segment",
            y="predicted_clv",
            title="Predicted CLV Distribution by Segment",
            labels={"segment": "Segment", "predicted_clv": "Predicted CLV"},
            template="plotly_white",
        )
        fig_clv_box.update_layout(title_x=0.02, margin=dict(l=10, r=10, t=60, b=10), xaxis_tickangle=-35)
        st.plotly_chart(fig_clv_box, use_container_width=True)

    with dist_col2:
        fig_profit_clv = px.scatter(
            clv_filtered.sample(min(len(clv_filtered), 2000), random_state=42) if len(clv_filtered) else clv_filtered,
            x="customer_profit",
            y="predicted_clv",
            color="segment",
            hover_data=["customer_id", "segment_churn_rate", "risk_adjusted_clv"],
            title="Customer Profit vs Predicted CLV",
            labels={"customer_profit": "Customer profit", "predicted_clv": "Predicted CLV"},
            template="plotly_white",
        )
        fig_profit_clv.update_layout(title_x=0.02, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_profit_clv, use_container_width=True)

    render_interpretation(
        "How to explain CLV without losing the ML angle",
        [
            "The model estimates future customer value; the dashboard then adjusts that value by churn exposure to support investment decisions.",
            "Deciles show value concentration: a small part of the base may explain a large share of future value.",
            "The segment and customer views translate predictions into defend, grow, automate or nurture actions.",
        ],
    )

    st.markdown("### Segment Value Plan")
    segment_value_plan = clv_by_segment.copy()
    segment_value_plan["recommended_action"] = np.select(
        [
            (segment_value_plan["avg_predicted_clv"] >= segment_value_plan["avg_predicted_clv"].quantile(0.75))
            & (segment_value_plan["avg_segment_churn_rate"] >= 0.20),
            segment_value_plan["avg_predicted_clv"] >= segment_value_plan["avg_predicted_clv"].quantile(0.75),
            segment_value_plan["avg_segment_churn_rate"] >= 0.20,
        ],
        [
            "Defend with proactive retention and service quality",
            "Grow with cross-sell, bundles and preferential offers",
            "Use automated retention; avoid expensive manual effort",
        ],
        default="Maintain through efficient lifecycle marketing",
    )
    show_dataframe(
        segment_value_plan[[
            "segment",
            "customers",
            "avg_predicted_clv",
            "avg_segment_churn_rate",
            "clv_value_at_risk",
            "total_risk_adjusted_clv",
            "recommended_action",
        ]]
    )

    st.markdown("### Customer CLV Ranking")
    ranking = clv_filtered.sort_values("risk_adjusted_clv", ascending=False).head(50)
    show_dataframe(
        ranking,
        columns=[
            "customer_id",
            "segment",
            "value_action",
            "customer_profit",
            "segment_churn_rate",
            "predicted_clv",
            "clv_value_at_risk",
            "risk_adjusted_clv",
            "residual",
        ],
    )

    st.markdown("### Filtered CLV Table")
    show_dataframe(
        clv_filtered.sort_values("risk_adjusted_clv", ascending=False),
        columns=[
            "customer_id",
            "segment",
            "value_action",
            "customer_profit",
            "segment_churn_rate",
            "segment_discount_rate",
            "constructed_clv",
            "predicted_clv",
            "clv_value_at_risk",
            "risk_adjusted_clv",
        ],
    )


st.divider()
with st.container():
    st.markdown('<span id="mic-logout-marker"></span>', unsafe_allow_html=True)
    if st.button("Sign Out", key="mic_logout"):
        logout()
        st.rerun()
