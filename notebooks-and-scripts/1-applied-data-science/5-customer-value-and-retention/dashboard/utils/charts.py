"""Reusable Plotly Express chart helpers."""

import pandas as pd
import plotly.express as px

PLOT_TEMPLATE = "plotly_white"


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, labels: dict | None = None):
    fig = px.bar(df, x=x, y=y, title=title, labels=labels or {}, template=PLOT_TEMPLATE)
    fig.update_layout(title_x=0.02, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: str | None = None,
    size: str | None = None,
    hover_data: list[str] | None = None,
    labels: dict | None = None,
):
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        size=size,
        hover_data=hover_data,
        title=title,
        labels=labels or {},
        template=PLOT_TEMPLATE,
    )
    fig.update_layout(title_x=0.02, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def histogram(df: pd.DataFrame, x: str, title: str, labels: dict | None = None, nbins: int = 30):
    fig = px.histogram(df, x=x, nbins=nbins, title=title, labels=labels or {}, template=PLOT_TEMPLATE)
    fig.update_layout(title_x=0.02, margin=dict(l=10, r=10, t=60, b=10))
    return fig
