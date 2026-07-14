"""Data loading utilities for precomputed portfolio outputs."""

from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_data_dir() -> Path:
    """Return the data directory used by the dashboard.

    The project currently ships with a ``data/`` folder. A ``datos/`` fallback is
    kept for backwards compatibility with earlier local versions of the app.
    """
    for candidate in (PROJECT_ROOT / "data", PROJECT_ROOT / "datos"):
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / "data"


DATA_DIR = _resolve_data_dir()


@st.cache_data
def load_customer_segmentation() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "customer-segmentation.csv")


@st.cache_data
def load_churn_prediction() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "customer-churn-prediction.csv")


@st.cache_data
def load_clv_prediction() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "customer-clv-prediction.csv")


@st.cache_data
def load_all_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        load_customer_segmentation(),
        load_churn_prediction(),
        load_clv_prediction(),
    )
