"""Formatting helpers used across the dashboard."""

import pandas as pd


def format_number(value: float, decimals: int = 0) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.{decimals}f}"


def format_currency(value: float, decimals: int = 0) -> str:
    if pd.isna(value):
        return "-"
    return f"${value:,.{decimals}f}"


def format_percent(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return "-"
    return f"{100 * value:.{decimals}f}%"


def risk_level(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"
