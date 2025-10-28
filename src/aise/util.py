"""Utility functions for AI Safety and Ethics analysis."""

from collections import Counter

import pandas as pd
from rich.table import Table
from rich.text import Text


def word_freqs(series: pd.Series) -> Counter:
    """Compute word frequencies from a pandas Series of text data."""
    words = " ".join(series).split()
    return Counter(words)


def pandas_to_rich(data, title: str = "Pandas Object") -> Table:
    """Convert a pandas DataFrame or Series to a Rich Table with nice formatting."""
    table = Table(
        title=title,
        header_style="bold magenta",
        show_lines=True,
        title_style="bold cyan",
    )

    # --- Handle DataFrame ---
    if isinstance(data, pd.DataFrame):
        # Add column headers
        for col in data.columns:
            justify = "right" if pd.api.types.is_numeric_dtype(data[col]) else "left"
            table.add_column(str(col), justify=justify, overflow="ellipsis",)

        # Add rows
        for _, row in data.iterrows():
            formatted = [_format_value(v) for v in row.to_numpy()]
            table.add_row(*formatted)

    # --- Handle Series ---
    elif isinstance(data, pd.Series):
        table.add_column("Index", justify="left")
        table.add_column(
            str(data.name or "Value"),
            justify="right" if pd.api.types.is_numeric_dtype(data) else "left",
            overflow="ellipsis",
        )

        for idx, val in data.items():
            table.add_row(str(idx), _format_value(val))

    else:
        err_msg = "Input must be a pandas DataFrame or Series."
        raise TypeError(err_msg)

    return table


def _format_value(value):
    """Help format numeric values with color and alignment."""
    if isinstance(value, (int, float)):
        text = f"{value:,.3f}" if isinstance(value, float) else f"{value:,}"
        if value > 0:
            return Text(text, style="green")
        if value < 0:
            return Text(text, style="red")
        return Text(text, style="grey50")
    return Text(str(value), style="white")
