"""The ``aise`` command: every stage of the analysis under one CLI.

    uv run aise analyze ...                  # preprocessing, topics, co-word networks
    uv run aise plot ...                     # word and category figures
    uv run aise engagement ...               # classify papers into the Figure 1 modes
    uv run aise engagement-plots ...         # figures of the classification
    uv run aise qualitative ...              # qualitative coding of the rare modes
    uv run aise qualitative-plots figures    # figures of the qualitative analysis

Each group is also a module that runs on its own, e.g. ``python -m aise.engagement``.
See aise.config for where inputs and outputs live and how to select earlier runs.
"""

import typer

from aise import analysis, engagement, plot, plot_modes, plot_qualitative, qualitative

app = typer.Typer(rich_markup_mode="rich", no_args_is_help=True,
                  help="Bridging research problems of AI safety and AI ethics.")
app.command()(analysis.analyze)
app.add_typer(plot.app, name="plot")
app.add_typer(engagement.app, name="engagement")
app.add_typer(plot_modes.app, name="engagement-plots")
app.add_typer(qualitative.app, name="qualitative")
app.add_typer(plot_qualitative.app, name="qualitative-plots")

if __name__ == "__main__":
    app()
