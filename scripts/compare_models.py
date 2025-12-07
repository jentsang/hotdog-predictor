"""
Command-line entry point for aggregating cross-validation score tables
across multiple models.

This script expects each model to have a subdirectory under ``table-dir``
containing an ``agg_cv_scores.csv`` file. It concatenates those tables,
selects a particular column level (e.g., "mean"), and writes a combined
comparison table to ``output-dir``. It is intended to be run from the
command line.
"""

from pathlib import Path

import click
import pandas as pd

# Default set of models expected to have `agg_cv_scores.csv` tables
models: list[str] = [
    "Dummy",
    "DecisionTree",
    "LogisticRegression",
    "NaiveBayes",
]


@click.command()
@click.option(
    "--table-dir",
    type=str,
    required=True,
    help="Directory containing model subfolders with `agg_cv_scores.csv`.",
)
@click.option(
    "--output-dir",
    type=str,
    required=True,
    help="Directory where the combined comparison CSV will be written.",
)
@click.option(
    "--param",
    type=str,
    default="mean",
    show_default=True,
    help=(
        "Column level to extract from the aggregated cross-validation "
        "scores ('mean', 'std')."
    ),
)
def main(table_dir: str, output_dir: str, param: str) -> None:
    """
    Aggregate per-model cross-validation score tables into a single CSV.

    Parameters
    ----------
    table_dir : str
        Path to the directory containing one subdirectory per model, each
        with an ``agg_cv_scores.csv`` file.
    output_dir : str
        Path to the directory where the combined CSV will be saved.
    param : str
        Column level to select from the MultiIndex columns (e.g., 'mean').

    Returns
    -------
    None
    """
    table_dir_path = Path(table_dir)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    model_comparison: dict[str, pd.DataFrame] = {}

    for model in models:
        csv_path = table_dir_path / model / "agg_cv_scores.csv"
        model_comparison[model] = pd.read_csv(csv_path, index_col=0)

    combined = (
        pd.concat(model_comparison, axis="columns")
        .xs(param, axis="columns", level=1)
    )

    output_file = output_dir_path / f"model_comparison_{param}.csv"
    combined.to_csv(output_file, index=True)

    click.echo(f"Combined comparison table saved to: {output_file}")


if __name__ == "__main__":
    main()
