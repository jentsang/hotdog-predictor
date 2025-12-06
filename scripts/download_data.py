"""
Download the Food Vendors data and save it as a CSV file.

This script is the first step of our analysis pipeline.
It reads the raw data from a URL (or local path) and writes it to disk.
"""

import pathlib

import click
import pandas as pd


@click.command()
@click.option(
    "--source",
    "source",
    required=True,
    help=(
        "Location of the raw Food Vendors data. "
        "This can be a URL (e.g., the City of Vancouver CSV export)."
    ),
)
@click.option(
    "--out-file",
    "out_file",
    required=True,
    type=click.Path(),
    help="Path where the downloaded CSV should be saved (e.g., data/raw/food_vendors_raw.csv).",
)
def main(source, out_file):
    """Download the Food Vendors data and save it to out_file."""
    out_path = pathlib.Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # The City of Vancouver export uses ';' as a separator
    df = pd.read_csv(source, sep=";")

    df.to_csv(out_path, index=False)

    click.echo(f"Saved raw data to {out_path}")


if __name__ == "__main__":
    main()
