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
def main(source, out_file) -> None:
    """
    Download the Food Vendors data and save it as a CSV file.

    The command reads a raw Food Vendors dataset from ``source``, which may
    be either a local file path or a URL (such as the City of Vancouver
    CSV export). The data is parsed using a semicolon (``;``) separator,
    and then written to ``out_file`` as a standard comma-separated CSV.
    Parent directories for ``out_file`` are created if they do not already
    exist.

    Parameters
    ----------
    source : str
        Location of the raw Food Vendors data. This can be a URL or a
        local file path. The file is expected to use ``;`` as a field
        separator.
    out_file : str
        Path where the processed CSV should be saved (for example,
        ``"data/raw/food_vendors_raw.csv"``). Any missing parent
        directories will be created.

    Returns
    -------
    None

    Examples
    --------
    From the command line:

    .. code-block:: bash

       python download_food_vendors.py \\
           --source="https://example.com/food_vendors.csv" \\
           --out-file="data/raw/food_vendors_raw.csv"
    """
    out_path = pathlib.Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # The City of Vancouver export uses ';' as a separator
    df = pd.read_csv(source, sep=";")

    df.to_csv(out_path, index=False)

    click.echo(f"Saved raw data to {out_path}")


if __name__ == "__main__":
    main()
