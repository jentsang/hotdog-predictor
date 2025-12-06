"""
Prepare the Food Vendors data for modelling.

This script:
- reads the raw data from a CSV file,
- runs a few simple checks,
- keeps only BUSINESS_NAME, DESCRIPTION and the target is_hotdog,
- fills missing BUSINESS_NAME with empty strings,
- splits into train and test sets, and
- saves them to CSV files.

Usage example:

    python scripts/prepare_data.py \
        --input-file data/raw/food_vendors_raw.csv \
        --train-out data/processed/vendors_train.csv \
        --test-out data/processed/vendors_test.csv
"""

import pathlib

import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command()
@click.option(
    "--input-file",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Path to the raw Food Vendors CSV (downloaded data).",
)
@click.option(
    "--train-out",
    "train_out",
    required=True,
    type=click.Path(),
    help="Path where the processed *training* data will be saved.",
)
@click.option(
    "--test-out",
    "test_out",
    required=True,
    type=click.Path(),
    help="Path where the processed *test* data will be saved.",
)
@click.option(
    "--train-size",
    "train_size",
    default=0.7,
    show_default=True,
    help="Proportion of the data to use for training.",
)
@click.option(
    "--seed",
    "seed",
    default=522,
    show_default=True,
    help="Random seed for the train/test split.",
)
def main(input_file, train_out, test_out, train_size, seed):
    """Load raw data, validate basic structure, engineer target, and split."""

    input_path = pathlib.Path(input_file)
    train_out_path = pathlib.Path(train_out)
    test_out_path = pathlib.Path(test_out)

    # Make sure output folders exist
    train_out_path.parent.mkdir(parents=True, exist_ok=True)
    test_out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load raw data -------------------------------------------------------
    # The City of Vancouver export uses ';' as a separator.
    food_vendors = pd.read_csv(input_path, sep=None, engine="python")

    # Strip any accidental leading/trailing spaces in column names
    food_vendors.columns = food_vendors.columns.str.strip()

    # --- Basic column checks -------------------------------------------------
    required_columns = ["BUSINESS_NAME", "DESCRIPTION"]
    missing = [col for col in required_columns if col not in food_vendors.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Columns found: {list(food_vendors.columns)}"
        )

    # --- Keep only columns needed for our analysis ---------------------------
    clean_food = food_vendors[required_columns].copy()

    # Binary target: True if description is exactly "Hot Dogs"
    clean_food["is_hotdog"] = clean_food["DESCRIPTION"] == "Hot Dogs"

    # Replace missing business names with empty strings (as in the notebook)
    clean_food["BUSINESS_NAME"] = clean_food["BUSINESS_NAME"].fillna("")

    # --- Train/test split ----------------------------------------------------
    train_data, test_data = train_test_split(
        clean_food, train_size=train_size, random_state=seed
    )

    # --- Save outputs --------------------------------------------------------
    train_data.to_csv(train_out_path, index=False)
    test_data.to_csv(test_out_path, index=False)

    click.echo(f"Saved training data to {train_out_path}")
    click.echo(f"Saved test data to {test_out_path}")


if __name__ == "__main__":
    main()