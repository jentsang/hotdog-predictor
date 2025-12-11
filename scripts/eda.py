"""
Command-line entry point for generating EDA plots on out training data set.

This script loads training data, generates 3 exploratory data plots 
and exports the corresponding plots to the results/EDA folder.
This script is  intended to be run from the command line.
"""

import os 
import click
import altair as alt
import pandas as pd

@click.command()
@click.option('--training-data', type=str, help="Path to training data")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")
def main(training_data, plot_to):
    """
    Generate exploratory data analysis (EDA) plots for the Hotdog Vendor dataset.

    This script reads the processed training data and produces three plots:
    1. A bar chart showing the most common cuisine types among food vendors.
    2. A bar chart illustrating class imbalance in the training data.
    3. A heat map showing if blank names have any relevance on classification. 

    Parameters
    ----------
    training_data : str
        Path to the CSV file containing the training data (located in data/processed/).
    plot_to: str
        The directory where our EDA figures will be saved as PNG's (located in results/figures/EDA). 

    Returns 
    -------
    None 

    Examples
    --------
    From the command line run (assuming you are in ./scripts/):

    python scripts/eda.py   
    --training-data ../data/processed/vendors_train.csv   
    --plot-to ../results/figures/EDA
    """
    train_data = pd.read_csv(training_data)

    train_data["BUSINESS_NAME"] = train_data["BUSINESS_NAME"].fillna("")

    # PLOT 1: Cuisine types among food vendors
    plot1 = alt.Chart(
        train_data,
        title=(
        "What are the most common cuisine types among food vendors "
        "in Downtown Vancouver?"
        )
    ).mark_bar(color="chocolate").encode(
        x=alt.X("count()", title="Total"),
        y=alt.Y("DESCRIPTION:N", sort='-x', title="Food type")
    ).properties(
        width=250,
        height=500
    )

    plot1_path = os.path.join(plot_to, "plot1_cuisine_types.png")
    plot1.save(plot1_path, scale_factor=2.0)
    click.echo(f"Saved Plot 1 to {plot1_path}")

    # PLOT 2: Class Imbalance? 
    plot2 = alt.Chart(
        train_data,
        title="Are we dealing with a class imbalance in our train data?").mark_bar(color="seagreen").encode(
        x=alt.X("is_hotdog", title="Is it a hot dog vendor?"),
        y=alt.Y("count()", title="Number of vendors")
    ).properties(
        width=85,
        height=495
    )

    plot2_path = os.path.join(plot_to, "plot2_class_imbalance.png")
    plot2.save(plot2_path, scale_factor=2.0)
    click.echo(f"Saved Plot 2 to {plot2_path}")

    #PLOT 3: Identifying missing and NAN values
    train_data["text_is_na"] = train_data["BUSINESS_NAME"] == ""

    plot3 = alt.Chart(
        train_data,
        title="Are blank names relevant for our classification?").mark_rect(color="seagreen").encode(
        x=alt.X("is_hotdog", title="Is it a hot dog vendor?"),
        y=alt.Y("text_is_na", title="Is it a blank BUSINESS_NAME?"),
        color=alt.Color("count()", title="# of observations")
    ).properties(
        width=200,
        height=200
    )


    plot3 = plot3 + alt.Chart(train_data).mark_text(
        fontSize=14, fontWeight="bold").encode(
        x="is_hotdog:N",
        y="text_is_na:N",
        text=alt.Text("count():Q", format="d")
    )

    plot3_path = os.path.join(plot_to, "plot3_blank_names_vs_hotdog.png")
    plot3.save(plot3_path, scale_factor=2.0)
    click.echo(f"Saved Plot 3 to {plot3_path}")
        
if __name__ == '__main__':
    main()