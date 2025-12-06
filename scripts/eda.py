# eda.py
# author: Zaki Aslam 
# date: 2025-12-01

import os 
import click
import altair as alt
import pandas as pd

@click.command()
@click.option('--processed-training-data', type=str, help="Path to processed training data")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")
def main(processed_training_data, plot_to):
    