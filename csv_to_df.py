import pandas as pd
import numpy as np
import csv

"""
Convert reaction data csv to a dataframe
Data to get for each row:
1) reaction ID (for sorting)
2) all listed reagent information
3) maximum absorption

Since the CSV format stores absorption @ time with each timestep being it's own column,
the script needs to get the data from all those columns per row, and find the max element

Dataframe format:
ID | reagenent conc.1 | reagenent conc.2 | ... | max absorbancy |
-----------------------------------------------------------------

"""


def csv_to_pdf(file_list):
    # Empty dataframe
    dataset = pd.DataFrame()

    # Script handles all CSVs at once to build a large, useful dataframe
    for file_name in file_list:
        try:
            with open(file_name, newline='') as file:
                csv_reader = csv.reader(file)
        except Exception as e:
            print(e)
