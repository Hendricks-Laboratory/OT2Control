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

Wavelength column IDs run from 300 to 1000 Reagent column IDs = ctab, trisodium_citrate, potassium_bromide, 
silver_nitrate, sodium_borohydride, hydrogen_peroxide, e_ctab, ectab

Dataframe format:
ID | reagenent conc.1 | reagenent conc.2 | ... | max absorbancy |
-----------------------------------------------------------------

"""

absorbancy_column_range = range(300, 1001)
reagent_column_ids = ["ctab", "trisodium_citrate", "potassium_bromide", "silver_nitrate",
                      "sodium_borohydride", "hydrogen_peroxide", "e_ctab", "ectab"]


def csv_to_pdf(file_list):
    # Empty dataframe
    df = pd.DataFrame(columns=["reaction_ID", "ctab", "trisodium_citrate", "potassium_bromide", "silver_nitrate",
                               "sodium_borohydride", "hydrogen_peroxide", "e_ctab", "ectab", "max_absorbancy"])

    # Script handles all CSVs at once to build a large, useful dataframe
    for file_name in file_list:
        try:
            with open(file_name, newline='') as file:
                csv_reader = csv.reader(file)

                row_index = 0
                for row in csv_reader:
                    # make array to store data for the row being constructed
                    row_data = [row_index]

                    # add all reagent data
                    for r in reagent_column_ids:
                        row_data.append(row[r])

                    # find and add max absorbency

                    # add row to df
                    df.loc[row_index] = row_data

        except Exception as e:
            print(e)
