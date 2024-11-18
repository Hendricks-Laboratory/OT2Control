import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


"""
plate(file: the csv output file of auto run, num_wells: the number of filled wells in the reaction, target: the lambda max target)
    This function takes the csv file and parses it for a lambda max between 400-900nm. 
    It subtracts the target lambda max to find closeness to target and it formats this 
    data into the dimensionsof a well plate and fills the empy wells with 0. Finally 
    returning the data with correct dimensions to be plotted
"""

def plate(file, num_wells, target):
    
    df = pd.read_csv(file)                                              # Convert csv to df 
    
    lambda_max = []                     
    for i in range(num_wells):          # For each row that is a well find the max absorbance and find the wavelength it is at 
        column_name = df.columns[df.loc[i] == max(df.iloc[i][105:606])].to_list()[0]
        lambda_max.append(int(column_name))
    
    zeros = []                                                          # Fill the empty wells with 0s as placeholders 
    for i in range(96-num_wells):
        zeros.append(0)
                                        
    data = np.array(lambda_max+zeros).reshape(8,12, order='F')         # A well plate is 8x12 and it is ordered tblr (Foran)
    data = abs(data-target)                                             # Calculate closeness to target for each well
    return data


"""
plot(data: correct wellplate shaped data that is ready to be plotted)
    This function plots a heatmap given wellplate shaped data using a custom colormap and 
    custom labels. 
"""

def heat_map(data):

    cmap = LinearSegmentedColormap.from_list("red_blue", ["#ff2222", "#00328b"], N=50)  # Custom colormap
    masked_data = np.ma.masked_where(data > 500, data)

    plt.imshow(masked_data, cmap=cmap, interpolation='nearest')
    plt.xticks(np.arange(12), np.arange(1, 13))                             # Custom labels that resemble our wellplates
    plt.yticks(np.arange(8), ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])

    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='both', which='both', length=0)                     # Removes tick marks

    plt.colorbar()
    plt.show()

