#INPUTS
run_kinetics_plotter = False
run_scan_plotter = False
run_all_data_plotter = True
run_abs_vs_time_plotter = False
run_lambda_max_plotter = False

#kinetics plots all scans of each provided well
#scan_plotter plot all wells of each provided scan
#all_data plots all data
#abs_vs_time plots data at provided wavelength for all scans for provided wells
#lambda_max plots data at provided time for all scans for provided wells




experiment_name = "CNH_008"
well_names = ["A05", "G05"], "G05", "H05"
scan_names = ["scan_1", "scan_3"], "scan_4"
wavelength_to_plot = 305, 400
wavelength_range = (450, 800)
save_or_not = True
# from matplotlib import rcParams
# rcParams.update({'figure.autolayout': True})
"""saving is currently not working due to: FileNotFoundError: [Errno 2] No such file or directory:
 '/content/drive/Shareddrives/Hendricks Lab Drive/Opentrons_Reactions/Protocol_Outputs/CNH_test/plots/CNH_test_A05, H05 at 305 nm.png'"""

#provide the name of the experiment, as spelled in the original sheet, in quotes. This is how the code will know where to find the data.
#ex: experiment_name = "CNH_test"
#provide the wells you want to plot (for the plotting methods that go by well). Provide each well name in quotes, and if you want multiple wells on the same plot include them together in brackets
#ex: well_names = ["A05", "G05"], "G05", "H05"
#provide the scans you want to plot (for the plotting methods that go by scan). Provide each scan name in quotes, and if you want multiple wells on the same plot include them together in brackets
#ex: scan_names = ["scan_1", "scan_3"], "scan_4"
#provide the wavelengths you want plotted as integers, separated by commas
#ex: wavelength_to_plot = 305, 400
#provide the wavelength range you want to find lambda max in, as integers in paranthesis separated by a comma with the lower bound first and upper bound second
#ex: wavelength_range = (450, 800)
#decide whether you want to save the plots with True or False



#THINGS TO CHANGE ABOUT STANDARD DATAFRAME
#add column to far left for experiment name (so we can compare multiple experiments in the future by merging the dstaframes)
#add column with reaction name (probably leave column with reaction well in there)
#add columns for time of last reagent added
#add columns for all reagent concentrations at the time of scan (so we can include this information in the plots)

#add option to work on local desktop rather than drive
#figure out how to subtract controls
#make GUI to make this more user-freindly
#update lambda_max_plotter to include two figures, the one of the left as-is, thee one on the right showing the position of lambda max vs. time
#add plate summary plot (in both normal and presentation modes)

#fix path/add saving


#Assumption: wavelengths are initially found in the top non-title row
#Assumption: the first data is always immediately to the right of the "Time" column

from posixpath import lexists
import math
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm



def load_df(path):
  df = pd.read_csv(path)
  wavelengths = list(df.iloc[0][df.columns.get_loc("Time")+1:])
  for i in range(1,len(wavelengths)+1):
    df = df.rename({str(i):wavelengths[i-1] }, axis='columns')
  df = df.rename({"Unnamed: 0":"Scan_Number" }, axis='columns')
  df["Scan ID"][0]="Wavelength"
  df["Well"][0]="Wavelength"

  wells = list(set(df.loc[1:,"Well"]))
  scans = list(set(df.loc[1:,"Scan ID"]))

  df = df.set_index(["Scan ID", "Well"]).sort_index()
  #print(df)
  #df.drop(df['Wavelength']['Wavelength'], inplace = True)
  df.to_csv('nicenice.csv')
  return df, wavelengths, scans, wells



def plot_prepper(wavelengths, plot_title):
  number_wavelength_tick_marks = 8
  number_y_tick_marks = 5
  y_max = 1
  y_min = 0
  y_tick_marks = [i/number_y_tick_marks for i in range(0,number_y_tick_marks+1,1)]
  y_tick_marks = [y_max*i for i in y_tick_marks]

  (wavelength_min, wavelength_max) = (wavelengths[0], wavelengths[-1])
  wavelength_range = wavelength_max - wavelength_min
  wavelength_tick_mark_separation = wavelength_range / (number_wavelength_tick_marks-1)
  wavelength_tick_marks = [wavelength_min]
  for i in range(number_wavelength_tick_marks-1):
    wavelength_tick_marks = wavelength_tick_marks + [wavelength_tick_marks[-1]+wavelength_tick_mark_separation]

  plt.figure(num=None, figsize=(4, 4),dpi=300, facecolor='w', edgecolor='k')
  plt.legend(loc="upper right",frameon = False, prop={"size":7},labelspacing = 0.5)
  plt.rc('axes', linewidth = 2)
  plt.xlabel('Wavelength (nm)',fontsize = 16)
  plt.ylabel('Absorbance (a.u.)', fontsize = 16)
  plt.tick_params(axis = "both", width = 2)
  #plt.tick_params(axis = "both", width = 2)
  plt.xticks(wavelength_tick_marks)
  plt.yticks(y_tick_marks)
  plt.axis([wavelength_min, wavelength_max, y_min , y_max])
  plt.xticks(fontsize = 14)
  plt.yticks(fontsize = 14)
  plt.title(str(plot_title), fontsize = 16, pad = 20)



def plot_prepper2_part1(plot_title):
  number_y_tick_marks = 5
  y_max = 1
  y_min = 0
  time_min = 0
  time_max = 10

  y_tick_marks = [i/number_y_tick_marks for i in range(0,number_y_tick_marks+1,1)]
  y_tick_marks = [y_max*i for i in y_tick_marks]


  plt.figure(num=None, figsize=(4, 4),dpi=300, facecolor='w', edgecolor='k')
  plt.legend(loc="upper right",frameon = False, prop={"size":7},labelspacing = 0.5)
  plt.rc('axes', linewidth = 2)
  plt.xlabel('Time (minutes)',fontsize = 16)
  plt.ylabel('Absorbance (a.u.)', fontsize = 16)
  plt.tick_params(axis = "both", width = 2)
  plt.yticks(y_tick_marks)
  plt.axis([time_min, time_max, y_min , y_max])
  plt.yticks(fontsize = 14)
  plt.title(str(plot_title), fontsize = 16, pad = 20)


def plot_prepper2_part2(total_time):
  number_time_tick_marks = 5
  time_min = 0
  y_max = 1
  y_min = 0

  #rounds the x-axis maximum up to the nearest 10 minute, i.e. both 11 min and 19 min round to 20 min
  time_max = (int(math.ceil(total_time/10)))*10
  x_tick_marks = [i/number_time_tick_marks for i in range(0,number_time_tick_marks+1,1)]
  x_tick_marks = [time_max*i for i in x_tick_marks]
  plt.axis([time_min, time_max, y_min , y_max])
  plt.tick_params(axis = "both", width = 2)
  plt.xticks(x_tick_marks)
  plt.xticks(fontsize = 14)
 

def kinetics_plotter(df, wavelengths, well_names, experiment_name, save_or_not, save_path):
  
  plot_title = ", ".join(well_names) + " Kinetics"
  plot_prepper(wavelengths, plot_title)

  list_of_time_strings = []
  list_of_temperatures = []
  well_name_check = ""
  data = df.loc[pd.IndexSlice[:, ["Wavelength"] + well_names], :]
  number_of_scans = data.shape[0]
  colors = list(cm.rainbow(np.linspace(0, 1, number_of_scans)))
  
  for i in range(1,number_of_scans):
    if str(data.index.tolist()[i][1]) != well_name_check:
      initial_time = datetime.strptime(data.iloc[i,data.columns.get_loc("Time")], "%m/%d/%y %H:%M")
      well_name_check = str(data.index.tolist()[i][1])
    scan_time = datetime.strptime(data.iloc[i,data.columns.get_loc("Time")], "%m/%d/%y %H:%M")
    time_difference = scan_time - initial_time
    if len(well_names) == 1:
      list_of_time_strings = list_of_time_strings + [str(time_difference.total_seconds()/60) +" minutes"]
    else: 
      list_of_time_strings = list_of_time_strings + [str(data.index.tolist()[i][1]) + " " + str(time_difference.total_seconds()/60) +" minutes"]
    list_of_temperatures = list_of_temperatures + [data.iloc[i,data.columns.get_loc("Temp")]]

    x_data = data.iloc[data.index.get_loc("Wavelength"),df.columns.get_loc("Time")+1:].transpose()
    y_data = data.iloc[i,df.columns.get_loc("Time")+1:]

    plt.plot(x_data, y_data,color=tuple(colors[i-1]))

  patches = [mpatches.Patch(color=color, label=list_of_time_strings) for list_of_time_strings, color in zip(list_of_time_strings, colors)]
  plt.legend(patches, list_of_time_strings, loc='upper right', frameon=False,prop={'size':8})
  
  max_temp = max(list_of_temperatures)
  min_temp = min(list_of_temperatures)
  metadata_text = experiment_name + "\n" + "Range of temperatures during this experiment: " + str(min_temp) + " - " + str(max_temp) + " C \n The first scan for this experiment occured at: " + str(initial_time)
  plt.gcf().text(0.5, -0.2, metadata_text, fontsize = 8, horizontalalignment="center")

  save_path = save_path + experiment_name + "_" + "_".join(well_names) + "_kinetics.png"
  if save_or_not: plt.savefig(save_path, bbox_inches='tight')


def scan_plotter(df, wavelengths, scan_names, experiment_name, save_or_not, save_path):
  plot_title = "Data from " + ", ".join(scan_names)
  plot_prepper(wavelengths, plot_title)

  data = df.loc[pd.IndexSlice[["Wavelength"] + scan_names],:, :]
  list_of_wells = []
  list_of_temperatures = []
  number_of_wells = data.shape[0]
  colors = list(cm.rainbow(np.linspace(0, 1, number_of_wells)))
  initial_time = datetime.strptime(data.iloc[1,data.columns.get_loc("Time")], "%m/%d/%y %H:%M")

  for i in range(1,number_of_wells):
    if len(scan_names) == 1:
      list_of_wells = list_of_wells + [data.index.tolist()[i][1]]
    else: 
      list_of_wells = list_of_wells + [data.index.tolist()[i][0] + " " + data.index.tolist()[i][1]]
    
    list_of_temperatures = list_of_temperatures + [data.iloc[i,data.columns.get_loc("Temp")]]

    x_data = data.iloc[data.index.get_loc("Wavelength"),df.columns.get_loc("Time")+1:].transpose()
    y_data = data.iloc[i,df.columns.get_loc("Time")+1:]

    plt.plot(x_data, y_data,color=tuple(colors[i-1]))

  patches = [mpatches.Patch(color=color, label=list_of_wells) for list_of_wells, color in zip(list_of_wells, colors)]
  plt.legend(patches, list_of_wells, loc='upper right', frameon=False,prop={'size':8})

  max_temp = max(list_of_temperatures)
  min_temp = min(list_of_temperatures)
  metadata_text = experiment_name + "\n" + "Range of temperatures during this experiment: " + str(min_temp) + " - " + str(max_temp) + " C \n The first scan for this experiment occured at: " + str(initial_time)
  plt.gcf().text(0.5, -0.2, metadata_text, fontsize = 8, horizontalalignment="center")

  save_path = save_path + experiment_name + "_" + "_".join(scan_names) + ".png"
  if save_or_not: plt.savefig(save_path, bbox_inches='tight')

def all_data_plotter(df, wavelengths, experiment_name, save_or_not, save_path):
  plot_prepper(wavelengths, "All Data")

  data = df
  list_of_scans = []
  list_of_temperatures = []
  number_of_scans = data.shape[0]
  colors = list(cm.rainbow(np.linspace(0, 1, number_of_scans)))
  #initial_time = datetime.strptime(data.iloc[1,data.columns.get_loc("Time")], "%m/%d/%y %H:%M:")
  initial_time = "x"

  for i in range(0,number_of_scans):
    list_of_scans = list_of_scans + [str(data.index.tolist()[i][0]) + " " + str(data.index.tolist()[i][1])]
    list_of_temperatures = list_of_temperatures + [data.iloc[i,data.columns.get_loc("Temp")]]
    x_data = data.iloc[data.index.get_loc("Wavelength"),df.columns.get_loc("Time")+1:].transpose()
    y_data = data.iloc[i,df.columns.get_loc("Time")+1:]

    plt.plot(x_data, y_data,color=tuple(colors[i-1]))

  patches = [mpatches.Patch(color=color, label=list_of_scans) for list_of_scans, color in zip(list_of_scans, colors)]
  plt.legend(patches, list_of_scans, loc='upper right', frameon=False,prop={'size':4})

  max_temp = max(list_of_temperatures)
  min_temp = min(list_of_temperatures)
  metadata_text = experiment_name + "\n" + "Range of temperatures during this experiment: " + str(min_temp) + " - " + str(max_temp) + " C \n The first scan for this experiment occured at: " + str(initial_time)
  plt.gcf().text(0.5, -0.2, metadata_text, fontsize = 8, horizontalalignment="center")
  save_path = save_path + experiment_name + "_all_data.png"
  if save_or_not: plt.savefig(save_path, bbox_inches='tight')

def abs_vs_time_plotter(df, wavelengths, well_names, wavelength_to_plot, experiment_name, save_or_not, save_path):
  
  
  title = ", ".join(well_names) + " at " + str(wavelength_to_plot) + " nm"
  plot_prepper2_part1(title)

  number_of_lines = len(well_names)
  list_of_temperatures = []
  colors = list(cm.rainbow(np.linspace(0, 1, number_of_lines)))

  #this is a bad hack to get total_time to be a datetime delta object equal to 0
  time_1 = datetime.strptime("00:10","%H:%M")
  time_2 = datetime.strptime("00:10","%H:%M")
  total_time = time_1 - time_2

  for wells in well_names:
    data = df.loc[pd.IndexSlice[:, ["Wavelength", wells], :]]
    print(data.columns.get_loc("Time"))
    fake = data
    fake.to_csv('fakke.csv')
    initial_time = datetime.strptime(data.iloc[1,data.columns.get_loc("Time")], "%m/%d/%y %H:%M")
    list_of_time_floats = []
    abs_data = []
    number_of_scans = data.shape[0]

    for i in range(1,number_of_scans):
      scan_time = datetime.strptime(data.iloc[i,data.columns.get_loc("Time")], "%m/%d/%y %H:%M")
      time_difference = scan_time - initial_time
      if time_difference > total_time: total_time = time_difference
      list_of_time_floats = list_of_time_floats + [time_difference.total_seconds()/60]
      abs_data = abs_data + [data.iloc[i,data.columns.get_loc(wavelength_to_plot)]]
      list_of_temperatures = list_of_temperatures + [data.iloc[i,data.columns.get_loc("Temp")]]

    x_data = list_of_time_floats
    y_data = abs_data
    plt.plot(x_data, y_data,linestyle='--', marker='o', color=tuple(colors[well_names.index(wells)]))

  total_time = total_time.total_seconds()/60
  plot_prepper2_part2(total_time)
  patches = [mpatches.Patch(color=color, label=well_names) for well_names, color in zip(well_names, colors)]
  if len(well_names) != 1:
    plt.legend(patches, well_names, loc='upper right', frameon=False,prop={'size':8})

  max_temp = max(list_of_temperatures)
  min_temp = min(list_of_temperatures)
  metadata_text = experiment_name + "\n" + "Range of temperatures during this experiment: " + str(min_temp) + " - " + str(max_temp) + " C \n The first scan for this experiment occured at: " + str(initial_time)
  plt.gcf().text(0.5, -0.2, metadata_text, fontsize = 8, horizontalalignment="center")

  save_path = save_path + experiment_name + "_" + title + ".png"
  if save_or_not: plt.savefig(save_path, bbox_inches='tight')

def lambda_max_plotter(df, wavelengths, well_names, wavelength_range, experiment_name, save_or_not, save_path):

  title = ", ".join(well_names) + " Max in Range" + str(wavelength_range) + " Plot"
  plot_prepper2_part1(title)

  number_of_lines = len(well_names)
  list_of_temperatures = []
  colors = list(cm.rainbow(np.linspace(0, 1, number_of_lines)))

  #this is a bad hack to get total_time to be a datetime delta object equal to 0
  time_1 = datetime.strptime("00:10","%H:%M")
  time_2 = datetime.strptime("00:10","%H:%M")
  total_time = time_1 - time_2

  for wells in well_names:
    data = df.loc[pd.IndexSlice[:, ["Wavelength", wells], :]]
    absorbance_data = data.loc[:, wavelength_range[0]:wavelength_range[1]]
    lambda_max_wavelengths = absorbance_data.idxmax(axis = 1).tolist()[1:]
    absorbance_max = absorbance_data.max(axis = 1).tolist()[1:]

    initial_time = datetime.strptime(data.iloc[1,data.columns.get_loc("Time")], "%m/%d/%y %H:%M")
    list_of_time_floats = []
    abs_data = []
    list_of_lambda_max = []
    number_of_scans = data.shape[0]

    for i in range(1,number_of_scans):
      scan_time = datetime.strptime(data.iloc[i,data.columns.get_loc("Time")], "%m/%d/%y %H:%M")
      time_difference = scan_time - initial_time
      if time_difference > total_time: total_time = time_difference
      list_of_time_floats = list_of_time_floats + [time_difference.total_seconds()/60]
      list_of_temperatures = list_of_temperatures + [data.iloc[i,data.columns.get_loc("Temp")]]

    x_data = list_of_time_floats
    y_1_data = lambda_max_wavelengths
    y_2_data = absorbance_max
    plt.plot(x_data, y_2_data,linestyle='--', marker='o', color=tuple(colors[well_names.index(wells)]))

  total_time = total_time.total_seconds()/60
  plot_prepper2_part2(total_time)
  patches = [mpatches.Patch(color=color, label=well_names) for well_names, color in zip(well_names, colors)]
  if len(well_names) != 1:
    plt.legend(patches, well_names, loc='upper right', frameon=False,prop={'size':8})

  max_temp = max(list_of_temperatures)
  min_temp = min(list_of_temperatures)
  metadata_text = experiment_name + "\n" + "Range of temperatures during this experiment: " + str(min_temp) + " - " + str(max_temp) + " C \n The first scan for this experiment occured at: " + str(initial_time)
  plt.gcf().text(0.5, -0.2, metadata_text, fontsize = 8, horizontalalignment="center")

  save_path = save_path + experiment_name + "_" + title + ".png"
  if save_or_not: plt.savefig(save_path, bbox_inches='tight')









def run_plotters(run_kinetics_plotter, run_scan_plotter, run_all_data_plotter, run_abs_vs_time_plotter, experiment_name, well_names, scan_names, wavelength_to_plot, wavelenth_range, save_or_not):

  #load_path = "/content/drive/Shareddrives/Hendricks Lab Drive/Opentrons_Reactions/Protocol_Outputs/" + experiment_name + "/pr_data/" + experiment_name + "_full_df.csv"
  #save_path = "/content/drive/Shareddrives/Hendricks Lab Drive/Opentrons_Reactions/Protocol_Outputs/" + experiment_name + "/plots/" 
  load_path = "/Users/grantdidway/Documents/OT2Robot/Personal Testing/"  + experiment_name + "_full_df2.csv"
  save_path = "/Users/grantdidway/Documents/OT2Robot/Personal Testing/" 


  well_names = list(well_names)
  scan_names = list(scan_names)
  if isinstance(wavelength_to_plot, tuple): wavelength_to_plot = list(wavelength_to_plot)
  if isinstance(wavelength_to_plot, int): wavelength_to_plot = [wavelength_to_plot]

  df, wavelengths, scans, wells =load_df(load_path)

  if run_kinetics_plotter:
    for i in well_names:
      if not isinstance(i, list): i = [i]
      kinetics_plotter(df,wavelengths, i, experiment_name, save_or_not, save_path)
  if run_scan_plotter:
    for i in scan_names:
      if not isinstance(i, list): i = [i]
      scan_plotter(df,wavelengths, i, experiment_name, save_or_not, save_path)
  if run_all_data_plotter:
    all_data_plotter(df,wavelengths, experiment_name, save_or_not, save_path)
  if run_abs_vs_time_plotter:
    for i in well_names:
      for j in wavelength_to_plot:
        if not isinstance(i, list): i = [i]
        abs_vs_time_plotter(df,wavelengths, i, j, experiment_name, save_or_not, save_path)
  if run_lambda_max_plotter:
    for i in well_names:
        if not isinstance(i, list): i = [i]
        lambda_max_plotter(df,wavelengths, i, wavelength_range, experiment_name, save_or_not, save_path)






run_plotters(run_kinetics_plotter, run_scan_plotter, run_all_data_plotter, run_abs_vs_time_plotter, experiment_name, well_names, scan_names, wavelength_to_plot, wavelength_range, save_or_not)

