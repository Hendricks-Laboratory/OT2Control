#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 13:39:14 2022

@author: grantdidway
"""


import json

import math
import os
import shutil
import webbrowser
from tempfile import NamedTemporaryFile
import logging
import asyncio
import threading
import time
import argparse
import re
import functools
import datetime



import pandas as pd
import numpy as np

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm



def main():
    '''
    prompts for input and then calls appropriate launcher
    '''
    TESTING_FILEPATH = "/mnt/c/Users/science_356_lab/Robot_Files/OT2Control/Personal Testing"

    
    data = ScanDataFrame(TESTING_FILEPATH, 'CNH_008',TESTING_FILEPATH)
    
    # for filename in os.listdir(TESTING_FILEPATH):
    #     if filename.endswith('.csv'):
    #         print(filename)
    #         data.AddToDF(os.path.join(os.curdir, filename))
    
    #data.df.to_csv('23232.csv')
    data.AddToDF("CNH_008_fullscan_1.csv")
    data.AddReagentInfo()
    data.AddToDF("CNH_008_fullscan_2.csv")
    data.AddReagentInfo()
    data.AddToDF("CNH_008_t0.csv")
    data.AddReagentInfo()
    
   
    
    
    #data.df.set_index(['Scan ID, Well'])
    #data.df = data.df[['Scan ID', 'Well', 'Time', 'Temp']+[str(i) for i in range(301,999)]]
    data.df.info()
    #data.df.index.set_names(['Scan ID', 'Well'], inplace=True)
    
   
    print(data.df.columns)
    
    
    #3data.df = data.df.groupby('Scan ID')
    #print(mean)
    print(data.df.head())
    #data.df.to_csv("CNH_008_full_df2.csv")
    print(data.df.index.names)
    print(data.df.index)
    

class ScanDataFrame():
    '''
    This class handles and saves data 
    
    ATTRIBUTES:  
        df df: Not passed in but created in init. Pandas Dataframe to be used 
            to store all scans from the run.
        str data_path: pathname for local platereader data.
    
    METHODS:  
        add_to_df() void: formats data from a scan and adds to the data frame.
        
    '''
    
    def __init__(self, data_path, header_data, eve_files_path):
        self.df = pd.DataFrame()
        self.data_path = data_path
        self.eve_files_path = eve_files_path
        #self.experiment_name = {row[0]:row[1] for row in header_data[1:]}['data_dir']
        self.experiment_name = "CNH_008"
        self.isFirst = True
        self.isFirst1 = True
        self.add = True
        
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        
    def AddToDF(self, file_name):
        
        temp_file = os.path.join(self.data_path,file_name)
    
 #Extracts and stores data/time metadata for the time column

        
        df_read_data_1 = pd.read_csv(temp_file,nrows = 35,skiprows = [7], header=None,na_values=["       -"],encoding = 'latin1')   
        am_pm = df_read_data_1.iloc[1][0].split(" ")[5]
        hour = int(df_read_data_1.iloc[1][0].split(" ")[4].split(":")[0]) + 12 if am_pm == "PM" else int(df_read_data_1.iloc[1][0].split(" ")[4].split(":")[0])
        date_time = datetime.datetime(int(df_read_data_1.iloc[1][0].split(" ")[1].split("/")[2]), int(df_read_data_1.iloc[1][0].split(" ")[1].split("/")[0]), int(df_read_data_1.iloc[1][0].split(" ")[1].split("/")[1]), hour, int(df_read_data_1.iloc[1][0].split(" ")[4].split(":")[1]), int(df_read_data_1.iloc[1][0].split(" ")[4].split(":")[2]))
        num_cycles = int(df_read_data_1.iloc[4][0][15:])
        if num_cycles != 1:
            raise Exception('Error due to Bad Scan Protocol: too many cycles')
        
#Extracts and stores wavelength metadata

        df_read_data_2 = pd.read_csv(temp_file,nrows = 3, skiprows = 43, header=None,na_values=["       -"],encoding = 'latin1')
        wavelength_blue = int(df_read_data_2[0][0][13:].split("nm", 2)[0])
        wavelength_red = int(df_read_data_2[0][0][13:].split("nm", 2)[1][3:])
        wavelength_steps = int(df_read_data_2[1][0].split("nm", 1)[0][1:])

#Extracts and stores temp metadata for the temp column
        df_read_data_3 = pd.read_csv(temp_file,nrows = 3, skiprows = 45, header=None,na_values=["       -"],encoding = 'latin1')    
        temp = float(df_read_data_3[0][2].split(" ")[-1])
        
    
#Extracts and stores absorbance data 
        
        data_df = pd.read_csv(temp_file,skiprows=48,header=None,na_values=["       -"],encoding = 'latin1',)
        
        g=data_df.iloc[:,0]
        g = [x.rstrip(':') for x in g]
        data_df = data_df.drop(data_df.columns[0], axis=1)
        wavvelengths = []
        for x in range (wavelength_blue,wavelength_red+1, wavelength_steps):
            wavvelengths = wavvelengths + [x]
    
        
        #Combines metadata with absorbance data
        data_df.columns = wavvelengths
        
        
        data_df.insert(0,"Time",date_time) 
        data_df.insert(0,"Temp",temp)
        data_df.insert(0, 'Well', g)
        data_df.insert(0,"Scan ID",file_name.rstrip('.csv'))
        data_df = data_df.set_index(['Scan ID','Well'])
        
        df1 = pd.read_csv(os.path.join(self.eve_files_path, 'translated_wellmap.tsv'), sep='\t')
        #df = pd.read_csv('CNH_008_full_df1.csv')

        # self.df = self.df.reset_index()

        # wavelengths = list(self.df.iloc[0][self.df.columns.get_loc("Time")+1:])
        # for i in range(1,len(wavelengths)+1):
        #   self.df = self.df.rename({str(i):wavelengths[i-1] }, axis='columns')
        # self.df = self.df.rename({"Unnamed: 0":"Scan_Number" }, axis='columns')

          
        # wells = list(set(self.df.loc[1:,"Well"]))
        # scans = list(set(self.df.loc[1:,"Scan ID"]))
        # self.df.drop('index', inplace=True, axis=1)
        # self.df = self.df.set_index(["Scan ID", "Well"]).sort_index()

        col_list  = data_df.index.get_level_values('Well').tolist()

        well_names = []

        for well_with_zeros_in_name in col_list:
            x = ''
            #Turn A01 into A1
            well = str(well_with_zeros_in_name)
            print(well)
            well = "{}{}".format(well[0], int(well[2:]))
           
            y = df1.loc[df1['loc'] == str(well), 'chem_name'].values[:]
            for i in y:
                if str(self.experiment_name) in i:
                    x=i
                if ('control'in i.lower()) or ('blank' in i.lower()):
                        x = 'blank'
                print(x)
            
            well_names.append(x)

        data_df.insert(0, 'Well Name', well_names)

        
        
        
        #data_df.to_csv(file_name+'{}'.format('yayaaa.csv'))
        #self.df.to_csv(file_name+'{}'.format('computer.csv'))
        
        
        if self.isFirst:
            self.df = data_df
            #full_df = pd.concat([wavvelength_df,data_df])
            self.isFirst = False
            
        else:
            full_df = data_df
        
            self.df = pd.concat([full_df,self.df])
        
        #self.df = self.df.sort_values(by=['Scan ID', 'Well'])
        self.df = self.df.sort_values(['Time', 'Well'])
        self.df.to_csv("CNH_008_full_df.csv")
        #self.df.to_csv(file_name+'{}'.format('yuss.csv'))
    def AddReagentInfo(self):
        
        reaction = self.experiment_name

        well_hist_df = pd.read_csv(os.path.join(self.eve_files_path,'well_history.tsv'), sep='\t')

        timess = well_hist_df.timestamp.values.tolist()
        for time in timess:
          
            index = timess.index(time)

            time = pd.Timestamp(time)
            given_time = time - pd.DateOffset(hours=7)
            given_time = given_time.strftime('%Y-%m-%d %H:%M:%S:%f')
            timess[index] = given_time


        well_hist_df['timestamp'] = timess

        df = pd.read_csv(reaction+'_full_df.csv')



        chem_names = ['KBR', 'unsure', 'AGNO3', 'H2O2', 'NABH4']


        KBR = 'potassium_bromide'
        unsure = 'trisodium_citrate'
        AGNO3 = 'silver_nitrate'
        H2O2 = 'hydrogen_peroxide'
        NABH4 = 'sodium_borohydride'

        reagents = [KBR, unsure, AGNO3, H2O2, NABH4]

        KBR_list = []
        unsure_list = [] 
        AGNO3_list = []
        H2O2_list = [] 
        NABH4_list = []

        reagent_lists = [KBR_list, unsure_list, AGNO3_list, H2O2_list, NABH4_list]

        containers = []
        times = []
        volumes =[]
        vols = []
        indices = []
        cons = []
        chems = []
        con_list  = well_hist_df['container'].tolist()
        for container in con_list:
            if (reaction in  container) or ('blank' in container) or ('control' in container):
                volume = well_hist_df.loc[well_hist_df['container'] == container].index.tolist()
                #print(volume)
                
                indices.append(container)
                indices = list(set(indices))

                vols.extend(volume)
                vols = list(set(vols))




        for index in vols:
            chemical = well_hist_df["chemical"].iloc[index]
            head, sep, tail = chemical[3:].partition('C')
            chemical = str(chemical[:3] + head)
            chems.append(chemical)
            
            concentration = tail
            cons.append(concentration)
            timestamp = well_hist_df["timestamp"].iloc[index]
            times.append(timestamp)
            volume = well_hist_df["vol"].iloc[index]
            volumes.append(volume)
            container = well_hist_df["container"].iloc[index]
            head, sep, tail = container[3:].partition('C')
            container = str(container[:3] + head +sep+tail)
            containers.append(container)
            
            
            
            

                
                
            
            
            
            #concentration = tail
        pddict = {'time':times, 'vol':volumes, 'cont':containers, 'chem':chems, 'conc': cons} 
        yay = pd.DataFrame(pddict)


        df2 = yay.sort_values(by = ['cont', 'time'], ascending = [True, True])


        n = len(pd.unique(df2['cont']))




        base = df2.loc[(df2['cont'].str.contains('blank'))|(df2['cont'].str.contains('control'))]
        base['KBR'],base['unsure'],base['AGNO3'],base['H2O2'],base['NABH4'] = 0,0,0,0,0
        for i in indices: 

            if 'blank' not in i and 'control' not in i:
                temp = df2.loc[df2['cont']==i]
                temp.sort_values(by='time')
                
                
                
                volumes = temp.vol.values.tolist()
              
                sum_volumes = [sum(volumes[0:i[0]+1]) for i in enumerate(volumes)]
            
                concentrations = temp.conc.tolist()
               
                count = 0
                
                KBR_conc, UNSURE_conc, AGNO3_conc, H2O2_conc, NABH4_conc = [0], [0], [0], [0], [0]
                reagent_conc = [KBR_conc, UNSURE_conc, AGNO3_conc, H2O2_conc, NABH4_conc]
                for chem in temp.chem.tolist():
                    
                   
                    flag = False
                    for reagent in reagents:
                        if reagent in chem:
                            flag = True
                  
                    if (chem in reagents) or (flag):
                        for reagent in reagents:
                           
                            if reagent in chem:
                               
                                reagent_conc[reagents.index(reagent)].append(float(volumes[count])*float(concentrations[count])/float(sum_volumes[count]))
                                for x in reagents:
                                    if x != reagent:
                                        
                                        reagent_conc[reagents.index(x)].append(reagent_conc[reagents.index(x)][count]*(float(sum_volumes[count-1]/float(sum_volumes[count]))))
                           
                    elif 'water' in chem.lower():
                       
                        for x in reagents:
                            reagent_conc[reagents.index(x)].append(reagent_conc[reagents.index(x)][count]*(float(sum_volumes[count-1]/float(sum_volumes[count]))))
                   
                    count += 1
            
                
                for i in reagent_conc:
                    temp[chem_names[reagent_conc.index(i)]] = i[1:]
             
                base = pd.concat([base, temp])
            
        full= base.sort_values(by = ['cont', 'time'], ascending = [True, True])



        weird = []
        last_reagent = []


        scans = list(set(df['Scan ID'].tolist()))
        reactions = list(set((df['Well Name'].tolist())))


        kbr_conc_list = []
        unsure_conc_list = []
        agno3_conc_list = []
        h202_conc_list = []
        nabh4_conc_list = []






        df.set_index('Scan ID',inplace = True)


        for scan in df.index.get_level_values('Scan ID').unique():
         

            for time in df.loc[scan]['Time']:
                time = time


            for react in df.loc[scan]['Well Name']:
                transfers_before_scans = []
                react = str(react)
                print(react)
                if 'blank' in react:
                    react = 'control'
                transfer_times = full.loc[full['cont'].str.contains(react),'time'].tolist()
                
          
                for transfer_time in transfer_times:
                    if transfer_time <= time:
                        print('heyeyeyyeye', transfer_time)
                        transfers_before_scans.append(transfer_time)
                latest_transfer_time = max(transfers_before_scans)
                
                
                
                kbr_conc = full[(full['cont'] == react) & (full['time'] == latest_transfer_time)]['KBR'].tolist()
                if len(kbr_conc)==0:
                    kbr_conc_list.append(0)
                else:
                    kbr_conc_list.append(kbr_conc[0])
                    
                    
                unsure_conc = full[(full['cont'] == react) & (full['time'] == latest_transfer_time)]['unsure'].tolist()
                if len(unsure_conc)==0:
                    unsure_conc_list.append(0)
                else:
                    unsure_conc_list.append(unsure_conc[0])
                    
                    
                agno3_conc = full[(full['cont'] == react) & (full['time'] == latest_transfer_time)]['AGNO3'].tolist()
                if len(agno3_conc)==0:
                    agno3_conc_list.append(0)
                else:
                    agno3_conc_list.append(agno3_conc[0])
                    
                h202_conc = full[(full['cont'] == react) & (full['time'] == latest_transfer_time)]['H2O2'].tolist()
                if len(h202_conc)==0:
                    h202_conc_list.append(0)
                else:
                    h202_conc_list.append(h202_conc[0])
                    
                nabh4_conc = full[(full['cont'] == react) & (full['time'] == latest_transfer_time)]['NABH4'].tolist()
                if len(h202_conc)==0:
                    nabh4_conc_list.append(0)
                else:
                    nabh4_conc_list.append(nabh4_conc[0])
                
                
                    
                   
                    
                   
                weird.append(latest_transfer_time)
                last_reagent_added = full[full['time']==latest_transfer_time]['chem'].item()
                print(last_reagent_added)
                last_reagent.append(last_reagent_added)
                
                
               
                        
                    

        df['time of last reagent added'] = weird
        df['last reagent added'] = last_reagent

        df['KBR'] = kbr_conc_list
        df['NA3C6H5O7'] = unsure_conc_list
        df['AGNO3'] = agno3_conc_list
        df['H2O2'] = h202_conc_list
        df['NABH4'] = nabh4_conc_list

        df.reset_index(inplace=True)
        df = df[['Scan ID', 'Well', 'Well Name', 'KBR', 'NA3C6H5O7', 'AGNO3', 'H2O2', 'NABH4', 'time of last reagent added','last reagent added', 'Temp', 'Time']+
                [c for c in df if c not in ['Scan ID', 'Well', 'Well Name', 'KBR', 'NA3C6H5O7', 'AGNO3', 'H2O2', 'NABH4', 'time of last reagent added','last reagent added', 'Temp', 'Time']] ]
                    
        df.to_csv(os.path.join(self.data_path, reaction + '_full.csv'))
       
        
#Prints export values    
        # print(" \n \nUseful Parameters:")
        # print("Number of cycles")
        # print(num_cycles)
        # print("Date Time")
        # print(date_time)  
        # print("wavelength_blue")
        # print(wavelength_blue)    
        # print("wavelength_red")
        # print(wavelength_red)    
        # print("wavelength_steps")
        # print(wavelength_steps)
        # print("Temperature")
        # print(temp)
        # print("Scan ID")
        # print(file_name)

        
    # def CombineWellNames(self,first):
    #         #self.df = self.df.set_index(["Scan ID", "Well"]).sort_index()
    #         df1 = pd.read_csv(os.path.join(self.eve_files_path, 'translated_wellmap.tsv'), sep='\t')
    #         #df = pd.read_csv('CNH_008_full_df1.csv')
    
    #         # self.df = self.df.reset_index()
    
    #         # wavelengths = list(self.df.iloc[0][self.df.columns.get_loc("Time")+1:])
    #         # for i in range(1,len(wavelengths)+1):
    #         #   self.df = self.df.rename({str(i):wavelengths[i-1] }, axis='columns')
    #         # self.df = self.df.rename({"Unnamed: 0":"Scan_Number" }, axis='columns')
    
              
    #         # wells = list(set(self.df.loc[1:,"Well"]))
    #         # scans = list(set(self.df.loc[1:,"Scan ID"]))
    #         # self.df.drop('index', inplace=True, axis=1)
    #         # self.df = self.df.set_index(["Scan ID", "Well"]).sort_index()
    
    #         col_list  = self.df.index.get_level_values('Well').tolist()

    #         well_names = []
    
    #         for well_with_zeros_in_name in col_list:
    #             x = ''
    #             #Turn A01 into A1
    #             well = str(well_with_zeros_in_name)
    #             print(well)
    #             well = "{}{}".format(well[0], int(well[2:]))
               
    #             y = df1.loc[df1['loc'] == str(well), 'chem_name'].values[:]
    #             #print(y)
    #             for i in y:
    #                 if str(self.experiment_name) in i:
    #                     x=i
    #                     if 'control' in x:
    #                         x = 'blank'
                
    #             well_names.append(x)
    
    #         self.df.insert(0, 'Well Name', well_names)


if __name__ == '__main__':
    main()