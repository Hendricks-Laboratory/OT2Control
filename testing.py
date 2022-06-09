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
    data = ScanDataFrame("/Users/grantdidway/Documents/OT2Robot/OT2Control/")
    #data.CreateDF()
    data.AddToDF("CNH_002_t0.csv", "final")
    data.AddToDF("CNH_002_fullscan_1.csv", "final")
    data.AddToDF("CNH_004_t0.csv", "final")
    data.AddToDF("CNH_004_fullscan_1.csv", "final")
    data.AddToDF("CNH_004_fullscan_2.csv", "final")
    
    
    #data.df.set_index(['Scan, Well'])
    #data.df = data.df[['Scan ID', 'Well', 'Time', 'Temp']+[str(i) for i in range(301,999)]]
    data.df.to_csv("hmm_very_nice.csv")
    print(data.df.index.names)
    
    

class ScanDataFrame():
    '''
    This class handles data output.
    '''
    
    def __init__(self, data_path):
        self.df = pd.DataFrame()
        self.data_path = data_path
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        self.isEmpty = True
        
    def load_reader_data(self, filename, loc_to_name):
        '''
        takes in the filename of a reader output and returns a dataframe with the scan data
        loaded, and a dictionary with relevant metadata.  
        Note that only the wells specified in loc_to_name will be returned.  
        params:  
            str filename: the name of the file to read without extension  
            df: the scan data for the wellnames supplied in loc_to_name for that file.  
        returns:  
            df: the scan data for that file  
            dict<str:obj>: holds the metadata  
                str filename: the filename as you passed in  
                int n_cycles: the number of cycles  
        '''
        filename = "{}.csv".format(filename)
        #parse the metadata
        start_i, metadata = self._parse_metadata(filename)
        # Read data ignoring first metadata lines
        df = pd.read_csv(os.path.join(self.data_path,filename), skiprows=start_i,
                header=None,index_col=0,na_values=["       -"],encoding = 'latin1').T
        headers = ["{}{}".format(x[0], int(x[1:-1])) for x in df.columns] #rename A01->A1
        df.columns = headers
        #get only the things we want
        df = df[loc_to_name.keys()]
        #rename by wellname
        df.rename(columns=loc_to_name, inplace=True)
        df.dropna(inplace=True)
        df = df.astype(float)
        return df, metadata

    def _parse_metadata(self, filename):
        '''
        parses the meta data of a platereader output, and returns a dataframe of the scans
        and a dictionary of parameters  
        params:  
            str filename: the name of the file to be read  
        returns:  
            int: the index to start reading the dataframe at  
            dict<str:obj>: holds the metadata  
                str filename: the filename as you passed in  
                int n_cycles: the number of cycles  
        '''
        found_start = False
        i = 0
        n_cycles = None
        line = 'dowhile'
        with open(os.path.join(self.data_path,filename), 'r',encoding='latin1') as file:
            while not found_start and line != '':
                line = file.readline()
                if bool(re.match(r'No\. of Cycles:',line)):
                    #is number of cycles
                    n_cycles = int((re.search(r'\d+', line)).group(0))
                if line[:6] == 'T[°C]:':
                    while not bool(re.match('\D\d',line)) and line != '':
                        #is not of form A1/B03 etc
                        line = file.readline()
                        i += 1
                    i -= 1 #cause you will increment once more 
                    found_start = True
                i+=1
        assert (line != ''), "corrupt reader file. ran out of file to read before finding a scanned well"
        assert (n_cycles != None), "corrupt reader file. num cycles not found."
        return i, {'n_cycles':n_cycles,'filename':filename}
    
    # def CreateDF(self):
    #     list_of_wells = ['Scan ID','wavelength (nm)', 'A01:', 'B01:', 'C01:', 'D01:', 'E01:', 'F01:', 'G01:', 'H01:', 'A02:', 'B02:', 'C02:', 'D02:', 'E02:', 'F02:', 'G02:', 'H02:', 'A03:', 'B03:', 'C03:', 'D03:', 'E03:', 'F03:', 'G03:', 'H03:', 'A04:', 'B04:', 'C04:', 'D04:', 'E04:', 'F04:', 'G04:', 'H04:', 'A05:', 'B05:', 'C05:', 'D05:', 'E05:', 'F05:', 'G05:', 'H05:', 'A06:', 'B06:', 'C06:', 'D06:', 'E06:', 'F06:', 'G06:', 'H06:', 'A07:', 'B07:', 'C07:', 'D07:', 'E07:', 'F07:', 'G07:', 'H07:', 'A08:', 'B08:', 'C08:', 'D08:', 'E08:', 'F08:', 'G08:', 'H08:', 'A09:', 'B09:', 'C09:', 'D09:', 'E09:', 'F09:', 'G09:', 'H09:', 'A10:', 'B10:', 'C10:', 'D10:', 'E10:', 'F10:', 'G10:', 'H10:', 'A11:', 'B11:', 'C11:', 'D11:', 'E11:', 'F11:', 'G11:', 'H11:', 'A12:', 'B12:', 'C12:', 'D12:', 'E12:', 'F12:', 'G12:', 'H12:']
    #     #self.df = pd.DataFrame(columns=["Scan ID", "Well"])
    #     #self.df = pd.MultiIndex.from_frame(self.df)
    #     full_df = pd.DataFrame()
    #     self.df = full_df
        #full_df.to_csv('my_new_file.csv')
        
        #print(full_df)
        
    def AddToDF(self, file_name, dst):
        # dst = dst+'.csv'
        # dst_path = os.path.join(self.data_path, dst)
        # #create the base file you're going to be writing to
        # shutil.copyfile(os.path.join(self.data_path,file_name), dst_path)
        # n_cycles = self._parse_metadata(file_name)[1]['n_cycles'] #n_cycles of first file
        # #iterate through the other files
        # #setup
        # filepath = os.path.join(self.data_path, file_name)
        # meta = self._parse_metadata(file_name)
        # #strip out just the data from the file
        # with open(filepath, 'r', encoding='latin1') as file:
        #     #these files are generally pretty small
        #     lines = file.read().split('\n')
        #     lines = lines[meta[0]:] #grab the raw data without preamble
        # #write the data to the dst file
        # with open(dst_path, 'a') as file:
        #     file.write('\n'.join(lines))
        #     print(file)
        # #cleanup
        # filepath = os.path.join(self.data_path, file_name)
        # #os.remove(filepath)
        
        
        # full_df = pd.concat([full_df, self.df], axis=1, join="inner")
        # full_df.to_csv('my_final_final.csv')
        
        temp_file = file_name
    
 #Creates first dataframe to extract date/time metadata   
    
        df_read_data_1 = pd.read_csv(temp_file,nrows = 35,skiprows = [7], header=None,na_values=["       -"],encoding = 'latin1')   
        am_pm = df_read_data_1.iloc[1][0].split(" ")[5]
        hour = int(df_read_data_1.iloc[1][0].split(" ")[4].split(":")[0]) + 12 if am_pm == "PM" else int(df_read_data_1.iloc[1][0].split(" ")[4].split(":")[0])
        date_time = datetime.datetime(int(df_read_data_1.iloc[1][0].split(" ")[1].split("/")[2]), int(df_read_data_1.iloc[1][0].split(" ")[1].split("/")[0]), int(df_read_data_1.iloc[1][0].split(" ")[1].split("/")[1]), hour, int(df_read_data_1.iloc[1][0].split(" ")[4].split(":")[1]), int(df_read_data_1.iloc[1][0].split(" ")[4].split(":")[2]))
        num_cycles = int(df_read_data_1.iloc[4][0][15:])
        if num_cycles != 1:
            raise Exception('Error due to Bad Scan Protocol: too many cycles')
        
#Creates second dataframe to extract wavelength metadata

        df_read_data_2 = pd.read_csv(temp_file,nrows = 3, skiprows = 43, header=None,na_values=["       -"],encoding = 'latin1')
        wavelength_blue = int(df_read_data_2[0][0][13:].split("nm", 2)[0])
        wavelength_red = int(df_read_data_2[0][0][13:].split("nm", 2)[1][3:])
        wavelength_steps = int(df_read_data_2[1][0].split("nm", 1)[0][1:])
        if file_name == 'CNH_004_fullscan_1.csv':
            wavelength_blue+=100
 
#Creates third dataframe to extract temp metadata
        df_read_data_3 = pd.read_csv(temp_file,nrows = 3, skiprows = 45, header=None,na_values=["       -"],encoding = 'latin1')    
        temp = float(df_read_data_3[0][2].split(" ")[-1])
        
    
#Creates fourth dataframe for data   
        
        data_df = pd.read_csv(temp_file,skiprows=48,header=None,na_values=["       -"],encoding = 'latin1',)
        
        g=data_df.iloc[:,0]
        g = [x.rstrip(':') for x in g]
        print(g)
        data_df = data_df.drop(data_df.columns[0], axis=1)
        wavvelengths = []
        for x in range (wavelength_blue-1,wavelength_red+1, wavelength_steps):
            wavvelengths = wavvelengths + [str(x)]
        wavvelength_df = pd.DataFrame()
        wavvelength_df.insert(0, "Wavelength (nm)", wavvelengths)
        
    
        wavvelength_df = wavvelength_df.T
        wavvelength_df=wavvelength_df.drop(wavvelength_df.columns[0], axis=1)
        
        
        
        data_df.insert(0,"Temp",temp)
        data_df.insert(0,"Time",date_time) 
        data_df.insert(0, 'Well', g)
        data_df.insert(0,"Scan ID",file_name)
        #data_df.loc['wavelength (nm)','temp'] = 0
        #data_df.loc['Scan ID','temp'] = 0
        
        
    
#Prints export values    
        print(" \n \nUseful Parameters:")
        print("Number of cycles")
        print(num_cycles)
        print("Date Time")
        print(date_time)  
        print("wavelength_blue")
        print(wavelength_blue)    
        print("wavelength_red")
        print(wavelength_red)    
        print("wavelength_steps")
        print(wavelength_steps)
        print("Temperature")
        print(temp)
        print("Scan ID")
        print(file_name)
    
        full_df = pd.concat([wavvelength_df,data_df])
        
        self.df = pd.concat([full_df,self.df])
        
        one= self.df.pop('Scan ID')
        two= self.df.pop('Well')
        three= self.df.pop('Time')
        four = self.df.pop('Temp')
        
        
        self.df.insert(0, 'Time', three)
        self.df.insert(0, 'Temp', four)
        self.df.insert(0, 'Well', two)
        self.df.insert(0, 'Scan ID', one)
        
        
            
        
            

        
        
    def merge_stuff(self, filenames, dst):
        filenames = ['{}.csv'.format(filename) for filename in filenames]
        dst = dst+'.csv'
        dst_path = os.path.join(self.data_path, dst)
        #create the base file you're going to be writing to
        shutil.copyfile(os.path.join(self.data_path,filenames[0]), dst_path)
        n_cycles = self._parse_metadata(filenames[0])[1]['n_cycles'] #n_cycles of first file
        #iterate through the other files
        for filename in filenames[1:]:
            #setup
            filepath = os.path.join(self.data_path, filename)
            meta = self._parse_metadata(filename)
            assert (n_cycles == meta[1]['n_cycles']), "scan files to merge, {} and {} had different n_cycles".format(filename, filenames[0])
            #strip out just the data from the file
            with open(filepath, 'r', encoding='latin1') as file:
                #these files are generally pretty small
                lines = file.read().split('\n')
                lines = lines[meta[0]:] #grab the raw data without preamble
            #write the data to the dst file
            with open(dst_path, 'a') as file:
                file.write('\n'.join(lines))
        #cleanup
        for filename in filenames:
            filepath = os.path.join(self.data_path, filename)
            os.remove(filepath)
        

    
    

if __name__ == '__main__':
    main()