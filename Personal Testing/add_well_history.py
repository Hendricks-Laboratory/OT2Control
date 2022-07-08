#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 13:14:57 2022

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

reaction = 'CNH_008'

well_hist_df = pd.read_csv('./well_history.tsv', sep='\t')

timess = well_hist_df.timestamp.values.tolist()
for time in timess:
  
    index = timess.index(time)

    time = pd.Timestamp(time)
    given_time = time - pd.DateOffset(hours=7)
    given_time = given_time.strftime('%Y-%m-%d %H:%M:%S:%f')
    timess[index] = given_time


well_hist_df['timestamp'] = timess

df = pd.read_csv('CNH_008_full_df2.csv')



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
      
        if 'blank' in react:
            react = 'control'
        transfer_times = full.loc[full['cont'].str.contains(react),'time'].tolist()
        
  
        for transfer_time in transfer_times:
            print(transfer_time, time)
            print(type(transfer_time), type(time))
            print(transfer_time>time)
            # transfer_time = datetime.datetime.strptime(transfer_time, '%Y-%m-%d %H:%M:%S:%f')
            # transfer_time = tran
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
            
df.to_csv('lolthisisatest.csv')


full.to_csv('yus.csv')





