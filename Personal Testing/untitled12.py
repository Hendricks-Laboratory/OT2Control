#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 18:47:26 2022

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

df1 = pd.read_csv('translated_wellmap.tsv', sep='\t')
df = pd.read_csv('CNH_008_full_df1.csv')

df = df.reset_index()

wavelengths = list(df.iloc[0][df.columns.get_loc("Time")+1:])
for i in range(1,len(wavelengths)+1):
  df = df.rename({str(i):wavelengths[i-1] }, axis='columns')
df = df.rename({"Unnamed: 0":"Scan_Number" }, axis='columns')

  
wells = list(set(df.loc[1:,"Well"]))
scans = list(set(df.loc[1:,"Scan ID"]))
df.drop('index', inplace=True, axis=1)
df = df.set_index(["Scan ID", "Well"]).sort_index()

col_list  = df.index.get_level_values('Well').tolist()

well_names = []

for well_with_zeros_in_name in col_list:
    x = ''
    #Turn A01 into A1
    well = str(well_with_zeros_in_name)
    well = "{}{}".format(well[0], int(well[2:]))
   
    y = df1.loc[df1['loc'] == str(well), 'chem_name'].values[:]
    #print(y)
    for i in y:
        if 'CNH_008' in i:
            x=i
            if 'control' in x:
                x = 'blank'
    
    well_names.append(x)

df.insert(0, 'Well Name', well_names)

df.to_csv('nicenice.csv')

          

