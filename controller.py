'''
This module contains everything that the server needs to run. Partly seperate from the OT2 because
it needs different packages (OT2 uses historic packages) and partly for organizational purposes.
The core of this module is the ProtocolExecutor class. The ProtocolExecutor is responsible for 
interfacing with the robot, the platereader, and googlesheets. It's purpose is to load a reaction
protocol from googlesheets and then execute that protocol line by line by communicating with the
robot and platereader. Attempts to do as much computation as possible before sending commands 
to those applications
The ProtocolExecutor uses a PlateReader.
PlateReader is a custom class that is built for controlling the platereader. 
In order to control the platereader, the software should be closed when PlateReader 
is instantiated, and (obviously) the software should exist on the machine you're running
This module also contains two launchers.
launch_protocol_exec runs a protocol from the sheets using a protocol executor
launch_auto runs in automatic machine learning mode
A main method is supplied that will run if you run this script. It will call one of the launchers
based on command line args. (run this script with -h)
'''
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from collections import namedtuple
import socket
import json
import dill
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

from bidict import bidict
import gspread
from df2gspread import df2gspread as d2g
from df2gspread import gspread2df as g2d
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import opentrons.execute
import opentrons.simulate
from opentrons import protocol_api, types
from boltons.socketutils import BufferedSocket
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from Armchair.armchair import Armchair
from ot2_robot import launch_eve_server
from df_utils import make_unique, df_popout, wslpath, error_exit
from optimizers import OptimizationModel
from exceptions import ConversionError

from heatmap import plate, heat_map
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build



def init_parser():
    parser = argparse.ArgumentParser()
    mode_help_str = 'mode=auto runs in ml, mode=protocol or not supplied runs protocol'
    parser.add_argument('-m','--mode',help=mode_help_str,default='protocol')
    parser.add_argument('-n','--name',help='the name of the google sheet')
    parser.add_argument('-c','--cache',help='flag. if supplied, uses cache',action='store_true')
    parser.add_argument('-s','--simulate',help='runs robot and pr in simulation mode',action='store_true')
    parser.add_argument('--no-sim',help='won\'t run simulation at the start.',action='store_true')
    parser.add_argument('--no-pr', help='won\'t invoke platereader, even in simulation mode',action='store_true')
    return parser

def main(serveraddr):
    '''
    prompts for input and then calls appropriate launcher
    '''
    parser = init_parser()
    args = parser.parse_args()
    if args.mode == 'protocol':
        print('launching in protocol mode')
        launch_protocol_exec(serveraddr,args.name,args.cache,args.simulate,args.no_sim,args.no_pr)
    elif args.mode == 'auto':
        print('launching in auto mode')
        launch_auto(serveraddr,args.name,args.cache,args.simulate,args.no_sim,args.no_pr)
    else:
        print("invalid argument to mode, '{}'".format(args.mode))
        parser.print_help()

def launch_protocol_exec(serveraddr, rxn_sheet_name, use_cache, simulate, no_sim, no_pr):
    '''
    main function to launch a controller and execute a protocol
    '''
    #instantiate a controller
    if not rxn_sheet_name:
        rxn_sheet_name = input('<<controller>> please input the sheet name ')
    my_ip = socket.gethostbyname(socket.gethostname())
    controller = ProtocolExecutor(rxn_sheet_name, my_ip, serveraddr, use_cache=use_cache)

    if not no_sim:
        controller.run_simulation(no_pr=no_pr)
    if input('would you like to run the protocol? [yn] ').lower() == 'y':
        controller.run_protocol(simulate, no_pr)

def launch_auto(serveraddr, rxn_sheet_name, use_cache, simulate, no_sim, no_pr):
    '''
    main function to launch an auto scientist that designs it's own experiments
    '''
    if not rxn_sheet_name:
        rxn_sheet_name = input('<<controller>> please input the sheet name ')
    my_ip = socket.gethostbyname(socket.gethostname())
    auto = AutoContr(rxn_sheet_name, my_ip, serveraddr, use_cache=use_cache)
    #note shorter iterations for testing
    #final_spectra = np.loadtxt("test_target_1.csv", delimiter=',', dtype=float).reshape(1,-1)
    print(auto.rxn_df.describe())
    print(auto.rxn_df.head(20))
    print(auto.rxn_df)
    y_shape = auto.y_shape# number of reagents to learn on
    print("starting with y_shape:", y_shape)
    reagent_info = auto.robo_params['reagent_df']
    fixed_reagents = auto.get_fixed_reagents()
    variable_reagents = auto.get_variable_reagents()
    target_value = np.random.randint(1, 200) # not being used? TODO delete 
    # Generate bounds for each reagent, assuming concentrations range from 0 to 1
    bounds = [{'name': f'reagent_{i+1}_conc', 'type': 'continuous', 'domain': (0.00025, 0.001)} for i in range(y_shape)]
    # final_spectra not used?
    model = OptimizationModel(bounds, auto.getModelInfo()["target"], reagent_info, fixed_reagents, variable_reagents, initial_design_numdata=auto.getModelInfo()["initial_data"], batch_size=1, max_iters=auto.getModelInfo()["max_iterations"])
    print(auto.getModelInfo()["target"])
    if not no_sim:
        auto.run_simulation(no_pr=no_pr)
    if input('would you like to run on robot and pr? [yn] ').lower() == 'y':
        auto.run_protocol(simulate=simulate, model=model,no_pr=no_pr)



class Controller(ABC):
    '''
    This class is a shared interface for the ProtocolExecutor and the ______AI__Executor___  

    ATTRIBUTES:  
        armchair.Armchair portal: the Armchair object to ship files across  
        rxn_sheet_name: the name of the reaction sheet  
        str cache_path: path to a directory for all cache files  
        bool use_cache: read from cache if possible  
        str eve_files_path: the path to put files from eve  
        str debug_path: the path to place debugging information  
        str my_ip: the ip of this controller  
        str server_ip: the ip of the server. This is modified for simulation, but returned to 
          original state at the end of simulation  
        dict<str:object> robo_params: convenient place for the parameters for the robot  
            + bool using_temp_ctrl: True if the temperature control is being used  
            + float temp: the temperature in celcius to keep the temp control at  
            + df reagent_df: holds information about reagents  
                + float conc: the concentration  
                + str loc: location on labware  
                + int deck_pos: the position on the deck  
                + float mass: the mass of the tube with reagent and cap  
            dict<str:str> instruments: maps 'left' and 'right' to the pipette names  
            df labware_df  
                + int deck_pos: the position of the labware on the deck  
                + str name: the name of the labware  
                + str first_usable: a location of the first usable tip/well on labware  
                + list<str> empty_list: a list of locations on the labware that have empty tubes  
            df product_df: This information is used to figure out where to put chemicals  
                + INDEX  
                + str chemical_name: the name of the chemical  
                + COLS  
                + str labware: the requested labware you want to put it in  
                + str container: the container you want to put it in  
                + float max_vol: the maximum volume you will put in the container  
        bool simulate: whether a simulation is being run or not. False by default. changed true 
          temporarily when simulating  
        int buff_size: this is the size of the buffer between Armchair commands. It's size
          corresponds to the number of commands you want to pile up in the socket buffer.
          Really more for developers  
    PRIVATE ATTRS:  
        dict<str:ChemCacheEntry> _cached_reader_locs: chemical information from the robot
            ChemCacheEntry is a named tuple with below attributes
            The tuple has following structure:  
            str loc: the loc of the well on it's labware (translated to human if on pr)  
            int deck_pos: the position of the labware it's on  
            float vol: the volume in the container  
            float aspiratible_vol: the volume minus dead vol  
    CONSTANTS:  
        bidict<str:tuple<str,str>> PLATEREADER_INDEX_TRANSLATOR: used to translate from locs on
        wellplate to locs on the opentrons object. Use a json viewer for more structural info  
    METHODS:  
        run_protocol(simulate, port) void: both args have good defaults. simulate can be used to
          simulate on the plate reader and robot, but generally you want false to actually run
          the protocol. port can be configured, but 50000 is default  
        run_simulation() int: runs a simulation on local machine. Tries plate reader, but
          not necessary. returns an error code  
        close_connection() void: automatically called by run_protocol. used to terminate a 
          connection with eve  
        init_robot(simulate): used to initialize the robot. called automatically in run. simulate
          is the same as used by the robot protocol  
        translate_wellmap() void: used to convert a wellmap.tsv from robot to wells locs 
          that correspond to platereader  
    '''
    #this has two keys, 'deck_pos' and 'loc'. They map to the plate reader and the loc on that plate
    #reader given a regular loc for a 96well plate.
    #Please do not read this. paste it into a nice json viewer.
    PLATEREADER_INDEX_TRANSLATOR = bidict({'A1': ('E1', 'platereader4'), 'A2': ('D1', 'platereader4'), 'A3': ('C1', 'platereader4'), 'A4': ('B1', 'platereader4'), 'A5': ('A1', 'platereader4'), 'A12': ('A1', 'platereader7'), 'A11': ('B1', 'platereader7'), 'A10': ('C1', 'platereader7'), 'A9': ('D1', 'platereader7'), 'A8': ('E1', 'platereader7'), 'A7': ('F1', 'platereader7'), 'A6': ('G1', 'platereader7'), 'B1': ('E2', 'platereader4'), 'B2': ('D2', 'platereader4'), 'B3': ('C2', 'platereader4'), 'B4': ('B2', 'platereader4'), 'B5': ('A2', 'platereader4'), 'B6': ('G2', 'platereader7'), 'B7': ('F2', 'platereader7'), 'B8': ('E2', 'platereader7'), 'B9': ('D2', 'platereader7'), 'B10': ('C2', 'platereader7'), 'B11': ('B2', 'platereader7'), 'B12': ('A2', 'platereader7'), 'C1': ('E3', 'platereader4'), 'C2': ('D3', 'platereader4'), 'C3': ('C3', 'platereader4'), 'C4': ('B3', 'platereader4'), 'C5': ('A3', 'platereader4'), 'C6': ('G3', 'platereader7'), 'C7': ('F3', 'platereader7'), 'C8': ('E3', 'platereader7'), 'C9': ('D3', 'platereader7'), 'C10': ('C3', 'platereader7'), 'C11': ('B3', 'platereader7'), 'C12': ('A3', 'platereader7'), 'D1': ('E4', 'platereader4'), 'D2': ('D4', 'platereader4'), 'D3': ('C4', 'platereader4'), 'D4': ('B4', 'platereader4'), 'D5': ('A4', 'platereader4'), 'D6': ('G4', 'platereader7'), 'D7': ('F4', 'platereader7'), 'D8': ('E4', 'platereader7'), 'D9': ('D4', 'platereader7'), 'D10': ('C4', 'platereader7'), 'D11': ('B4', 'platereader7'), 'D12': ('A4', 'platereader7'), 'E1': ('E5', 'platereader4'), 'E2': ('D5', 'platereader4'), 'E3': ('C5', 'platereader4'), 'E4': ('B5', 'platereader4'), 'E5': ('A5', 'platereader4'), 'E6': ('G5', 'platereader7'), 'E7': ('F5', 'platereader7'), 'E8': ('E5', 'platereader7'), 'E9': ('D5', 'platereader7'), 'E10': ('C5', 'platereader7'), 'E11': ('B5', 'platereader7'), 'E12': ('A5', 'platereader7'), 'F1': ('E6', 'platereader4'), 'F2': ('D6', 'platereader4'), 'F3': ('C6', 'platereader4'), 'F4': ('B6', 'platereader4'), 'F5': ('A6', 'platereader4'), 'F6': ('G6', 'platereader7'), 'F7': ('F6', 'platereader7'), 'F8': ('E6', 'platereader7'), 'F9': ('D6', 'platereader7'), 'F10': ('C6', 'platereader7'), 'F11': ('B6', 'platereader7'), 'F12': ('A6', 'platereader7'), 'G1': ('E7', 'platereader4'), 'G2': ('D7', 'platereader4'), 'G3': ('C7', 'platereader4'), 'G4': ('B7', 'platereader4'), 'G5': ('A7', 'platereader4'), 'G6': ('G7', 'platereader7'), 'G7': ('F7', 'platereader7'), 'G8': ('E7', 'platereader7'), 'G9': ('D7', 'platereader7'), 'G10': ('C7', 'platereader7'), 'G11': ('B7', 'platereader7'), 'G12': ('A7', 'platereader7'), 'H1': ('E8', 'platereader4'), 'H2': ('D8', 'platereader4'), 'H3': ('C8', 'platereader4'), 'H4': ('B8', 'platereader4'), 'H5': ('A8', 'platereader4'), 'H6': ('G8', 'platereader7'), 'H7': ('F8', 'platereader7'), 'H8': ('E8', 'platereader7'), 'H9': ('D8', 'platereader7'), 'H10': ('C8', 'platereader7'), 'H11': ('B8', 'platereader7'), 'H12': ('A8', 'platereader7')})

    ChemCacheEntry = namedtuple('ChemCacheEntry',['loc','deck_pos','vol','aspirable_vol'])
    DilutionParams = namedtuple('DilultionParams', ['cont','vol'])

    def __init__(self, rxn_sheet_name, my_ip, server_ip, buff_size=4, use_cache=False, cache_path='Cache'):
        '''
        Note that init does not initialize the portal. This must be done explicitly or by calling
        a run function that creates a portal. The portal is not passed to init because although
        the code must not use more than one portal at a time, the portal may change over the 
        lifetime of the class
        Note that pr cannot be initialized until you know if you're simulating or not, so it
        is instantiated in run
        '''
        #set according to input
        self.cache_path=cache_path
        self._make_cache()
        self.use_cache = use_cache
        self.my_ip = my_ip
        self.server_ip = server_ip
        self.buff_size = 4
        self.rxn_sheet_name = rxn_sheet_name
        self.simulate = False #by default will be changed if a simulation is run
        self._cached_reader_locs = {} #maps wellname to loc on platereader
        #this will be gradually filled
        self.robo_params = {}
        #necessary helper params
        self._check_cache_metadata(rxn_sheet_name)
        credentials = self._init_credentials(rxn_sheet_name)
        self.drive_service = self._init_google_drive(credentials) # Terence   
        self.wks_key_pairs = self._get_wks_key_pairs(credentials, rxn_sheet_name)
        self.name_key_wks = self._get_key_wks(credentials)
        wks_key = self._get_wks_key(credentials, rxn_sheet_name)
        rxn_spreadsheet = self._open_sheet(rxn_sheet_name, credentials)
        header_data = self._download_sheet(rxn_spreadsheet,0)
        self.header_data = header_data
        input_data = self._download_sheet(rxn_spreadsheet,1)
        deck_data = self._download_sheet(rxn_spreadsheet, 2)
        self._init_robo_header_params(header_data)
        self._make_out_dirs(header_data)
        self.reaction_folder_name = None
        self.rxn_df = self._load_rxn_df(input_data) #products init here
        self.tot_vols = self._get_tot_vols(input_data) #NOTE we're moving more and more info
        #to the controller. It may make sense to build a class at some point
        self._query_reagents(wks_key, credentials)
        raw_reagent_df = self._download_reagent_data(wks_key, credentials)#will be replaced soon
        #with a parsed reagent_df. This is exactly as is pulled from gsheets
        empty_containers = self._get_empty_containers(raw_reagent_df)
        self.robo_params['dry_containers'] = self._get_dry_containers(raw_reagent_df)
        products_to_labware = self._get_products_to_labware(input_data)
        self.robo_params['reagent_df'] = self._parse_raw_reagent_df(raw_reagent_df)
        self.robo_params['instruments'] = self._get_instrument_dict(deck_data)
        self.robo_params['labware_df'] = self._get_labware_df(deck_data, empty_containers)
        self.robo_params['product_df'] = self._get_product_df(products_to_labware)

    def _insert_tot_vol_transfer(self):
        '''
        inserts a row into self.rxn_df that transfers volume from WaterC1.0 to fill
        the necessary products  
        Postconditions:  
            has inserted a row into the rxn_df to transfer WaterC1.0  
            If the reaction has already overflowed the total volume, will add negative volume
            (which is impossible. The caller of this function must account for this.)  
            If no total vols were specified, no transfer step will be inserted.  
        '''
        #if there are no total vols, don't insert the row, just return
        if self.tot_vols:
            end_vols = pd.Series(self.tot_vols)
            start_vols = pd.Series([self._vol_calc(name) 
                                    for name in end_vols.index], index=end_vols.index)
            del_vols = end_vols - start_vols
            #begin building a dictionary for the row to insert
            transfer_row_dict = {col:del_vols[col] if col in del_vols else np.nan 
                                for col in self.rxn_df.columns}
            #now have dict maps every col to '' except chemicals to add, which are mapped to float to add
            transfer_row_dict.update(
                {'op':'transfer',
                'reagent':'Water',
                'conc':1.0,
                'chemical_name':'WaterC1.0',
                'callbacks':''}
            )
            for chem_name in self._products:
                if pd.isna(transfer_row_dict[chem_name]):
                    transfer_row_dict[chem_name] = 0.0
            #convert the row to a dataframe
            transfer_row_df = pd.DataFrame(transfer_row_dict, index=[-1], columns=self.rxn_df.columns)
            self.rxn_df = pd.concat((transfer_row_df, self.rxn_df)) #add in column
            self.rxn_df.index += 1 #update index to go 0-n instead of -1-n-1

    def _get_tot_vols(self, input_data):
        '''
        params:  
            list<obj> input_data: as parsed from the google sheets  
        returns:  
            dict<str:float>: maps product names to their appropriate total volumes if specified  
        Preconditions:  
            self._products has been initialized  
        '''
        product_start_i = input_data[0].index('reagent (must be uniquely named)')+1
        product_tot_vols = input_data[3][product_start_i:]
        return {product:float(tot_vol) for product, tot_vol in zip(self._products, product_tot_vols) if tot_vol}

    def _check_cache_metadata(self, rxn_sheet_name):
        '''
        Checks a file, .metadata.txt with the cache path.
        Postconditions:
            If use_cache is true:
                reads .metadata.txt
                asserts that the rxn_sheet_name matches the name in sheet
                prints the timestamp that the cache was last written
            If use_cache is false:
                writes .metadata.txt with the sheet name and a timestamp
        '''
        if self.use_cache:
            assert (os.path.exists(os.path.join(self.cache_path, '.metadata.json'))), \
                    "tried to read metadata in cache, but file does not exist"
            with open(os.path.join(self.cache_path, '.metadata.json'), 'r') as file:
                metadata = json.load(file)
            assert (metadata['name'] == rxn_sheet_name), "desired sheet was, '{}', but cached data is for '{}'".format(rxn_sheet_name, metadata['name'])
            print("<<controller>> using cached data for '{}', last updated '{}'".format(
                    metadata['name'],metadata['timestamp']))
        else:
            metadata = {'timestamp':datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S:%f'),
                        'name':rxn_sheet_name}
            with open(os.path.join(self.cache_path, '.metadata.json'), 'w') as file:
                json.dump(metadata, file)

    def _get_key_wks(self, credentials):
        gc = gspread.authorize(credentials)
        name_key_wks = gc.open_by_url('https://docs.google.com/spreadsheets/d/1m2Uzk8z-qn2jJ2U1NHkeN7CJ8TQpK3R0Ai19zlAB1Ew/edit#gid=0').get_worksheet(0)
        return name_key_wks

    def _get_wks_key_pairs(self, credentials, rxn_sheet_name):
        '''
        open and search a sheet that tells you which sheet is associated with the reaction
        Or read from cache if cache is enabled  
        params:  
            ServiceAccountCredentials credentials: to access the sheets  
            str rxn_sheet_name: the name of sheet  
        returns:  
            list<list<str>> name_key_pairs: the data in the wks_key spreadsheet  
        Postconditions:  
            If cached data could not be found, will dump spreadsheet data to name_key_pairs.pkl 
            in cache path  
        '''
        if self.use_cache:
            #load cache
            with open(os.path.join(self.cache_path, 'name_key_pairs.pkl'), 'rb') as name_key_pairs_cache:
                name_key_pairs = dill.load(name_key_pairs_cache)
        else:
            #pull down data
            gc = gspread.authorize(credentials)
            name_key_wks = gc.open_by_url('https://docs.google.com/spreadsheets/d/1m2Uzk8z-qn2jJ2U1NHkeN7CJ8TQpK3R0Ai19zlAB1Ew/edit#gid=0').get_worksheet(0)
            name_key_pairs = name_key_wks.get_all_values() #list<list<str name, str key>>
            #Note the key is a unique identifier that can be used to access the sheet
            #d2g uses it to access the worksheet
            #dump to cache
            with open(os.path.join(self.cache_path, 'name_key_pairs.pkl'), 'wb') as name_key_pairs_cache:
                dill.dump(name_key_pairs, name_key_pairs_cache)
        return name_key_pairs

    def _init_pr(self, simulate, no_pr):
        '''
        params:  
            bool simulate: True indicates that the platereader should be launched in simulation
              mode
            bool no_pr: True indicates that even if platereader can be run in simulation mode,
              it should not be. This should be run only for the marginal speedup that can be
              gained by not using the platereader for certain tests
        Postconditions:  
            self.pr is initialized with either a connection to the SPECTROstar if possible and
              no_pr is false, otherwise, a Dummy with no connection, but the same interface
              is supplied
        '''
        if no_pr:
            self.pr = DummyReader(os.path.join(self.out_path, 'pr_data'))
        else:
            try:
                self.pr = PlateReader(os.path.join(self.out_path, 'pr_data'), self.header_data, self.eve_files_path, simulate)
            except:
                print('<<controller>> failed to initialize platereader, initializing dummy reader')
                self.pr = DummyReader(os.path.join(self.out_path, 'pr_data'))

    def _download_sheet(self, rxn_spreadsheet, index):
        '''
        pulls down the sheet at the index  
        params:  
            gspread.Spreadsheet rxn_spreadsheet: the sheet with all the reactions  
            int index: the index of the sheet to pull down  
        returns:  
            list<list<str>> data: the input template sheet pulled down into a list  
        '''
        if self.use_cache:
            with open(os.path.join(self.cache_path,'wks_data{}.pkl'.format(index)), 'rb') as rxn_wks_data_cache:
                data = dill.load(rxn_wks_data_cache)
        else:
            rxn_wks = rxn_spreadsheet.get_worksheet(index)
            data = rxn_wks.get_all_values()
            with open(os.path.join(self.cache_path,'wks_data{}.pkl'.format(index)),'wb') as rxn_wks_data_cache:
                dill.dump(data, rxn_wks_data_cache)
        return data


    def _make_out_dirs(self, header_data):
        '''
        params:  
            list<list<str>> header_data: data from the header  
        Postconditions:  
            All paths used by this class have been initialized if they were not before
            They are not overwritten if they already exist. paths variables of this class
            have also been initialized
        '''

        out_path = 'Ideally this would be a gdrive path, but for now everything is local'
        if not os.path.exists(out_path):
            #not on the laptop
            out_path = '/mnt/c/Users/science_356_lab/Robot_Files/Protocol_Outputs'
        #get the root folder
        header_dict = {row[0]:row[1] for row in header_data[1:]}
        data_dir = header_dict['data_dir']
        self.reaction_folder_name = os.path.basename(os.path.dirname(data_dir))
        self.out_path = os.path.join(out_path, data_dir)
        #if the folder doesn't exist yet, make it
        self.eve_files_path = os.path.join(self.out_path, 'Eve_Files')
        self.debug_path = os.path.join(self.out_path, 'Debug')
        self.plot_path = os.path.join(self.out_path, 'Plots')
        paths = [self.out_path, self.eve_files_path, self.debug_path, self.plot_path]
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    def _make_cache(self):
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

    def _init_credentials(self, rxn_sheet_name):
        '''
        this function reads a local json file to get the credentials needed to access other funcs  
        params:  
            str rxn_sheet_name: the name of the reaction sheet to run  
        returns:  
            ServiceAccountCredentials: the credentials to access that sheet  
        '''
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive', 
                 'https://www.googleapis.com/auth/drive.file']
        #get login credentials from local file. Your json file here
        path = 'Credentials/hendricks-lab-jupyter-sheets-5363dda1a7e0.json'
        credentials = ServiceAccountCredentials.from_json_keyfile_name(path, scope) 
        return credentials

    def _init_google_drive(self, credentials): 
        try:
            drive_service = build('drive', 'v3', credentials=credentials)
            return drive_service
        except HttpError as error:
            print(f"Google Drive API Error: {error}")
            return None

    def _get_wks_key(self, credentials, rxn_sheet_name):
        '''
        open and search a sheet that tells you which sheet is associated with the reaction  
        params:  
            ServiceAccountCredentials credentials: to access the sheets  
            str rxn_sheet_name: the name of sheet  
        returns:  
            if self.use_cache:  
                str wks_key: the key associated with the sheet. It functions similar to a url  
            else:  
                None: this is ok because the wks key will not be used if caching  
        '''
        name_key_pairs = self.wks_key_pairs
        try:
            i=0
            wks_key = None
            while not wks_key and i <= len(name_key_pairs):
                row = name_key_pairs[i]
                if row[0] == rxn_sheet_name:
                    wks_key = row[1]
                i+=1
        except IndexError:
            raise Exception('Spreadsheet Name/Key pair was not found. Check the dict spreadsheet \
            and make sure the spreadsheet name is spelled exactly the same as the reaction \
            spreadsheet.')
        return wks_key

    def _open_sheet(self, rxn_sheet_name, credentials):
        '''
        open the google sheet  
        params:  
            str rxn_sheet_name: the title of the sheet to be opened  
            oauth2client.ServiceAccountCredentials credentials: credentials read from a local json  
        returns:  
            if self.use_cache:  
                gspread.Spreadsheet the spreadsheet (probably of all the reactions)  
            else:  
                None: this is fine because the wks should never be used if cache is true  
        '''
        gc = gspread.authorize(credentials)
        try:
            if self.use_cache:
                wks = None
            else:
                wks = gc.open(rxn_sheet_name)
        except: 
            raise Exception('Spreadsheet Not Found: Make sure the spreadsheet name is spelled correctly and that it is shared with the robot ')
        return wks

    def _init_robo_header_params(self, header_data):
        '''
        loads the header data into self.robo_params  
        params:  
            list<list<str> header_data: as in gsheets  
        Postconditions:  
            simulate, using_temp_ctrl, and temp have been initialized according to values in 
            excel  
        '''
        header_dict = {row[0]:row[1] for row in header_data[1:]}
        self.robo_params['using_temp_ctrl'] = header_dict['using_temp_ctrl'] == 'yes'
        self.robo_params['temp'] = float(header_dict['temp']) if self.robo_params['using_temp_ctrl'] else None
        if self.robo_params['temp'] != None:
            assert( self.robo_params['temp'] >= 4 and self.robo_params['temp'] <= 95), "invalid temperature"
        self.dilution_params = self.DilutionParams(header_dict['dilution_cont'], 
                float(header_dict['dilution_vol']))
        self.robo_params['target'] = float(header_dict['target'])
        self.robo_params['max_iterations'] = float(header_dict['max_iterations'])
        self.robo_params['initial_data'] = int(header_dict['initial_data'])

    def getModelInfo(self): 
        return self.robo_params

    def _plot_setup_overlay(self,title):
        '''
        Sets up a figure for an overlay plot  
        params:  
            str title: the title of the reaction  
        '''
        #formats the figure nicely
        plt.figure(num=None, figsize=(4, 4),dpi=300, facecolor='w', edgecolor='k')
        plt.legend(loc="upper right",frameon = False, prop={"size":7},labelspacing = 0.5)
        plt.rc('axes', linewidth = 2)
        plt.xlabel('Wavelength (nm)',fontsize = 16)
        plt.ylabel('Absorbance (a.u.)', fontsize = 16)
        plt.tick_params(axis = "both", width = 2)
        plt.tick_params(axis = "both", width = 2)
        plt.xticks([300,400,500,600,700,800,900,1000])
        plt.yticks([i/10 for i in range(0,11,1)])
        plt.axis([300, 1000, 0.0 , 1.0])
        plt.xticks(fontsize = 10)
        plt.yticks(fontsize = 10)
        plt.title(str(title), fontsize = 16, pad = 20)
        
    def plot_LAM_overlay(self,df,wells,filename=None):
        '''
        plots overlayed spectra of wells in the order that they are specified  
        params:  
            df df: dataframe with columns = chem_names, and values of each column is a series
              of scans in 701 intervals.  
            str filename: the title of the plot, and the file  
            list<str> wells: an ordered list of all of the chem_names you want to plot.  
        Postconditions:  
            plot has been written with name "overlay.png" to the plotting dir. or 
            {filename}.png if filename was supplied  
        '''
        if not filename:
            filename = "overlay"
        x_vals = list(range(300,1001))
        #overlays only things you specify
        y = []
        #df = df[df_reorder]
        #headers = [well_key[k] for k in df.columns]
        #legend_colors = []
        for chem_name in wells:
            y.append(df[chem_name].iloc[-701:].to_list())
        self._plot_setup_overlay(filename)
        colors = list(cm.rainbow(np.linspace(0, 1,len(y))))
        for i in range(len(y)):
            plt.plot(x_vals,y[i],color = tuple(colors[i]))
        patches = [mpatches.Patch(color=color, label=label) for label, color in zip(wells, colors)]
        plt.legend(patches, wells, loc='upper right', frameon=False,prop={'size':3})
        legend = pd.DataFrame({'Color':patches,'Labels': wells})
        plt.savefig(os.path.join(self.plot_path, '{}.png'.format(filename)))
        plt.close()
       
    # below until ~end is all not used yet needs to be worked up
    def plot_kin_subplots(self,df,n_cycles,wells,filename=None):
        '''
        TODO this function doesn't save properly, but it does show. Don't know issue  
        plots kinetics for each well in the order given by wells.  
        params:  
            df df: the scan data  
            int n_cycles: the number of cycles for the scan data  
            list<str> wells: the wells you want to plot in order
        Postconditions:  
            plot has been written with name "{filename}_overlay.png" to the plotting dir.  
            If filename is not supplied, name is kin_subplots
        '''
        if not filename:
            filename=kin_subplots
        x_vals = list(range(300,1001))
        colors = list(cm.rainbow(np.linspace(0, 1, n_cycles)))
        fig, axes = plt.subplots(8, 12, dpi=300, figsize=(50, 50),subplot_kw=dict(box_aspect=1,sharex = True,sharey = True))
        for idx, (chem_name, ax) in enumerate(zip(wells, axes.flatten())):
            ax.set_title(chem_name)
            self._plot_kin(ax, df, n_cycles, chem_name)
            plt.subplots_adjust(wspace=0.3, hspace= -0.1)
        
            ax.tick_params(
                which='both',
                bottom='off',
                left='off',
                right='off',
                top='off'
            )
            ax.set_xlim((300,1000))
            ax.set_ylim((0,1.0))
            ax.set_xlabel("Wavlength (nm)")
            ax.set_ylabel("Absorbance (A.U.)")
            ax.set_xticks(range(301, 1100, 100))
            #ax.set_aspect(adjustable='box')
            #ax.set_yticks(range(0,1))
        else:
            [ax.set_visible(False) for ax in axes.flatten()[idx+1:]]
        plt.savefig(os.path.join(self.plot_path, '{}.png'.format(filename)))
        plt.close()

    def _plot_kin(self, ax, df, n_cycles, chem_name):
        '''
        helper method for kinetics plotting methods  
        params:  
            plt.axes ax: or anything with a plot func. the place you want ot plot  
            df df: the scan data  
            int n_cycles: the number of cycles in per well scanned  
            str chem_name: the name of the chemical to be plotted  
        Postconditions:  
            a kinetics plot of the well has been plotted on ax  
        '''
        x_vals = list(range(300,1001))
        colors = list(cm.rainbow(np.linspace(0, 1, n_cycles)))
        kin = 0
        col = df[chem_name]
        for kin in range(n_cycles):
            ax.plot(x_vals, df[chem_name].iloc[kin*701:(kin+1)*701],color=tuple(colors[kin]))
        
    
    def plot_single_kin(self, df, n_cycles, chem_name, filename=None):
        '''
        plots one kinetics trace. 
        params:  
            df df: the scan data  
            int n_cycles: the number of cycles in per well scanned  
            str chem_name: the name of the chemical to be plotted  
            str filename: the name of the file to write  
        Postconditions:  
            A kinetics trace of the well has been written to the Plots directory.
            under the name filename. If filename was None, the filename will be 
            {chem_name}_kinetics.png
        '''
        if not filename:
            filename = '{}_kinetics'.format(chem_name)
        self._plot_setup_overlay('Kinetics {}: '.format(chem_name))
        self._plot_kin(plt,df, n_cycles, chem_name)
        plt.savefig(os.path.join(self.plot_path, '{}.png'.format(filename)))
        plt.close()

    def _get_empty_containers(self, raw_reagent_df):
        '''
        only one line, but there's a lot going on. extracts the empty lines from the raw_reagent_df  
        params:  
            df raw_reagent_df: as in reagent_info of excel  
        returns:  
            df empty_containers:  
                + INDEX:  
                + int deck_pos: the position on the deck  
                + COLS:  
                + str loc: location on the labware  
        '''
        return raw_reagent_df.loc['empty' == raw_reagent_df.index].set_index('deck_pos').drop(columns=['conc', 'mass'])

    def _get_dry_containers(self, raw_reagent_df):
        '''
        params:  
            df raw_reagent_df: the reagent dataframe as recieved from excel  
        returns:  
            df dry_containers:  
                note: cannot be sent over pickle as is because the index has duplicates.
                  solution is to reset the index for shipping
                + str index: the chemical name
                + float conc: the concentration once built
                + str loc: the location on the labware
                + int deck_pos: position on the deck
                + float required_vol: the volume of water needed to turn this into a reagent
        '''
        #other rows will be empty str unless dry
        dry_containers = raw_reagent_df.loc[raw_reagent_df['molar_mass'].astype(bool)].astype(
                {'deck_pos':int,'mass':float,'molar_mass':float})
        dry_containers.drop(columns='conc',inplace=True)
        dry_containers.reset_index(inplace=True)
        dry_containers['index'] = dry_containers['index'].apply(lambda x: x.replace(' ','_'))
        return dry_containers


    
    def _parse_raw_reagent_df(self, raw_reagent_df):
        '''
        parses the raw_reagent_df into final form for reagent_df  
        params:  
            df raw_reagent_df: as in excel  
        returns:  
            df reagent_df: empties ignored, columns with correct types  
        '''
        # incase not on axis
        reagent_df = raw_reagent_df.drop(['empty'], errors='ignore')
        reagent_df = reagent_df.loc[~reagent_df['molar_mass'].astype(bool)] #drop dry
        reagent_df.drop(columns='molar_mass',inplace=True)
        try:
            reagent_df = reagent_df.astype({'conc':float,'deck_pos':int,'mass':float})
        except ValueError as e:
            raise ValueError("Your reagent info could not be parsed. Likely you left out a required field, or you did not specify a concentration on the input sheet")
        return reagent_df

    def _get_instrument_dict(self, deck_data):
        '''
        uses data from deck sheet to return the instrument params  
        Preconditions:  
            The second sheet in the worksheet must be initialized with where you've placed reagents 
            and the first thing not being used  
        params:  
            list<list<str>>deck_data: the deck data as in excel  
        returns:  
            Dict<str:str>: key is 'left' or 'right' for the slots. val is the name of instrument  
        '''
        #the format google fetches this in is funky, so we convert it into a nice df
        #make instruments
        instruments = {}
        instruments['left'] = deck_data[13][0]
        instruments['right'] = deck_data[13][1]
        return instruments
    
    def _get_labware_df(self, deck_data, empty_containers):
        '''
        uses data from deck sheet to get information about labware locations, first tip, etc.  
        Preconditions:  
            The second sheet in the worksheet must be initialized with where you've placed reagents 
            and the first thing not being used  
        params:  
            list<list<str>>deck_data: the deck data as in excel  
            df empty_containers: this is used for tubes. it holds the containers that can be used  
                + int index: deck_pos  
                + str position: the position of the empty container on the labware  
        returns:  
            df:  
                + str name: the common name of the labware  
                + str first_usable: the first tip/well to use  
                + int deck_pos: the position on the deck of this labware  
                + str empty_list: the available slots for empty tubes format 'A1,B2,...' No specific
                  order  
        '''
        labware_dict = {'name':[], 'first_usable':[],'deck_pos':[]}
        for row_i in range(0,10,3):
            for col_i in range(3):
                labware_dict['name'].append(deck_data[row_i+1][col_i])
                labware_dict['first_usable'].append(deck_data[row_i+2][col_i])
                labware_dict['deck_pos'].append(deck_data[row_i][col_i])
        labware_df = pd.DataFrame(labware_dict)
        #platereader positions need to be translated, and they shouldn't be put in both
        #slots
        platereader_rows = labware_df.loc[(labware_df['name'] == 'platereader7') | \
                (labware_df['name'] == 'platereader4')]
        usable_rows = platereader_rows.loc[platereader_rows['first_usable'].astype(bool), 'first_usable']
        assert (not usable_rows.empty), "please specify a first tip/well for the platereader"
        assert (usable_rows.shape[0] == 1), "too many first wells specified for platereader"
        platereader_input_first_usable = usable_rows.iloc[0]
        platereader_name = self.PLATEREADER_INDEX_TRANSLATOR[platereader_input_first_usable][1]
        platereader_first_usable = self.PLATEREADER_INDEX_TRANSLATOR[platereader_input_first_usable][0]
        if platereader_name == 'platereader7':
            platereader4_first_usable = 'F8' #anything larger than what is on plate
            platereader7_first_usable = platereader_first_usable
        else:
            platereader4_first_usable = platereader_first_usable
            platereader7_first_usable = 'G1'
        labware_df.loc[labware_df['name']=='platereader4','first_usable'] = platereader4_first_usable
        labware_df.loc[labware_df['name']=='platereader7','first_usable'] = platereader7_first_usable
        labware_df = labware_df.loc[labware_df['name'] != ''] #remove empty slots
        labware_df.set_index('deck_pos', inplace=True)
        #add empty containers in list form
        #there's some fancy formating here that gets you a series with deck as the index and
        #comma seperated loc strings eg 'A1,A3,B2' as values
        grouped = empty_containers['loc'].apply(lambda pos: pos+',').groupby('deck_pos')
        labware_locs = grouped.sum().apply(lambda pos: pos[:len(pos)-1])
        labware_df = labware_df.join(labware_locs, how='left')
        labware_df['loc'] = labware_df['loc'].fillna('')
        labware_df.rename(columns={'loc':'empty_list'},inplace=True)
        labware_df.reset_index(inplace=True)
        labware_df['deck_pos'] = pd.to_numeric(labware_df['deck_pos'])
        return labware_df

    def save(self):
        self.portal.send_pack('save')
        #server will initiate file transfer
        files = self.portal.recv_ftp()
        for filename, file_bytes in files:
            local_path = os.path.join(self.eve_files_path, filename)
            with open(local_path, 'wb') as write_file:
                write_file.write(file_bytes)

            #try saving to google drive
            try:
                self.save_to_google_drive('Eve_Files', local_path, filename)
            except Exception as e:
                print(f"<<controller>> Warning: Failed to save {filename} to Google Drive: {str(e)}")
    
        self.translate_wellmap()
    

    def create_google_drive_folder(self, folder_name, parent_folder_id=None):
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_folder_id:
            folder_metadata['parents'] = [parent_folder_id]
        folder = self.drive_service.files().create(body=folder_metadata, fields='id').execute()
        print(f"Created folder: {folder_name} with ID: {folder['id']}")
        return folder['id']

    def get_google_drive_folder_id(self, folder_name, parent_id=None):
        if parent_id:
            query = f"'{folder_name}' in parents and mimeType = 'application/vnd.google-apps.folder' and '{parent_id}' in parents"
        else:
            query = f"'{folder_name}' in parents and mimeType = 'application/vnd.google-apps.folder'"
        response = self.drive_service.files().list(q=query, spaces='drive', fields='nextPageToken, files(id, name)').execute()
        folders = response.get('files', [])
        if folders:
            return folders[0]['id']
        else:
            return None
  
    def save_to_google_drive(self, folder_name, file_path, filename):

        if not self.drive_service:
            print("<<controller.save_to_google_drive>> warning: Google Drive service not initialized. Skipping upload.")
            return


        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file or directory at {file_path} does not exist")
        
        try:
            #get or create folder
            reaction_folder_id = self.get_google_drive_folder_id(self.reaction_folder_name)
            folder_id = self.get_google_drive_folder_id(folder_name, parent_id = reaction_folder_id)

            if folder_id is None:
                parent_folder_id = self.get_google_drive_folder_id(os.path.basename(os.path.dirname(file_path)))
                folder_id = self.create_google_drive_folder(folder_name, parent_folder_id)


            
            file_metadata = {
                'name': os.path.basename(filename),
                'parents': [folder_id]
            }

            mimetypes = {
                '.png': 'image/png',
                '.csv': 'text/csv',
                '.tsv': 'text/tab-separated-values',
                '.txt': 'text/plain'
            }
            file_type = os.path.splitext(filename)[1].lower()
            mimetype = mimetypes.get(file_type, 'application/octet-stream') # default to binary if unknown

            media = MediaIoBaseUpload(file_path, mimetype=mimetype)
            self.drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            print(f"Uploaded file: {filename} to folder with ID: {folder_id}")
        
        except Exception as e:
            print(f"<<controller.save_to_google_drive>> warning: Error uploading file to Google Drive: {e}")

        
    def delete_wks_key(self):
        '''
        deletes key from the reaction key pair google sheet to prevent accidental
        runs in the future
        Postconditions:    
            if the key pair still exists, the key is deleted 
        '''
        wks = self.name_key_wks
        cell_list = wks.findall(str(self.rxn_sheet_name))
        for cell in cell_list   : 
            if cell:
                wks.batch_clear(['B'+str(cell.row)])

    def close_connection(self):
        '''
        runs through closing procedure with robot    
        Postconditions:    
            Log files have been written to self.out_path  
            Connection has been closed  
        '''
        
        print('<<controller>> initializing breakdown')
        self.save()
        #server should now send a close command
        self.portal.send_pack('close')
        print('<<controller>> shutting down')
        self.portal.close()
        self.delete_wks_key()
        
    def translate_wellmap(self):
        '''
        Preconditions:  
            there exists a file wellmap.tsv in self.eve_files, and that file has eve level
            machine labels  
        Postconditions:  
            translated_wellmap.tsv has been created. translated is a copy of wellmap with   
            it's locations translated to human locs, but the labware pos remains the same  
        '''
        df = pd.read_csv(os.path.join(self.eve_files_path,'wellmap.tsv'), sep='\t')
        df['loc'] = df.apply(lambda r: r['loc'] if (r['deck_pos'] not in [4,7]) else self.PLATEREADER_INDEX_TRANSLATOR.inv[(r['loc'],'platereader'+str(r['deck_pos']))],axis=1)
        df.to_csv(os.path.join(self.eve_files_path,'translated_wellmap.tsv'),sep='\t',index=False)

    def init_robot(self, simulate):
        '''
        this does the dirty work of sending accumulated params over network to the robot  
        params:  
            bool simulate: whether the robot should run a simulation  
        Postconditions:  
            robot has been initialized with necessary params  
        '''
        #send robot data to initialize itself
        #note reagent_df can have index with same name so index is reset for transfer
        cid = self.portal.send_pack('init', simulate, 
                self.robo_params['using_temp_ctrl'], self.robo_params['temp'],
                self.robo_params['labware_df'].to_dict(), self.robo_params['instruments'],
                self.robo_params['reagent_df'].reset_index().to_dict(), self.my_ip,
                self.robo_params['dry_containers'].to_dict())

    @abstractmethod
    def run_simulation(self):
        pass

    @abstractmethod
    def run_protocol(self,simulate):
        pass


    def _error_handler(self, e):
        '''
        When an error is thrown from a public method, it will be sent here and handled
        '''
        #handle the error
        if self.portal.state == 1:
            #Armchair recieved an error packet, so eve had a problem
            try:
                eve_error = self.portal.error_payload[0]
                print('''<<controller>>----------------Eve Error----------------
                Eve threw error '{}'
                Attempting to save state on exit
                '''.format(eve_error))
                self.portal.reset_error()
                self.close_connection()
                self.pr.shutdown()
            finally:
                raise eve_error
        else:
            try:
                print('''<<controller>> ----------------Controller Error----------------
                <<controller>> Attempting to save state on exit''')
                self.close_connection()
                self.pr.shutdown()
            finally:
                time.sleep(.5) #this is just for printing format. Not critical
                raise e

    def _load_rxn_df(self, input_data):
        '''
        reaches out to google sheets and loads the reaction protocol into a df and formats the df
        adds a chemical name (primary key for lots of things. e.g. robot dictionaries)
        renames some columns to code friendly as opposed to human friendly names  
        params:  
            list<list<str>> input_data: as recieved in excel  
        returns:  
            pd.DataFrame: the information in the rxn_spreadsheet w range index. spreadsheet cols  
        Postconditions:  
            self._products has been initialized to hold the names of all the products  
        '''
        cols = make_unique(pd.Series(input_data[0])) 
        rxn_df = pd.DataFrame(input_data[4:], columns=cols)
        #rename some of the clunkier columns 
        rxn_df.rename({'operation':'op', 'dilution concentration':'dilution_conc','max number of scans':'max_num_scans','concentration (mM)':'conc', 'reagent (must be uniquely named)':'reagent', 'plot protocol':'plot_protocol', 'pause time (s)':'pause_time', 'comments (e.g. new bottle)':'comments','scan protocol':'scan_protocol', 'scan filename (no extension)':'scan_filename', 'plot filename (no extension)':'plot_filename'}, axis=1, inplace=True)
        rxn_df.drop(columns=['comments'], inplace=True)#comments are for humans
        rxn_df.replace('', np.nan,inplace=True)
        rxn_df[['pause_time','dilution_conc','conc','max_num_scans']] = rxn_df[['pause_time','dilution_conc','conc','max_num_scans']].astype(float)
        rxn_df['reagent'] = rxn_df['reagent'].apply(lambda s: s if pd.isna(s) else s.replace(' ', '_'))
        rxn_df['chemical_name'] = rxn_df[['conc', 'reagent']].apply(self._get_chemical_name,axis=1)
        self._rename_products(rxn_df)
        #go back for some non numeric columns
        rxn_df['callbacks'].fillna('',inplace=True)
        self._products = rxn_df.loc[:,'reagent':'chemical_name'].drop(columns=['chemical_name', 'reagent']).columns
        #make the reagent columns floats
        rxn_df.loc[:,self._products] =  rxn_df[self._products].astype(float)
        rxn_df.loc[:,self._products] = rxn_df[self._products].fillna(0)
        return rxn_df

    @abstractmethod
    def _rename_products(self, rxn_df):
        '''
        Different for Protocol Executor vs auto
        renames dilutions acording to the reagent that created them
        and renames rxns to have a concentration  
        Preconditions:  
            dilution cols are named dilution_1/2 etc  
            callback is the last column in the dataframe  
            rxn_df is not expected to be initialized yet. This is a helper for the initialization  
        params:  
            df rxn_df: the dataframe with all the reactions  
        Postconditions:  
            the df has had it's dilution columns renamed to a chemical name
        '''
        pass

    def _get_products_to_labware(self, input_data):
        '''
        create a dictionary mapping products to their requested labware/containers  
        Preconditions:  
            self.rxn_df must have been initialized already  
        params:  
            list<list<str>> input data: the data from the excel sheet  
        returns:  
            Dict<str,list<str,str>>: effectively the 2nd and 3rd rows in excel. Gives 
                    labware and container preferences for products  
        '''
        cols = self.rxn_df.columns.to_list()
        product_start_i = cols.index('reagent')+1
        requested_containers = input_data[2][product_start_i+1:]
        requested_labware = input_data[1][product_start_i+1:]#add one to account for the first col (labware)
        #in df this is an index, so size cols is one less
        products_to_labware = {product:[labware,container] for product, labware, container in zip(self._products, requested_labware,requested_containers)}
        return products_to_labware

    def _query_reagents(self, spreadsheet_key, credentials):
        '''
        query the user with a reagent sheet asking for more details on locations of reagents, mass
        etc  
        Preconditions:  
            self.rxn_df should be initialized  
        params:  
            str spreadsheet_key: this is the a unique id for google sheet used for i/o with sheets
            ServiceAccount Credentials credentials: to access sheets  
        PostConditions:  
            reagent_sheet has been constructed  
        '''
        #you might make a reaction you don't want to specify at the start
        reagent_df = self.rxn_df.loc[self.rxn_df['op'] != 'make', ['reagent', 'conc']]
        reagent_df = reagent_df.groupby(['reagent','conc'], dropna=False).first().reset_index()
        reagent_df.dropna(how='all',inplace=True)
        rows_to_drop = []
        duplicates = reagent_df['reagent'].duplicated(keep=False)
        for i, reagent, conc in reagent_df.itertuples():
            if duplicates[i] and pd.isna(conc):
                rows_to_drop.append(i)
        reagent_df.drop(index=rows_to_drop, inplace=True)
        reagent_df.set_index('reagent',inplace=True)
        reagent_df.fillna('',inplace=True)

        #add water if necessary
        needs_water = self.rxn_df['op'].apply(lambda x: x in ['make', 'dilution']).any()
        if needs_water:
            if 'Water' not in reagent_df.index:
                reagent_df = reagent_df.append(pd.Series({'conc':1.0}, name='Water'))
            else:
                reagent_df.loc['Water','conc'] = 1.0
        #start dropping products
        rxn_names = self._products.copy() #going to drop template, hence copy
        rxn_names = rxn_names.drop('Template', errors='ignore') #Template will throw error
        #we now need to split the rxn_names into reagent names and concs.
        #There may be duplicate reagents, so we will make a dictionary with list values of 
        #concs
        rxn_name_dict = {}
        for name in rxn_names:
            reagent = self._get_reagent(name)
            conc = self._get_conc(name)
            if reagent in rxn_name_dict:
                #already exists, append to list
                rxn_name_dict[reagent].append(conc)
            else:
                #doesn't exist, create list
                rxn_name_dict[reagent] = [conc]
        rxn_names = pd.Series(rxn_name_dict, name='conc',dtype=object)
        #rxn_names is now a series of concentrations with reagents as keys
        reagent_df = reagent_df.join(rxn_names, how='left', rsuffix='2') 
        reagent_df = reagent_df.loc[
                reagent_df.apply(lambda r: (not isinstance(r['conc2'],list)) 
                or r['conc'] not in r['conc2'], axis=1)
                ].drop(columns='conc2')
        reagent_df[['loc', 'deck_pos', 'mass', 'molar_mass (for dry only)', 'comments']] = ''
        if not self.use_cache:
            if reagent_df.empty:
                #d2g has weird upload behavior so must add a blank row
                blanks = ['' for i in range(reagent_df.shape[1])]
                reagent_df = reagent_df.append(pd.DataFrame([blanks],
                        columns=reagent_df.columns,index=pd.Index([''],name='chemical_name')))
            d2g.upload(reagent_df.reset_index().rename(columns={'index':'chemical_name'}),spreadsheet_key,wks_name = 'reagent_info', row_names=False , credentials = credentials)

    def _get_product_df(self, products_to_labware):
        '''
        Creates a df to be used by robot to initialize containers for the products it will make  
        params:  
            df products_to_labware: as passed to init_robot  
        returns:  
            df products:  
                + INDEX:  
                + str chemical_name: the name of this rxn  
                + COLS:  
                + str labware: the labware to put this rxn in or None if no preference  
                + float max_vol: the maximum volume that will ever ocupy this container  
        '''
        products = products_to_labware.keys()
        max_vols = [self._get_rxn_max_vol(product, products) for product in products]
        product_df = pd.DataFrame(products_to_labware, index=['labware','container']).T
        product_df['max_vol'] = max_vols
        return product_df

    @abstractmethod
    def _get_rxn_max_vol(self, name, products):
        '''
        This needs to be implemented to as a helper for _get_product_df.
        It calculates the maximum volume that a container will hold at a time
        '''
        pass

    def execute_protocol_df(self):
        '''
        takes a protocol df and sends every step to robot to execute  
        params:  
            int buff: the number of commands allowed in flight at a time  
        Postconditions:  
            every step in the protocol has been sent to the robot  
        '''
        for i, row in self.rxn_df.iterrows():
            print("<<controller>> executing command {} of the protocol df with operation {}.".format(i+4, row['op'])) # added 4 to align with order in Gsheets
            if row['op'] == 'transfer':
                self._send_transfer_command(row,i)
            elif row['op'] == 'pause':
                cid = self.portal.send_pack('pause',row['pause_time'])
            elif row['op'] == 'stop':
                self._stop(i)
            elif row['op'] == 'scan':
                self._execute_scan(row, i)
            elif row['op'] == 'dilution':
                self._send_dilution_commands(row, i)
            elif row['op'] == 'mix':
                self._mix(row, i)
            elif row['op'] == 'make':
                self._send_make(row, i)
            elif row['op'] == 'save':
                self.save()
            elif row['op'] == 'plot':
                self._create_plot(row, i)
            elif row['op'] == 'print':
                self._execute_print(row,i)
            elif row['op'] == 'scan_until_complete':
                self._scan_until_complete(row,i)
            else:
                raise Exception('invalid operation {}'.format(row['op']))

    def _execute_print(self, row, i):
        print(row['message'])

    def _create_plot(self, row, i):
        '''
        exectues a plot command  
        params:  
            pd.Series row: a row of self.rxn_df  
            int i: index of this row  
        '''
        wellnames = row[self._products][row[self._products].astype(bool)].index
        plot_type = row['plot_protocol']
        filename = row['plot_filename']
        #make sure you have mapping for all files

        self._update_cached_locs(wellnames)
        pr_dict = {self._cached_reader_locs[wellname].loc: wellname for wellname in wellnames}
        #it's not safe to plot in simulation because the scan file may not exist yet
        df, metadata = self.pr.load_reader_data(row['scan_filename'], pr_dict)
        #execute the plot depending on what was specified
        if plot_type == 'single_kin':
            for wellname in wellnames:
                self.plot_single_kin(df, metadata['n_cycles'], wellname, "{}_{}".format(wellname, filename))
        elif plot_type == 'overlay':
            self.plot_LAM_overlay(df, wellnames, filename)
        elif plot_type == 'multi_kin':
            self.plot_kin_subplots(df, metadata['n_cycles'], wellnames, filename)

    def _download_reagent_data(self, spreadsheet_key, credentials):
        '''
        This is almost line for line inherited, but we need to input in the middle. 
        What can you do?  
        params:  
            str spreadsheet_key: this is the a unique id for google sheet used for i/o with sheets  
            ServiceAccount Credentials credentials: to access sheets  
        returns:  
            df reagent_info: dataframe as pulled from gsheets (with comments dropped)  
        '''
        
        if self.use_cache:
            #if you've already seen this don't pull it
            with open(os.path.join(self.cache_path, 'reagent_info_sheet.pkl'), 'rb') as reagent_info_cache:
                reagent_info = dill.load(reagent_info_cache)
        else:
            input("<<controller>> please press enter when you've completed the reagent sheet")
            #pull down from the cloud
            reagent_info = g2d.download(spreadsheet_key, 'reagent_info', col_names = True, 
                row_names = True, credentials=credentials).drop(columns=['comments'])
            #cache the data
            with open(os.path.join(self.cache_path, 'reagent_info_sheet.pkl'), 'wb') as reagent_info_cache:
                dill.dump(reagent_info, reagent_info_cache)
        #need to rename only the chemicals that were specified with their <name>C<conc> name
        #this is delicate because the indices will not be unique when it is first pulled.
        reagent_info.index = reagent_info.apply(lambda r: "{}C{}".format(r.name,float(r['conc'])) if r['conc'] else r.name,axis=1)
        reagent_info.rename(columns={'molar_mass (for dry only)': 'molar_mass'}, inplace=True)
        return reagent_info

    def _send_make(self, row, i):
        '''
        sends a make command to the robot  
        params:  
            pd.Series row: a row of self.rxn_df  
            int i: index of this row  
        '''
        self.portal.send_pack('make', row['reagent'].replace(' ','_'), row['conc'])

    def _execute_scan(self,row,i):
        '''
        There are a few things entailed in a scan command  
        1) send home to robot  
        2) block until you run out of waits  
        3) figure out what wells you want to scan  
        4) query the robot for those wells, or use cache if you have it  
            a) if you had to query robot, send request of reagents  
            b) wait on robot response  
            c) translate robot response to human readable  
        5) update layout to scanner and scan  
        params:  
            pd.Series row: a row of self.rxn_df  
            int i: index of this row  
        '''
        
        #1)
        self.portal.send_pack('home')
        #2)
        self.portal.burn_pipe()
        #3)
        wellnames = row[self._products][row[self._products].astype(bool)].index
        self._update_cached_locs(wellnames)
        #4)
        #update the locs on the well
        well_locs = []
        for well, entry in [(well, self._cached_reader_locs[well]) for well in wellnames]:
            assert (entry.deck_pos in [4,7]), "tried to scan {}, but {} is on {} in deck pos {}".format(well, well, entry.deck_pos, entry.loc)
            assert (well not in self.tot_vols or math.isclose(entry.vol, self.tot_vols[well])), "tried to scan {}, but {} has a bad volume. Vol was {}, but 200 is required for a scan".format(well, well, entry.vol)
            well_locs.append(entry.loc)
        #5
        self.pr.exec_macro('PlateIn')
        self.pr.run_protocol(row['scan_protocol'], row['scan_filename'], layout=well_locs)
        self.pr.exec_macro('PlateOut')

    def _update_cached_locs(self, wellnames):
        '''
        A query will be
        made to Eve for the wellnames, and data for those will be stored in the cache  
        params:  
            listlike<str> wellnames: the names of the wells you want to lookup  
        Postconditions:  
            The wellnames are in the cache  
        '''
        if not isinstance(wellnames,str):
            #can't send pandas objects over socket for package differences on robot vs laptop
            wellnames = [wellname for wellname in wellnames]
        #couldn't find in the cache, so we got to make a query
        self.portal.send_pack('loc_req', wellnames)
        pack_type, _, payload = self.portal.recv_pack()
        assert (pack_type == 'loc_resp'), 'was expecting loc_resp but recieved {}'.format(pack_type)
        returned_well_locs = payload[0]
        #update the cache
        for well_entry in returned_well_locs:
            if well_entry[2] in [4,7]:
                #is on reader. Need to translate index
                self._cached_reader_locs[well_entry[0]] = self.ChemCacheEntry(*(self.PLATEREADER_INDEX_TRANSLATOR.inv[(well_entry[1],'platereader{}'.format(well_entry[2]))],)+well_entry[2:])
            else:
                #not on reader, just use vanilla index
                self._cached_reader_locs[well_entry[0]] = self.ChemCacheEntry(*well_entry[1:])

    def _mix(self,row,i):
        '''
        this method mixes everything on the platereader with a shake. it mixes other things
        by pipette
        params:  
            pd.Series row: the row with the mix operation
            index i: index of the row in the dataframe
        '''
        wells_to_mix = row[self._products].loc[row[self._products].astype(bool)].astype(int)
        wells_to_mix.name = 'mix_code'
        self._update_cached_locs(wells_to_mix.index)
        deck_poses = pd.Series({wellname:self._cached_reader_locs[wellname].deck_pos for 
                wellname in wells_to_mix.index}, name='deck_pos', dtype=int)
        wells_to_mix_df = pd.concat((wells_to_mix, deck_poses),axis=1)
        #get platereader rows. true if pr
        wells_to_mix_df['platereader'] = wells_to_mix_df['deck_pos'].apply(lambda x: x in [4,7]) 
        if wells_to_mix_df['platereader'].sum() > 0:
            #TODO technically, you could be mixing the other stuff by hand while you're mixing
            #the stuff in the reader, but if you miscalculated and accidently hand mix on the
            #platereader because of a bug, Mark will be mad, so apart for now. After testing
            #you should burn pipe, then send the handmix command, then mix the platereader
            #to multitask

            #at least one well nees a shake
            self.portal.send_pack('home')
            self.portal.burn_pipe() # can't be pulling plate in if you're still mixing
            self.pr.exec_macro('PlateIn')
            if (row.loc[self._products] == 2).any():
                self.pr.shake(60)
            else:
                self.pr.shake(30)
            self.pr.exec_macro('PlateOut')
        if (~wells_to_mix_df['platereader']).sum() > 0:
            #at least one needs to be mixed by hand
            #still df
            hand_mix_wells = wells_to_mix_df.loc[~wells_to_mix_df['platereader']].reset_index()
            #convert to list of tuples
            hand_mix_wells = [tuple(t) for t in hand_mix_wells[['index','mix_code']].itertuples(index=False)]
            self.portal.send_pack('mix', hand_mix_wells)

    def _send_dilution_commands(self,row,i):
        '''
        used to execute a dilution. This is analogous to microcode. This function will send two
          commands. Water is always added first.
            transfer: transfer water into the container
            transfer: transfer reagent into the container  
        params:  
            pd.Series row: a row of self.rxn_df  
            int i: index of this row  
        Preconditions:  
            The buffer has room for at least one command  
        Postconditions:  
            Two transfer commands have been sent to the robot to: 1) add water. 2) add reagent.  
            Will block on ready if the buffer is filled  
        '''
        water_transfer_row, reagent_transfer_row = self._get_dilution_transfer_rows(row)
        prod_transfer = reagent_transfer_row.loc[self._products].ne(0)
        product_val = reagent_transfer_row[self._products][prod_transfer].index
        mix_row = row.copy()
        mix_row['op'] = 'mix'
        mix_row.loc[product_val] = 2 
        self._send_transfer_command(water_transfer_row, i)
        self._send_transfer_command(reagent_transfer_row, i)
        self._mix(mix_row,i)
    def _get_dilution_transfer_rows(self, row):
        '''
        Takes in a dilution row and builds two transfer rows to be used by the transfer command.  
        This command will communicate with the robot to get the current deck position of the
        thing being diluted.  
        This is required because if that thing is on a temperature controller, ColdWater shall
        be used instead of Water.  
        params:  
            pd.Series row: a row of self.rxn_df  
        returns:  
            tuple<pd.Series>: rows to be passed to the send transfer command. water first, then
              reagent
              see self._construct_dilution_transfer_row for details  
            Note the second row (the reagent row) will have have whichever callbacks are passed.
        Preconditions:  
            robot has been initialized  
            Water or ColdWater is on the deck (depending on if this is on temperature module
            or not.  
        '''
        reagent = row['chemical_name']
        #figure out if it is on temperature module
        self._update_cached_locs([reagent])
        deck_pos = self._cached_reader_locs[reagent].deck_pos
        df = self.robo_params['labware_df'] #cause typing hurts
        #iloc is necessary because will give a series by default, but always has one element
        is_temp_cont = df.loc[df['deck_pos'] == deck_pos,'name'].iloc[0] == 'temp_mod_24_tube'
        water_src = 'ColdWaterC1.0' if is_temp_cont else 'WaterC1.0'
        product_cols = row.loc[self._products]
        dilution_name_vol = product_cols.loc[~product_cols.apply(lambda x: math.isclose(x,0,abs_tol=1e-9))]
        #TODO investigate if this works
        #assert (dilution_name_vol.size == 1), "Failure on row {} of the protocol. It seems you tried to dilute into multiple containers"
        target_name = dilution_name_vol.index[0]
        vol_water, vol_reagent = self._get_dilution_transfer_vols(row)
        water_transfer_row = self._construct_dilution_transfer_row(water_src, target_name, vol_water)

        reagent_transfer_row = self._construct_dilution_transfer_row(reagent, target_name, vol_reagent)
        reagent_transfer_row['callbacks'] = row['callbacks'] #give the second row whatever callbacks you had
        return water_transfer_row, reagent_transfer_row

    def _get_dilution_transfer_vols(self, row):
        '''
        calculates the amount of reagent volume needed for a dilution  
        params:  
            float target_conc: the concentration desired at the end  
            float reagent_conc: the concentration of the reagent  
            float total_vol: the total volume requested  
        returns:  
            tuple<float>: size 2
                volume of water to transfer
                volume of reagent to transfer  
        '''
        reagent_conc = row['conc']
        product_cols = row.loc[self._products]
        dilution_name_vol = product_cols.loc[~product_cols.apply(lambda x: math.isclose(x,0,abs_tol=1e-9))]
        total_vol = dilution_name_vol.iloc[0]
        target_conc = row['dilution_conc']

        mols_reagent = total_vol*target_conc #mols (not really mols if not milimolar. whatever)
        vol_reagent = mols_reagent/reagent_conc
        vol_water = total_vol - vol_reagent
        return vol_water, vol_reagent

    def _construct_dilution_transfer_row(self, reagent_name, target_name, vol):
        '''
        The transfer command expects a nicely formated row of the rxn_df, so here we create a row
        with everything in it to ship to the transfer command.  
        params:  
            str reagent_name: used as the chemical_name field  
            str target_name: used as the product_name field  
            str vol: the volume to transfer  
        returns:  
            pd.Series: has all the fields of a regular row, but only [chemical_name, target_name,
              op] have been initialized. The other fields are empty/NaN  
        '''
        template = self.rxn_df.iloc[0].copy()
        template[:] = np.nan
        template[self._products] = 0.0
        template['op'] = 'transfer'
        template['chemical_name'] = reagent_name
        template[target_name] = vol
        template['callbacks'] = ''
        return template

    def _stop(self, i):
        '''
        used to execute a stop operation. reads through buffer and then waits on user input  
        params:  
            int i: the index of the row in the protocol you're stopped on  
        Postconditions:  
            self._inflight_packs has been cleaned  
        '''
        self.portal.send_pack('stop')
        pack_type, _, _ = self.portal.recv_pack()
        assert (pack_type == 'stopped'), "sent stop command and expected to recieve stopped, but instead got {}".format(pack_type)
        if not self.simulate:
            input("stopped on line {} of protocol. Please press enter to continue execution".format(i+1))
        self.portal.send_pack('continue')

    def _send_transfer_command(self, row, i):
        '''
        params:  
            pd.Series row: a row of self.rxn_df
              uses the chemical_name, callbacks (and associated args), product_columns  
            int i: index of this row  
        Postconditions:  
            a transfer command has been sent to the robot  
        '''
        src = row['chemical_name']
        containers = row[self._products].loc[row[self._products] != 0]
        transfer_steps = [name_vol_pair for name_vol_pair in containers.iteritems()]
        #temporarilly just the raw callbacks
        callbacks = row['callbacks'].replace(' ', '').split(',') if row['callbacks'] else []
        if callbacks:
            #if there were callbacks, you must send transfer one at a time, breaking up into
            #iterate through each transfer_step we're doing.
            for callback_num, transfer_step in enumerate(transfer_steps):
                #send just that transfer step
                self.portal.send_pack('transfer', src, [transfer_step])
                #then send a callback for each callback you've got 
                for callback in callbacks:
                    self._send_callback(callback, transfer_step[0], callback_num, row, i)

            #merge all the scans into a single file if there were any scans
            #get the names of all the scan files
            if 'scan' in callbacks:
                dst = row['scan_filename'] #also the base name for all files to be merged
                scan_names = ['{}-{}'.format(dst, chr(i+97)) for i in range(len(transfer_steps))] + ['{}-{}'.format(dst, chr(i+97)+chr(i+97)) for i in range(len(transfer_steps))]
                if len(transfer_steps) <= 25:
                    scan_names = ['{}-{}'.format(dst, chr(i+97)) for i in range(len(transfer_steps))]
                elif len(transfer_steps) > 25:
                    scan_names = ['{}-{}'.format(dst, chr(i+97)) for i in range(26)] + ['{}-{}'.format(dst, chr(i+97)+chr(i+97)) for i in range(len(transfer_steps)-26)]
                    callback_alph = chr(callback_num + ord('a')) + chr(callback_num + ord('a')) #convert the number to alpha
                self.pr.merge_scans(scan_names, dst)
        else:
            self.portal.send_pack('transfer', src, transfer_steps)
        
        self.save()

    def _send_callback(self, callback, product, callback_num, row, i):
        '''
        This method is used to send (or execute) a single callback.  
        params:  
            str callback: the string name of the callback  
            str product: the name of the product. Required to generate things like a 
              scan row.  
            int callback_num: the number of the callback. i.e. 0 if this is the first transfer,
              1 if second, etc. If multiple callbacks, they will all be 0 for a product
            pd.Series row: the row of this operation. (used to extract metaparameters)  
            int i: the index of this command in rxn_df. This will be the same for all the
              callbacks of a single transfer.  
        Postconditions:  
            the callback has been executed/sent
        Preconditions:  
            callback_num must not be larger than 26 (alpha numeric characters are used. If you
              go larger than 26, you'll exceed alpha numeric)
        '''
        if callback_num <= 25:
            callback_alph = chr(callback_num + ord('a')) #convert the number to alpha
        elif callback_num > 25:
            callback_num -= 26
            callback_alph = chr(callback_num + ord('a')) + chr(callback_num + ord('a')) #convert the number to alpha
        i_ext = 'i-{}'.format(callback_alph) #extended index with callback
        if callback == 'stop':
            self._stop(i)
        if callback == 'pause':
            self.portal.send_pack('pause',row['pause_time'])
        if callback == 'scan':
            template = row.copy()
            template.loc[self._products] = 0 
            template.loc[product] = 1
            template['op'] = 'scan'
            #rename the scans with the callback_alph appended
            template['scan_filename'] = '{}-{}'.format(template['scan_filename'], callback_alph)
            #note that there will be some miscellaneous crap left in the row, but shouldn't affect
            #the scan
            self._execute_scan(template, i_ext)
        if callback == 'mix':
            template = row.copy()
            template.loc[self._products] = 0
            template.loc[product] = 1
            template['op'] = 'mix'
            self._mix(template, i_ext)
    
    def _get_chemical_name(self,row):
        '''
        create a chemical name
        from a row in a pandas df. (can be just the two columns, ['conc', 'reagent'])  
        params:  
            pd.Series row: a row in the rxn_df  
        returns:  
            chemical_name: the name for the chemical "{}C{}".format(name, conc) or name if
              has no concentration, or nan if no name  
        '''
        if pd.isnull(row['reagent']) or pd.isnull(row['conc']):
            #this must not be a transfer. this operation has no chemical name
            return np.nan
        else:
            #this uses a chemical with a conc. Probably a stock solution
            return "{}C{}".format(row['reagent'], row['conc'])
        return pd.Series(new_cols)

    def run_all_checks(self):
        '''
        runs all checks on a rxn_df converted to volumes.  
        This code will probably be overridden by children of this class to add more checks.  
        returns:  
            int found_errors:  
                code:  
                0: OK.  
                1: Some Errors, but could run  
                2: Critical. Abort  
        '''
        found_errors = 0
        found_errors = max(found_errors, self.check_rxn_df())
        found_errors = max(found_errors, self.check_labware())
        found_errors = max(found_errors, self.check_reagents())
        found_errors = max(found_errors, self.check_tot_vol())
        found_errors = max(found_errors,self.check_conc())
        return found_errors

    def check_labware(self):
        '''
        checks to ensure that the labware has been correctly initialized  
        returns  
            int found_errors:  
                code:  
                0: OK.  
                1: Some Errors, but could run  
                2: Critical. Abort  
        '''
        found_errors = 0
        for i, r in self.robo_params['labware_df'].iterrows():
            #check that everything has afirst well if it's not a tube
            if not 'tube' in r['name'] and not r['first_usable']:
                print('<<controller>> specified labware {} on deck_pos {}, but did not specify first usable tip/well.'.format(r['name'], r['deck_pos']))
                found_errors = max(found_errors,2)
            #if you're not a tube and you have an empty_list, that's also bad
            if not 'tube' in r['name'] and r['empty_list']:
                print('<<controller>> An empty list for {} on deck pos {} was specified, but {} takes only a first usable tip/well.'.format(r['name'], r['deck_pos'], r['name']))
                found_errors = max(found_errors,2)
            #check for no duplicates in the empty list
            if r['empty_list']:
                locs = r['empty_list'].replace(' ','').split(',')
                if len(set(locs)) < len(locs):
                    print('<<controller>> empty list for {} on deck pos {} had duplicates. List was {}'.format(r['name'],r['deck_pos'], r['empty_list']))
                    found_errors = max(found_errors,2)
        return found_errors 

    def check_reagents(self):
        '''
        checks to ensure that you've specified reagents correctly, and also checks that
        you did not double book empty containers onto reagents  
        returns  
            int found_errors:  
                code:  
                0: OK.  
                1: Some Errors, but could run  
                2: Critical. Abort  
        '''
        found_errors = 0
        #This is a little hefty. We're checking to see if any reagents/empty containers 
        #were double booked onto the same location on the same deck position
        labware_w_empties = self.robo_params['labware_df'].loc[self.robo_params['labware_df']['empty_list'].astype(bool)]
        loc_pos_empty_pairs = [] # will become series
        for i, row in labware_w_empties.iterrows():
            for loc in row['empty_list'].replace(' ','').split(','):
                loc_pos_empty_pairs.append((loc, row['deck_pos']))
        loc_pos_empty_pairs = pd.Series(loc_pos_empty_pairs, dtype=object)
        loc_deck_pos_pairs = self.robo_params['reagent_df'].apply(lambda r: (r['loc'], r['deck_pos']),axis=1)
        loc_deck_pos_pairs = loc_deck_pos_pairs.append(loc_pos_empty_pairs)
        val_counts = loc_deck_pos_pairs.value_counts()
        for i in val_counts.loc[val_counts > 2].index:
            print('<<controller>> location {} on deck position has multiple reagents/empty containers assigned to it')
            found_errors = max(found_errors,2)
        return found_errors

    def check_rxn_df(self):
        '''
        Runs error checks on the reaction df to ensure that formating is correct. Illegal/Ill 
        Advised options are printed and if an error code is returned
        Will run through and check all rows, even if errors are found
        returns  
            int found_errors:  
                code:  
                0: OK.  
                1: Some Errors, but could run  
                2: Critical. Abort  
        '''
        found_errors = 0
        if self.rxn_df.loc[self.rxn_df['op']=='scan']['scan_filename'].duplicated().sum() > 0:
            print("<<controller>> Multiple scans use same filename. It will be overwritten. Do you wish to proceed?")
            found_errors = max(found_errors, 1)
        if self.rxn_df.loc[self.rxn_df['op']=='plot']['plot_filename'].duplicated().sum() > 0:
            print("<<controller>> Multiple plots use same filename. They will be overwritten. Do you wish to proceed?")
            found_errors = max(found_errors, 1)
        for i, r in self.rxn_df.iterrows():
            r_num = i+1
            #check pauses
            if (not ('pause' in r['op'] or 'pause' in r['callbacks'] or r['op'] == 'scan_until_complete')) == (not pd.isna(r['pause_time'])):
                print("<<controller>> You asked for a pause in row {}, but did not specify the pause_time or vice versa".format(r_num))
                found_errors = max(found_errors, 2)
            #check that there's always a volume when you transfer
            if (r['op'] == 'transfer' and math.isclose(r[self._products].sum(), 0,abs_tol=1e-9)):
                print("<<controller>> You executed a transfer step in row {}, but you did not transfer any volume.".format(r_num))
                found_errors = max(found_errors, 1)
            #check that you have a reagent if you're transfering
            if r['op'] == 'transfer' and pd.isna(r['reagent']):
                print('<<controller>> transfer specified without reagent in row {}'.format(r_num))
                found_errors = max(found_errors,2)
            #check that scans have a scan file
            if (r['op'] == 'scan' or 'scan' in r['callbacks']) and pd.isna(r['scan_filename']):
                print('<<controller>> scan without scan filename in row {}'.format(r_num))
                found_errors = max(found_errors,2)
            #check no multiple scans on one callback
            callbacks = r['callbacks'].replace(' ', '').split(',')
            if 'scan' in callbacks:
                callbacks.remove('scan')
                if 'scan' in callbacks:
                    print('<<controller>> multiple scans in a callback on line {}'.format(r_num))
                    found_errors = max(found_errors,2)
            #check that plots have scans
            if r['op'] == 'plot':
                if pd.isna(r['scan_filename']):
                    print("<<controller>> please specify a scan filename in row '{}'".format(r_num))
                    found_errors = max(found_errors,2)
                if pd.isna(r['plot_filename']):
                    print("<<controller>> please specify a plot filename in row '{}'".format(r_num))
                    found_errors = max(found_errors,2)
                rows_above = self.rxn_df.loc[:i,:]
                scan_rows = rows_above.loc[(rows_above['scan_filename'] == r['scan_filename']) &\
                        ((rows_above['op'] == 'scan')| (rows_above['op'] == 'scan_until_complete'))]
                
                
                if scan_rows.empty:
                        print("<<controller>> row {} plots using nonexistent scan file\
                                ".format(r_num))
                        found_errors = max(found_errors, 2)
                else:
                    last_scan_row = scan_rows.iloc[-1,:]
                    last_scan_products = last_scan_row[self._products]
                    scanned_products=last_scan_products.loc[last_scan_products.astype(bool)].index
                    scanned_products = set(scanned_products)
                    plotted_products = r[self._products]
                    plotted_products = set(plotted_products[plotted_products.astype(bool)])
                    if plotted_products.issubset(scanned_products):
                        print("<<controller>> row {} plots products that have not been scanned\
                        ".format(r_num))
                        found_errors = max(found_errors, 2)
        return found_errors

    def check_tot_vol(self):
        '''
        This check ensures that the inserted total volume row does not contain negative floats.
        returns:  
            int found_errors:  
                code:  
                0: OK.  
                1: Some Errors, but could run  
                2: Critical. Abort  
        '''        
        found_errors = 0
        
        #checks for negative input in tot_vol rows
        for key,val in self.tot_vols.items():
            product_volumes = self.rxn_df[key]
            if val < 0:
                print("<<controller>> Error in total volume row: value " + str(val) + " is negative. We cannot have negative values as input.")
                found_errors = max(found_errors,2)
            
        #checks for scan errors
        check_scan = self.rxn_df.loc[(self.rxn_df['op'] == 'scan')]
        #make sure if you're scanning you have a total volume
        cols_w_scans = check_scan[self._products].astype(int).any() #bool arr if product is scaned
        cols_w_scans = cols_w_scans.loc[cols_w_scans].index #just the cols that are scanned
        for col in cols_w_scans:
            if col not in self.tot_vols:
                print("<<controller>> {} is scanned, but does not have a specified total volume. Will be scanned at whatever volume it has at the time of scan.".format(col))
                found_errors = max(found_errors,1)
        #check more scan issues
        first_scans_i = check_scan[check_scan.eq(check_scan.max(1),0)&check_scan.ne(0)].stack()   
        scan_products = []
        #Creates list for products that have scans
        for prod in self.tot_vols.keys():
            for sc in  first_scans_i.index:
                if prod == sc[1]:
                    scan_products.append([prod,sc[0]])  
        #checks if all transfers happen before scan
        for products in scan_products:
            specific_prod = self.rxn_df[products[0]]
            scan_index = products[1]
            while (scan_index < len(specific_prod)):
                if self.rxn_df['op'][scan_index] == 'transfer' and specific_prod[scan_index] != 0:
                    print("<<controller>> Error in product: " +str(products[0]) +" in index: " +str(scan_index) + ", cannot make transfers after scan when total volume column is specified.")
                    found_errors = max(found_errors,2)
                    break
                else:
                    scan_index +=1
                
        #check for illegal dilutions in total vol
        check_dilutions = self.rxn_df.loc[(self.rxn_df['op'] == 'dilution')]
        check_dilutions_name = self.rxn_df.loc[(self.rxn_df['op'] == 'dilution'),'chemical_name']
        first_dilutions_i = check_dilutions[check_dilutions.eq(check_dilutions.max(1),0)&check_dilutions.ne(0)].stack()
        for prod in self.tot_vols.keys():
            for dil in first_dilutions_i.index:
                if prod == dil[1]:
                    print("<<controller>> Error in product: " + str(prod) + " in index: " +str(dil[0]) + ", cannot dilute products that have a given total volume")
                    found_errors = max(found_errors,2)
                    break
        #checks for dilutions in reagent slot--illegal!
        for idx,dil_prod in enumerate(check_dilutions_name):
            if dil_prod in self.tot_vols.keys():
                print("<<controller>> Error in reagent row index "+str(idx) +" with product "+  str(dil_prod) + ": cannot have dilutions out of product with total volume specified.")
                found_errors = max(found_errors,2)
                
        #Checks reagents to see if there is a transfer that transfers a product with tot_vol
        check_transfer = self.rxn_df.loc[(self.rxn_df['op'] == 'transfer'),'chemical_name']
        for idx,trans_prod in enumerate(check_transfer):
            if trans_prod in self.tot_vols.keys():
                print("<<controller>> Error in reagent row index "+str(idx) +" with product "+  str(trans_prod) + ": cannot have transfer out of product with total volume specified.")
                found_errors= max(found_errors,2)
        
        return found_errors 

    def _get_transfer_container(self,reagent,molarity,total_vol,ratio=1.0):
        '''
        This function is responsible for converting from a reagent (without concentration) to
        a uniquely identified container that holds that reagent. This is used when rows are
        specified as molarities as opposed to volumes because the container must be chosen
        from a number of containers that may hold that reagent at different concentations.
        There are a number of ways to optimize which container should be chosen. This 
        algorithm will always take the most concentrated solution unless there is not sufficient
        volume, or the volume that would be required to pipette is less than the minimum
        pipettable volume. defined here as 2uL.  
        params:  
            str reagent: the name of the reagent that you are searching for a container for  
            float molarity: the desired molarity at end of reaction.  
            float total_vol: the total volume that this well will have at end of the reaction.  
            float ratio: between 1 and 0 if specified, this specifies that this addition 
              will only add the ratio of the reagent, (important because it affects the min
              vol that would be added with this transfer. effectively multiplies total_vol 
              by ratio)  
        returns:  
            tuple<str, float>: if a match was found for the reagent   
                str: the container name.  
                float: the volume that must be transfered with this container.  
        raises:  
            ConversionError: when the molarity cannot be acheived without overdrawing from
              container, or by pipetting less than min_vol  
        Preconditions:
            the cached_reader_locs should be up to date  
        '''
        min_vol = 5
        containers = [key for key in self._cached_reader_locs.keys() 
                if re.fullmatch(reagent+'C\d*\.\d*', key)]
        containers.sort(key=self._get_conc)
        filtered_conts = [] #this will hold the containers that are diluted enough to be able
        #to transfer without exceeding min_vol
        for cont in containers:
            vol = self._get_transfer_vol(cont,molarity,total_vol,ratio)
            if vol > min_vol:
                filtered_conts.append(cont)
                if vol < self._cached_reader_locs[cont].aspirable_vol:
                    return cont, vol
        raise ConversionError(reagent, molarity, total_vol, ratio, filtered_conts)


    def _convert_conc_to_vol(self, rxn_df, products):
        '''
        This function converts any molarity rows into volume rows  
        params:  
            df rxn_df: the reaction dataframe with some concentration rows  
            str products: the names of the products  
        returns:  
            df: the rxn_df with all concentrations converted to volumes if things went well  
        raises:  
            ConversionError: for too small vol transfer, run out of vol in a reagent, or 
              overflow  
        '''
        #We now need to iterate through df and for each column, calculate the container to pull
        #from, and volume. Since one row may now pull from muliple reagents, this causes a
        #rebuild of the dataframe. We accumulate a list of series and then rebuild
        disassembled_df = [] # list of series

        for i, row in rxn_df.iterrows():
            if row['op'] == 'transfer' and pd.isna(row['conc']):
                #needs the concentration to be converted
                testCont = row[products].reset_index()                
                cont_vol_key = row[products].reset_index().apply(lambda r:
                        pd.Series({x: y for x, y in 
                        zip(['chem_name', 'vol'], 
                                (np.nan,np.nan) if math.isclose(r.iloc[1], 0, abs_tol=1e-9) else
                                    self._get_transfer_container(row['reagent'], r.iloc[1],
                                        self.tot_vols[r['index']],ratio=1.0))}),axis=1)
                cont_vol_key.index = products
                conts = cont_vol_key['chem_name'].dropna().unique()
                for cont in conts:
                    new_row = row.copy()
                    new_row['chemical_name'] = cont
                    new_row['conc'] = self._get_conc(cont)
                    for product in products:
                        new_row[product] = cont_vol_key.loc[product,'vol'] if \
                            cont_vol_key.loc[product,'chem_name'] == cont else 0
                    disassembled_df.append(new_row)
            else:
                disassembled_df.append(row)
        return pd.DataFrame(disassembled_df)

    def _get_transfer_vol(self,reagent,molarity,total_vol,ratio):
        '''
        helper function to calculate the necessary volume for a transfer given a reagent and
        desired molarity and a volume (and some other stuff)  
        params:  
            str reagent: the chemical name of the reagent fullname that you are searching for  
            float molarity: the desired molarity at end of reaction.  
            float total_vol: the total volume that this well will have at end of the reaction.  
            float ratio: between 1 and 0 if specified, this specifies that this addition 
              will only add the ratio of the reagent, (important because it affects the min
              vol that would be added with this transfer. effectively multiplies total_vol 
              by ratio)  
        returns:  
            float: the volume to transfer from the reagent for desired end molarity  
        '''
        conc = self._get_conc(reagent)
        vol = molarity * (total_vol*ratio) / conc
        return vol
        
    def _vol_calc(self, name):
        '''
        calculates the total volume of a column at the end of rxn  
        params:
            str name: chem_name
        returns:
            volume at end in that name
        '''
        dispenses = self.rxn_df.loc[(self.rxn_df['op'] == 'dilution') |
                (self.rxn_df['op'] == 'transfer')][name].sum()
        transfer_aspirations = self.rxn_df.loc[(self.rxn_df['op']=='transfer') &\
                (self.rxn_df['chemical_name'] == name),self._products].sum().sum()
        dilution_rows = self.rxn_df.loc[(self.rxn_df['op']=='dilution') &\
                (self.rxn_df['chemical_name'] == name),:]
        def calc_dilution_vol(row):
            return self._get_dilution_transfer_vols(row)[1]

        if dilution_rows.empty:
            dilution_aspirations = 0.0
        else:
            dilution_vols = dilution_rows.apply(lambda r: calc_dilution_vol(r),axis=1)
            dilution_aspirations = dilution_vols.sum()
        return dispenses - transfer_aspirations - dilution_aspirations
    
    def _get_conc(self, chem_name):
        '''
        handy method for getting the concentration from a chemical name  
        params:  
            str chem_name: the chemical name to strip a concentration from  
        returns:  
            float: the concentration parsed from the chem_name  
        '''
        return float(re.search('C\d*\.\d*$', chem_name).group(0)[1:])

    def _get_reagent(self, chem_name):
        '''
        handy method for getting the reagent from a chemical name  
        The foil of _get_conc  
        params:  
            str chem_name: the chemical name to strip a reagent name from  
        returns:  
            str: the reagent name parsed from the chem_name  
        '''
        
        return chem_name[:re.search('C\d*\.\d*$', chem_name).start()]

    def _handle_conversion_err(self,e):
        '''
        This function will handle errors caught in the conversion process from molarity to
        volume reaction dataframe.  
        params:  
            ConversionError e: the conversion error raised  
        Postconditions:  
            If the error was pipetting infinitesimal volume, a dilution has been performed on
            the robot to dilute by 2X   
        Raises:  
            NotImplementedError: If you ran out of a reagent you probably need to have Mark
              restock (or you could dilute a stock maybe)  
        '''
        #TO DO!! IMPLEMENT HANDLE_CONVERSION ERROR INTO ABSTRACT, Currently it will produce many errors as we try to integrate this functionality.
        raise NotImplementedError("We need to implement the handling of dilution errors into the controller. Currently it does not work.")
           


    def _execute_single_dilution(self, end_conc, reagent):
        '''
        This function creates a single dilution row and executes that row.  
        This involves:  
        + 1 inititializing a new product with the desired name  
        + 2 constructing a new dilution row (series), and then turn that into a dataframe  
        + 3 save rxn_df and associated metadata and overwrite with the dilution row.
            restore immediately after execution
        params:  
            float end_conc: the end concentration of the dilution  
            str reagent: the full chemical name of the reagent to be diluted  
            float vol: the end volume of the dilution  
        Postconditions:  
            a command has been sent to the robot requesting initialization of a container for
            this dilution  
            a command has been sent to the robot to perform a dilution  
        '''
        #1 initialize the new product on the robot
        product = '{}C{}'.format(self._get_reagent(reagent), end_conc)
        product_df = pd.DataFrame(
                    {'labware':'',
                    'container':self.dilution_params.cont,
                    'max_vol':self.dilution_params.vol}, index=[product])
        self.portal.send_pack('init_containers', product_df.to_dict())
        #2 construct a new dilution row (series)
        colList = self.rxn_df.loc[:,:'reagent'].columns        
        row = pd.Series(np.nan, colList)
        row['op'] = 'dilution'
        row['callbacks'] = ''
        row['dilution_conc'] = end_conc
        row['chemical_name'] = reagent
        row['conc'] = self._get_conc(reagent)
        row['reagent'] = self._get_reagent(reagent)
        row['Template'] = self.dilution_params.vol
        row.rename({'Template':product},inplace=True)
        #print(row)
        #3 call send_dilution
        #here we're appropriating a method that was designed to be run on the dataframe with
        #associated metaparameters (esp _products). We temporarilly overwrite products and restore
        #immediately afterwards
        cached_products = self._products
        cached_rxn_df = self.rxn_df
        self._products = [product]
        self.rxn_df = pd.DataFrame([row])
        self.execute_protocol_df()
        self._products = cached_products
        self.rxn_df = cached_rxn_df

    def _scan_until_complete(self,row,i):
        """
        This function handles the scan_until_complete operation.
        This involves:
            executing and creating a new scan row, that takes in the scan row,
            executes the scan, and then compares the oldScan to the newScan with a different
            filename to allow spacing between the two scans until they are indifferentiably the same
        """
        #Count is declared to track how long we want the process to run if it is going to take too long for there to be distinction
        count = 1
        
        #Eps represents the difference variable that we want in order to check if the scans are similar enough
        eps = 3/700 
            
        scan_product_index = row[self._products].ne(0)
        
        wellnames = row[self._products][scan_product_index].index
        oldScan = self._build_suc_row(row,count)
        self._execute_scan(oldScan,i)
        
        #cached reader locs updated by scan
        pr_dict = {self._cached_reader_locs[wellname].loc: wellname for wellname in wellnames}
        
        old_scan_data, metadata = self.pr.load_reader_data(oldScan['scan_filename'], pr_dict)
        
        if not self.simulate: 
            time.sleep(row['pause_time'])
        
        count += 1
        
        newScan = self._build_suc_row(row,count)
        self._execute_scan(newScan,i)
        new_scan_data,metadata =  self.pr.load_reader_data(newScan['scan_filename'], pr_dict)
         
        #checks difference, defines old_scan to new scan, until they are similar
        #Divide by 700 eps should = 3/700
        #divide result by 700 aswell
        while (((((((new_scan_data - old_scan_data)**2)/700)>eps).any()).any()) and (count < row['max_num_scans'])):    
            oldScan = newScan
            old_scan_data = new_scan_data
            
            if not self.simulate:
                time.sleep(row['pause_time'])
            
            newScan = self._build_suc_row(row,count)
            self._execute_scan(newScan,i)
            new_scan_data,metadata = self.pr.load_reader_data(newScan['scan_filename'], pr_dict) 
            count += 1
        #Renames the unique filename back to what it was declared as in the sheet    
        self.pr._rename_scan(newScan['scan_filename'],row['scan_filename'])
        

    def _build_suc_row(self,row,count):
       #Builds a row for the scan_until_complete function
        
        newFilename =  "{}_suc_{}".format(row['scan_filename'], count)
        newRow = row.copy()
        newRow['op'] = 'scan'
        newRow['scan_filename'] = newFilename
        
        return newRow

    def check_conc(self):
        """
        Makes checks about concentration to see if the concentrations declared are legal declarations
        """

        found_errors = 0
        #Check to make sure water always has a concentration defined
        check_water_conc = (self.rxn_df.loc[(self.rxn_df['reagent']=='Water'),'conc'].isna())
        if check_water_conc.any():
            print("<<controller>> Error in index: "+ str(check_water_conc.loc[check_water_conc].index[0])+ " Water needs to always have a concentration defined.")
            found_errors = max(found_errors,2)
        #Check to make sure you don't transfer a reagent with a concentration into a reagent with a volume
        #boolean list of all concentrations that are nan
                
        check_conc = (self.rxn_df.loc[(self.rxn_df['op']== 'transfer'),'conc'].isna())
        transfer_df = self.rxn_df.loc[(self.rxn_df['op'] == 'transfer')]
        #Check_nan a list of all reagents that dont have a concentration
        check_nan = (transfer_df.loc[(check_conc),'reagent'].unique())
        #check_vol list of all reagents that dont have a volume
        check_vol = (transfer_df.loc[(~check_conc),'reagent'].unique())
        
        for prod in self._products:
            col = self.rxn_df[prod].ne(0)
            product_df = self.rxn_df.loc[col]
            check_concs = (product_df.loc[(product_df['op'] == 'transfer'),'conc'].isna())
            transfer_dfs = product_df.loc[(product_df['op'] == 'transfer')]
            check_nans = (transfer_dfs.loc[(check_concs),'reagent'].unique())
            check_vols = (transfer_dfs.loc[(~check_concs),'reagent'].unique())
            for val in check_nans:
                if val in check_vols:
                    print("<<controller>> Error in reagent " + val + ", cannot transfer a reagent without a concentration into the same product with a reagent with concentration.")
        
        #Checks to make sure all reagents with molarity get transferred into products with total volume
        tot_vol_mol = transfer_df.loc[check_conc,self._products]
        if not tot_vol_mol.empty:
            tot_vol_mol = tot_vol_mol.sum().apply(lambda x: not math.isclose(x, 0, abs_tol=1e-9))
            tot_vol_mol = tot_vol_mol.loc[tot_vol_mol].index
            for i in tot_vol_mol:
                if i not in self.tot_vols.keys():
                    print("<<controller>> Error in product: " + str(i) + " you can only transfer reagents with molarity into products with total volume specified.")
                    found_errors = max(found_errors, 2)
        return found_errors

    def create_connection(self, simulate, no_pr, port):
        self._init_pr(simulate, no_pr)
        #create a connection
        sock = socket.socket(socket.AF_INET)
        sock.connect((self.server_ip, port))
        buffered_sock = BufferedSocket(sock, maxsize=1e9, timeout=None)
        print("<<controller>> connected")
        self.portal = Armchair(buffered_sock,'controller','Armchair_Logs', buffsize=4)
        self.init_robot(simulate)

class AutoContr(Controller):
    '''
    This is a completely automated controller. It takes as input a layout sheet, and then does
    it's own experiments, pulling data etc  
    We're adding in self.rxn_df_template, which uses the same parsing style as rxn_df
    but it's only a template, so we give it a new name and use self.rxn_df to change for the current batch we're trying to make
    '''

    def _clean_template(self):
        '''
        There are some traces of the template column that must be removed from the rxn_df and 
        associated data structures at this point before further processing.  
        Preconditions:  
            self._products includes 'Template'  
            self.tot_vols includes 'Template'  
            self.robo_params['product_df'] holds the product info for Template  
        Postconditions:  
            'Template' has been removed from self._products  
            'Template' has been removed from self.tot_vols  
            self.template_meta has been initialized to a dictionary with meta data for template
            The key 'product_df' has been removed from self.robo_params (you should never have
              need to access it.  
        '''
        self.template_meta = {
                'tot_vol':self.tot_vols['Template'],
                'cont':self.robo_params['product_df'].loc['Template', 'container'],
                'labware':self.robo_params['product_df'].loc['Template', 'labware']
                }
        del self.robo_params['product_df']
        self._products = []
        del self.tot_vols['Template']

    def __init__(self, rxn_sheet_name, my_ip, server_ip, buff_size=4, use_cache=False, cache_path='Cache', num_duplicates=3):
        super().__init__(rxn_sheet_name, my_ip, server_ip, buff_size, use_cache, cache_path)
        self.variable_reagents = self.get_variable_reagents()
        print(f'variable reagents: {self.variable_reagents}')
        self.fixed_reagents = self.get_fixed_reagents()
        self.y_shape = len(self.variable_reagents)
        print(f"y-shape is {self.y_shape}")
        print(self.robo_params['reagent_df'])
        self.run_all_checks()
        self.rxn_df_template = self.rxn_df
        self.reagent_order = self.rxn_df['reagent'].dropna().loc[self.rxn_df['conc'].isna()].unique()
        print(f'reagent_order: {self.reagent_order}')
        self._clean_template() #moves template data out of the data for rxn_df
        self.experiment_data = pd.DataFrame(columns = ["Recipes", "Wellnames", "Experiment_result"])
        self.num_duplicates = num_duplicates
    
    # Update experiment_data DataFrame after each batch
    def _update_experiment_data(self, wellnames, recipes, Experiment_result):
        
        new_data = pd.DataFrame({
            'Recipes': recipes.reshape(1,-1).tolist()[0],
            'WellNames': wellnames,
            'Experiment Result': Experiment_result
        })
        self.experiment_data = pd.concat([self.experiment_data, new_data], ignore_index=True)

    def get_variable_reagents(self):

        # Find unique reagents where 'conc' is NaN and 'op' equals 'transfer'
        unique_reagents = self.rxn_df.loc[self.rxn_df['conc'].isna() & (self.rxn_df['op'] == 'transfer'), 'reagent'].unique()

        # Calculate the number of unique reagents
        num_unique_reagents = len(unique_reagents)

        print(f"Number of unique variable reagents: {num_unique_reagents}")
        print(f"List of unique variable reagent names: {unique_reagents}")
        return unique_reagents
    
    def get_fixed_reagents(self):
        # Find unique reagents where 'conc' is NaN and 'op' equals 'transfer'
        fixed_reagents = self.rxn_df.loc[self.rxn_df['conc'].notna() & (self.rxn_df['op'] == 'transfer'), 'reagent'].unique() 
        # Calculate the number of unique reagents
        num_unique_reagents = len(fixed_reagents)

        print(f"Number of fixed reagents: {num_unique_reagents}")
        print(f"List of fixed reagent names: {fixed_reagents}")
        return fixed_reagents       
 
    def run_simulation(self,model=None,no_pr=False):
        '''
        runs a full simulation of the protocol on local machine
        Temporarilly overwrites the self.server_ip with loopback, but will restore it at
        end of function  
        params:
            MLModel model: the model to use when training and predicting  
        Returns:  
            bool: True if all tests were passed  
        '''
        #cache some things before you overwrite them for the simulation
        stored_server_ip = self.server_ip
        stored_simulate = self.simulate
        self.server_ip = '127.0.0.1'
        self.simulate = True
        if model == None:
            #you're simulating with a dummy model.
            print('<<controller>> running with dummy ml')
            #TODO fix sim flow
            model = DummyMLModel(self.reagent_order.shape[0], max_iters=2)
        print('<<controller>> ENTERING SIMULATION')
        port = 50000
        #launch an eve server in background for simulation purposes
        b = threading.Barrier(2,timeout=20)
        eve_thread = threading.Thread(target=launch_eve_server, kwargs={'my_ip':'','barrier':b},name='eve_thread')
        eve_thread.start()
        #do create a connection
        b.wait()
        self._run(port, True, model, no_pr)

        #collect the eve thread
        eve_thread.join()

        #restore changed vars
        self.server_ip = stored_server_ip
        self.simulate = stored_simulate
        print('<<controller>> EXITING SIMULATION')
        return True

    def run_protocol(self, model=None, simulate=False, port=50000, no_pr=False):
        '''
        The real deal. Input a server addr and port if you choose and protocol will be run  
        params:  
            str simulate: (this should never be used in normal operation. It is for debugging
              on the robot)  
            bool no_pr: if True, will not use the plate reader even if possible to simulate
            MLModel model: the model to use when training and predicting  
        NOTE: the simulate here is a little different than running run_simulation(). This simulate
          is sent to the robot to tell it to simulate the reaction, but that it all. The other
          simulate changes some things about how code is run from the controller
        '''
        print('<<controller>> RUNNING')
        if model == None:
            #you're simulating with a dummy model.
            print('<<controller>> running with dummy ml')
            model = DummyMLModel(self.reagent_order.shape[0], max_iters=2)
        self._run(port, simulate, model, no_pr)
        print('<<controller>> EXITING')

    def _rename_products(self, rxn_df):
        '''
        required for class compatibility, but not used by the Auto  
        '''
        pass

    @error_exit
    def _run(self, port, simulate, model, no_pr):
        '''
        private function to run
        '''
        #TODO Normalize Inputs
        def normalize(x, min_val, max_val):
           return (x - min_val) / (max_val - min_val)

        def denormalize(x_normalized, min_val, max_val):
            return x_normalized * (max_val - min_val) + min_val
        
        self.batch_num = 0 #used internally for unique filenames
        self.well_count = 0 #used internally for unique wellnames
        self.create_connection(simulate, no_pr, port)
        # Begin optimization
        print('<<controller>> executing batch {}'.format(self.batch_num))

        # Generate initial design and simulate experiments to get initial data
        X_initial = model.generate_initial_design()
        print(f"X_initial: {X_initial}")
        X_Normalized = normalize(X_initial,0,0.002)
        print(f"X Normalized: {X_Normalized}")

        recipes =  self.duplicate_list_elements(X_initial, self.num_duplicates)    

        print(f'Initial seeds as follows: {recipes}') 

        #generate wellnames for this batch
        wellnames = [self._generate_wellname() for i in range(recipes.shape[0])]
        #plan and execute a reaction with duplicates.
        print(f'creating samples at wellnames: {wellnames}')

        self._create_samples(wellnames, recipes)

        print('samples created')
        #pull in the scan data
        filenames = self.rxn_df[
                (self.rxn_df['op'] == 'scan') |
                (self.rxn_df['op'] == 'scan_until_complete')
                ].reset_index()
        #TODO filenames is empty. dunno why

        print(f'filenames: {filenames}')
        last_filename = filenames.loc[filenames['index'].idxmax(),'scan_filename']
        scan_data = self._get_sample_data(wellnames, last_filename)

        def find_max(scan_data):
            find_max_df = scan_data



            list_of_scans = find_max_df.columns
            range_list = [*range(300,1001,1)]
            find_max_df.index = range_list
            list_of_blank_data = [0.491,0.45,0.416,0.387,0.364,0.344,0.327,0.311,0.298,0.287,0.277,0.269,0.263,0.258,0.253,0.247,0.242,0.237,0.232,0.227,0.222,0.218,0.213,0.209,0.205,0.201,0.196,0.192,0.191,0.187,0.183,0.179,0.174,0.172,0.168,0.161,0.159,0.156,0.151,0.147,0.145,0.142,0.14,0.137,0.133,0.131,0.13,0.128,0.126,0.123,0.122,0.12,0.117,0.116,0.115,0.112,0.109,0.108,0.105,0.105,0.103,0.1,0.096,0.094,0.093,0.091,0.09,0.087,0.084,0.08,0.077,0.074,0.072,0.068,0.064,0.062,0.059,0.057,0.055,0.053,0.053,0.051,0.05,0.049,0.049,0.047,0.046,0.045,0.045,0.045,0.044,0.044,0.043,0.044,0.043,0.043,0.042,0.042,0.043,0.043,0.042,0.041,0.041,0.041,0.04,0.04,0.04,0.04,0.04,0.04,0.039,0.039,0.039,0.04,0.038,0.039,0.037,0.038,0.038,0.038,0.038,0.037,0.037,0.036,0.037,0.037,0.037,0.037,0.036,0.036,0.037,0.037,0.036,0.036,0.036,0.036,0.035,0.035,0.035,0.035,0.035,0.035,0.036,0.035,0.035,0.035,0.035,0.035,0.035,0.035,0.034,0.034,0.034,0.034,0.034,0.034,0.034,0.034,0.034,0.034,0.034,0.033,0.034,0.033,0.034,0.034,0.034,0.034,0.034,0.033,0.033,0.033,0.033,0.033,0.033,0.033,0.033,0.033,0.033,0.033,0.033,0.033,0.033,0.033,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.033,0.032,0.032,0.032,0.032,0.032,0.032,0.033,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.031,0.032,0.032,0.031,0.032,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.032,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.032,0.031,0.031,0.032,0.032,0.032,0.031,0.031,0.032,0.031,0.031,0.032,0.031,0.031,0.032,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.03,0.031,0.031,0.031,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.031,0.031,0.031,0.03,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.03,0.03,0.03,0.031,0.031,0.031,0.03,0.031,0.031,0.031,0.031,0.031,0.03,0.03,0.031,0.03,0.031,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.031,0.031,0.03,0.03,0.03,0.03,0.03,0.029,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.029,0.029,0.029,0.03,0.03,0.03,0.03,0.03,0.03,0.031,0.03,0.03,0.03,0.029,0.03,0.029,0.03,0.03,0.029,0.03,0.029,0.029,0.029,0.029,0.029,0.029,0.029,0.029,0.03,0.029,0.029,0.029,0.029,0.029,0.029,0.029,0.03,0.029,0.029,0.029,0.029,0.029,0.03,0.028,0.028,0.029,0.029,0.029,0.029,0.028,0.029,0.029,0.029,0.029,0.029,0.029,0.03,0.029,0.029,0.029,0.029,0.029,0.029,0.029,0.03,0.03,0.029,0.029,0.03,0.03,0.029,0.03,0.031,0.029,0.03,0.03,0.031,0.031,0.03,0.031,0.03,0.031,0.031,0.031,0.031,0.031,0.031,0.032,0.031,0.032,0.032,0.032,0.033,0.033,0.033,0.033,0.034,0.034,0.033,0.033,0.034,0.034,0.034,0.034,0.033,0.033,0.035,0.034,0.034,0.035,0.034,0.034,0.034,0.035,0.035,0.034,0.034,0.033,0.035,0.035,0.034,0.034,0.034,0.034,0.034,0.034,0.034,0.034,0.034,0.035,0.035,0.034,0.034,0.034,0.034,0.034,0.034,0.034,0.034,0.034,0.034,0.034,0.034,0.034,0.034,0.034,0.033,0.033,0.033,0.033,0.033,0.034,0.033,0.033,0.033,0.032,0.032,0.032,0.032,0.032,0.032,0.033,0.032,0.034,0.033,0.033,0.032,0.032,0.033,0.032,0.033,0.032,0.033,0.033,0.033,0.033,0.033,0.033,0.033,0.032,0.032,0.033,0.031,0.031,0.03,0.032,0.032,0.033,0.032,0.033,0.034,0.034,0.036,0.036,0.035,0.035,0.035,0.035,0.036,0.036,0.036,0.036,0.035,0.036,0.037,0.037,0.038,0.038,0.038,0.039,0.038,0.038,0.038,0.038,0.039,0.037,0.039,0.037,0.039,0.039,0.039,0.038,0.04,0.04,0.04,0.04,0.04,0.041,0.041,0.041,0.042,0.043,0.044,0.044,0.044,0.046,0.045,0.045,0.044,0.044,0.044,0.043,0.04,0.041,0.045,0.04,0.04,0.042,0.042,0.041,0.042,0.042,0.041,0.04,0.04,0.04,0.04,0.041,0.041,0.041,0.041,0.041,0.041,0.041,0.041,0.042,0.042,0.042,0.043,0.044,0.044,0.043,0.044,0.045,0.045,0.046,0.045,0.044,0.045,0.047,0.048,0.049,0.049,0.05,0.051,0.053,0.052,0.053,0.055,0.056,0.058,0.059,0.062,0.062,0.064,0.065,0.066,0.068,0.07,0.071,0.074,0.075,0.076,0.079,0.082,0.085,0.086,0.089,0.091,0.095,0.098,0.101,0.103,0.108,0.112,0.116,0.121,0.124,0.126,0.132,0.133,0.136,0.137,0.138,0.138,0.14,0.141,0.141,0.141,0.14,0.142,0.143,0.142,0.142,0.142,0.142,0.142,0.143,0.142,0.142,0.141,0.141,0.141,0.14,0.141,0.14,0.141,0.14,0.139,0.136,0.136,0.135,0.132,0.132,0.131,0.131,0.13,0.13,0.128,0.128,0.127]

            for x in list_of_scans:
                list_of_data = np.array(find_max_df[x])
                subtracted_data = np.subtract(list_of_data, list_of_blank_data)
                
                find_max_df[x] = subtracted_data
                del list_of_data, subtracted_data

            find_max_df=find_max_df.T

            lambda_max_wavelengths = []
            lambda_max_abs = []
            for x in list_of_scans:
                lambda_max_wavelengths = lambda_max_wavelengths + [find_max_df.loc[(x),:].idxmax()]
                lambda_max_abs = lambda_max_abs + [find_max_df.loc[(x),:].max()]


            return lambda_max_wavelengths

        lambda_maxes = find_max(scan_data)
        print(lambda_maxes)
        
        Y_initial = normalize(np.array(lambda_maxes),300,900).reshape(-1,1)
        self.batch_num += 1

        self._update_experiment_data(wellnames, recipes, lambda_maxes)
        # Optimizer update method not yet available so add initial design to records.
        model.experiment_data['X'] = list(recipes)
        model.experiment_data['Y'] = list(Y_initial)
        # Initialize the optimizer with initial experimental data
        model.initialize_optimizer(recipes, Y_initial)

        self.batch_num += 1

        Y_best = np.min(abs(denormalize(model.optimizer.Y,300,900) - model.target_value))

        print(f"Model X: {model.optimizer.X}")
        print(f"Model Y: {model.optimizer.Y}")

        #enter iterative while loop now that we have data
        while not model.quit:

            # get new recipes
            X_new = model.suggest()
            recipes =  self.duplicate_list_elements(X_new, self.num_duplicates)
            print(f'<<controller>> executing batch {self.batch_num}, Suggested Location: {X_new}')
            # do the experiments
            #generate new wellnames for next batch
            wellnames = [self._generate_wellname() for i in range(recipes.shape[0])]
            # plan and execute a reaction with duplicate reactions.
            self._create_samples(wellnames, recipes)

            #pull in the scan data
            filenames = self.rxn_df[
                    (self.rxn_df['op'] == 'scan') |
                    (self.rxn_df['op'] == 'scan_until_complete')
                    ].reset_index()
            last_filename = filenames.loc[filenames['index'].idxmax(),'scan_filename']
            scan_data = self._get_sample_data(wellnames, last_filename) 
            #TODO convert scan data into list of single values. Add method of optimizer? Or maybe just incorporate into calc_obj method
            # update model with data

            lambda_maxes = find_max(scan_data)
            print(lambda_maxes)
            Y_new = normalize(np.array(lambda_maxes),300,900).reshape(-1,1)
            model.update_experiment_data(recipes, Y_new)

            # print results
            # To get the best observed X values (parameters)
            X_best = model.optimizer.X[np.argmin(model.optimizer.Y)]
            # To get the best observed Y value (function value)
            Y_best = np.min(model.optimizer.Y) # if zero we need to quit?
            if Y_best < 5:
                print("Exit due to meeting target")
            print(f"Best recipe: {X_best}, Best lambda max: {Y_best}")

            self.batch_num += 1
            self._update_experiment_data(wellnames, recipes, lambda_maxes)     
            
        self.experiment_data.to_csv("experiment_data.csv", index=True)
        print("Success!!!")

        print(self.well_count) # for heatmap debugging

        # Add plotting before closing connections
        try:
                    #plate(file, num_wells, target)
            data = plate(os.path.join(self.data_path, f"{self.experiment_name}full_df.csv"),  
                        self.well_count, 
                        550)  # target is hardcoded for now
            plt.figure()
            heat_map(data)
            plt.savefig(os.path.join(self.data_path, f"{self.experiment_name}_heatmap.png"))
            plt.close()
        except Exception as e:
            print(f"<<controller>> Failed to generate heatmap: {str(e)}")
        
        self.close_connection()
        self.pr.shutdown()
            
        return
    
    def duplicate_list_elements(self, list1, factor):
        """Duplicates the elements of a list by a factor.

        Args:
            list1: The list to duplicate the elements of.
            factor: The factor by which to duplicate the elements.

        Returns:
            A new list with the elements of list1 duplicated by factor.
        """

        new_list = []
        for element in list1:
            for i in range(factor):
                new_list.append(element)
        return np.array(new_list)

    
    def _get_sample_data(self,wellnames, filename):
        '''
        loads the spectra for the wells specified from the scan file specified  
        params:  
            list<str> wellnames: the names of the wells to be scanned  
            str filename: the name of the file that holds the scans  
        returns:  
            df: n_wells, by size of spectra, the scan data.  
        ''' 
        self._update_cached_locs(wellnames)
        pr_dict = {self._cached_reader_locs[wellname].loc: wellname for wellname in wellnames}
        unordered_data, metadata = self.pr.load_reader_data(filename, pr_dict)
        #reorder according to order of wellnames
        return unordered_data[wellnames]

    def _create_samples(self, wellnames, recipes):
        '''
        creates the desired reactions on the platereader  
        params:  
            str wellnames: the ordered names of the wells you want to produce  
            np.array recipes: shape(n_predicted, n_reagents). Holds ratios of all the reagents
              you can use for each reaction you want to perform  
        returns:  
            list<str> wellnames: the names of the wells produced ordered in accordance to the
              order of recipes
        Postconditions:
        '''
     

        self.portal.send_pack('init_containers', pd.DataFrame(
                {'labware':self.template_meta['labware'],
                'container':self.template_meta['cont'], 
                'max_vol':self.template_meta['tot_vol']}, index=wellnames).to_dict())
        #clean and update metadata from last reaction
        self._clean_meta(wellnames)
        successful_build = False #Flag True when a self.rxn_df using volumes has been generated
        #from the concentrations
        while not successful_build:
            try:


                #build new df
                self.rxn_df = self._build_rxn_df(wellnames, recipes)
                self._insert_tot_vol_transfer()

                print('trying to build df.')
                print(f'products list:{self._products}')
                print(f'rxn_df: {self.rxn_df}')

                if self.tot_vols: #has at least one element
                    if (self.rxn_df.loc[0,self._products] < 0).any():
                        raise NotImplementedError("A product overflowed it's container using the most concentrated solutions on the deck. Future iterations will ask Mark to add a more concentrated solution")
                successful_build = True
            except ConversionError as e:
                self._handle_conversion_err(e)
        self.execute_protocol_df()



    def _clean_meta(self, wellnames):
        '''
        In addition to replacing the rxn_df, there is some metadata associated with a reaction
        and it's reagents that must be cleaned after a reaction.  
        params:  
            str wellnames: the ordered names of the wells you want to produce  
        Preconditions:  
            self._products_contains products from last reaction  
            self._tot_vols has products from last reaction as keys  
        Postconditions:  
            self._products has been reset to be wellnames  
            self.tot_vols has been reset to have only the wellnames as keys and template vol
              as the value  
        '''
        #remove old products
        for product in self._products:
            del self.tot_vols[product]
        #add new keys
        self.tot_vols.update({wellname:self.template_meta['tot_vol'] for wellname in wellnames})
        #update products
        self._products = wellnames
            

    def _generate_wellname(self):
        '''
        returns:  
            str: a unique name for a new well
        '''
        wellname = "autowell{}C1.0".format(self.well_count)
        self.well_count += 1
        return wellname

    def _get_rxn_max_vol(self, name, products):
        '''
        This is used right now because it's best I've got. Ideally, you could drop the part 
        of init that constructs product_df
        '''
        return self.tot_vols['Template']

    def _build_rxn_df(self,wellnames,recipes):
        '''
        used to construct a rxn_df for this batch of reactions
        Postconditions:  
            self.tot_vols has been updated to 
        '''

        print(f'recipes {recipes}')
        rxn_df = self.rxn_df_template.copy() #starting point. still neeeds products
        recipe_df = pd.DataFrame(recipes, index=wellnames, columns=self.reagent_order)
        n_wellnames = np.array(wellnames)
        #n_wellnames_reshaped = n_wellnames.reshape(2,2)
        #n_reagent_order = self.reagent_order.reshape(2,2)
        #n_recipes = recipes.reshape(2,2)

        print("Shapes:")
        print("wellnames:", n_wellnames.shape)
        print("recipes:", recipes.shape)
        print("self.reagent_order:", self.reagent_order.shape)

        print("Contents:")
        print("wellnames:", n_wellnames)
        print("recipes:", recipes)
        print("self.reagent_order:", self.reagent_order)
        
        

        self._update_cached_locs('all')
        def build_product_rows(row):
            '''
            params:  
                pd.Series row: a row of the template df  
            returns:  
                pd.Series: a row for the new df
            '''
            d = {}
            if row['op'] == 'transfer' and pd.isna(row['conc']):
                #is a transfer, so we want to lookup the volume of that reagent in recipe_df
                return recipe_df.loc[:, row['reagent']]
            else:
                #if not a tranfer, we want to keep whatever value was there
                return pd.Series(row['Template'], index=recipe_df.index)
        rxn_df = rxn_df.join(self.rxn_df_template.apply(build_product_rows, axis=1))
        rxn_df = self._convert_conc_to_vol(rxn_df, wellnames)
        rxn_df['scan_filename'] = rxn_df['scan_filename'].apply(lambda x: np.nan if pd.isna(x) 
                else "{}-{}".format(x, self.batch_num))
        rxn_df['plot_filename'] = rxn_df['plot_filename'].apply(lambda x: np.nan if pd.isna(x) 
                else "{}-{}".format(x, self.batch_num))
        rxn_df.drop(columns='Template',inplace=True) #no longer need template
        return rxn_df

    def run_all_checks(self): 
        found_errors = super().run_all_checks()
        found_errors = max(found_errors,self.check_conc())
        if found_errors == 0:
            print("<<controller>> All prechecks passed!")
            return
        elif found_errors == 1:
            if 'y'==input("<<controller>> Please check the above errors and if you would like to ignore them and continue enter 'y' else any key "):
                return
            else:
                raise Exception('Aborting base on user input')
        elif found_errors == 2:
            raise Exception('Critical Errors encountered during prechecks. Aborting')

    def _handle_conversion_err(self,e):
        '''
        This function will handle errors caught in the conversion process from molarity to
        volume reaction dataframe.  
        params:  
            ConversionError e: the conversion error raised  
        Postconditions:  
            If the error was pipetting infinitesimal volume, a dilution has been performed on
            the robot to dilute by 2X   
        Raises:  
            NotImplementedError: If you ran out of a reagent you probably need to have Mark
              restock (or you could dilute a stock maybe)  
        '''
        print('<<controller>> handling conversion error')
        if e.empty_reagents:
            #You ran out of something
            #query the user
            #It is also possible here that you might be able to perform dilution
            raise NotImplementedError("You ran out of a reagent. Future functionality will call Mark at this point")
        else:
            #you're trying to pipette an infinitesimal volume
            #send a single dilution column to the robot that will solve this problem
            #we have the data here to do something smart with how much we want to dilute, but
            #for now lets do something dumb like dilute 2x

            #generate necessary parameters
            containers = [key for key in self._cached_reader_locs.keys() 
                if re.fullmatch(e.reagent+'C\d*\.\d*', key)]
            stock_cont = max(containers, key=self._get_conc)
            min_conc = min(map(self._get_conc, containers))
            new_conc = min_conc / 2
            #execute dilution
            self._execute_single_dilution(new_conc, stock_cont)



    def check_rxn_df(self):
        '''
        Runs error checks on the reaction df to ensure that formating is correct. Illegal/Ill 
        Advised options are printed and if an error code is returned
        Will run through and check all rows, even if errors are found
        Preconditions:
            self.rxn_df is rxn_df template at this point  
        returns  
            int found_errors:  
                code:  
                0: OK.  
                1: Some Errors, but could run  
                2: Critical. Abort  
        '''
        #at this point self.rxn_df
        found_errors = super().check_rxn_df()
        reagent_ratios  = self.rxn_df.loc[(self.rxn_df['conc'].isna()) & (self.rxn_df['op'] == 'transfer'),\
                ['Template','reagent']].groupby('reagent').sum()['Template']
        has_invalid_ratio = reagent_ratios.apply(lambda x: not math.isclose(x, 1.0,
                abs_tol=1e-9)).any()
        if has_invalid_ratio:
            print('<<controller>> precheck error: invalid ratio of reagents (doesn\'t add to 1)')
            print('  ratios were {}'.format(reagent_ratios))
            found_errors = max(found_errors, 2)
        return found_errors

class ProtocolExecutor(Controller): 
    '''
    class to execute a protocol from the docs  
    ATTRIBUTES:  
    ATTRIBUTES:  
    class to execute a protocol from the docs  
    ATTRIBUTES:  
        df rxn_df: the reaction df. Not passed in, but created in init  
    INHERITED ATTRIBUTES:  
        armchair.Armchair portal, str rxn_sheet_name, str cache_path, bool use_cache,   
        str eve_files_path, str debug_path, str my_ip, str server_ip,  
        dict<str:object> robo_params, bool simulate, int buff_size  
    PRIVATE ATTRS:  
        pd.index _products: the product columns  
    INHERITED PRIVATE ATTRS:  
        dict<str:tuple<obj>> _cached_reader_locs  
    METHODS:  
        execute_protocol_df() void: used to execute a single row of the reaction df  
        run_all_checks() void: wrapper for pre rxn error checking to handle any found errors
          run automatically when you run your simulation  
        CHECKS: all print messages for errors and return error codes  
        check_rxn_df() int: checks for errors in input.  
        check_labware() int: checks for errors in labware/labware assignments.   
        check_products() int: checks for errors in the product placement.  
        check_reagents() int: checks for errors in the reagent_info tab.   
    INHERITED METHODS:  
        run_protocol(simulate, port) void, close_connection() void, init_robot(simulate), 
        translate_wellmap() void, run_simulation() bool  
    '''

    def __init__(self, rxn_sheet_name, my_ip, server_ip, buff_size=4, use_cache=False):
        '''
        Note that init does not initialize the portal. This must be done explicitly or by calling
        a run function that creates a portal. The portal is not passed to init because although
        the code must not use more than one portal at a time, the portal may change over the 
        lifetime of the class
        NOte that pr cannot be initialized until you know if you're simulating or not, so it
        is instantiated in run
        '''
        super().__init__(rxn_sheet_name, my_ip, server_ip, buff_size, use_cache)
        self.run_all_checks() 

    def run_simulation(self, no_pr=False):
        '''
        runs a full simulation of the protocol with
        Temporarilly overwrites the self.server_ip with loopback, but will restore it at
        end of function  
        Returns:  
            bool: True if all tests were passed  
        '''
        
        #cache some things before you overwrite them for the simulation
        stored_server_ip = self.server_ip
        stored_simulate = self.simulate
        stored_cached_reader_locs = self._cached_reader_locs
        self.server_ip = '127.0.0.1'
        self.simulate = True
        print('<<controller>> ENTERING SIMULATION')
        port = 50000
        #launch an eve server in background for simulation purposes
        b = threading.Barrier(2,timeout=20)
        eve_thread = threading.Thread(target=launch_eve_server, kwargs={'my_ip':'','barrier':b},name='eve_thread')
        eve_thread.start()

        #do create a connection
        b.wait()
        self._run(port, simulate=True, no_pr=no_pr)



        #collect the eve thread
        eve_thread.join()

        #restore changed vars
        self.server_ip = stored_server_ip
        self.simulate = stored_simulate
        self._cached_reader_locs = stored_cached_reader_locs
        print('<<controller>> EXITING SIMULATION')
        # delete later.
        return True 
    
    def run_protocol(self, simulate=False, no_pr=False, port=50000):
        '''
        The real deal. Input a server addr and port if you choose and protocol will be run  
        params:  
            bool simulate: (this should never be used in normal operation. It is for debugging
              on the robot)  
            bool no_pr: This should be false normally, but can be set to true to deliberately
              not use the platereader even if on the laptop  
        NOTE: the simulate here is a little different than running run_simulation(). This simulate
          is sent to the robot to tell it to simulate the reaction, but that it all. The other
          simulate changes some things about how code is run from the controller
        '''
        print('<<controller>> RUNNING PROTOCOL')
        self._run(port, simulate=simulate, no_pr=no_pr)
        print('<<controller>> EXITING PROTOCOL')
        
    @error_exit
    def _run(self, port, simulate, no_pr):
        '''
        params:  
            int port: the port number to connect on  
            bool simulate: (this should never be used in normal operation. It is for debugging
              on the robot)  
            bool no_pr: This should be false normally, but can be set to true to deliberately
              not use the platereader even if on the laptop  
        Returns:  
            bool: True if all tests were passed  
        '''
        self.create_connection(simulate, no_pr, port)
        successful_build = False
        while not successful_build:
            try:
                self._update_cached_locs('all')
                #build new df
                self.rxn_df = self._convert_conc_to_vol(self.rxn_df,self._products)
                self._insert_tot_vol_transfer()
                if self.tot_vols: #has at least one element
                    if (self.rxn_df.loc[0,self._products] < 0).any():
                        raise NotImplementedError("A product overflowed it's container using the most concentrated solutions on the deck. Future iterations will ask Mark to add a more concentrated solution")
                successful_build = True
            except ConversionError as e:
                self._handle_conversion_err(e)        
        self.execute_protocol_df()

        try:
            data = plate(os.path.join(self.data_path, f"{self.experiment_name}full_df.csv"),
                        len(self._products),  # Use number of products instead of well_count
                        550)  # target is hardcoded for now. needs to be configurable
            plt.figure()
            heat_map(data)
            plt.savefig(os.path.join(self.data_path, f"{self.experiment_name}_heatmap.png"))
            plt.close()
        except Exception as e:
            print(f"<<controller>> Failed to generate heatmap: {str(e)}")
        
        self.close_connection()
        self.pr.shutdown()

    def init_robot(self,simulate):
        '''
        calls super init robot, and then sends an init_containers command to initialize all the
        prodcuts  
        params:  
            bool simulate: whether the robot should run a simulation  
        '''
        super().init_robot(simulate)
        #send robot data to initialize empty product containers. Because we know things like total
        #vol and desired labware, this makes sense for a planned experiment
        self.portal.send_pack('init_containers', self.robo_params['product_df'].to_dict())
    
    def _rename_products(self, rxn_df):
        '''
        renames dilutions acording to the reagent that created them
        and renames rxns to have a concentration  
        Preconditions:  
            dilution cols are named dilution_1/2 etc  
            callback is the last column in the dataframe  
            rxn_df is not expected to be initialized yet. This is a helper for the initialization  
        params:  
            df rxn_df: the dataframe with all the reactions  
        Postconditions:  
            the df has had it's dilution columns renamed to the chemical used to produce it + C<conc>  
            rxn columns have C1 appended to them  
        '''
        dilution_cols = [col for col in rxn_df.columns if 'dilution_placeholder' in col]
        #get the rxn col names
        rxn_cols = rxn_df.loc[:, 'reagent':'chemical_name'].drop(columns=['reagent','chemical_name']).columns
        rename_key = {}
        for col in rxn_cols:
            if 'dilution_placeholder' in col:
                row = rxn_df.loc[rxn_df['op'] == 'dilution'].loc[~rxn_df[col].isna()].squeeze()
                reagent_name = row['chemical_name']
                assert (isinstance(reagent_name, str)), "dilution placeholder was used twice"
                name = reagent_name[:reagent_name.rfind('C')+1]+str(row['dilution_conc'])
                rename_key[col] = name
            else:
                rename_key[col] = "{}C1.0".format(col).replace(' ','_')
        rxn_df.rename(rename_key, axis=1, inplace=True)

    def _get_rxn_max_vol(self, name, products):
        '''
        Preconditions:  
            volume in a container can change only during a 'transfer' or 'dilution'. Easy to add more
            by changing the vol_change_rows
            self.rxn_df is initialized  
        params:  
            str name: the column name to be searched  
            list<str> products: the column names of all reagents (we could look this up in rxn_df, but
              convenient to pass it in)  
        returns:  
            float: the maximum volume that this container will ever hold at one time, not taking into 
              account aspirations for dilutions  
        '''
        if name in self.tot_vols:
            return self.tot_vols[name]
        else:
            vol_change_rows = self.rxn_df.loc[self.rxn_df['op'].apply(lambda x: x in ['transfer','dilution'])]
            aspirations = vol_change_rows['chemical_name'] == name
            max_vol = 0
            current_vol = 0
            for i, is_aspiration in aspirations.iteritems():
                if is_aspiration and self.rxn_df.loc[i,'op'] == 'transfer':
                    #This is a row where we're transfering from this well
                    current_vol -= self.rxn_df.loc[i, products].sum()
                elif is_aspiration and self.rxn_df.loc[i, 'op'] == 'dilution':
                    current_vol -= self._get_dilution_transfer_vols(self.rxn_df.loc[i])[1]
                else:
                    current_vol += self.rxn_df.loc[i,name]
                    max_vol = max(max_vol, current_vol)
            return max_vol

    
    #TESTING
    #PRE Simulation
    def run_all_checks(self):
        found_errors = super().run_all_checks()
        found_errors = max(found_errors, self.check_products())
        if found_errors == 0:
            print("<<controller>> All prechecks passed!")
            return
        elif found_errors == 1:
            if 'y'==input("<<controller>> Please check the above errors and if you would like to ignore them and continue enter 'y' else any key"):
                return
            else:
                raise Exception('Aborting base on user input')
        elif found_errors == 2:
            raise Exception('Critical Errors encountered during prechecks. Aborting')

                
    def check_products(self):
        '''
        checks to ensure that the products were correctly initialized  
        returns  
            int found_errors:  
                code:  
                0: OK.  
                1: Some Errors, but could run  
                2: Critical. Abort  
        '''
        found_errors = 0
        for i, r in self.robo_params['product_df'].loc[\
                ~self.robo_params['product_df']['labware'].astype(bool) & \
                ~self.robo_params['product_df']['container'].astype(bool)].iterrows():
            found_errors = max(found_errors,1)
            print('<<controller>> {} has no specified labware or container. It could end up in anything that has enough volume to contain it. Are you sure that\'s what you want? '.format(i))
        return found_errors

    #POST Simulation

class AbstractPlateReader(ABC):
    '''
    This class is responsible for executing platereader commands. When instantiated, this
    class changes the config file  
    METHODS:  
        edit_layout(protocol_name, layout) void: changes the layout for a protocol  
        run_protocol(protocol_name, filename, data_path, layout) void: executes a protocol  
        shutdown() void: kills the platereader and restores default config  
        shake() void: shakes the platereader  
        exec_macro(macro, *args) void: low level method to send a command to platereader with
          arguments  
        load_reader_data(str filename, dict<str:str> loc_to_name, str path) tuple<df, dict>:
          reads the platereader data into a df and returns a dictionary of interesting 
          metadata.  
    ATTRIBUTES:
        str data_path: a linux path to where all the data is 
    '''
    SPECTRO_ROOT_PATH = "/mnt/c/Program Files/SPECTROstar Nano V5.50/"
    PROTOCOL_PATH = r"C:\Program Files\SPECTROstar Nano V5.50\User\Definit"
    SPECTRO_DATA_PATH = "/mnt/c/Users/science_356_lab/Robot_Files/Plate Reader Data"

    def __init__(self, data_path):
        self.data_path = data_path
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        #self.data = ScanDataFrame(data_path, header_data, eve_files_path)
        
    def exec_macro(self, macro, *args):
        '''
        sends a macro command to the platereader and blocks waiting for response. If response
        not ok, it'll crash and burn  
        params:  
            str macro: should be a macro from the documentation  
            *args: associated arguments of the macto  
        Postconditions:  
            The command has been sent to the PlateReader, if the return status was not 0 (good)  
            an error will be thrown  
        '''
        pass

    def shake(self, shake_time):
        '''
        executes a shake
        '''
        pass

    def edit_layout(self, protocol_name, layout):
        '''
        params:  
            str protocol_name: the name of the protocol that will be edited  
            list<str> wells: the wells that you want to be used for the protocol ordered.
              (first will be X1, second X2 etc. If layout is all, all wells will be made X  
        Postcondtions:  
            The protocol has had it's layout updated to include only the wells specified  
        '''
        pass

    def run_protocol(self, protocol_name, filename, layout=None):
        r'''
        In the abstract version, a dummy file will be written.  
        params:  
            str protocol_name: the name of the protocol that will be edited  
            list<str> layout: the wells that you want to be used for the protocol ordered.
              (first will be X1, second X2 etc. If not specified will not alter layout)  
        '''
        
        filename = '{}.csv'.format(filename)
        filepath = os.path.join(self.data_path,filename)
        if os.path.exists(filepath):
            os.system('rm {}'.format(filepath))

        data = pd.DataFrame(.42*np.random.rand(701,len(layout)), columns=layout)
        

        with open(filepath, 'a+', encoding='latin1') as file:
            file.write('No. of Cycles: 1\nT[°C]: \n23.5\n')
            for name, col in data.iteritems():
                write_str = name[0] + name[1:].zfill(2) + ':, '
                write_str += ', '.join([str(i) for i in col])
                write_str += '\n'
                file.write(write_str)
        


    def _rename_scan(self,new_scan_file,old_scan_file):
        """
        Helper function for scan until complete,
        renames the filename back to the original to help deal with
        scan until complete rows
        """

        shutil.move(os.path.join(self.data_path, "{}.csv".format(new_scan_file)),
        os.path.join(self.data_path, "{}.csv".format(old_scan_file)))
    
    def shutdown(self):
        '''
        closes connection. Use this if you're done with this object at cleanup stage
        '''
        pass

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
    
    def merge_scans(self, filenames, dst):
        '''
        merges the specified files together into a single scan file.  
        params:  
            list<str> filenames: a list of all the files you want to merge without extensions.  
            str dst: the filename of the output file without extension.  
        Postconditions:  
            A new file has been created with the data from all the files.  
            NOTE metadata may change across scans. the metadata of only the first scan to
              be merged shall be preserved.
        Preconditions:  
            n_cycles must be the same for each scan file.  
        '''
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

class DummyReader(AbstractPlateReader):
    '''
    Inherits from AbstractPlateReader, so it has all of it's methods, but doesn't actually do
    anything. useful for some simulations
    '''
    pass


class PlateReader(AbstractPlateReader):
    '''
    This class handles all platereader interactions. Inherits from the interface
    '''

    def __init__(self, data_path, header_data, eve_files_path, simulate=False):
        super().__init__(data_path)
        self.experiment_name = {row[0]:row[1] for row in header_data[1:]}['data_dir']
        self.simulate=simulate
        self._set_config_attr('Configuration','SimulationMode', str(int(simulate)))
        self._set_config_attr('ControlApp','AsDDEserver', 'True')
        self.exec_macro("dummy")
        self.exec_macro("init")
        self.exec_macro('PlateOut')
        self.data = ScanDataFrame(data_path, self.experiment_name, eve_files_path)
        
    def exec_macro(self, macro, *args):
        '''
        sends a macro command to the platereader and blocks waiting for response. If response
        not ok, it'll crash and burn  
        params:  
            str macro: should be a macro from the documentation  
            *args: associated arguments of the macto  
        Postconditions:  
            The command has been sent to the PlateReader, if the return status was not 0 (good)
            an error will be thrown  
        '''
        exec_str = "'{}Cln/DDEClient.exe' {}".format(self.SPECTRO_ROOT_PATH, macro)
        #add arguments
        for arg in args:
            exec_str += " '{}'".format(arg)
        print('<<Reader>> executing: {}'.format(exec_str))
        exit_code = os.system(exec_str)
        try:
            assert (exit_code == 0)
        except:
            if exit_code < 1000:
                raise Exception("PlateReader rejected command Error")
            elif exit_code == 1000:
                raise Exception("PlateReader Nonexistent Protocol Name Error")
            elif exit_code == 2000:
                raise Exception("PlateReader Communication Error")
            else:
                raise Exception("PlateReader Error. Exited with code {}".format(exit_code))

    def shake(self, shake_time):
        '''
        executes a shake
        '''
        macro = "Shake"
        shake_type = 2
        shake_freq = 300
        self.exec_macro(macro, shake_type, shake_freq, shake_time)

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
        if self.simulate:
            return super().load_reader_data(filename, loc_to_name) #return dummy data
        else:
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


    def edit_layout(self, protocol_name, layout):
        '''
        This protocol creates a temporary file, .temp_ot2_bmg_layout.lb
        in the SPECTROstar root. It is also possible (theoretically) to 
        send a literal 'edit_layout' command, but this fails for long
        strings. (not sure why, maybe windows limited sized strings?
        but the file works). It removes the file after importing  
        params:  
            str protocol_name: the name of the protocol that will be edited  
            list<str> wells: the wells that you want to be used for the protocol ordered.
              (first will be X1, second X2 etc. If layout is all, all wells will be made X  
        Postcondtions:  
            The protocol has had it's layout updated to include only the wells specified  
        '''
        if layout == 'all':
            #get a list of all the wellanmes
            layout = [a+str(i) for a in list('ABCDEFGH') for i in range(1,13,1)]
        well_entries = []
        for i, well in enumerate(layout):
            well_entries.append("{}=X{}".format(well, i+1))
        filepath_lin = os.path.join(self.SPECTRO_ROOT_PATH,'.temp_ot2_bmg_layout.lb')
        filepath_win = os.path.join(wslpath(self.SPECTRO_ROOT_PATH,'w'),'.temp_ot2_bmg_layout.lb')
        with open(filepath_lin, 'w+') as layout:
            layout.write('EmptyLayout')
            for entry in well_entries:
                layout.write("\n{}".format(entry))
        self.exec_macro('ImportLayout', protocol_name, self.PROTOCOL_PATH, filepath_win)
        os.remove(filepath_lin)

    def run_protocol(self, protocol_name, filename,layout=None):
        r'''
        params:  
            str protocol_name: the name of the protocol that will be edited  
            list<str> layout: the wells that you want to be used for the protocol ordered.
              (first will be X1, second X2 etc. If not specified will not alter layout)  
        '''
        if layout:
            self.edit_layout(protocol_name, layout)
        macro = 'run'
        #three '' are plate ids to pad. data_path specified once for ascii and once for other
        self.exec_macro(macro, protocol_name, self.PROTOCOL_PATH, wslpath(self.SPECTRO_DATA_PATH,'w'), '', '', '', '', filename)
        #Note, here I am clearly passing in a save path for the file, but BMG tends to ignore
        #that, so we move it from the default landing zone to where I actually want it
        if self.simulate:
            super().run_protocol(protocol_name, filename, layout)
        else:
            shutil.copyfile(os.path.join(self.SPECTRO_DATA_PATH, "{}.csv".format(filename)), 
                    os.path.join(self.data_path, "{}.csv".format(filename)))
        
       
            self.data.AddToDF("{}.csv".format(filename))

            self.data.df.to_csv(os.path.join(self.data_path, "{}{}.csv".format(self.experiment_name, 'full_df')))
            
            self.data.AddReagentInfo()
        


    def _set_config_attr(self, header, attr, val):
        '''
        opens the Spectrostar nano config file and replaces the value of attr under header
        with val
        There are better ways to build this function, but it's not something you'll use much
        so I'm leaving it here  
        params:  
            str header: the header in the config file [header]  
            str attr: the attribute you want to change  
            obj val: the value to set the attribute to  
        Postconditions:  
            The SPECTROstar Nano.ini has had the attribute under the header overwritten with val
            or appended to end if it wasn't found   
        '''
        with open(os.path.join(self.SPECTRO_ROOT_PATH, r'SPECTROstar Nano.ini'), 'r') as config:
            file_str = config.readlines()
            write_str = ''
            header_exists = False
            i = 0
            while i < len(file_str): #iterating through lines
                line = file_str[i]
                write_str += line
                if line[1:-2] == header:
                    header_exists = True#you found the appropriate header
                    i += 1
                    found_attr = False
                    line = file_str[i] #do
                    while '[' != line[0] and i < len(file_str): #not a header and not EOF
                        if line[:line.find('=')] == attr:
                            found_attr = True
                            write_str += '{}={}\n'.format(attr, val)
                        else:
                            write_str += line
                        i += 1
                        if i < len(file_str):
                            line = file_str[i]
                    if not found_attr:
                        write_str += '{}={}\n'.format(attr, val)
                else:
                    i += 1
            if not header_exists:
                write_str += '[{}]\n'.format(header)
                write_str += '{}={}\n'.format(attr, val)

        with open(os.path.join(self.SPECTRO_ROOT_PATH, r'SPECTROstar Nano.ini'), 'w+') as config:
            config.write(write_str)

    def shutdown(self):
        '''
        closes connection. Use this if you're done with this object at cleanup stage
        '''
        #self.exec_macro('PlateIn')
        self.exec_macro('Terminate')
        self._set_config_attr('ControlApp','AsDDEserver','False')
        self._set_config_attr('ControlApp', 'DisablePlateCmds','False')
        self._set_config_attr('Configuration','SimulationMode', str(0))

  
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
    
    def __init__(self, data_path, experiment_name, eve_files_path):
        self.df = pd.DataFrame()
        self.data_path = data_path
        self.eve_files_path = eve_files_path
        self.experiment_name = experiment_name
        self.isFirst = True
        
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        
    def AddToDF(self, file_name):
        temp_file = os.path.join(self.data_path,file_name)
    
        #Extracts and stores data/time metadata for the time column

        df_read_data_1 = pd.read_csv(temp_file,nrows = 35,skiprows = [7], header=None,na_values=["       -"],encoding = 'latin1')   
        am_pm = df_read_data_1.iloc[1][0].split(" ")[5]
        
        
        if am_pm == "PM":
            temphour = int(df_read_data_1.iloc[1][0].split(" ")[4].split(":")[0])
            hour = temphour if temphour == 12  else temphour +12
        elif am_pm == "AM":
            hour = int(df_read_data_1.iloc[1][0].split(" ")[4].split(":")[0])
        
        
        
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
        
        data_df.insert(0,"Scan ID",file_name.replace(".csv", ""))
        data_df = data_df.set_index(['Scan ID','Well'])
        
        df1 = pd.read_csv(os.path.join(self.eve_files_path, 'translated_wellmap.tsv'), sep='\t')
        

        col_list  = data_df.index.get_level_values('Well').tolist()

        well_names = []

        for well_with_zeros_in_name in col_list:
            x = ''
            
            if well_with_zeros_in_name[1] == '0':
                #print('yes')
                #Turn A01 into A1 if A9 or less
                well = str(well_with_zeros_in_name)
                well = "{}{}".format(well[0], int(well[2:]))
            
            else:
                #Leave A10 and greater alone
                well = str(well_with_zeros_in_name)
           
            y = df1.loc[df1['loc'] == str(well), 'chem_name'].values[:]
            for i in y:
                if str(self.experiment_name) in i:
                    x=i
                if ('control' in i.lower()):
                        x = 'control'
                if ('blank' in i.lower()):
                        x = 'blank'
               
            well_names.append(x)

        data_df.insert(0, 'Well Name', well_names)

    
        
        
        if self.isFirst:
            self.df = data_df
            #full_df = pd.concat([wavvelength_df,data_df])
            self.isFirst = False
            
        else:
            full_df = data_df
        
            self.df = pd.concat([full_df,self.df])
        
        self.df = self.df.sort_values(['Time', 'Well'])
    

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

        df = pd.read_csv(os.path.join(self.data_path, reaction+'full_df.csv'))

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
                
                
                indices.append(container)
                indices = list(set(indices))

                vols.extend(volume)
                vols = list(set(vols))


        for index in vols:
            chemical = well_hist_df["chemical"].iloc[index]
            
            chem_c_index = chemical.rfind('C')
            head = chemical[:chem_c_index]
            sep = chemical[chem_c_index]
            tail = chemical[chem_c_index+1:]
            
            chemical =  head
            chems.append(chemical)
            
            concentration = tail
            cons.append(concentration)
            timestamp = well_hist_df["timestamp"].iloc[index]
            times.append(timestamp)
            volume = well_hist_df["vol"].iloc[index]
            volumes.append(volume)
            
            container = well_hist_df["container"].iloc[index]
            
            cont_c_index = container.rfind('C')
            head = container[:cont_c_index]
            sep = container[cont_c_index]
            tail = container[cont_c_index+1:]
            
            container = head +sep + tail
            containers.append(container)
                  
        
        chems_unique = list(set(chems))
    
        chem_info = {}
        for chem in chems_unique:
            chem_info[chem] = [0]
            
            

                
                
            
            
        """    
            #concentration = tail
        pddict = {'time':times, 'vol':volumes, 'cont':containers, 'chem':chems, 'conc': cons} 
        yay = pd.DataFrame(pddict)


        df2 = yay.sort_values(by = ['cont', 'time'], ascending = [True, True])


        n = len(pd.unique(df2['cont']))




        base = df2.loc[(df2['cont'].str.contains('blank'))|(df2['cont'].str.contains('control'))]
       
        for chem in chems_unique:
            base[chem] = 0
        for i in indices: 
           
            if 'blank' not in i and 'control' not in i:
                temp = df2.loc[df2['cont']==i]
                temp.sort_values(by='time')
                
                
                
                volumes = temp.vol.values.tolist()
              
                sum_volumes = [sum(volumes[0:i[0]+1]) for i in enumerate(volumes)]
            
                concentrations = temp.conc.tolist()
               
                count = 0
                
               
                for chem in chems_unique:
                    chem_info[chem] = [0]
                
                
                for chem in temp.chem.tolist():
                  
                    if 'water' in chem.lower():
                        for i in chems_unique:
                            chem_info[i].append(chem_info[i][count]*(float(sum_volumes[count-1]/float(sum_volumes[count]))))
                    else:
                            
                        chem_info[chem].append(float(volumes[count])*float(concentrations[count])/float(sum_volumes[count]))
                        for x in chems_unique:
                            if x != chem:
                                
                                chem_info[x].append(chem_info[x][count]*(float(sum_volumes[count-1]/float(sum_volumes[count]))))
                   
                 
                    count += 1
            
                
                for i in chem_info:
                   
                    temp[i] = chem_info[i][1:]
             
                base = pd.concat([base, temp])
            
        full= base.sort_values(by = ['cont', 'time'], ascending = [True, True])



        weird = []
        last_reagent = []


        scans = list(set(df['Scan ID'].tolist()))
        reactions = list(set((df['Well Name'].tolist())))

        #more lists
        another_dict = {}
        for x in chems_unique:
            another_dict[x] = []
            

        df.set_index('Scan ID',inplace = True)
        scan_list=df.index.get_level_values('Scan ID').unique()
        df.reset_index(inplace = True)
      
        for scan in scan_list:
     
            
            
            
            for time in df.loc[df['Scan ID']==scan, 'Time']:
                time = time


            for react in df.loc[df['Scan ID']==scan, 'Well Name']:
                transfers_before_scans = []
                react = str(react)
                transfer_times = full.loc[full['cont'].str.contains(react),'time'].tolist()             
                print("transfer_times list = ", transfer_times)
                for transfer_time in transfer_times:
                    print("transfer_time = ", transfer_time)
                    print("time = ", time)
                    if transfer_time <= time:
                        print("yes, transfer_time<=time")
                        transfers_before_scans.append(transfer_time)
                latest_transfer_time = max(transfers_before_scans)
                
                
                
                
                
                for i in chems_unique:
                    current_chem_conc_list = full[(full['cont'] == react) & (full['time'] == latest_transfer_time)][i].tolist()
                    if len(current_chem_conc_list)==0:
                        another_dict[i].append(0)
                    else:
                        another_dict[i].append(current_chem_conc_list[0])
                
                   
                weird.append(latest_transfer_time)
                
               
                
               
                
                if "water" in str(full[full['time']==latest_transfer_time]['chem'].item()).lower():
                    stillWater = True
                    while stillWater:
                        if len(transfers_before_scans)>1:
                            transfers_before_scans.remove(transfers_before_scans.index(latest_transfer_time))
                            latest_transfer_time = max(transfers_before_scans)
                        elif len(transfers_before_scans) ==1:
                            latest_transfer_time = transfers_before_scans[0]
                            if "water" in str(full[full['time']==latest_transfer_time]['chem'].item()).lower():
                                stillWater = False
                                last_reagent_added = full[full['time']==latest_transfer_time]['chem'].item()
                        if "water" not in str(full[full['time']==latest_transfer_time]['chem'].item()).lower():
                            stillWater = False
                            last_reagent_added = full[full['time']==latest_transfer_time]['chem'].item()
                            
                    
                    
                else:
                    last_reagent_added = full[full['time']==latest_transfer_time]['chem'].item()
                    
                    
                    
                
                last_reagent.append(last_reagent_added)
                

        df['time of last reagent added'] = weird
        df['last reagent added'] = last_reagent

        for x in another_dict:
            
            df[x] = another_dict[x]
     
        df.reset_index(inplace=True)
        left_list = ['Scan ID', 'Well', 'Well Name']+[i for i in another_dict if 'water' not in i.lower()]+['time of last reagent added','last reagent added', 'Temp', 'Time']
        right_list = [c for c in df if c not in ['Scan ID', 'Well', 'Well Name']+[i for i in another_dict if 'water' not in i.lower()]+['time of last reagent added','last reagent added', 'Temp', 'Time']]
        df = df[left_list + right_list]
        df= df.rename(columns=str.lower)
        if 'index' in df.columns:
            df.drop('index', axis=1,inplace=True)
        if 'water' in df.columns:
            df.drop('water', axis=1,inplace=True)
        
#         wellnames = df['well name'].tolist()
#         wellnamenumbers = []
#         for i in wellnames:
#             print(i,'you')
#             a = i.lower()
#             if 'blank' not in i and 'control' not in i:
                
#                 l_index = re.search('rxn', a).end()
#                 r_index = a.rfind('c')
#                 wellnamenumber = a[l_index:r_index]
               
#                 wellnamenumbers.append(wellnamenumber)
#             else:
#                 wellnamenumbers.append(0)
#             #print(l)
        
#         df['wellnameorder'] = wellnamenumbers
        
#         # Split the 'wellnameorder' into two columns: 'num' and 'alpha'
#         df['num'] = df['col'].str.extract('(\d+)').astype(int)
#         df['alpha'] = df['col'].str.extract('([a-zA-Z]+)')
        
#          # Sort by 'time', then 'num' and 'alpha'
        df.sort_values(by=['time'], inplace = True)
        
        # Drop the 'num' and 'alpha' and 'wellnameorder' columns, they are no longer needed
#         df = df.drop(columns=['num', 'alpha'])
#         df.drop('wellnameorder', axis=1,inplace=True)
        
        df.to_csv(os.path.join(self.data_path, reaction + '_full.csv'))
"""
        
    
class Plotter():
    '''
    This class creates and saves plots 
    
    ATTRIBUTES:  
        df df: Not passed in but created in init. Pandas Dataframe to be used 
            to store all scans from the run.
        str data_path: pathname for local platereader data.
    
    METHODS:  
        add_to_df() void: formats data from a scan and adds to the data frame.
    '''
    
    def __init__(self, filename):
        self.filename = filename
    
        
        
            

        

    
    

if __name__ == '__main__':
    SERVERADDR = "169.254.44.249"
    main(SERVERADDR)
