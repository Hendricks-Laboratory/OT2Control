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
from datetime import datetime
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

from bidict import bidict
import gspread
from df2gspread import df2gspread as d2g
from df2gspread import gspread2df as g2d
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import opentrons.execute
from opentrons import protocol_api, simulate, types
from boltons.socketutils import BufferedSocket
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from Armchair.armchair import Armchair
from ot2_robot import launch_eve_server
from df_utils import make_unique, df_popout, wslpath, error_exit
from ml_models import DummyMLModel


def init_parser():
    parser = argparse.ArgumentParser()
    mode_help_str = 'mode=auto runs in ml, mode=protocol or not supplied runs protocol'
    parser.add_argument('-m','--mode',help=mode_help_str,default='protocol')
    parser.add_argument('-n','--name',help='the name of the google sheet')
    parser.add_argument('-c','--cache',help='flag. if supplied, uses cache',action='store_true')
    parser.add_argument('-s','--simulate',help='runs robot and pr in simulation mode',action='store_true')
    return parser

def main(serveraddr):
    '''
    prompts for input and then calls appropriate launcher
    '''
    parser = init_parser()
    args = parser.parse_args()
    if args.mode == 'protocol':
        print('launching in protocol mode')
        launch_protocol_exec(serveraddr,args.name,args.cache,args.simulate)
    elif args.mode == 'auto':
        print('launching in auto mode')
        launch_auto(serveraddr,args.name,args.cache,args.simulate)
    else:
        print("invalid argument to mode, '{}'".format(args.mode))
        parser.print_help()

def launch_protocol_exec(serveraddr, rxn_sheet_name=None, use_cache=False, simulate=False):
    '''
    main function to launch a controller and execute a protocol
    '''
    #instantiate a controller
    if not rxn_sheet_name:
        rxn_sheet_name = input('<<controller>> please input the sheet name ')
    if not use_cache:
        #using the cache bypasses google docs communication and uses the last rxn you loaded
        use_cache = 'y' == input('<<controller>> would you like to use spreadsheet cache? [yn] ')
    my_ip = socket.gethostbyname(socket.gethostname())
    controller = ProtocolExecutor(rxn_sheet_name, my_ip, serveraddr, use_cache=use_cache)

    tests_passed = controller.run_simulation()

    if tests_passed:
        if input('would you like to run the protocol? [yn] ').lower() == 'y':
            controller.run_protocol(simulate)
    else:
        print('Failed Some Tests. Please fix your errors and try again')

def launch_auto(serveraddr, rxn_sheet_name, use_cache, simulate):
    '''
    main function to launch an auto scientist that designs it's own experiments
    '''
    if not rxn_sheet_name:
        rxn_sheet_name = input('<<controller>> please input the sheet name ')
    if not use_cache:
        #using the cache bypasses google docs communication and uses the last rxn you loaded
        use_cache = 'y' == input('<<controller>> would you like to use spreadsheet cache? [yn] ')
    my_ip = socket.gethostbyname(socket.gethostname())
    auto = AutoContr(rxn_sheet_name, my_ip, serveraddr, use_cache=use_cache)
    model = DummyMLModel(2, 3)
    auto.run_simulation(model)
    if input('would you like to run on robot and pr? [yn] ').lower() == 'y':
        #need a new model because last is fit to sim
        model = DummyMLModel(2, 3)
        auto.run_protocol(model, simulate)


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

    def __init__(self, rxn_sheet_name, my_ip, server_ip, buff_size=4, use_cache=False, cache_path='Cache'):
        '''
        Note that init does not initialize the portal. This must be done explicitly or by calling
        a run function that creates a portal. The portal is not passed to init because although
        the code must not use more than one portal at a time, the portal may change over the 
        lifetime of the class
        NOte that pr cannot be initialized until you know if you're simulating or not, so it
        is instantiated in run
        '''
        #set according to input
        self.cache_path=cache_path
        self._make_cache()
        self.use_cache = use_cache
        self.my_ip = my_ip
        self.server_ip = server_ip
        self.buff_size=4
        self.simulate = False #by default will be changed if a simulation is run
        self._cached_reader_locs = {} #maps wellname to loc on platereader
        #this will be gradually filled
        self.robo_params = {}
        #necessary helper params
        credentials = self._init_credentials(rxn_sheet_name)
        wks_key = self._get_wks_key(credentials, rxn_sheet_name)
        rxn_spreadsheet = self._open_sheet(rxn_sheet_name, credentials)
        header_data = self._download_sheet(rxn_spreadsheet,0)
        input_data = self._download_sheet(rxn_spreadsheet,1)
        deck_data = self._download_sheet(rxn_spreadsheet, 2)
        self._init_robo_header_params(header_data)
        self._make_out_dirs(header_data)
        self.rxn_df = self._load_rxn_df(input_data)
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

    def _init_pr(self,simulate):
        try:
            self.pr = PlateReader(os.path.join(self.out_path, 'pr_data'),simulate)
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
            out_path = './Controller_Out'
        #get the root folder
        header_dict = {row[0]:row[1] for row in header_data[1:]}
        data_dir = header_dict['data_dir']
        self.out_path = os.path.join(out_path, data_dir)
        #if the folder doesn't exist yet, make it
        self.eve_files_path = os.path.join(self.out_path, 'Eve_Files')
        self.debug_path = os.path.join(self.out_path, 'Debug')
        paths = [self.out_path, self.eve_files_path, self.debug_path]
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    def _make_cache(self):
        if not os.path.exists(self.cache_path):
            os.path.makedirs(self.cache_path)

    def _init_credentials(self, rxn_sheet_name):
        '''
        this function reads a local json file to get the credentials needed to access other funcs  
        params:  
            str rxn_sheet_name: the name of the reaction sheet to run  
        returns:  
            ServiceAccountCredentials: the credentials to access that sheet  
        '''
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        #get login credentials from local file. Your json file here
        path = 'Credentials/hendricks-lab-jupyter-sheets-5363dda1a7e0.json'
        credentials = ServiceAccountCredentials.from_json_keyfile_name(path, scope) 
        return credentials

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
        name_key_pairs = self._get_wks_key_pairs(credentials, rxn_sheet_name)
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

    def _plot_setup_overlay(name):
        '''
        Sets up a figure for an overlay plot
        '''
        #formats the figure nicely
        plt.figure(num=None, figsize=(4, 4),dpi=300, facecolor='w', edgecolor='k')
        plt.legend(loc="upper right",frameon = False, prop={"size":7},labelspacing = 0.5)
        plt.rc('axes', linewidth = 2)
        plt.xlabel('Wavelength (nm)',fontsize = 16, fontfamily = 'Arial',fontname="Arial")
        plt.ylabel('Absorbance (a.u.)', fontsize = 16,fontname="Arial")
        plt.tick_params(axis = "both", width = 2)
        plt.tick_params(axis = "both", width = 2)
        plt.xticks([300,400,500,600,700,800,900,1000])
        plt.yticks([i/10 for i in range(0,11,1)])
        plt.axis([300, 1000, 0.0 , 1.0])
        plt.xticks(fontsize = 10)
        plt.yticks(fontsize = 10)
        plt.title(str(name), fontsize = 16, pad = 20,fontname="Arial")


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
            with open(os.path.join(self.eve_files_path,filename), 'wb') as write_file:
                write_file.write(file_bytes)
        self.translate_wellmap()
        

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
        cid = self.portal.send_pack('init', simulate, 
                self.robo_params['using_temp_ctrl'], self.robo_params['temp'],
                self.robo_params['labware_df'].to_dict(), self.robo_params['instruments'],
                self.robo_params['reagent_df'].to_dict(), self.my_ip,
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
        rxn_df = pd.DataFrame(input_data[3:], columns=cols)
        #rename some of the clunkier columns 
        rxn_df.rename({'operation':'op', 'dilution concentration':'dilution_conc','concentration (mM)':'conc', 'reagent (must be uniquely named)':'reagent', 'pause time (s)':'pause_time', 'comments (e.g. new bottle)':'comments','scan protocol':'scan_protocol', 'scan filename (no extension)':'scan_filename'}, axis=1, inplace=True)
        rxn_df.drop(columns=['comments'], inplace=True)#comments are for humans
        rxn_df.replace('', np.nan,inplace=True)
        rxn_df[['pause_time','dilution_conc','conc']] = rxn_df[['pause_time','dilution_conc','conc']].astype(float)
        #rename chemical names
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
        rxn_names = self.rxn_df.loc[:, 'reagent':'chemical_name'].drop(columns=['reagent','chemical_name']).columns
        reagent_df = self.rxn_df[['chemical_name', 'conc']].groupby('chemical_name').first()
        reagent_df.drop(rxn_names, errors='ignore', inplace=True) #not all rxns are reagents
        reagent_df[['loc', 'deck_pos', 'mass', 'molar_mass (for dry only)', 'comments']] = ''
        if not self.use_cache:
            d2g.upload(reagent_df.reset_index(),spreadsheet_key,wks_name = 'reagent_info', row_names=False , credentials = credentials)

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
            if row['op'] == 'transfer':
                self._send_transfer_command(row,i)
            elif row['op'] == 'pause':
                cid = self.portal.send_pack('pause',row['pause_time'])
            elif row['op'] == 'stop':
                #read through the inflight packets
                self.portal.send_pack('stop')
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
            else:
                raise Exception('invalid operation {}'.format(row['op']))

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
            #DEBUG
            with open(os.path.join(self.cache_path, 'reagent_info_sheet.pkl'), 'wb') as reagent_info_cache:
                dill.dump(reagent_info, reagent_info_cache)
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
            assert (math.isclose(entry.vol, 200)), "tried to scan {}, but {} has a bad volume. Vol was {}, but 200 is required for a scan".format(well, well, entry.vol)
            well_locs.append(entry.loc)
        #5
        self.pr.exec_macro('PlateIn')
        self.pr.run_protocol(row['scan_protocol'], row['scan_filename'], layout=well_locs)
        self.pr.exec_macro('PlateOut')

    def _update_cached_locs(self, wellnames):
        '''
        checks the cache to see if wellnames are in the cache. If they aren't, a query will be
        made to Eve for the wellnames, and data for those will be stored in the cache  
        params:  
            list<str> wellnames: the names of the wells you want to lookup  
        Postconditions:  
            The wellnames are in the cache  
        '''
        unknown_wellnames = [wellname for wellname in wellnames if wellname not in self._cached_reader_locs]
        if unknown_wellnames:
            #couldn't find in the cache, so we got to make a query
            self.portal.send_pack('loc_req', unknown_wellnames)
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
        For now this function just shakes the whole plate.
        In the future, we may want to mix
        things that aren't on the platereader, in which case a new argument should be made in 
        excel for the wells to scan, and we should make a function to pipette mix.
        '''
        wells_to_mix = row[self._products].loc[row[self._products].astype(bool)].astype(int)
        wells_to_mix.name = 'mix_code'
        #wells_to_mix = [t for t in wells_to_mix.astype(int).iteritems()]
        self._update_cached_locs(wells_to_mix.index)
        deck_poses = pd.Series({wellname:self._cached_reader_locs[wellname].deck_pos for 
                wellname in wells_to_mix.index}, name='deck_pos')
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
            self.pr.shake()
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
        self._send_transfer_command(water_transfer_row, i)
        self._send_transfer_command(reagent_transfer_row, i)

    def _get_dilution_transfer_rows(self, row):
        '''
        Takes in a dilution row and builds two transfer rows to be used by the transfer command  
        params:  
            pd.Series row: a row of self.rxn_df  
        returns:  
            tuple<pd.Series>: rows to be passed to the send transfer command. water first, then
              reagent
              see self._construct_dilution_transfer_row for details  
        '''
        reagent = row['chemical_name']
        reagent_conc = row['conc']
        product_cols = row.loc[self._products]
        dilution_name_vol = product_cols.loc[~product_cols.apply(lambda x: math.isclose(x,0,abs_tol=1e-9))]
        #assert (dilution_name_vol.size == 1), "Failure on row {} of the protocol. It seems you tried to dilute into multiple containers"
        total_vol = dilution_name_vol.iloc[0]
        target_name = dilution_name_vol.index[0]
        target_conc = row['dilution_conc']
        vol_water, vol_reagent = self._get_dilution_transfer_vols(target_conc, reagent_conc, total_vol)
        water_transfer_row = self._construct_dilution_transfer_row('WaterC1.0', target_name, vol_water)
        reagent_transfer_row = self._construct_dilution_transfer_row(reagent, target_name, vol_reagent)
        return water_transfer_row, reagent_transfer_row

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
        returns:  
            int: the cid of this command  
        Postconditions:  
            a transfer command has been sent to the robot  
        '''
        src = row['chemical_name']
        containers = row[self._products].loc[row[self._products] != 0]
        transfer_steps = [name_vol_pair for name_vol_pair in containers.iteritems()]
        #temporarilly just the raw callbacks
        callbacks = row['callbacks'].replace(' ', '').split(',') if row['callbacks'] else []
        has_stop = 'stop' in callbacks
        callbacks = [(callback, self._get_callback_args(row, callback)) for callback in callbacks]
        cid = self.portal.send_pack('transfer', src, transfer_steps, callbacks)
        if has_stop:
            n_stops = containers.shape[0]
            for _ in range(n_stops):
                self._stop(i)

    
    def _get_callback_args(self, row, callback):
        '''
        params:  
            pd.Series row: a row of self.rxn_df  
        returns:  
            list<object>: the arguments associated with the callback or None if no arguments  
        '''
        if callback == 'pause':
            return [row['pause_time']]
        return None
    

    def _get_dilution_transfer_vols(self, target_conc, reagent_conc, total_vol):
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
        mols_reagent = total_vol*target_conc #mols (not really mols if not milimolar. whatever)
        vol_reagent = mols_reagent/reagent_conc
        vol_water = total_vol - vol_reagent
        return vol_water, vol_reagent

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
            return "{}C{}".format(row['reagent'], row['conc']).replace(' ', '_')
        return pd.Series(new_cols)

class AutoContr(Controller):
    '''
    This is a completely automated controller. It takes as input a layout sheet, and then does
    it's own experiments, pulling data etc  
    We're adding in self.rxn_df_template, which uses the same parsing style as rxn_df
    but it's only a template, so we give it a new name and use self.rxn_df to change for the current batch we're trying to make
    '''
    def __init__(self, rxn_sheet_name, my_ip, server_ip, buff_size=4, use_cache=False, cache_path='Cache'):
        super().__init__(rxn_sheet_name, my_ip, server_ip, buff_size, use_cache, cache_path)
        self.rxn_df_template = self.rxn_df
        self.reagent_order = self.robo_params['reagent_df'].index.to_numpy()
        self.well_count = 0 #used internally for unique wellnames
        self.batch_num = 0 #used internally for unique filenames

    def run_simulation(self, model):
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

        print('<<controller>> ENTERING SIMULATION')
        port = 50000
        #launch an eve server in background for simulation purposes
        b = threading.Barrier(2,timeout=20)
        eve_thread = threading.Thread(target=launch_eve_server, kwargs={'my_ip':'','barrier':b},name='eve_thread')
        eve_thread.start()

        #do create a connection
        b.wait()
        self._run(port, True, model)

        #collect the eve thread
        eve_thread.join()

        #restore changed vars
        self.server_ip = stored_server_ip
        self.simulate = stored_simulate
        print('<<controller>> EXITING SIMULATION')
        return True

    def run_protocol(self, model, simulate=False, port=50000):
        '''
        The real deal. Input a server addr and port if you choose and protocol will be run  
        params:  
            str simulate: (this should never be used in normal operation. It is for debugging
              on the robot)  
            MLModel model: the model to use when training and predicting  
        NOTE: the simulate here is a little different than running run_simulation(). This simulate
          is sent to the robot to tell it to simulate the reaction, but that it all. The other
          simulate changes some things about how code is run from the controller
        '''
        print('<<controller>> RUNNING')
        self._run(port, simulate, model)
        print('<<controller>> EXITING')

    def _rename_products(self, rxn_df):
        '''
        required for class compatibility, but not used by the Auto
        '''
        pass

    def _get_product_df(self, products_to_labware):
        '''
        required for class compatibility, but not used by Auto
        TODO would be nice to return something that would blow up if accessed
        '''
        return np.nan

    @error_exit
    def _run(self, port, simulate, model):
        '''
        private function to run
        Returns:  
            bool: True if all tests were passed  
        '''
        self._init_pr(simulate)
        #create a connection
        sock = socket.socket(socket.AF_INET)
        sock.connect((self.server_ip, port))
        buffered_sock = BufferedSocket(sock, timeout=None)
        print("<<controller>> connected")
        self.portal = Armchair(buffered_sock,'controller','Armchair_Logs', buffsize=1)
        
        self.init_robot(simulate)
        while not model.quit:
            recipes = model.predict(5)
            #generate new wellnames for next batch
            wellnames = [self.generate_wellname() for i in range(recipes.shape[0])]
            #plan and execute a reaction
            self._create_samples(wellnames, recipes)
            #pull in the scan data
            scan_data = self._get_sample_data(wellnames, self.rxn_df['scan_filename']) 
            #train on scans
            model.train(scan_data.T.to_numpy(),recipes)
            self.batch_num += 1
        self.close_connection()
        self.pr.shutdown()
        return
    
    def _get_sample_data(self,wellnames, filename):
        '''
        TODO
        SKELETON
        scans a sample of wells specified by wellnames, and returns their spectra  
        params:  
            list<str> wellnames: the names of the wells to be scanned  
            str filename: the name of the file that holds the scans  
        returns:  
            df: n_wells, by size of spectra, the scan data.  
        ''' 
        self._update_cached_locs(wellnames)
        pr_dict = {self._cached_reader_locs[wellname].loc: wellname for wellname in wellnames}
        unordered_data = self.pr.load_reader_data(filename, pr_dict)
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
        '''
        self.portal.send_pack('init_containers', pd.DataFrame({'labware':'platereader',
                'container':'Well96', 'max_vol':200.0}, index=wellnames).to_dict())
        self.rxn_df = self._build_rxn_df(wellnames, recipes)
        self._products = wellnames
        self.execute_protocol_df()

    def generate_wellname(self):
        '''
        returns:  
            str: a unique name for a new well
        '''
        wellname = "autowell{}".format(self.well_count)
        self.well_count += 1
        return wellname

    def _get_rxn_max_vol(self, name, products):
        '''
        This is used right now because it's best I've got. Ideally, you could drop the part 
        of init that constructs product_df
        '''
        return 200.0

    def _build_rxn_df(self,wellnames,recipes):
        '''
        used to construct a rxn_df for this batch of reactions
        TODO test bejesus out of this method
        TODO need some sort of key to map from name of reagent to index of the 
        recipes, and then mul by 200 and then lookup to mul by the percentages in the reagent df
        '''
        rxn_df = self.rxn_df_template.copy() #starting point. still neeeds products
        recipe_df = pd.DataFrame(recipes, index=wellnames, columns=self.reagent_order)
        def build_product_rows(row):
            '''
            params:  
                pd.Series row: a row of the template df  
            returns:  
                pd.Series: a row for the new df
            '''
            d = {}
            if row['op'] == 'transfer':
                #is a transfer, so we want to lookup the volume of that reagent in recipe_df
                return recipe_df.loc[:, row['chemical_name']] * row['template'] * 200.0
            else:
                #if not a tranfer, we want to keep whatever value was there
                return pd.Series(row['template'], index=recipe_df.index)
        rxn_df = rxn_df.join(self.rxn_df_template.apply(build_product_rows, axis=1)).drop(
                columns='template')
        rxn_df['scan_filename'] = rxn_df['scan_filename'].apply(lambda x: "{}-{}".format(
                x, self.batch_num))
        return rxn_df

class ProtocolExecutor(Controller): 
    '''
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
        TESTS: These are run after a reaction concludes to make sure things went well  
        run_all_tests() bool: True if you passed, else false. run when at end of simulation  
        test_vol_lab_cont() bool: tests that labware volume and containers are correct  
        test_contents() bool: tests that the contents of each container is ok  
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

    def run_simulation(self):
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
        self._run(port, simulate=True)

        #run post execution tests
        tests_passed = self.run_all_tests()

        #collect the eve thread
        eve_thread.join()

        #restore changed vars
        self.server_ip = stored_server_ip
        self.simulate = stored_simulate
        print('<<controller>> EXITING SIMULATION')
        return tests_passed

    def run_protocol(self, simulate=False, port=50000):
        '''
        The real deal. Input a server addr and port if you choose and protocol will be run  
        params:  
            str simulate: (this should never be used in normal operation. It is for debugging
              on the robot)  
        NOTE: the simulate here is a little different than running run_simulation(). This simulate
          is sent to the robot to tell it to simulate the reaction, but that it all. The other
          simulate changes some things about how code is run from the controller
        '''
        print('<<controller>> RUNNING PROTOCOL')
        self._run(port, simulate=simulate)
        print('<<controller>> EXITING PROTOCOL')
        
    @error_exit
    def _run(self, port, simulate):
        '''
        Returns:  
            bool: True if all tests were passed  
        '''
        self._init_pr(simulate)
        #create a connection
        sock = socket.socket(socket.AF_INET)
        sock.connect((self.server_ip, port))
        buffered_sock = BufferedSocket(sock, timeout=None)
        print("<<controller>> connected")
        self.portal = Armchair(buffered_sock,'controller','Armchair_Logs', buffsize=1)

        self.init_robot(simulate)
        self.execute_protocol_df()
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
                row = rxn_df.loc[~rxn_df[col].isna()].squeeze()
                reagent_name = row['chemical_name']
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
        vol_change_rows = self.rxn_df.loc[self.rxn_df['op'].apply(lambda x: x in ['transfer','dilution'])]
        aspirations = vol_change_rows['chemical_name'] == name
        max_vol = 0
        current_vol = 0
        for i, is_aspiration in aspirations.iteritems():
            if is_aspiration and self.rxn_df.loc[i,'op'] == 'transfer':
                #This is a row where we're transfering from this well
                current_vol -= self.rxn_df.loc[i, products].sum()
            elif is_aspiration and self.rxn_df.loc[i, 'op'] == 'dilution':
                _, transfer_row = self._get_dilution_transfer_rows(self.rxn_df.loc[i])
                vol = transfer_row[self._products].sum() 
                current_vol -= vol
            else:
                current_vol += self.rxn_df.loc[i,name]
                max_vol = max(max_vol, current_vol)
        return max_vol

    
    #TESTING
    #PRE Simulation
    def run_all_checks(self):
        found_errors = 0
        found_errors = max(found_errors, self.check_rxn_df())
        found_errors = max(found_errors, self.check_labware())
        found_errors = max(found_errors, self.check_reagents())
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
        for i, r in self.rxn_df.iterrows():
            r_num = i+1
            #check pauses
            if (not ('pause' in r['op'] or 'pause' in r['callbacks'])) == (not pd.isna(r['pause_time'])):
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


    #POST Simulation
    def run_all_tests(self):
        '''
        runs all post rxn tests  
        Returns:  
            bool: True if all tests were passed  
        '''
        print('<<controller>> running post execution tests')
        valid = True
        valid = valid and self.test_vol_lab_cont()
        valid = valid and self.test_contents()
        return valid

    def test_vol_lab_cont(self):
        '''
        tests that vol, labware, and containers are correct for a row of a side by side df with
        those attributes  
        Preconditions:  
            labware_df, reagent_df, and products_df are all initialized as vals in robo_params  
            self.rxn_df is initialized  
            df labware_df:  
            df rxn_df: as from excel  
            df reagent_df: info on reagents. columns from sheet. See excel specification  
            df product_df:  
            self.eve_files_path + wellmap.tsv exists (this is a file output by eve that is shipped
              over in close step  
        Postconditions:  
            Any errors will be printed to the screen.  
            If errors were found, a pkl of the sbs will be written  
        Returns:  
            bool: True if all tests were passed  
        '''
        sbs = self._get_vol_lab_cont_sbs()
        sbs['flag'] = sbs.apply(lambda row: self._is_valid_vol_lab_cont_sbs(row), axis=1)
        filtered_sbs = sbs.loc[~sbs['flag']]
        if filtered_sbs.empty:
            print('<<controller>> congrats! Volumes, labware, containers, and deck_poses look good!')
        else:
            print('<<controller>> volume/deck pos/labware/container errors')
            with open(os.path.join(self.debug_path,'vol_lab_cont_sbs.pkl'), 'wb') as sbs_pkl:
                dill.dump(sbs, sbs_pkl)
            if input('<<controller>> would you like to view the full sbs? [yn] ').lower() == 'y':
                print(sbs)
            return False
        return True

    def test_contents(self):
        '''
        tests to ensure that the contents of each container is correct
        note does not work for dilutions, and does not check reagents  
        params:  
            df rxn_df: from excel  
            bool use_cache: True if data is cached  
            str eve_logpath: the path to the eve logfiles  
        Postconditions:  
            if a difference was found it will be displayed,  
            if no differences are found, a friendly print message will be displayed  
        Returns:  
            bool: True if all tests were passed  
        '''
        sbs = self._create_contents_sbs()
        sbs['flag'] = sbs.apply(self._is_valid_contents_sbs,axis=1)
        filtered_sbs = sbs.loc[~sbs['flag']]
        if filtered_sbs.empty:
            print('<<controller>> congrats! Contents are correct!')
        else:
            print('<<controller>> there ere some content errors')
            with open(os.path.join(self.debug_path,'contents_sbs.pkl'), 'wb') as sbs_pkl:
                dill.dump(sbs, sbs_pkl)
            if input('<<controller>> would you like to view the full sbs? [yn] ').lower() == 'y':
                print(sbs)
            return False
        return True

    def _is_valid_vol_lab_cont_sbs(self, row):
        '''
        params:  
            pd.Series row: a row of a sbs dataframe:  
        returns:  
            Bool: True if it is a valid row  
        '''
        if row['deck_pos_t'] != 'any' and row['deck_pos'] not in row['deck_pos_t']:
            print('<<controller>> deck_pos_error:')
            print(row.to_frame().T)
            print()
            return False
        if row['vol_t'] != 'any' and not math.isclose(row['vol'],row['vol_t'], abs_tol=1e-9):
            print('<<controller>> volume error:')
            print(row.to_frame().T)
            print()
            return False
        if row['container_t'] != 'any' and not row['container'] == row['container_t']:
            print('<<controller>> container error:')
            print(row.to_frame().T)
            print()
            return False
        if row['loc_t'] != 'any' and row['loc'] not in row['loc_t']:
            print('<<controller>> loc error:')
            print(row.to_frame().T)
            print()
            return False
        return True
    
    def _get_vol_lab_cont_sbs(self):
        '''
        This is for comparing the volumes, labwares, and containers  
        params:  
        Preconditions:  
            labware_df, reagent_df, and products_df are all initialized as vals in robo_params  
            self.rxn_df is initialized  
            df labware_df:  
            df rxn_df: as from excel  
            df reagent_df: info on reagents. columns from sheet. See excel specification  
            df product_df:  
            self.eve_files_path + wellmap.tsv exists (this is a file output by eve that is shipped
              over in close step  
        returns  
            df  
                + INDEX
                + chemical_name: the containers name
                + COLS: symmetric. Theoretical are suffixed _t
                + str deck_pos: position on deck
                + float vol: the volume in the container
                + list<tuple<str, float>> history: the chem_name paired with the amount or
                  keyword 'aspirate' and vol
        '''
        #copy the locals cause we're changing them
        labware_df = self.robo_params['labware_df'].set_index('name').rename(index={'platereader7':'platereader','platereader4':'platereader'}) #converting to dict like
        product_df = self.robo_params['product_df'].copy()
        reagent_df = self.robo_params['reagent_df'].copy()
        #create a df with sets of allowable locs and deck_poses
        def get_dry_container_cols(df):
            '''
            apply helper func to combine the rows of the dry_containers_df
            '''
            d = {'loc':set(),'deck_pos':set()}
            for i, r in df.iterrows():
                d['loc'].add(r['loc'])
                d['deck_pos'].add(r['deck_pos'])
            return pd.Series(d)
        dry_containers = self.robo_params['dry_containers'].groupby('index').apply(get_dry_container_cols)
        def get_deck_pos(labware):
            '''
            apply helper func to get the deck position for products
            '''
            if labware:
                deck_pos = labware_df.loc[labware,'deck_pos']
                if isinstance(deck_pos,np.int64):
                    return [deck_pos]
                else:
                    #for platereader with two indices
                    return deck_pos.to_list()
            else:
                return 'any'
        product_df['deck_pos'] = product_df['labware'].apply(get_deck_pos)
        product_df['vol'] = [self._vol_calc(name) for name in product_df.index]
        product_df['loc'] = 'any'
        product_df.replace('','any', inplace=True)

        #because reagents can be built, we now need to ensure that you end up with something 
        #that could be on a new set of labware for reagents
        reagent_df['deck_pos'] = reagent_df['deck_pos'].apply(lambda x: {x})
        reagent_df['loc'] = reagent_df['loc'].apply(lambda x: {x})
        reagent_df['vol'] = 'any' #I'm not checking this because it's harder to check, and works fine
        reagent_df['container'] = 'any' #actually fixed, but checked by combo deck_pos and loc
        def merge_dry(row):
            '''
            apply helper to merge a reagent_df with a dry_container_df
            '''
            d = {'loc':{}, 'deck_pos':{}}
            found_match = False
            for name in dry_containers.index.unique():
                if name in row.name:
                    d['loc'] = dry_containers.loc[name,'loc'].union(row['loc'])
                    d['deck_pos'] = dry_containers.loc[name,'deck_pos'].union(row['deck_pos'])
                    found_match = True
            if not found_match:
                d['loc'] = row['loc']
                d['deck_pos'] = row['deck_pos']
            return pd.Series(d)
        reagent_df[['loc', 'deck_pos']] = reagent_df.apply(merge_dry, axis=1)
                    
        theoretical_df = pd.concat((reagent_df.loc[:,['loc', 'deck_pos',\
                'vol','container']], product_df.loc[:,['loc', 'deck_pos','vol','container']]))
        result_df = pd.read_csv(os.path.join(self.eve_files_path,'wellmap.tsv'), sep='\t').set_index('chem_name')
        sbs = result_df.join(theoretical_df, rsuffix='_t') #side by side
        return sbs

    def _vol_calc(self, name):
        '''
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
            _, reagent_transfer_row = self._get_dilution_transfer_rows(row) #the _ is water
            return reagent_transfer_row[self._products].sum()

        if dilution_rows.empty:
            dilution_aspirations = 0.0
        else:
            dilution_vols = dilution_rows.apply(lambda r: calc_dilution_vol(r),axis=1)
            dilution_aspirations = dilution_vols.sum()
        return dispenses - transfer_aspirations - dilution_aspirations
    
    def _is_valid_contents_sbs(self, row):
        '''
        tests if a row of contents sbs is valid
        params:  
            pd.Series row: has vol_t and vol  
        returns:  
            False if vol_t!=vol else True  
        Postconditions:  
            If vol_t!=vol the row will be printed  
            
        '''
        if not math.isclose(row['vol_t'], row['vol']):
            print('<<controller>> contents error:')
            print(row.to_frame().T)
            print()
            return False
        return True


        if not sbs.loc[~sbs['flag']].empty:
            print('<<controller>> found some invalid contents. Displaying rows')
            container_index = sbs.loc[~sbs['flag']].index.get_level_values('container')
            print(sbs.loc[container_index])
        else:
            print('<<controller>> Well done! Product have correct ratios of reagents')

    def _create_contents_sbs(self):
        '''
        constructs a side by side frame from the history in well_history.tsv and the reaction
        df
        NOTE: completely ignores aspiration, but if all of your dispenses are correct, and your
        final contents are correct you're looking pretty good
        '''
        history = pd.read_csv(os.path.join(self.eve_files_path, 'well_history.tsv'),na_filter=False,sep='\t').rename(columns={'chemical':'chem_name'})
        disp_hist = history.loc[history['chem_name'].astype(bool)]
        contents = disp_hist.groupby(['container','chem_name']).sum()
        theoretical_his_list = []
        for _, row in self.rxn_df.loc[(self.rxn_df['op'] == 'transfer') | \
                (self.rxn_df['op'] == 'dilution')].iterrows():
            if row['op'] == 'transfer':
                for product in self._products:
                    theoretical_his_list.append((product, row[product], row['chemical_name']))
            else: #row['op'] == 'dilution'
                water_transfer_row, reagent_transfer_row = self._get_dilution_transfer_rows(row) #the _ is water
                product_vols = water_transfer_row[self._products]
                target_reagent = product_vols.loc[~product_vols.apply(lambda x: \
                        math.isclose(x,0,abs_tol=1e-9))].index[0]
                theoretical_his_list.append((target_reagent, water_transfer_row[target_reagent], \
                        'WaterC1.0'))
                theoretical_his_list.append((target_reagent, \
                        reagent_transfer_row[target_reagent], \
                        reagent_transfer_row['chemical_name']))
        theoretical_his = pd.DataFrame(theoretical_his_list, \
                columns=['container', 'vol', 'chem_name'])
        theoretical_contents = theoretical_his.groupby(['container','chem_name']).sum()
        theoretical_contents = theoretical_contents.loc[~theoretical_contents['vol'].apply(lambda x:\
                math.isclose(x,0))]
        sbs = theoretical_contents.join(contents, how='left',lsuffix='_t')
        return sbs

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
    SPECTRO_ROOT_PATH = None
    PROTOCOL_PATH = None

    def __init__(self, data_path):
        pass
        
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

    def shake(self):
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
        params:  
            str protocol_name: the name of the protocol that will be edited  
            list<str> layout: the wells that you want to be used for the protocol ordered.
              (first will be X1, second X2 etc. If not specified will not alter layout)  
        '''
        pass

    def shutdown(self):
        '''
        closes connection. Use this if you're done with this object at cleanup stage
        '''
        pass

    def load_reader_data(self, filename,loc_to_name):
        '''
        takes in the filename of a reader output and returns a dataframe with the scan data
        loaded, and a dictionary with relevant metadata.  
        params:  
            str filename: the name of the file to read  
            dict<str:str> loc_to_name: maps location to name of reaction  
        returns:  
            df: the scan data for that file  
            dict<str:obj>: holds the metadata  
                str filename: the filename as you passed in  
                int n_cycles: the number of cycles  
        '''
        pass

    def load_reader_data(self, filename, loc_to_name):
        '''
        doesn't actually read data, generates some dummy data for you to use  
        returns:  
            df: a bunch of -1s in a nice dataframe with size 701 per well  
        '''
        wellnames = loc_to_name.values()
        return pd.DataFrame(np.ones((701,len(wellnames))), columns=wellnames)

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
    SPECTRO_ROOT_PATH = "/mnt/c/Program Files/SPECTROstar Nano V5.50/"
    PROTOCOL_PATH = r"C:\Program Files\SPECTROstar Nano V5.50\User\Definit"
    SPECTRO_DATA_PATH = "/mnt/c/Hendricks Lab/Plate Reader Data Backup"

    def __init__(self, data_path, simulate=False):
        self.data_path = data_path
        self.simulate = simulate
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        self._set_config_attr('Configuration','SimulationMode', str(int(simulate)))
        self._set_config_attr('ControlApp','AsDDEserver', 'True')
        self.exec_macro("dummy")
        self.exec_macro("init")
        self.exec_macro('PlateOut')
        
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

    def shake(self):
        '''
        executes a shake
        '''
        macro = "Shake"
        shake_type = 2
        shake_freq = 300
        shake_time = 60
        self.exec_macro(macro, shake_type, shake_freq, shake_time)

    def load_reader_data(self,filename, loc_to_name):
        '''
        takes in the filename of a reader output and returns a dataframe with the scan data
        loaded, and a dictionary with relevant metadata.  
        params:  
            str filename: the name of the file to read  
            dict<str:str> loc_to_name: maps location to name of reaction  
        returns:  
            df: the scan data for that file  
            dict<str:obj>: holds the metadata  
                str filename: the filename as you passed in  
                int n_cycles: the number of cycles  
        '''
        if self.simulate:
            return super().load_reader_data(filename, loc_to_name) #return dummy data
        else:
            #parse the metadata
            start_i, metadata = self._parse_metadata(filename, self.data_path)
            # Read data ignoring first metadata lines
            df = pd.read_csv(os.path.join(self.data_path,filename), skiprows=start_i,
                    header=None,index_col=0,na_values=["       -"],encoding = 'latin1').T
            headers = [loc_to_name[x[:-1]] for x in df.columns]
            df.columns = headers
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

    def run_protocol(self, protocol_name, filename, layout=None):
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
        if not self.simulate:
            shutil.move(os.path.join(self.SPECTRO_DATA_PATH, "{}.csv".format(filename)), 
                    os.path.join(self.data_path, "{}.csv".format(filename)))


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
        self.exec_macro('PlateIn')
        self.exec_macro('Terminate')
        self._set_config_attr('ControlApp','AsDDEserver','False')
        self._set_config_attr('ControlApp', 'DisablePlateCmds','False')
        self._set_config_attr('Configuration','SimulationMode', str(0))


if __name__ == '__main__':
    SERVERADDR = "10.25.16.146"
    main(SERVERADDR)