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
import copy
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
import sys
import traceback

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
from matplotlib.lines import Line2D
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


class TeeTerminalOutput:

    '''

    Mirrors terminal output to a text file while preserving normal terminal

    printing.

    This is used to save a copy of the Auto run terminal output into the Debug

    folder. It behaves like sys.stdout/sys.stderr, but writes each message to

    both the original stream and a log file.

    '''

    def __init__(self, stream, log_file_handle):

        self.stream = stream

        self.log_file_handle = log_file_handle

    def write(self, message):

        self.stream.write(message)

        self.log_file_handle.write(message)

        self.log_file_handle.flush()

    def flush(self):

        self.stream.flush()

        self.log_file_handle.flush()

    def isatty(self):

        return self.stream.isatty()

def terminal_output_capture_guard(func):
    '''
    Ensures terminal output capture is finalized whether the run succeeds or
    errors.
    '''
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)

        finally:
            stop_capture = getattr(
                self,
                '_stop_terminal_output_capture',
                None
            )

            if callable(stop_capture):
                stop_capture()

    return wrapper

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
    auto = None

    try:
        if not rxn_sheet_name:
            rxn_sheet_name = input('<<controller>> please input the sheet name ')

        my_ip = socket.gethostbyname(socket.gethostname())
        auto = AutoContr(rxn_sheet_name, my_ip, serveraddr, use_cache=use_cache)

        #note shorter iterations for testing
        #final_spectra = np.loadtxt("test_target_1.csv", delimiter=',', dtype=float).reshape(1,-1)
        #print(auto.rxn_df.describe())
        #print(auto.rxn_df.head(20))
        #print(auto.rxn_df)

        y_shape = auto.y_shape# number of reagents to learn on
        #print("starting with y_shape:", y_shape)

        reagent_info = auto.robo_params['reagent_df']
        fixed_reagents = auto.get_fixed_reagents()
        variable_reagents = auto.get_variable_reagents()
        target_value = auto.getModelInfo()["target"] 
        acquisition_modes = list(
            auto.robo_params.get(
                'acquisition_modes',
                [auto.robo_params.get('acquisition_mode', 'exploit')]
            )
        )

        try:
            min_conc = auto.get_min_conc()
            if auto.robo_params.get(
                'auto_terminal_verbosity',
                'standard'
            ) == 'diagnostic':
                print(
                    "<<controller diagnostic>> minimum variable "
                    f"concentrations: {min_conc}"
                )
            min_conc = list(min_conc.values())
        except Exception as e:
            print(f'Error getting min_conc: {e}')

        # Generate bounds for each reagent, assuming concentrations range from 0 to 1
        bounds = [{'name': f'reagent_{i+1}_conc', 'type': 'continuous', 'domain': (0, 1)} for i in range(y_shape)]

        # final_spectra not used?
        if auto.robo_params.get(
            'auto_terminal_verbosity',
            'standard'
        ) != 'essential':
            print("<<controller>> setting up Auto optimization model")
        
        model = OptimizationModel(
            bounds,
            target_value,
            reagent_info,
            fixed_reagents,
            variable_reagents,
            initial_design_numdata=auto.getModelInfo()["initial_data"],
            # A portfolio produces one unique condition for each requested
            # mode, while num_duplicates remains the physical replicate count
            # for every one of those conditions.
            batch_size=len(acquisition_modes),
            max_iters=auto.getModelInfo()["max_iterations"],
            min_conc=auto.min_conc,
            max_conc=auto.max_conc,
            total_volume=auto.template_meta['tot_vol'],
            fixed_reagent_volumes=auto._get_fixed_reagent_volumes(),
            allow_true_zero=auto.robo_params.get('allow_true_zero', False),
            acquisition_mode=auto.robo_params.get(
                'acquisition_mode',
                'exploit'
            ),
            # The balanced weight is intentionally a named code-level setting
            # with a backward-compatible default. A separate workbook control
            # is not required for the initial balanced-mode implementation.
            balanced_exploration_weight=auto.robo_params.get(
                'balanced_exploration_weight',
                1.0
            ),
            terminal_verbosity=auto.robo_params.get(
                'auto_terminal_verbosity',
                'standard'
            )
        )

        model.acquisition_modes = acquisition_modes
        model.portfolio_min_distance = auto.robo_params.get(
            'portfolio_min_distance',
            0.05
        )

        if (
            model.acquisition_mode
            not in model.IMPLEMENTED_ACQUISITION_MODES
        ):
            raise NotImplementedError(
                "Auto acquisition mode "
                f"{model.acquisition_mode!r} is configured, but its recipe "
                "selection behavior is not implemented yet. The Auto "
                "protocol will not start until that mode is implemented and "
                "validated."
            )
        
        print(f"Target: {target_value}")

        if not no_sim:
            auto.run_simulation(no_pr=no_pr)

        if input('would you like to run on robot and pr? [yn] ').lower() == 'y':
            auto._check_auto_well_capacity(model)
            auto._check_auto_pipette_tip_capacity(model)
            auto.run_protocol(simulate=simulate, model=model, no_pr=no_pr)

    finally:
        if auto is not None:
            auto._stop_terminal_output_capture()



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
            float aspirable_vol: the volume minus dead volume
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
        self.terminal_log_file_handle = None
        self.original_stdout = None
        self.original_stderr = None
        
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
        #self.drive_service = self._init_google_drive(credentials) # Terence   
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
        self._start_terminal_output_capture()

        try:
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

        except Exception:
            print("<<controller>> ERROR during controller initialization")
            traceback.print_exc()
            self._stop_terminal_output_capture()
            raise

    def _start_terminal_output_capture(self):
        '''
        Starts mirroring stdout/stderr to Debug/terminal_output.txt.

        The terminal still prints normally. This only adds a file copy of the
        current Python process output for debugging.
        '''
        if self.terminal_log_file_handle is not None:
            return
        
        debug_dir = getattr(self, 'debug_path', None)

        if debug_dir is None:
            return

        terminal_log_path = os.path.join(
            debug_dir,
            'terminal_output.txt'
        )

        self.terminal_log_file_handle = open(
            terminal_log_path,
            'w',
            encoding='utf-8'
        )

        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        sys.stdout = TeeTerminalOutput(
            self.original_stdout,
            self.terminal_log_file_handle
        )

        sys.stderr = TeeTerminalOutput(
            self.original_stderr,
            self.terminal_log_file_handle
        )

        print(
            f"<<controller>> saving terminal output to "
            f"{terminal_log_path}"
        )
    
    def _stop_terminal_output_capture(self):
        '''
        Restores stdout/stderr and closes the terminal output log file.

        Safe to call more than once. If terminal capture was never started,
        this returns without doing anything.
        '''
        terminal_log_file_handle = getattr(
            self,
            'terminal_log_file_handle',
            None
        )

        if terminal_log_file_handle is None:
            return

        original_stdout = getattr(self, 'original_stdout', None)
        original_stderr = getattr(self, 'original_stderr', None)

        try:
            print("<<controller>> terminal output capture complete")
        finally:
            if original_stdout is not None:
                sys.stdout = original_stdout

            if original_stderr is not None:
                sys.stderr = original_stderr

            try:
                terminal_log_file_handle.close()
            finally:
                self.terminal_log_file_handle = None
                self.original_stdout = None
                self.original_stderr = None
    
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

    def get_min_conc(self):
        """
        Gets the minimum target concentration for each variable reagent.

        By default, Auto uses the 5 uL-equivalent concentration as the lower
        bound for each variable reagent. This keeps the optimizer inside the
        normal continuous pipetting region.

        If Header allow_true_zero is enabled, the lower bound becomes 0 for
        each variable reagent. The true-zero repair logic then handles the
        forbidden 0-5 uL transfer region by mapping candidates to either true
        zero or the minimum executable transfer volume.

        Input:
            None

        Output:
            dict:
                Reagent names as keys and minimum target concentrations as values.
        """
        min_concs = {}

        if self.robo_params.get('allow_true_zero', False):
            for var_reagent in self.get_variable_reagents():
                min_concs[var_reagent] = 0.0

            print(
                "<<controller>> true-zero search enabled: "
                "variable reagent lower bounds set to 0"
            )
            return min_concs

        # Default behavior: use the concentration produced by a 5 uL transfer
        # from the stock reagent into the template reaction volume.
        for var_reagent in self.get_variable_reagents():
            stock_conc = self._get_variable_reagent_stock_conc(var_reagent)
            total_volume = float(self.template_meta['tot_vol'])

            min_conc = stock_conc * 5.0 / total_volume
            min_concs[var_reagent] = min_conc

        print(
            "<<controller>> true-zero search disabled: "
            "variable reagent lower bounds use 5 uL-equivalent concentrations"
        )
        return min_concs
    
    def get_max_conc(self):
        """
        Calculates the maximum target concentration for each variable reagent.

        Each variable reagent is given an independent maximum based on the
        concentration it could reach if that reagent alone used the full
        remaining well volume after fixed reagents are added.

        This intentionally does not split the remaining volume evenly between
        variable reagents. Splitting evenly makes the search space behave like a
        ratio-constrained space and cuts off valid high/high combinations that
        may still physically fit.

        Physical overflow is handled separately by the Auto volume-feasibility
        checks. Water fills whatever volume remains after fixed and variable
        reagent transfers.

        Returns:
            dict:
                Maximum target concentrations for each variable reagent.
        """
        fixed_vols = self._get_fixed_reagent_volumes()
        print(f"Fixed Volumes: {fixed_vols}")

        total_fixed_vol = float(sum(fixed_vols.values()))
        print(f"Total volume of fixed reagents: {total_fixed_vol}")

        total_volume = float(self.template_meta['tot_vol'])
        remaining_vol = total_volume - total_fixed_vol

        if remaining_vol < 0:
            raise ValueError(
                "Fixed reagent volumes exceed the final reaction volume. "
                f"Fixed volume = {total_fixed_vol:.4f} uL, "
                f"final reaction volume = {total_volume:.4f} uL."
            )

        max_concs = {}

        for var_reagent in self.get_variable_reagents():
            stock_conc = self._get_variable_reagent_stock_conc(var_reagent)
            print(f"Concentration on deck for {var_reagent}: {stock_conc}")

            # Calculate the maximum concentration using M1V1 = M2V2.
            # This is the concentration the reagent could reach if it alone
            # used the full remaining volume after fixed reagents.
            max_conc = stock_conc * remaining_vol / total_volume
            max_concs[var_reagent] = max_conc

        return max_concs


   
   
    def _make_out_dirs(self, header_data):
        '''
        Creates output directories both locally and in Google Drive
        params:  
            list<list<str>> header_data: data from the header  
        Postconditions:  
            All paths used by this class have been initialized locally.
            They are not overwritten if they already exist
        '''

        local_out_path = '/mnt/c/Users/science_356_lab/Robot_Files/Protocol_Outputs'
        
        header_dict = {row[0]:row[1] for row in header_data[1:]}
        data_dir = header_dict['data_dir']
        self.reaction_folder_name = os.path.basename(os.path.dirname(data_dir))
        
        
        self.out_path = os.path.join(local_out_path, data_dir)
        self.eve_files_path = os.path.join(self.out_path, 'Eve_Files')
        self.debug_path = os.path.join(self.out_path, 'Debug')
        self.plot_path = os.path.join(self.out_path, 'Plots')
        
       
        local_paths = [self.out_path, self.eve_files_path, self.debug_path, self.plot_path]
        for path in local_paths:
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
        Loads controller and Auto settings from the Header worksheet.

        Auto acquisition selection is controlled by the optional
        acquisition_mode setting:

            exploit:
                Selects the feasible recipe whose GP-predicted mean lambda max
                is closest to the requested target. This preserves the current
                Auto recipe-selection behavior.

            explore:
                Selects the feasible recipe with the greatest GP predictive
                uncertainty.

            balanced:
                Uses a target-aware hybrid score that rewards both proximity
                to the requested lambda-max target and predictive uncertainty.

            target_ei:
                Uses target-aware expected improvement to select the recipe
                expected to reduce the best QC-approved target error.

        Auto plotting is controlled by the optional auto_plot_profile setting:

            standard:
                Generates applicable per-batch and final Auto plots.

            final_only:
                Generates final-run Auto plots but suppresses per-batch Auto
                diagnostic plots.

            off:
                Suppresses automatic Auto diagnostic and summary plots.

        Older spreadsheets that do not contain acquisition_mode default to
        exploit so their recipe-selection behavior remains unchanged.

        Older spreadsheets that do not contain auto_plot_profile default to
        standard for backward compatibility.

        Condition-level early stopping is controlled by the optional
        target_tolerance_nm setting. It is the maximum absolute error, in
        nanometers, allowed between the requested target and a QC-approved
        condition mean. Older spreadsheets default to 10 nm. This setting
        does not bypass replicate-QC, model-training, or target-stop
        eligibility requirements.

        Auto terminal output is controlled by the optional
        auto_terminal_verbosity setting:

            essential:
                Shows safety warnings, QC exclusions, stop decisions, and
                final run status. This is the least verbose setting; it never
                suppresses safety-relevant or scientifically consequential
                information.

            standard:
                Adds normal Auto configuration, batch, acquisition-selection,
                volume-balance, GP-update, and output-summary messages. This
                is the default for older spreadsheets.

            diagnostic:
                Adds raw seed/model arrays, full controller provenance JSON,
                and a result line for every explored reagent mask. Complete
                audit records remain available in Auto CSV/report artifacts at
                every verbosity level.

        User-friendly aliases are accepted. In particular, off means
        essential (not silent), limited means standard, and all means
        diagnostic.

        Source-volume protection is controlled by the optional
        auto_source_volume_check setting:

            off:
                Preserves legacy behavior. No declared source-inventory
                preflight is performed.

            required:
                Before every Auto batch, verifies aggregate planned source
                use against the robot's current mass-derived aspiratable
                source inventory. A failed check stops before any liquid
                transfer is sent. Individual same-stock tube allocation
                remains the existing robot runtime's responsibility.

        The robot already derives each source's volume, dead volume, and
        aspiration height from the existing reagent_info mass entry and its
        physical tube type. No duplicate per-reagent volume entry is needed.
        An optional auto_source_reserve_volume_uL setting protects additional
        liquid beyond the robot's established dead-volume calculation.

        The optional pi_legacy_tare_offset_g setting supports a Raspberry Pi
        deployment whose tube tare constants are known to be lower than the
        corrected laboratory values. A positive offset is subtracted only
        from the mass payload sent to that legacy Pi, preserving the actual
        measured mass in the controller. Older worksheets default to 0 g.
        Do not enable this setting after the Raspberry Pi receives corrected
        tare constants, or its liquid-volume estimate would be double-corrected.
        '''
        header_dict = {
            row[0]: row[1]
            for row in header_data[1:]
        }

        self.robo_params['using_temp_ctrl'] = (
            header_dict['using_temp_ctrl'] == 'yes'
        )

        self.robo_params['temp'] = (
            float(header_dict['temp'])
            if self.robo_params['using_temp_ctrl']
            else None
        )

        if self.robo_params['temp'] is not None:
            assert (
                self.robo_params['temp'] >= 4
                and self.robo_params['temp'] <= 95
            ), "invalid temperature"

        self.dilution_params = self.DilutionParams(
            header_dict['dilution_cont'],
            float(header_dict['dilution_vol'])
        )

        self.robo_params['target'] = float(
            header_dict['target']
        )

        # Optional condition-level early-stop threshold. The same canonical
        # value is also used by the target-probability plots, so their stated
        # success region always matches the controller's actual stop rule.
        target_tolerance_value = str(
            header_dict.get(
                'target_tolerance_nm',
                10.0
            )
        ).strip()

        try:
            target_tolerance_nm = float(target_tolerance_value)
        except (TypeError, ValueError):
            raise ValueError(
                "Header value target_tolerance_nm must be a finite, "
                "nonnegative number in nm. "
                f"Received: {target_tolerance_value!r}."
            )

        if (
            not math.isfinite(target_tolerance_nm)
            or target_tolerance_nm < 0.0
        ):
            raise ValueError(
                "Header value target_tolerance_nm must be a finite, "
                "nonnegative number in nm. "
                f"Received: {target_tolerance_value!r}."
            )

        self.robo_params['target_tolerance_nm'] = target_tolerance_nm

        print(
            "<<controller>> Auto condition-level target tolerance: "
            f"{target_tolerance_nm:g} nm"
        )

        # Terminal verbosity intentionally controls presentation only. It
        # cannot hide safety warnings, QC outcomes, condition-level stop
        # decisions, or the persistent CSV/report audit trail.
        terminal_verbosity_value = str(
            header_dict.get(
                'auto_terminal_verbosity',
                'standard'
            )
        ).strip().lower()

        terminal_verbosity_value = (
            terminal_verbosity_value
            .replace('-', '_')
            .replace(' ', '_')
        )

        terminal_verbosity_aliases = {
            '': 'standard',
            'default': 'standard',
            'on': 'standard',
            'yes': 'standard',
            'true': 'standard',
            '1': 'standard',
            'essential': 'essential',
            'minimum': 'essential',
            'minimal': 'essential',
            'quiet': 'essential',
            'off': 'essential',
            'no': 'essential',
            'false': 'essential',
            '0': 'essential',
            'standard': 'standard',
            'limited': 'standard',
            'normal': 'standard',
            'diagnostic': 'diagnostic',
            'debug': 'diagnostic',
            'verbose': 'diagnostic',
            'all': 'diagnostic'
        }

        if terminal_verbosity_value not in terminal_verbosity_aliases:
            raise ValueError(
                "Header value auto_terminal_verbosity must be one of: "
                "essential, standard, or diagnostic. "
                "Aliases include off, limited, and all. "
                f"Received: {terminal_verbosity_value!r}."
            )

        self.robo_params['auto_terminal_verbosity'] = (
            terminal_verbosity_aliases[terminal_verbosity_value]
        )

        print(
            "<<controller>> Auto terminal verbosity: "
            f"{self.robo_params['auto_terminal_verbosity']}"
        )

        # Optional source-inventory hard stop. It remains disabled for legacy
        # worksheets. When enabled, it deliberately reuses the robot's
        # mass-derived aspiratable-volume cache, so the spreadsheet continues
        # to have one source of truth for every reagent's starting liquid.
        source_volume_check_value = str(
            header_dict.get(
                'auto_source_volume_check',
                'off'
            )
        ).strip().lower()

        source_volume_check_value = (
            source_volume_check_value
            .replace('-', '_')
            .replace(' ', '_')
        )

        source_volume_check_aliases = {
            '': 'off',
            'off': 'off',
            'none': 'off',
            'disabled': 'off',
            'no': 'off',
            'false': 'off',
            '0': 'off',
            'required': 'required',
            'on': 'required',
            'enabled': 'required',
            'yes': 'required',
            'true': 'required',
            '1': 'required'
        }

        if source_volume_check_value not in source_volume_check_aliases:
            raise ValueError(
                "Header value auto_source_volume_check must be off or "
                "required. Received: "
                f"{source_volume_check_value!r}."
            )

        source_volume_check = source_volume_check_aliases[
            source_volume_check_value
        ]
        self.robo_params['auto_source_volume_check'] = source_volume_check

        def parse_source_reserve_volume_setting():
            raw_value = str(
                header_dict.get('auto_source_reserve_volume_uL', 0.0)
            ).strip()

            try:
                parsed_value = float(raw_value)
            except (TypeError, ValueError):
                raise ValueError(
                    "Header value auto_source_reserve_volume_uL must be a "
                    "finite, nonnegative volume "
                    f"in uL. Received: {raw_value!r}."
                )

            if not math.isfinite(parsed_value) or parsed_value < 0.0:
                raise ValueError(
                    "Header value auto_source_reserve_volume_uL must be a "
                    "finite, nonnegative volume "
                    f"in uL. Received: {raw_value!r}."
                )

            return float(parsed_value)

        self.robo_params['auto_source_reserve_volume_uL'] = (
            parse_source_reserve_volume_setting()
        )

        print(
            "<<controller>> Auto source-volume preflight: "
            f"{source_volume_check}"
        )

        if source_volume_check == 'required':
            print(
                "<<controller>> Auto source-volume reserve beyond robot "
                "dead volume: "
                f"{self.robo_params['auto_source_reserve_volume_uL']:g} uL"
            )

        # Optional compatibility shim for the deployed Raspberry Pi. The Pi
        # currently uses older tube tare values that are 0.3 g too low. A
        # positive value makes its old calculation reproduce the corrected
        # liquid mass, while keeping the spreadsheet/controller record as the
        # true measured tube-plus-solution mass.
        pi_tare_offset_value = str(
            header_dict.get('pi_legacy_tare_offset_g', 0.0)
        ).strip()

        try:
            pi_tare_offset_g = float(pi_tare_offset_value)
        except (TypeError, ValueError):
            raise ValueError(
                "Header value pi_legacy_tare_offset_g must be a finite, "
                "nonnegative mass in g. "
                f"Received: {pi_tare_offset_value!r}."
            )

        if (
            not math.isfinite(pi_tare_offset_g)
            or pi_tare_offset_g < 0.0
        ):
            raise ValueError(
                "Header value pi_legacy_tare_offset_g must be a finite, "
                "nonnegative mass in g. "
                f"Received: {pi_tare_offset_value!r}."
            )

        self.robo_params['pi_legacy_tare_offset_g'] = pi_tare_offset_g

        if pi_tare_offset_g > 0.0:
            print(
                "<<controller>> Applying Raspberry Pi legacy tube-tare "
                f"payload correction: -{pi_tare_offset_g:g} g"
            )

        self.robo_params['max_iterations'] = int(
            header_dict['max_iterations']
        )

        self.robo_params['initial_data'] = int(
            header_dict['initial_data']
        )

        # Optional Auto setting. Defaults to 3 replicates for backward
        # compatibility with older Header sheets.
        num_duplicates_value = str(
            header_dict.get(
                'num_duplicates',
                ''
            )
        ).strip()

        if num_duplicates_value:
            self.robo_params['num_duplicates'] = int(
                num_duplicates_value
            )

        else:
            self.robo_params['num_duplicates'] = 3

        if self.robo_params['num_duplicates'] < 1:
            raise ValueError(
                "Header value num_duplicates must be at least 1."
            )

        # Optional Auto setting. Defaults to False for backward compatibility.
        # Inputs such as true, TRUE, yes, y, and 1 enable true-zero search.
        allow_true_zero_value = str(
            header_dict.get(
                'allow_true_zero',
                ''
            )
        ).strip().lower()

        self.robo_params['allow_true_zero'] = (
            allow_true_zero_value
            in [
                '1',
                'true',
                'yes',
                'y'
            ]
        )

        # The singular acquisition_mode remains the legacy interface. New
        # portfolio-enabled Header sheets must make the inactive interface
        # explicit with ``off`` so an experiment cannot start from an
        # ambiguous mixture of single-mode and portfolio settings.
        acquisition_mode_value = str(
            header_dict.get(
                'acquisition_mode',
                'exploit'
            )
        ).strip().lower()

        acquisition_mode_value = (
            acquisition_mode_value
            .replace('-', '_')
            .replace(' ', '_')
        )

        acquisition_mode_aliases = {
            '': 'exploit',
            'default': 'exploit',
            'exploit': 'exploit',
            'exploitation': 'exploit',
            'target': 'exploit',
            'target_distance': 'exploit',
            'closest_to_target': 'exploit',

            'explore': 'explore',
            'exploration': 'explore',
            'uncertainty': 'explore',
            'maximum_uncertainty': 'explore',
            'max_uncertainty': 'explore',
            'maximum_variance': 'explore',
            'max_variance': 'explore',

            'balanced': 'balanced',
            'balance': 'balanced',
            'hybrid': 'balanced',
            'straddle': 'balanced',
            'target_straddle': 'balanced',

            'target_ei': 'target_ei',
            'ei': 'target_ei',
            'expected_improvement': 'target_ei',
            'target_expected_improvement': 'target_ei',

            # ``off`` is a Header-interface sentinel, not an optimizer
            # acquisition mode. It makes the active configuration visible in
            # the spreadsheet when acquisition_modes is being used.
            'off': 'off',
            'none': 'off',
            'disabled': 'off'
        }

        if (
            acquisition_mode_value
            not in acquisition_mode_aliases
        ):
            raise ValueError(
                "Header value acquisition_mode must be one of: off, "
                "exploit, explore, balanced, or target_ei. "
                f"Received: {acquisition_mode_value!r}."
            )

        singular_acquisition_mode = acquisition_mode_aliases[
            acquisition_mode_value
        ]

        # acquisition_modes is an optional ordered portfolio interface. It is
        # deliberately semicolon-separated rather than comma-separated so
        # spreadsheet locale formatting cannot be confused with a list.
        # Older sheets lacking this field retain exact singular behavior.
        acquisition_modes_present = 'acquisition_modes' in header_dict
        acquisition_modes_value = str(
            header_dict.get('acquisition_modes', '')
        ).strip().lower()
        acquisition_modes_value = (
            acquisition_modes_value
            .replace('-', '_')
            .replace(' ', '_')
        )

        if acquisition_modes_value == 'core3':
            portfolio_acquisition_modes = [
                'exploit',
                'explore',
                'balanced'
            ]
        elif acquisition_modes_value in ['', 'off', 'none', 'disabled']:
            portfolio_acquisition_modes = []
        else:
            portfolio_acquisition_modes = []

            for raw_mode in acquisition_modes_value.split(';'):
                normalized_mode = raw_mode.strip().lower()
                normalized_mode = (
                    normalized_mode
                    .replace('-', '_')
                    .replace(' ', '_')
                )

                if normalized_mode not in acquisition_mode_aliases:
                    raise ValueError(
                        "Header value acquisition_modes contains an "
                        "unsupported mode. Use canonical modes exploit, "
                        "explore, balanced, target_ei, or the standalone "
                        "alias core3. Received: "
                        f"{raw_mode!r}."
                    )

                canonical_mode = acquisition_mode_aliases[normalized_mode]

                if canonical_mode == 'off':
                    raise ValueError(
                        "Header acquisition_modes may be off only as the "
                        "complete field value, not as one portfolio member."
                    )

                portfolio_acquisition_modes.append(canonical_mode)

        if len(portfolio_acquisition_modes) != len(
            set(portfolio_acquisition_modes)
        ):
            raise ValueError(
                "Header acquisition_modes must not repeat a canonical "
                "acquisition mode. Replicates are controlled only by "
                "num_duplicates."
            )

        if not acquisition_modes_present:
            if singular_acquisition_mode == 'off':
                raise ValueError(
                    "Header acquisition_mode is off, but acquisition_modes "
                    "is not present. Select one singular mode or add an "
                    "active acquisition_modes portfolio."
                )

            resolved_acquisition_modes = [singular_acquisition_mode]
            using_acquisition_portfolio = False

        elif singular_acquisition_mode == 'off' and portfolio_acquisition_modes:
            resolved_acquisition_modes = portfolio_acquisition_modes
            using_acquisition_portfolio = True

        elif singular_acquisition_mode != 'off' and not portfolio_acquisition_modes:
            resolved_acquisition_modes = [singular_acquisition_mode]
            using_acquisition_portfolio = False

        elif singular_acquisition_mode == 'off':
            raise ValueError(
                "Header acquisition_mode and acquisition_modes are both off. "
                "Activate exactly one acquisition interface."
            )

        else:
            raise ValueError(
                "Header acquisition_mode and acquisition_modes are both "
                "active. Set the unused interface to off."
            )

        # OptimizationModel retains a singular active mode for scoring one
        # candidate at a time. Portfolio orchestration switches that mode only
        # while evaluating immutable pre-batch selection records.
        self.robo_params['acquisition_mode'] = (
            resolved_acquisition_modes[0]
        )
        self.robo_params['acquisition_modes'] = list(
            resolved_acquisition_modes
        )
        self.robo_params['using_acquisition_portfolio'] = (
            using_acquisition_portfolio
        )

        portfolio_min_distance_value = str(
            header_dict.get('portfolio_min_distance', '0.05')
        ).strip()

        # The numeric threshold remains available for advanced users, while
        # these spreadsheet-friendly labels make the portfolio-diversity
        # setting understandable in a Header dropdown.  The stored parameter
        # is always the corresponding normalized RMS distance.
        portfolio_distance_aliases = {
            'none': 0.00,
            'modest': 0.05,
            'strong': 0.10,
            'very_strong': 0.15,
            'very strong': 0.15,
            'default': 0.05
        }
        normalized_portfolio_distance_value = (
            portfolio_min_distance_value.lower()
            .replace('-', '_')
            .replace(' ', '_')
        )

        if normalized_portfolio_distance_value in portfolio_distance_aliases:
            portfolio_min_distance = portfolio_distance_aliases[
                normalized_portfolio_distance_value
            ]
        else:
            try:
                portfolio_min_distance = float(
                    portfolio_min_distance_value
                )
            except ValueError:
                raise ValueError(
                    "Header portfolio_min_distance must be one of: none, "
                    "modest, strong, very_strong, or a finite number between "
                    "0 and 1 in normalized RMS recipe space. Received: "
                    f"{portfolio_min_distance_value!r}."
                )

        if (
            not math.isfinite(portfolio_min_distance)
            or portfolio_min_distance < 0.0
            or portfolio_min_distance > 1.0
        ):
            raise ValueError(
                "Header portfolio_min_distance must be a finite value "
                "between 0 and 1 in normalized RMS recipe space. Received: "
                f"{portfolio_min_distance!r}."
            )

        self.robo_params['portfolio_min_distance'] = (
            portfolio_min_distance
        )

        # Target-aware expected improvement needs a statistically defensible
        # condition-level incumbent. A single well can train the GP, but it
        # cannot establish replicate agreement or a replicate standard
        # deviation. Reject this configuration while parsing the Header so the
        # protocol cannot reach simulation or robot execution with an
        # ill-defined target-EI incumbent policy.
        if (
            'target_ei' in self.robo_params['acquisition_modes']
            and self.robo_params['num_duplicates'] < 2
        ):
            raise ValueError(
                "Header acquisition_mode(s) target_ei requires "
                "num_duplicates to be at least 2 so its incumbent is based "
                "on a replicate-validated condition."
            )
        elif self.robo_params['num_duplicates'] < 2:
            print(
                "<<controller warning>> num_duplicates is 1: valid single "
                "measurements may train the GP, but no condition can qualify "
                "for replicate-validated target stopping. Auto will continue "
                "until max_iterations unless stopped manually."
            )

        print(
            "<<controller>> Auto acquisition mode: "
            f"{self.robo_params['acquisition_mode']}"
        )

        print(
            "<<controller>> Auto acquisition portfolio: "
            + ';'.join(self.robo_params['acquisition_modes'])
            + " (active)"
            if self.robo_params['using_acquisition_portfolio']
            else "<<controller>> Auto acquisition portfolio: off"
        )

        # Optional general Auto plotting setting. The value is normalized so
        # users may enter spaces or hyphens without causing an avoidable error.
        auto_plot_profile_value = str(
            header_dict.get(
                'auto_plot_profile',
                'standard'
            )
        ).strip().lower()

        auto_plot_profile_value = (
            auto_plot_profile_value
            .replace('-', '_')
            .replace(' ', '_')
        )

        auto_plot_profile_aliases = {
            '': 'standard',
            'all': 'standard',
            'default': 'standard',
            'on': 'standard',
            'true': 'standard',
            'yes': 'standard',
            '1': 'standard',
            'standard': 'standard',
            'final': 'final_only',
            'final_only': 'final_only',
            'none': 'off',
            'disabled': 'off',
            'disable': 'off',
            'off': 'off',
            'false': 'off',
            'no': 'off',
            '0': 'off'
        }

        if (
            auto_plot_profile_value
            not in auto_plot_profile_aliases
        ):
            raise ValueError(
                "Header value auto_plot_profile must be one of: "
                "standard, final_only, or off. "
                f"Received: {auto_plot_profile_value!r}."
            )

        self.robo_params['auto_plot_profile'] = (
            auto_plot_profile_aliases[
                auto_plot_profile_value
            ]
        )

        print(
            "<<controller>> Auto plot profile: "
            f"{self.robo_params['auto_plot_profile']}"
        )
    
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
       
    def _get_2d_gpr_feasibility_overlay_data(
        self,
        model,
        x_values,
        y_values
    ):
        '''
        Calculates physical-executability masks for a 2D GP heatmap grid.

        The ordinary GP heatmaps intentionally show the complete configured
        concentration rectangle. This helper supplies separate diagnostic
        overlay plots with the portions that cannot be executed by the robot
        marked explicitly. It evaluates raw concentration-grid values rather
        than applying true-zero repair, so the non-executable interval between
        0 and 5 uL remains visible.

        A grid point is excluded when a variable-reagent transfer is below
        5 uL (except an exact zero when true-zero search is enabled), water
        top-off is between 0 and 5 uL, or the recipe overflows the configured
        final volume. Exact-zero water remains feasible.

        params:
            OptimizationModel model:
                Optimizer providing stock concentrations and volume settings.

            np.ndarray x_values:
                Physical concentrations for the first variable reagent.

            np.ndarray y_values:
                Physical concentrations for the second variable reagent.

        returns:
            dict:
                Boolean feasibility masks, water-volume grid, and
                reagent-specific transfer-volume grids. All arrays use rows
                for y_values and columns for x_values.
        '''
        reagent_names = list(self.variable_reagents)

        if len(reagent_names) != 2:
            raise ValueError(
                "2D GP feasibility overlays require exactly two variable "
                "reagents."
            )

        if model is None:
            raise ValueError(
                "2D GP feasibility overlays require an optimizer model."
            )

        x_values = np.asarray(x_values, dtype=float).reshape(-1)
        y_values = np.asarray(y_values, dtype=float).reshape(-1)

        if x_values.size == 0 or y_values.size == 0:
            raise ValueError(
                "2D GP feasibility overlays require nonempty axis values."
            )

        total_volume = float(getattr(model, 'total_volume', np.nan))
        fixed_reagent_volumes = getattr(
            model,
            'fixed_reagent_volumes',
            None
        )
        stock_concentration_getter = getattr(
            model,
            '_get_variable_reagent_stock_conc',
            None
        )

        if (
            not np.isfinite(total_volume)
            or total_volume <= 0
            or not isinstance(fixed_reagent_volumes, dict)
            or not callable(stock_concentration_getter)
        ):
            raise ValueError(
                "2D GP feasibility overlays require total volume, fixed "
                "reagent volumes, and variable-reagent stock concentrations."
            )

        fixed_volume_total = float(
            sum(
                float(volume)
                for volume in fixed_reagent_volumes.values()
            )
        )

        x_grid, y_grid = np.meshgrid(
            x_values,
            y_values,
            indexing='xy'
        )
        concentration_grids = [x_grid, y_grid]
        transfer_volume_grids = {}
        variable_transfer_infeasible = np.zeros(
            x_grid.shape,
            dtype=bool
        )
        volume_tolerance = 1e-9
        true_zero_allowed = bool(
            getattr(model, 'allow_true_zero', False)
        )

        for reagent_name, concentration_grid in zip(
            reagent_names,
            concentration_grids
        ):
            stock_concentration = float(
                stock_concentration_getter(reagent_name)
            )

            if math.isclose(
                stock_concentration,
                0.0,
                rel_tol=0,
                abs_tol=volume_tolerance
            ):
                raise ValueError(
                    "2D GP feasibility overlay cannot calculate transfer "
                    f"volume for {reagent_name}: stock concentration is 0."
                )

            transfer_volume_grid = (
                concentration_grid
                * total_volume
                / stock_concentration
            )
            transfer_volume_grids[reagent_name] = transfer_volume_grid

            is_exact_zero = np.isclose(
                transfer_volume_grid,
                0.0,
                rtol=0,
                atol=volume_tolerance
            )
            below_executable_minimum = (
                transfer_volume_grid < 5.0 - volume_tolerance
            )

            if true_zero_allowed:
                variable_transfer_infeasible |= (
                    below_executable_minimum & ~is_exact_zero
                )
            else:
                variable_transfer_infeasible |= (
                    below_executable_minimum
                )

        variable_volume_total = sum(
            transfer_volume_grids.values()
        )
        water_volume_grid = (
            total_volume
            - fixed_volume_total
            - variable_volume_total
        )
        water_is_exact_zero = np.isclose(
            water_volume_grid,
            0.0,
            rtol=0,
            atol=volume_tolerance
        )
        overflow = water_volume_grid < -volume_tolerance
        water_transfer_infeasible = (
            (water_volume_grid > volume_tolerance)
            & (water_volume_grid < 5.0 - volume_tolerance)
        )

        return {
            'x_values': x_values,
            'y_values': y_values,
            'infeasible': (
                variable_transfer_infeasible
                | water_transfer_infeasible
                | overflow
            ),
            'variable_transfer_infeasible': (
                variable_transfer_infeasible
            ),
            'water_transfer_infeasible': water_transfer_infeasible,
            'overflow': overflow,
            'water_volume_uL': water_volume_grid,
            'water_is_exact_zero': water_is_exact_zero,
            'transfer_volume_uL_by_reagent': transfer_volume_grids
        }

    def plot_2D_GPR(
        self,
        model,
        batch_number=None
    ):
        '''
        Saves square 2D GP prediction and uncertainty heatmaps.

        The scientific plotting panel is physically square. The colorbar
        remains outside that panel, so the complete saved image may be slightly
        wider than it is tall without stretching the underlying heatmap.

        Both axes use the configured executable concentration bounds for the
        two variable reagents. The physical x and y ranges are not forced to
        use identical data-unit scaling when the reagents have different
        concentration ranges.

        An explicit batch number may be supplied by the general Auto plotting
        coordinator. When omitted, the current self.batch_num value is used for
        backward compatibility.

        params:
            OptimizationModel model:
                Auto optimizer model containing two-dimensional prediction and
                uncertainty grids.

            int or None batch_number:
                Batch number used in titles and output filenames.

        returns:
            list:
                Paths of successfully generated GP heatmap files.
        '''
        generated_plot_paths = []

        if len(self.variable_reagents) != 2:
            print(
                "<<controller>> skipping 2D GP plots because there are not "
                "exactly two variable reagents"
            )
            return generated_plot_paths

        if (
            model is None
            or not hasattr(model, 'predictions')
            or model.predictions is None
        ):
            print(
                "<<controller>> skipping 2D GP plots because model "
                "predictions are not available"
            )
            return generated_plot_paths

        if batch_number is None:
            batch_number = getattr(
                self,
                'batch_num',
                0
            )

        try:
            plot_batch_number = int(
                batch_number
            )

        except (TypeError, ValueError, OverflowError):
            print(
                "<<controller warning>> skipping 2D GP plots because the "
                f"batch number is invalid: {batch_number!r}"
            )
            return generated_plot_paths

        if plot_batch_number < 0:
            print(
                "<<controller warning>> skipping 2D GP plots because the "
                f"batch number is negative: {plot_batch_number}"
            )
            return generated_plot_paths

        prediction_array = np.asarray(
            model.predictions,
            dtype=float
        )

        if (
            prediction_array.ndim != 2
            or prediction_array.size == 0
        ):
            print(
                "<<controller warning>> skipping 2D GP plots because the "
                "prediction grid is not a nonempty two-dimensional array"
            )
            return generated_plot_paths

        def _read_configured_bound(
            raw_bounds,
            dimension_index,
            reagent_name
        ):
            '''
            Reads one finite configured Auto concentration bound.
            '''
            bound_helper = getattr(
                self,
                '_get_auto_design_bound_value',
                None
            )

            if callable(bound_helper):
                return bound_helper(
                    raw_bounds=raw_bounds,
                    dimension_index=dimension_index,
                    reagent_name=reagent_name
                )

            try:
                if isinstance(raw_bounds, dict):
                    return float(
                        raw_bounds[reagent_name]
                    )

                return float(
                    raw_bounds[dimension_index]
                )

            except Exception:
                return np.nan

        x_minimum = _read_configured_bound(
            raw_bounds=getattr(
                self,
                'min_conc',
                None
            ),
            dimension_index=0,
            reagent_name=self.variable_reagents[0]
        )

        x_maximum = _read_configured_bound(
            raw_bounds=getattr(
                self,
                'max_conc',
                None
            ),
            dimension_index=0,
            reagent_name=self.variable_reagents[0]
        )

        y_minimum = _read_configured_bound(
            raw_bounds=getattr(
                self,
                'min_conc',
                None
            ),
            dimension_index=1,
            reagent_name=self.variable_reagents[1]
        )

        y_maximum = _read_configured_bound(
            raw_bounds=getattr(
                self,
                'max_conc',
                None
            ),
            dimension_index=1,
            reagent_name=self.variable_reagents[1]
        )

        bounds_are_valid = (
            np.isfinite(x_minimum)
            and np.isfinite(x_maximum)
            and x_maximum > x_minimum
            and np.isfinite(y_minimum)
            and np.isfinite(y_maximum)
            and y_maximum > y_minimum
        )

        if not bounds_are_valid:
            print(
                "<<controller warning>> skipping 2D GP plots because valid "
                "executable concentration bounds were not available"
            )
            return generated_plot_paths

        x_values = np.linspace(
            x_minimum,
            x_maximum,
            prediction_array.shape[1]
        )

        y_values = np.linspace(
            y_minimum,
            y_maximum,
            prediction_array.shape[0]
        )

        feasibility_overlay_data = None
        try:
            # The GP itself is evaluated on its established plotting grid.
            # Feasibility is purely geometric, so a denser independent grid
            # produces smooth physical-boundary overlays without changing any
            # GP predictions or the original heatmap pixels.
            feasibility_x_values = np.linspace(
                x_minimum,
                x_maximum,
                max(401, prediction_array.shape[1])
            )
            feasibility_y_values = np.linspace(
                y_minimum,
                y_maximum,
                max(401, prediction_array.shape[0])
            )
            feasibility_overlay_data = (
                self._get_2d_gpr_feasibility_overlay_data(
                    model=model,
                    x_values=feasibility_x_values,
                    y_values=feasibility_y_values
                )
            )
        except Exception as feasibility_error:
            # The established GP heatmaps remain available if optional
            # diagnostic overlay inputs are incomplete in an older workflow.
            print(
                "<<controller warning>> skipping 2D GP feasibility-overlay "
                f"plots: {feasibility_error}"
            )

        font_helper = getattr(
            self,
            '_get_auto_design_plot_font_sizes',
            None
        )

        if callable(font_helper):
            font_sizes = font_helper()

        else:
            font_sizes = {
                'title': 13.75,
                'axis_label': 12.5,
                'tick_label': 12.5
            }

        def _format_2d_gpr_axis(ax):
            '''
            Applies shared laboratory formatting to one GP heatmap axis.
            '''
            ax.set_xlabel(
                f"{self.variable_reagents[0]} (mM)",
                fontsize=font_sizes['axis_label']
            )

            ax.set_ylabel(
                f"{self.variable_reagents[1]} (mM)",
                fontsize=font_sizes['axis_label']
            )

            ax.set_xlim(
                x_minimum,
                x_maximum
            )

            ax.set_ylim(
                y_minimum,
                y_maximum
            )

            ax.tick_params(
                axis='both',
                which='both',
                direction='out',
                top=False,
                right=False,
                width=0.9,
                labelsize=font_sizes['tick_label']
            )

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.9)
                spine.set_color('0.2')

            square_helper = getattr(
                self,
                '_apply_auto_design_square_box_aspect',
                None
            )

            if callable(square_helper):
                square_helper(ax)

            else:
                set_box_aspect = getattr(
                    ax,
                    'set_box_aspect',
                    None
                )

                if callable(set_box_aspect):
                    set_box_aspect(1.0)

        def _save_2d_gpr_heatmap(
            heatmap_array,
            cmap_name,
            plot_title,
            colorbar_label,
            plot_filename,
            plot_description,
            feasibility_overlay=None,
            target_contour_nm=None
        ):
            '''
            Renders and saves one square GP heatmap.

            returns:
                str:
                    Saved plot path.
            '''
            fig, ax = plt.subplots(
                figsize=(6.4, 5.8),
                dpi=300
            )

            heatmap_mesh = ax.pcolormesh(
                x_values,
                y_values,
                heatmap_array,
                cmap=cmap_name,
                shading='auto'
            )

            feasibility_legend_handles = []

            if feasibility_overlay is not None:
                feasibility_x_values = np.asarray(
                    feasibility_overlay['x_values'],
                    dtype=float
                )
                feasibility_y_values = np.asarray(
                    feasibility_overlay['y_values'],
                    dtype=float
                )
                infeasible_mask = np.asarray(
                    feasibility_overlay['infeasible'],
                    dtype=bool
                )

                expected_feasibility_shape = (
                    feasibility_y_values.size,
                    feasibility_x_values.size
                )

                if infeasible_mask.shape != expected_feasibility_shape:
                    raise ValueError(
                        "Feasibility overlay shape does not match its "
                        "physical concentration axes."
                    )

                if np.any(infeasible_mask):
                    # Contour fill on a dense independent feasibility grid
                    # avoids the stair-step edge and small cell gaps produced
                    # by a coarse binary pcolormesh beside water boundaries.
                    ax.contourf(
                        feasibility_x_values,
                        feasibility_y_values,
                        infeasible_mask.astype(float),
                        levels=[0.5, 1.5],
                        colors=['0.70'],
                        alpha=0.55,
                        antialiased=True,
                        corner_mask=False,
                        zorder=2
                    )
                    feasibility_legend_handles.append(
                        mpatches.Patch(
                            facecolor='0.70',
                            alpha=0.55,
                            label=(
                                'Excluded: overflow or non-executable '
                                'transfer'
                            )
                        )
                    )

                def _grid_spans_contour_level(grid, level):
                    finite_values = np.asarray(
                        grid,
                        dtype=float
                    )
                    finite_values = finite_values[
                        np.isfinite(finite_values)
                    ]

                    return (
                        finite_values.size > 0
                        and np.min(finite_values) < level
                        and np.max(finite_values) > level
                    )

                water_volume_grid = feasibility_overlay[
                    'water_volume_uL'
                ]

                if _grid_spans_contour_level(water_volume_grid, 0.0):
                    ax.contour(
                        feasibility_x_values,
                        feasibility_y_values,
                        water_volume_grid,
                        levels=[0.0],
                        colors='#0072B2',
                        linewidths=1.35,
                        linestyles='solid',
                        zorder=4
                    )
                    water_zero_handle, = ax.plot(
                        [],
                        [],
                        color='#0072B2',
                        linewidth=1.35,
                        label='Water = 0 uL boundary'
                    )
                    feasibility_legend_handles.append(
                        water_zero_handle
                    )

                if _grid_spans_contour_level(water_volume_grid, 5.0):
                    ax.contour(
                        feasibility_x_values,
                        feasibility_y_values,
                        water_volume_grid,
                        levels=[5.0],
                        colors='#D55E00',
                        linewidths=1.35,
                        linestyles='dashed',
                        zorder=4
                    )
                    water_minimum_handle, = ax.plot(
                        [],
                        [],
                        color='#D55E00',
                        linewidth=1.35,
                        linestyle='dashed',
                        label='Water = 5 uL boundary'
                    )
                    feasibility_legend_handles.append(
                        water_minimum_handle
                    )

                for reagent_i, reagent_name in enumerate(
                    self.variable_reagents
                ):
                    transfer_volume_grid = feasibility_overlay[
                        'transfer_volume_uL_by_reagent'
                    ][reagent_name]

                    if _grid_spans_contour_level(
                        transfer_volume_grid,
                        5.0
                    ):
                        reagent_color = (
                            '#009E73'
                            if reagent_i == 0
                            else '#CC79A7'
                        )
                        ax.contour(
                            feasibility_x_values,
                            feasibility_y_values,
                            transfer_volume_grid,
                            levels=[5.0],
                            colors=reagent_color,
                            linewidths=1.15,
                            linestyles='dotted',
                            zorder=4
                        )
                        reagent_minimum_handle, = ax.plot(
                            [],
                            [],
                            color=reagent_color,
                            linewidth=1.15,
                            linestyle='dotted',
                            label=(
                                f'{reagent_name} = 5 uL boundary'
                            )
                        )
                        feasibility_legend_handles.append(
                            reagent_minimum_handle
                        )

                if (
                    target_contour_nm is not None
                    and _grid_spans_contour_level(
                        heatmap_array,
                        target_contour_nm
                    )
                ):
                    ax.contour(
                        x_values,
                        y_values,
                        heatmap_array,
                        levels=[target_contour_nm],
                        colors='#000000',
                        linewidths=1.1,
                        linestyles='dashdot',
                        zorder=5
                    )
                    target_handle, = ax.plot(
                        [],
                        [],
                        color='#000000',
                        linewidth=1.1,
                        linestyle='dashdot',
                        label=(
                            f'Target = {target_contour_nm:.0f} nm'
                        )
                    )
                    feasibility_legend_handles.append(target_handle)

            _format_2d_gpr_axis(
                ax
            )

            if feasibility_overlay is None:
                ax.set_title(
                    plot_title,
                    fontsize=font_sizes['title'],
                    fontweight='normal',
                    pad=14
                )
            else:
                fig.suptitle(
                    plot_title,
                    fontsize=font_sizes['title'],
                    fontweight='normal',
                    y=0.975
                )

            colorbar = fig.colorbar(
                heatmap_mesh,
                ax=ax,
                fraction=0.046,
                pad=0.05
            )

            colorbar.set_label(
                colorbar_label,
                fontsize=font_sizes['axis_label']
            )

            colorbar.ax.tick_params(
                labelsize=font_sizes['tick_label'],
                width=0.9
            )

            if len(feasibility_legend_handles) > 0:
                fig.legend(
                    feasibility_legend_handles,
                    [
                        handle.get_label()
                        for handle in feasibility_legend_handles
                    ],
                    loc='upper center',
                    bbox_to_anchor=(0.5, 0.925),
                    ncol=2,
                    frameon=False,
                    fontsize=font_sizes['tick_label'] * 0.68,
                    handlelength=1.7,
                    columnspacing=0.9
                )

            fig.subplots_adjust(
                left=0.15,
                right=0.86,
                bottom=0.14,
                top=(
                    0.75
                    if feasibility_overlay is not None
                    else 0.88
                )
            )

            full_plot_path = os.path.join(
                self.plot_path,
                plot_filename
            )

            fig.savefig(
                full_plot_path,
                bbox_inches='tight'
            )

            plt.close(
                fig
            )

            print(
                f"<<controller>> saved {plot_description} to "
                f"{full_plot_path}"
            )

            return full_plot_path

        os.makedirs(
            self.plot_path,
            exist_ok=True
        )

        prediction_plot_path = _save_2d_gpr_heatmap(
            heatmap_array=prediction_array,
            cmap_name='inferno',
            plot_title=(
                rf'2D GP Predicted $\lambda_{{\max}}$ '
                f'After Batch {plot_batch_number}'
            ),
            colorbar_label=(
                r'Predicted $\lambda_{\max}$ (nm)'
            ),
            plot_filename=(
                f'gpr_predictions_batch_'
                f'{plot_batch_number}.png'
            ),
            plot_description='2D GP prediction plot'
        )

        generated_plot_paths.append(
            prediction_plot_path
        )

        target_contour_nm = None
        try:
            target_contour_candidate = float(
                self.robo_params['target']
            )

            if np.isfinite(target_contour_candidate):
                target_contour_nm = target_contour_candidate
        except (AttributeError, KeyError, TypeError, ValueError):
            pass

        if feasibility_overlay_data is not None:
            prediction_feasibility_plot_path = _save_2d_gpr_heatmap(
                heatmap_array=prediction_array,
                cmap_name='inferno',
                plot_title=(
                    rf'2D GP Predicted $\lambda_{{\max}}$ with '
                    f'Feasibility Overlay After Batch {plot_batch_number}'
                ),
                colorbar_label=(
                    r'Predicted $\lambda_{\max}$ (nm)'
                ),
                plot_filename=(
                    f'gpr_predictions_feasibility_batch_'
                    f'{plot_batch_number}.png'
                ),
                plot_description=(
                    '2D GP prediction feasibility-overlay plot'
                ),
                feasibility_overlay=feasibility_overlay_data,
                target_contour_nm=target_contour_nm
            )

            generated_plot_paths.append(
                prediction_feasibility_plot_path
            )

        if (
            not hasattr(model, 'prediction_uncertainty')
            or model.prediction_uncertainty is None
        ):
            print(
                "<<controller>> skipping 2D GP uncertainty plot because "
                "prediction_uncertainty is not available"
            )
            return generated_plot_paths

        uncertainty_array = np.asarray(
            model.prediction_uncertainty,
            dtype=float
        )

        if uncertainty_array.shape != prediction_array.shape:
            print(
                "<<controller warning>> skipping 2D GP uncertainty plot "
                "because its grid shape does not match the prediction grid"
            )
            return generated_plot_paths

        uncertainty_plot_path = _save_2d_gpr_heatmap(
            heatmap_array=uncertainty_array,
            cmap_name='viridis',
            plot_title=(
                '2D GP Predictive Uncertainty '
                f'After Batch {plot_batch_number}'
            ),
            colorbar_label='GP predictive SD (nm)',
            plot_filename=(
                f'gpr_uncertainty_batch_'
                f'{plot_batch_number}.png'
            ),
            plot_description='2D GP uncertainty plot'
        )

        generated_plot_paths.append(
            uncertainty_plot_path
        )

        if feasibility_overlay_data is not None:
            uncertainty_feasibility_plot_path = _save_2d_gpr_heatmap(
                heatmap_array=uncertainty_array,
                cmap_name='viridis',
                plot_title=(
                    '2D GP Predictive Uncertainty with Feasibility Overlay '
                    f'After Batch {plot_batch_number}'
                ),
                colorbar_label='GP predictive SD (nm)',
                plot_filename=(
                    f'gpr_uncertainty_feasibility_batch_'
                    f'{plot_batch_number}.png'
                ),
                plot_description=(
                    '2D GP uncertainty feasibility-overlay plot'
                ),
                feasibility_overlay=feasibility_overlay_data
            )

            generated_plot_paths.append(
                uncertainty_feasibility_plot_path
            )

        return generated_plot_paths
    
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
        
        platereader_input_first_usable = str(usable_rows.iloc[0]).strip().upper()

        # Preserve the user-selected 96-well plate starting well before translating
        # it into the internal platereader4/platereader7 coordinate system. Auto mode
        # uses this original well, such as A4, for the pre-run well capacity check.
        self.robo_params['platereader_input_first_usable'] = platereader_input_first_usable

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
    
        self.translate_wellmap()
    

    
        
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
        Runs through closing procedure with robot and uploads all data to Google Drive
        
        Postconditions:    
            - Log files have been written to self.out_path
            - Connection has been closed  
        '''
        self.save()
        
        
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

    def _get_pi_compatible_reagent_payload(self):
        '''
        Returns a copy of reagent data adjusted only for the deployed
        Raspberry Pi's legacy tube-tare calculation.

        ``reagent_info.mass`` remains the actual measured tube-plus-solution
        mass everywhere on the controller. When a positive
        ``pi_legacy_tare_offset_g`` is configured, the controller subtracts
        it from each liquid-reagent mass only in this outbound payload. If the
        Pi tare is lower than the corrected tare by the same amount, its
        existing ``mass - legacy_tare`` calculation then yields the corrected
        liquid mass without changing Pi code.
        '''
        reagent_df = self.robo_params['reagent_df']
        payload_df = reagent_df.copy(deep=True)
        tare_offset_g = float(
            self.robo_params.get('pi_legacy_tare_offset_g', 0.0)
        )

        if tare_offset_g > 0.0:
            measured_masses_g = pd.to_numeric(
                payload_df['mass'],
                errors='raise'
            )
            payload_df['mass'] = measured_masses_g - tare_offset_g

        return payload_df.reset_index().to_dict()

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
                self._get_pi_compatible_reagent_payload(), self.my_ip,
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

    def execute_protocol_df(self, model=None):
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
                self._create_plot(row, i, model)
            elif row['op'] == 'print':
                self._execute_print(row,i)
            elif row['op'] == 'scan_until_complete':
                self._scan_until_complete(row,i)
            else:
                raise Exception('invalid operation {}'.format(row['op']))

    def _execute_print(self, row, i):
        print(row['message'])

    def _create_plot(
        self,
        row,
        i,
        model=None
    ):
        '''
        Executes one spreadsheet-controlled, scan-derived plot command.

        Spreadsheet plot rows are reserved for visualizations that are derived
        directly from a named plate-reader scan:

            SINGLE_KIN:
                Generates one kinetics plot for each selected well.

            OVERLAY:
                Generates one spectral overlay for the selected wells.

            MULTI_KIN:
                Generates a multi-well kinetics figure.

        Automatic Auto diagnostics are not generated here. GP prediction and
        uncertainty heatmaps, lambda progress plots, replicate plots,
        dimension-aware design-space plots, and the final Auto report are
        controlled by the Header auto_plot_profile setting and generated by
        _generate_auto_plot_suite() at the scientifically appropriate lifecycle
        stage.

        The legacy 2D_GPR spreadsheet protocol is recognized only for backward
        compatibility. It is ignored before any plate-reader data are loaded,
        preventing stale or duplicate GP heatmaps.

        params:
            pandas.Series row:
                Plot-operation row from self.rxn_df.

            int i:
                Index of the plot-operation row.

            OptimizationModel or None model:
                Retained for backward compatibility with execute_protocol_df().
                Automatic model plots are no longer generated in this method.

        returns:
            None
        '''
        plot_type = str(
            row['plot_protocol']
        ).strip().upper()

        print(
            f"<<controller>> creating plot using protocol: "
            f"{plot_type}"
        )

        # Legacy compatibility only. GP plots must be generated after the GP
        # has incorporated a completed batch, not from a plot row inside the
        # physical scan/transfer protocol.
        if plot_type == '2D_GPR':
            warning_already_printed = getattr(
                self,
                '_legacy_2d_gpr_warning_printed',
                False
            )

            if not warning_already_printed:
                print(
                    "<<controller warning>> spreadsheet plot protocol "
                    "2D_GPR is deprecated and will be ignored. Automatic GP "
                    "prediction and uncertainty plots are now controlled by "
                    "Header auto_plot_profile and generated after the fitted "
                    "model is updated."
                )

                self._legacy_2d_gpr_warning_printed = True

            return

        supported_scan_plot_types = {
            'SINGLE_KIN',
            'OVERLAY',
            'MULTI_KIN'
        }

        if plot_type not in supported_scan_plot_types:
            raise ValueError(
                "Unsupported spreadsheet plot protocol "
                f"{plot_type!r}. Supported scan-derived protocols are: "
                "SINGLE_KIN, OVERLAY, and MULTI_KIN. Automatic Auto plots "
                "are controlled by Header auto_plot_profile."
            )

        wellnames = row[
            self._products
        ][
            row[self._products].astype(bool)
        ].index

        if len(wellnames) == 0:
            raise ValueError(
                f"Plot protocol {plot_type} did not select any product wells "
                f"in protocol row {i}."
            )

        filename = row[
            'plot_filename'
        ]

        scan_filename = row[
            'scan_filename'
        ]

        self._update_cached_locs(
            wellnames
        )

        pr_dict = {
            self._cached_reader_locs[wellname].loc: wellname
            for wellname in wellnames
        }

        # These supported spreadsheet plots genuinely depend on a completed
        # plate-reader scan, so the scan file is loaded only after the plot
        # protocol has been validated.
        df, metadata = self.pr.load_reader_data(
            scan_filename,
            pr_dict
        )

        if plot_type == 'SINGLE_KIN':
            for wellname in wellnames:
                self.plot_single_kin(
                    df,
                    metadata['n_cycles'],
                    wellname,
                    f"{wellname}_{filename}"
                )

        elif plot_type == 'OVERLAY':
            self.plot_LAM_overlay(
                df,
                wellnames,
                filename
            )

        elif plot_type == 'MULTI_KIN':
            self.plot_kin_subplots(
                df,
                metadata['n_cycles'],
                wellnames,
                filename
            )

        return

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

            # Print only when cached volume differs from the expected scan volume.
            # This keeps normal output clean while making tiny bookkeeping differences
            # visible during scan-volume debugging.
            if well in self.tot_vols and entry.vol != self.tot_vols[well]:
                print(
                    f"<<controller>> scan volume difference for {well}: "
                    f"cached={entry.vol}, expected={self.tot_vols[well]}, "
                    f"diff={entry.vol - self.tot_vols[well]}"
                )

            # Allow a tiny tolerance for floating-point/cached-volume artifacts.
            # For example, a well intended to contain 200.0 uL may be tracked as
            # 200.00007 uL after several computed transfer steps. This prevents
            # harmless numeric noise from blocking scans while still catching
            # meaningful volume errors.
            assert (
                well not in self.tot_vols or
                math.isclose(entry.vol, self.tot_vols[well], rel_tol=0, abs_tol=1e-3)
            ), "tried to scan {}, but {} has a bad volume. Vol was {}, but {} is required for a scan".format(
                well, well, entry.vol, self.tot_vols[well]
            )
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

    def _round_transfer_volume(self, vol, decimals=9, artifact_tol=1e-9):
        """
        Clean tiny floating-point artifacts from transfer volumes before sending
        them to the robot.

        This function is intentionally conservative. It only changes a volume
        when the difference between the original value and the rounded value is
        extremely small, which indicates normal binary floating-point noise.

        This prevents values like:
            20.000000000000004 -> 20.0
            10.000000000000002 -> 10.0

        while avoiding unnecessary rounding of meaningful fractional transfer
        volumes that may be needed to preserve the intended final well volume.

        Parameters:
            vol:
                Computed transfer volume.

            decimals:
                Number of decimal places used for artifact cleanup. This is not
                meant to impose robot precision broadly; it is only used to
                identify whether a value is extremely close to a cleaner decimal
                representation.

            artifact_tol:
                Maximum allowed difference between the original value and the
                rounded value for the rounded value to be used. If the difference
                is larger than this tolerance, the original volume is preserved.

        Returns:
            float:
                The cleaned transfer volume if the change is only a tiny
                floating-point artifact; otherwise, the original volume.

        Postconditions:
            - Values that are effectively whole/simple decimal volumes are cleaned.
            - Meaningful fractional transfer volumes are preserved.
            - The returned value is always a float.
        """
        # Convert to float so math.isclose and round behave predictably even if
        # the input comes from a pandas/numpy scalar.
        vol = float(vol)

        # Preserve true zero exactly. Zero-volume transfers should remain zero
        # and should not be affected by rounding logic.
        if vol == 0:
            return 0.0

        # Create a cleaned candidate value. This candidate is only used if it is
        # nearly identical to the original value within artifact_tol.
        rounded_vol = round(vol, decimals)

        # Only return the rounded value when the difference is tiny enough to be
        # considered binary floating-point noise, such as 20.000000000000004.
        if math.isclose(vol, rounded_vol, rel_tol=0, abs_tol=artifact_tol):
            return rounded_vol

        # If rounding would meaningfully change the volume, keep the original.
        # This avoids accumulating small volume shifts across multiple reagents.
        return vol
    
    
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

        # Product columns should contain numeric transfer volumes for each product well.
        # Convert them to numeric values defensively so blanks, strings, or unexpected
        # nonnumeric entries become NaN instead of causing unclear downstream errors.
        #
        # NaN product volumes should not be sent to the robot as transfer commands.
        # Treat NaN as 0 here because a missing product-well volume means there is no
        # meaningful transfer to perform for that well.
        product_volumes = pd.to_numeric(row[self._products], errors='coerce').fillna(0)

        # Treat tiny floating-point artifacts as zero so meaningless near-zero
        # transfer volumes are not sent to the robot. This prevents values like
        # 1e-15 from being interpreted as real transfer steps.
        containers = product_volumes.loc[
            ~product_volumes.apply(
                lambda x: math.isclose(float(x), 0.0, rel_tol=0, abs_tol=1e-9)
            )
        ]

        transfer_steps = [(name, self._round_transfer_volume(vol)) for name, vol in containers.iteritems()]

        if not transfer_steps:
            print(f"<<controller>> skipping transfer from {src}: all destination volumes are 0 uL")
            return
        
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
        pipettable volume. defined here as 5uL.  
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
                if re.fullmatch(reagent+r'C\d*\.\d*', key)]
        containers.sort(key=self._get_conc)
        filtered_conts = [] #this will hold the containers that are diluted enough to be able
        #to transfer without exceeding min_vol
        for cont in containers:
            vol = self._get_transfer_vol(cont,molarity,total_vol,ratio)

            # Allow exactly-minimum transfers and protect against tiny floating-point
            # artifacts around the 5 uL lower limit.
            if vol >= min_vol - 1e-9:
                filtered_conts.append(cont)

                # Allow tiny floating-point artifacts when comparing calculated
                # transfer volume to cached aspiratable volume.
                if vol <= self._cached_reader_locs[cont].aspirable_vol + 1e-9:
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
        return float(re.search(r'C\d*\.\d*$', chem_name).group(0)[1:])

    def _get_reagent(self, chem_name):
        '''
        handy method for getting the reagent from a chemical name  
        The foil of _get_conc  
        params:  
            str chem_name: the chemical name to strip a reagent name from  
        returns:  
            str: the reagent name parsed from the chem_name  
        '''
        
        return chem_name[:re.search(r'C\d*\.\d*$', chem_name).start()]

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
        #print(f'variable reagents: {self.variable_reagents}')
        self.fixed_reagents = self.get_fixed_reagents()
        self.y_shape = len(self.variable_reagents)
        #print(f"y-shape is {self.y_shape}")
        if (
            self.robo_params.get(
                'auto_terminal_verbosity',
                'standard'
            )
            == 'diagnostic'
        ):
            print(self.robo_params['reagent_df'])
        self.run_all_checks()
        self.rxn_df_template = self.rxn_df
        self.reagent_order = self.rxn_df['reagent'].dropna().loc[self.rxn_df['conc'].isna()].unique()
        #print(f'reagent_order: {self.reagent_order}')
        self._clean_template() #moves template data out of the data for rxn_df
        self.experiment_data = pd.DataFrame(columns=[str(reagent) for reagent in self.variable_reagents] + ["Experiment_result"])
        self.num_duplicates = int(self.robo_params.get('num_duplicates', num_duplicates))
        print(f"<<controller>> using {self.num_duplicates} replicate wells per unique Auto recipe")
        self.max_conc = list(self.get_max_conc().values())
        self.min_conc = list(self.get_min_conc().values())
        # Auto reporting layer.
        # This list stores one row per unique reaction condition, not one row
        # per physical duplicate well. It is separate from experiment_data.csv,
        # which remains row-per-well for raw output and model training.
        self.auto_model_performance_rows = []
        self.auto_condition_counter = 0

    # Update experiment_data DataFrame after each batch
    def _update_experiment_data(self, recipes, Experiment_result, axis=1):
        # TODO: Reformat the instructions for csv output to make it export 
        # TODO: Test this implementation of formatting this csv
        '''
        Updates self.experiment_data with the recipes tested in the current batch
        and the corresponding experimental results.

        Parameters:
            recipes:
                Recipe values for the current batch. Expected shape is:
                    number_of_experiments x number_of_variable_reagents

                Each row is one tested well/experiment. Each column corresponds
                to one variable reagent in self.variable_reagents.

            Experiment_result:
                Experimental result values for the current batch. Expected length
                is one result per recipe row.

            axis:
                Kept in the function signature for compatibility with older calls.
                The current implementation appends rows to self.experiment_data
                and does not use axis.

        Postconditions:
            - Converts recipes and Experiment_result into numpy arrays with
              predictable shapes.
            - Verifies that each recipe has exactly one corresponding result.
            - Verifies that the number of recipe columns matches the number of
              variable reagents, so each recipe column can be correctly labeled.
            - Creates a new DataFrame with one column per variable reagent.
            - Adds Experiment_result as the final column.
            - Appends the new batch rows to self.experiment_data.
        '''

        # Convert recipes to a numpy array so that shape checks and indexing work
        # consistently, regardless of whether recipes was passed in as a list or array.
        recipes = np.asarray(recipes)

        # If a single recipe was passed as a 1D array, reshape it into a 2D array
        # with one row. This keeps the rest of the function compatible with both
        # single-recipe and multi-recipe batches.
        if recipes.ndim == 1:
            recipes = recipes.reshape(1, -1)

        # Convert experiment results to a flat 1D array. This avoids dataframe
        # construction errors and ensures there is one result value per recipe row.
        experiment_result = np.asarray(Experiment_result).reshape(-1)

        # Safety check: the number of measured results must match the number of
        # recipe rows. If not, experiment_data would be misaligned, so stop here
        # with a clear error message.
        if len(experiment_result) != len(recipes):
            raise ValueError(
                f"Number of experiment results ({len(experiment_result)}) does not match "
                f"number of recipes ({len(recipes)})."
            )
        
        # Safety check: the number of recipe columns must match the number of
        # variable reagents. Each recipe column is labeled using self.variable_reagents,
        # so a mismatch here would mean the dataframe columns could be mislabeled
        # or that a reagent value is missing/extra.
        if recipes.shape[1] != len(self.variable_reagents):
            raise ValueError(
                f"Number of recipe columns ({recipes.shape[1]}) does not match "
                f"number of variable reagents ({len(self.variable_reagents)})."
            )
        
        # Build a dataframe for the current batch. The recipe matrix supplies the
        # values, and self.variable_reagents supplies the column names, such as
        # silver_nitrate and potassium_bromide.
        new_data = pd.DataFrame(
            recipes,
            columns=[str(reagent) for reagent in self.variable_reagents]
        )

        # Add the measured experimental result as its own column. This is separate
        # from the recipe matrix because recipes contains input conditions, while
        # Experiment_result contains the measured output.
        new_data["Experiment_result"] = experiment_result

        # Append the current batch to the running experiment_data dataframe.
        # ignore_index=True creates a clean continuous row index after appending.
        self.experiment_data = pd.concat(
            [self.experiment_data, new_data],
            ignore_index=True
        )

        if (
            self.robo_params.get(
                'auto_terminal_verbosity',
                'standard'
            )
            == 'diagnostic'
        ):
            print("<<controller diagnostic>> experiment data dataframe:")
            print(self.experiment_data)
        if self.robo_params.get(
            'auto_terminal_verbosity',
            'standard'
        ) != 'essential':
            print(
                "<<controller>> experiment data updated successfully with "
                f"{len(new_data)} new rows"
            )

    def _safe_float_or_none(self, value):
        '''
        Converts a numeric value to float while preserving missing values as
        None.

        This is used by the Auto performance log so optional values, such as
        seed-recipe predictions, can remain blank in the exported CSV instead
        of causing conversion errors.

        params:
            value:
                Numeric value, None, NaN, or other value convertible to float.

        returns:
            float or None:
                Float-converted value, or None if the input is missing.
        '''
        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass

        return float(value)

    def _serialize_auto_audit_value(self, value):
        '''Serializes nested Auto audit metadata as deterministic JSON.'''
        def make_json_safe(item):
            if isinstance(item, np.ndarray):
                return make_json_safe(item.tolist())

            if isinstance(item, np.generic):
                return make_json_safe(item.item())

            if isinstance(item, dict):
                return {
                    str(key): make_json_safe(nested_value)
                    for key, nested_value in item.items()
                }

            if isinstance(item, (list, tuple)):
                return [make_json_safe(nested_value) for nested_value in item]

            if isinstance(item, float) and not math.isfinite(item):
                return None

            if item is None or isinstance(
                item,
                (bool, int, float, str)
            ):
                return item

            # Scipy result messages and other diagnostic-only scalar objects
            # are represented textually instead of making audit export fail.
            return str(item)

        return json.dumps(
            make_json_safe(value),
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=False,
            allow_nan=False
        )
    
    def _format_mask_for_report(self, mask):
        '''
        Converts a selected optimizer reagent mask into a compact string for
        the Auto performance log.

        Example:
            np.array([1, 0]) becomes "[1, 0]"

        params:
            mask:
                List, numpy array, or None representing the selected variable
                reagent ON/OFF mask.

        returns:
            str:
                Compact mask string, or an empty string if no mask is present.
        '''
        if mask is None:
            return ""

        mask_array = np.asarray(mask).astype(int).reshape(-1)

        return "[" + ", ".join(str(int(x)) for x in mask_array) + "]"

    def _get_active_variable_reagents_from_mask(self, mask):
        '''
        Converts a selected optimizer reagent mask into a comma-separated list
        of active variable reagent names for the Auto performance log.

        Example:
            variable_reagents = ["silver_nitrate", "potassium_bromide"]
            mask = np.array([1, 0])

            returns:
                "silver_nitrate"

        params:
            mask:
                List, numpy array, or None representing the selected variable
                reagent ON/OFF mask.

        returns:
            str:
                Comma-separated active variable reagent names, or an empty
                string if no mask is present.
        '''
        if mask is None:
            return ""

        mask_array = np.asarray(mask).astype(int).reshape(-1)

        active_reagents = []

        for reagent_i, is_active in enumerate(mask_array):
            if is_active and reagent_i < len(self.variable_reagents):
                active_reagents.append(str(self.variable_reagents[reagent_i]))

        return ", ".join(active_reagents)
    
    def _summarize_duplicate_lambda_values(self, lambda_values):
        '''
        Computes the mean, sample standard deviation, and standard error of
        the mean for duplicate lambda max measurements.

        This is used by the Auto performance log to summarize multiple
        physical replicate wells as one unique reaction condition.

        If only one replicate is present, standard deviation and SEM are set
        to 0.0 because replicate variability cannot be estimated.

        params:
            list lambda_values:
                Lambda max values from the physical replicate wells for one
                unique reaction condition.

        returns:
            tuple(float, float, float):
                Mean lambda max, sample standard deviation, and standard error
                of the mean. Returns (None, None, None) if no valid lambda max
                values are present.
        '''
        finite_lambda_values = []

        for value in lambda_values:
            if value is None or pd.isna(value):
                continue

            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            if math.isfinite(value):
                finite_lambda_values.append(value)

        lambda_values = finite_lambda_values

        if len(lambda_values) == 0:
            return None, None, None

        actual_mean = float(np.mean(lambda_values))

        if len(lambda_values) >= 2:
            actual_sd = float(np.std(lambda_values, ddof=1))
            actual_sem = float(actual_sd / math.sqrt(len(lambda_values)))
        else:
            actual_sd = 0.0
            actual_sem = 0.0

        return actual_mean, actual_sd, actual_sem
    
    def _get_auto_replicate_outlier_threshold_nm(self):
        '''
        Returns the lambda max replicate-outlier threshold in nm.

        This controls the conservative 3+ replicate QC rule used by Auto mode.
        A replicate is only excluded when it is clearly isolated from the other
        replicates by more than this threshold.
        '''
        return float(
            self.robo_params.get(
                'replicate_outlier_threshold_nm',
                50.0
            )
        )

    def _get_auto_replicate_sd_tolerance_nm(self):
        '''
        Returns the maximum replicate SD allowed for target decisions.

        GP model-training eligibility is intentionally broader than target
        incumbent and stopping eligibility. Ambiguous or single-replicate
        observations may still be scientifically useful model data, while a
        target decision must demonstrate agreement between at least two
        QC-included finite replicates.
        '''
        tolerance_nm = float(
            self.robo_params.get(
                'replicate_sd_tolerance_nm',
                25.0
            )
        )

        if not math.isfinite(tolerance_nm) or tolerance_nm < 0.0:
            raise ValueError(
                "replicate_sd_tolerance_nm must be a finite, nonnegative "
                f"number. Received: {tolerance_nm!r}."
            )

        return tolerance_nm

    def _get_auto_target_eligibility_decision(
        self,
        replicate_qc,
        model_training_decision,
        qc_replicate_sd_nm
    ):
        '''
        Decides whether a condition may set an incumbent or stop Auto.

        This policy is deliberately separate from GP model-training
        eligibility. The GP may retain a valid single observation or a noisy
        condition as information, but target EI and controller stopping need a
        condition-level result supported by replicate agreement.

        A condition is eligible for either target decision only when:
            1. the condition is accepted for GP model training;
            2. at least two finite, QC-included replicates remain; and
            3. their sample SD is finite and no greater than the configured
               replicate_sd_tolerance_nm (25 nm by default).

        returns:
            dict:
                Explicit incumbent/stopping eligibility, status, reason, and
                the SD tolerance used for the decision.
        '''
        tolerance_nm = self._get_auto_replicate_sd_tolerance_nm()
        n_replicates_used = int(
            replicate_qc.get('n_replicates_used', 0)
        )

        if not model_training_decision.get('use_for_model_training', False):
            status = 'ineligible_not_model_approved'
            reason = (
                'Condition was not approved for GP model training.'
            )
            eligible = False
        elif n_replicates_used < 2:
            status = 'ineligible_fewer_than_2_qc_replicates'
            reason = (
                'At least two finite QC-included replicates are required for '
                'a target incumbent or validated stop.'
            )
            eligible = False
        else:
            try:
                replicate_sd_nm = float(qc_replicate_sd_nm)
            except (TypeError, ValueError):
                replicate_sd_nm = None

            if (
                replicate_sd_nm is None
                or not math.isfinite(replicate_sd_nm)
            ):
                status = 'ineligible_nonfinite_replicate_sd'
                reason = (
                    'Replicate SD must be finite for a target incumbent or '
                    'validated stop.'
                )
                eligible = False
            elif replicate_sd_nm > tolerance_nm:
                status = 'ineligible_replicate_sd_above_tolerance'
                reason = (
                    f'Replicate SD {replicate_sd_nm:.4f} nm exceeds the '
                    f'{tolerance_nm:.4f} nm target-decision tolerance.'
                )
                eligible = False
            else:
                status = 'eligible_replicate_validated_condition'
                reason = (
                    f'{n_replicates_used} finite QC-included replicates have '
                    f'SD {replicate_sd_nm:.4f} nm, within the '
                    f'{tolerance_nm:.4f} nm target-decision tolerance.'
                )
                eligible = True

        return {
            'eligible_for_target_incumbent': bool(eligible),
            'eligible_for_target_stop': bool(eligible),
            'target_eligibility_status': status,
            'target_eligibility_reason': reason,
            'n_finite_qc_replicates_for_target_validation': (
                n_replicates_used
            ),
            'replicate_sd_tolerance_nm': tolerance_nm
        }

    def _run_lambda_replicate_qc(self, lambda_values):
        '''
        Runs conservative replicate-level QC for Auto lambda max values.

        Raw values are always preserved. This method only decides which
        replicate lambda max values should be used for model learning and
        condition-level summaries.

        Automatic exclusion is only applied when at least 3 valid replicate
        lambda max values are present.

        For exactly 3 valid replicates, the rule is:

            1. Find the closest pair of lambda max values.
            2. Only treat that pair as a reliable agreement if the pair
               distance is less than or equal to
               replicate_outlier_threshold_nm.
            3. Compute the mean of that closest pair.
            4. If the remaining third value is farther than
               replicate_outlier_threshold_nm from the closest-pair mean,
               exclude the third value.
            5. If no pair is within replicate_outlier_threshold_nm, flag the
               condition as suspicious but do not exclude automatically.

        For 4 or more valid replicates, a conservative median-distance rule is
        used. Values farther than replicate_outlier_threshold_nm from the
        median are excluded only if at least two valid replicates would remain.
        If the rule would leave fewer than two included replicates, the
        condition is flagged but not automatically excluded.

        For fewer than 3 valid values, no automatic exclusion is applied.

        params:
            list lambda_values:
                Raw replicate lambda max values for one unique reaction
                condition.

        returns:
            dict:
                QC result containing raw values, included values, excluded
                indices, excluded values, status, and reason.
        '''
        raw_values = [
            None if value is None or pd.isna(value) else float(value)
            for value in lambda_values
        ]

        valid_pairs = [
            (index, value)
            for index, value in enumerate(raw_values)
            if value is not None and math.isfinite(value)
        ]

        included_indices = [index for index, value in valid_pairs]
        excluded_indices = []
        excluded_values = []
        threshold_nm = self._get_auto_replicate_outlier_threshold_nm()

        qc_status = 'not_applied'
        qc_reason = ''

        if len(valid_pairs) < 3:
            return {
                'raw_values': raw_values,
                'included_indices': included_indices,
                'included_values': [value for index, value in valid_pairs],
                'excluded_indices': excluded_indices,
                'excluded_values': excluded_values,
                'n_replicates_total': len(raw_values),
                'n_replicates_valid': len(valid_pairs),
                'n_replicates_used': len(included_indices),
                'n_replicates_excluded': 0,
                'qc_status': qc_status,
                'qc_reason': 'fewer_than_3_valid_replicates',
                'replicate_outlier_threshold_nm': threshold_nm
            }

        if len(valid_pairs) == 3:
            closest_pair = None
            closest_pair_distance = None

            for i in range(len(valid_pairs)):
                for j in range(i + 1, len(valid_pairs)):
                    pair_distance = abs(valid_pairs[i][1] - valid_pairs[j][1])

                    if (
                        closest_pair_distance is None or
                        pair_distance < closest_pair_distance
                    ):
                        closest_pair_distance = pair_distance
                        closest_pair = (i, j)

            pair_i, pair_j = closest_pair
            pair_mean = float(
                (
                    valid_pairs[pair_i][1] +
                    valid_pairs[pair_j][1]
                ) / 2.0
            )

            third_position = list(
                set(range(3)) - set([pair_i, pair_j])
            )[0]

            third_index, third_value = valid_pairs[third_position]
            third_distance = float(abs(third_value - pair_mean))

            if (
                closest_pair_distance <= threshold_nm
                and third_distance > threshold_nm
            ):
                included_indices = [
                    valid_pairs[pair_i][0],
                    valid_pairs[pair_j][0]
                ]
                excluded_indices = [third_index]
                excluded_values = [third_value]
                qc_status = 'excluded_replicate'
                qc_reason = (
                    'lambda_max_outlier: closest_pair_distance='
                    f'{closest_pair_distance:.2f} nm; closest_pair_mean='
                    f'{pair_mean:.2f} nm; excluded_value='
                    f'{third_value:.2f} nm; distance='
                    f'{third_distance:.2f} nm; threshold='
                    f'{threshold_nm:.2f} nm'
                )
            elif closest_pair_distance > threshold_nm:
                qc_status = 'flagged_not_excluded'
                qc_reason = (
                    'triplicate_no_tight_pair: closest_pair_distance='
                    f'{closest_pair_distance:.2f} nm; threshold='
                    f'{threshold_nm:.2f} nm'
                )
            else:
                qc_status = 'passed'
                qc_reason = (
                    'no_replicate_excluded: closest_pair_distance='
                    f'{closest_pair_distance:.2f} nm; farthest_value_distance='
                    f'{third_distance:.2f} nm; threshold='
                    f'{threshold_nm:.2f} nm'
                )

        else:
            # For 4+ valid replicates, use a conservative median rule. This is
            # less central for the current workflow, which usually uses
            # triplicates, but keeps the function correct for n > 3.
            values = np.asarray([value for index, value in valid_pairs], dtype=float)
            median_value = float(np.median(values))
            distances = np.abs(values - median_value)

            outlier_positions = np.where(distances > threshold_nm)[0].tolist()

            if len(outlier_positions) > 0:
                excluded_indices = [
                    valid_pairs[position][0]
                    for position in outlier_positions
                ]
                excluded_values = [
                    valid_pairs[position][1]
                    for position in outlier_positions
                ]
                included_indices = [
                    index
                    for index, value in valid_pairs
                    if index not in excluded_indices
                ]

                # Keep at least two included replicates. If the rule would
                # exclude too many values, flag but do not exclude automatically.
                if len(included_indices) < 2:
                    included_indices = [index for index, value in valid_pairs]
                    excluded_indices = []
                    excluded_values = []
                    qc_status = 'flagged_not_excluded'
                    qc_reason = (
                        'median_rule_would_leave_fewer_than_2_replicates'
                    )
                else:
                    qc_status = 'excluded_replicate'
                    qc_reason = (
                        'lambda_max_outlier_median_rule: median='
                        f'{median_value:.2f} nm; threshold='
                        f'{threshold_nm:.2f} nm'
                    )
            else:
                qc_status = 'passed'
                qc_reason = (
                    'no_replicate_excluded_by_median_rule: median='
                    f'{median_value:.2f} nm; threshold='
                    f'{threshold_nm:.2f} nm'
                )

        included_values = [
            raw_values[index]
            for index in included_indices
            if raw_values[index] is not None
        ]

        return {
            'raw_values': raw_values,
            'included_indices': included_indices,
            'included_values': included_values,
            'excluded_indices': excluded_indices,
            'excluded_values': excluded_values,
            'n_replicates_total': len(raw_values),
            'n_replicates_valid': len(valid_pairs),
            'n_replicates_used': len(included_values),
            'n_replicates_excluded': len(excluded_values),
            'qc_status': qc_status,
            'qc_reason': qc_reason,
            'replicate_outlier_threshold_nm': threshold_nm
        }
    
    def _get_auto_model_training_decision_from_replicate_qc(
        self,
        replicate_qc
    ):
        '''
        Converts replicate QC results into an explicit model-training decision.

        This separates two scientific questions:

            1. Which replicate values are included in the condition-level QC
               summary?
            2. Which replicate values should be used to train the GP model?

        The policy is intentionally data-preserving:

            - Clear isolated replicate artifacts are excluded from model
              training.
            - Ambiguous flagged conditions are retained for model training
              because they may still contain useful information about the
              experimental response or reproducibility.
            - Conditions with no valid replicate values are skipped.

        params:
            dict replicate_qc:
                Output from _run_lambda_replicate_qc().

        returns:
            dict:
                Model-training decision metadata.
        '''
        qc_status = replicate_qc.get('qc_status')
        n_replicates_used = int(replicate_qc.get('n_replicates_used', 0))
        n_replicates_excluded = int(
            replicate_qc.get('n_replicates_excluded', 0)
        )
        n_replicates_valid = int(replicate_qc.get('n_replicates_valid', 0))

        if n_replicates_valid == 0:
            return {
                'use_for_model_training': False,
                'model_training_status': 'skipped_no_valid_replicates',
                'n_replicates_used_for_model_training': 0,
                'model_training_reason': (
                    'No valid replicate lambda max values were available.'
                )
            }

        if n_replicates_used == 0:
            return {
                'use_for_model_training': False,
                'model_training_status': 'skipped_no_qc_included_replicates',
                'n_replicates_used_for_model_training': 0,
                'model_training_reason': (
                    'Replicate QC left no included values for model training.'
                )
            }

        if qc_status == 'flagged_not_excluded':
            return {
                'use_for_model_training': True,
                'model_training_status': 'used_flagged_condition',
                'n_replicates_used_for_model_training': n_replicates_used,
                'model_training_reason': (
                    'Condition was flagged as suspicious or ambiguous by '
                    'replicate QC, but no replicate was automatically excluded; '
                    'all QC-included valid replicate values are retained for '
                    'GP model training.'
                )
            }

        if n_replicates_excluded > 0:
            return {
                'use_for_model_training': True,
                'model_training_status': 'used_qc_included_only',
                'n_replicates_used_for_model_training': n_replicates_used,
                'model_training_reason': (
                    'Clear replicate outlier(s) were excluded; only '
                    'QC-included replicate values are used for GP model '
                    'training.'
                )
            }

        return {
            'use_for_model_training': True,
            'model_training_status': 'used_all_valid_replicates',
            'n_replicates_used_for_model_training': n_replicates_used,
            'model_training_reason': (
                'Replicate QC passed or was not required; all valid replicate '
                'values are used for GP model training.'
            )
        }
    
    def _build_auto_qc_model_training_data(
        self,
        unique_recipes,
        lambda_max_values
    ):
        '''
        Builds replicate-level model-training arrays after Auto replicate QC.

        Raw replicate results are preserved elsewhere. This method returns only
        the recipe/lambda pairs that should be added to the GP model after
        conservative replicate QC and model-training eligibility review.

        For each unique recipe condition:
            - split its duplicate lambda max values
            - run replicate QC
            - convert QC results into an explicit model-training decision
            - keep QC-included replicate values for model-used conditions
            - repeat the unique recipe once for each model-used replicate value

        The policy is data-preserving: ambiguous flagged conditions are still
        used for model training, while clear excluded replicate artifacts are
        not.

        params:
            np.ndarray unique_recipes:
                One row per unique denormalized recipe condition.

            list lambda_max_values:
                Raw lambda max values from physical duplicate wells.

        returns:
            tuple:
                X_model_denormalized, Y_model_lambda_nm

                X_model_denormalized:
                    Denormalized recipe rows repeated only for replicate values
                    used for model training.

                Y_model_lambda_nm:
                    Raw nm lambda max values corresponding to the replicate
                    rows used for model training.
        '''
        unique_recipes = np.array(unique_recipes, dtype=float, copy=True)

        if unique_recipes.ndim == 1:
            unique_recipes = unique_recipes.reshape(1, -1)

        lambda_max_values = list(lambda_max_values)

        expected_lambda_count = unique_recipes.shape[0] * self.num_duplicates

        if len(lambda_max_values) != expected_lambda_count:
            raise ValueError(
                "Cannot build QC model training data because the number of "
                f"lambda max values ({len(lambda_max_values)}) does not match "
                f"unique recipes ({unique_recipes.shape[0]}) x "
                f"num_duplicates ({self.num_duplicates}) = "
                f"{expected_lambda_count}."
            )

        model_recipes = []
        model_lambda_values = []

        skipped_condition_count = 0

        for recipe_i, recipe in enumerate(unique_recipes):
            start_i = recipe_i * self.num_duplicates
            end_i = start_i + self.num_duplicates
            replicate_lambda_values = lambda_max_values[start_i:end_i]

            replicate_qc = self._run_lambda_replicate_qc(
                replicate_lambda_values
            )

            model_training_decision = (
                self._get_auto_model_training_decision_from_replicate_qc(
                    replicate_qc
                )
            )

            if not model_training_decision['use_for_model_training']:
                skipped_condition_count += 1
                print(
                    "<<controller warning>> skipping Auto condition "
                    f"{recipe_i} for GP model training: "
                    f"{model_training_decision['model_training_status']}; "
                    f"{replicate_qc['qc_reason']}"
                )
                continue

            for included_value in replicate_qc['included_values']:
                if included_value is None or pd.isna(included_value):
                    continue

                model_recipes.append(recipe.copy())
                model_lambda_values.append(float(included_value))

        if len(model_lambda_values) == 0:
            raise ValueError(
                "Auto replicate QC/model-training review removed or skipped "
                "all lambda max values. The GP model cannot be updated without "
                "at least one trusted observation."
            )

        if skipped_condition_count > 0:
            print(
                "<<controller warning>> skipped "
                f"{skipped_condition_count} Auto condition(s) from GP model "
                "training because no valid QC-included replicate values were "
                "available."
            )

        return (
            np.asarray(model_recipes, dtype=float),
            np.asarray(model_lambda_values, dtype=float)
        )
    
    def _append_auto_model_performance_rows(
        self,
        unique_recipes,
        lambda_max_values,
        condition_type,
        batch_number,
        prediction_metadata=None
    ):
        '''
        Appends condition-level rows to the Auto model performance log.

        This log stores one row per unique reaction condition. It does not
        store one row per physical duplicate well. Raw duplicate-well results
        remain preserved in experiment_data.csv and in the replicate-specific
        columns of this performance log.

        Duplicate wells are assumed to be contiguous because Auto mode expands
        each unique recipe with duplicate_list_elements() before execution.

        Replicate-level QC is applied before calculating the condition-level
        actual lambda max used by Auto performance summaries. Raw replicate
        values are preserved, while QC-cleaned values are used for:
            - actual_lambda_mean_nm
            - actual_lambda_sd_nm
            - actual_lambda_sem_nm
            - target_error_nm
            - prediction_error_nm

        params:
            np.ndarray unique_recipes:
                One row per unique denormalized recipe condition.

            list lambda_max_values:
                Lambda max values from the physical replicate wells.

            str condition_type:
                Type of condition being logged. Expected values are:
                    seed
                    optimizer_selected

            int batch_number:
                Auto batch number for these conditions.

            dict prediction_metadata:
                Optional metadata captured before experiment execution, such
                as acquisition mode and score, selected mask, predicted lambda
                distribution, predicted target error, and target-EI incumbent.

        returns:
            None
        '''
        unique_recipes = np.array(unique_recipes, dtype=float, copy=True)

        if unique_recipes.ndim == 1:
            unique_recipes = unique_recipes.reshape(1, -1)

        lambda_max_values = list(lambda_max_values)

        expected_lambda_count = unique_recipes.shape[0] * self.num_duplicates

        if len(lambda_max_values) != expected_lambda_count:
            raise ValueError(
                "Cannot append Auto model performance rows because the number "
                f"of lambda max values ({len(lambda_max_values)}) does not "
                f"match unique recipes ({unique_recipes.shape[0]}) x "
                f"num_duplicates ({self.num_duplicates}) = "
                f"{expected_lambda_count}."
            )

        if prediction_metadata is None:
            prediction_metadata = {}

        if isinstance(prediction_metadata, list):
            if len(prediction_metadata) != unique_recipes.shape[0]:
                raise ValueError(
                    "Portfolio prediction metadata must contain exactly one "
                    "selection record per unique recipe."
                )

            prediction_metadata_by_recipe = prediction_metadata
        else:
            prediction_metadata_by_recipe = [
                prediction_metadata
                for _ in range(unique_recipes.shape[0])
            ]

        def get_recipe_component(recipe_values, reagent_index):
            if recipe_values is None:
                return None

            try:
                recipe_array = np.asarray(recipe_values, dtype=float)

                if recipe_array.ndim == 1:
                    recipe_array = recipe_array.reshape(1, -1)

                if (
                    recipe_array.ndim != 2
                    or recipe_array.shape[0] == 0
                    or reagent_index >= recipe_array.shape[1]
                ):
                    return None

                value = float(recipe_array[0, reagent_index])
            except (TypeError, ValueError):
                return None

            return value if math.isfinite(value) else None

        def get_first_volume_balance(metadata_key):
            balances = prediction_metadata.get(metadata_key)

            if not balances or not isinstance(balances, (list, tuple)):
                return None

            first_balance = balances[0]

            return first_balance if isinstance(first_balance, dict) else None

        target_lambda = self.getModelInfo()["target"]

        for recipe_i, recipe in enumerate(unique_recipes):
            prediction_metadata = prediction_metadata_by_recipe[recipe_i]
            start_i = recipe_i * self.num_duplicates
            end_i = start_i + self.num_duplicates
            replicate_lambda_values = lambda_max_values[start_i:end_i]

            raw_mean, raw_sd, raw_sem = (
                self._summarize_duplicate_lambda_values(
                    replicate_lambda_values
                )
            )

            replicate_qc = self._run_lambda_replicate_qc(
                replicate_lambda_values
            )

            model_training_decision = (
                self._get_auto_model_training_decision_from_replicate_qc(
                    replicate_qc
                )
            )

            qc_mean, qc_sd, qc_sem = (
                self._summarize_duplicate_lambda_values(
                    replicate_qc['included_values']
                )
            )

            target_eligibility_decision = (
                self._get_auto_target_eligibility_decision(
                    replicate_qc=replicate_qc,
                    model_training_decision=model_training_decision,
                    qc_replicate_sd_nm=qc_sd
                )
            )

            if replicate_qc['n_replicates_excluded'] > 0:
                print(
                    "<<controller>> Auto replicate QC excluded "
                    f"{replicate_qc['n_replicates_excluded']} replicate(s) "
                    f"from batch {batch_number}, condition "
                    f"{self.auto_condition_counter}: "
                    f"{replicate_qc['qc_reason']}"
                )

            predicted_mean = self._safe_float_or_none(
                prediction_metadata.get('predicted_lambda_mean_nm')
            )
            predicted_std = self._safe_float_or_none(
                prediction_metadata.get('predicted_lambda_std_nm')
            )
            predicted_target_error = self._safe_float_or_none(
                prediction_metadata.get('predicted_target_error_nm')
            )
            acquisition_score = self._safe_float_or_none(
                prediction_metadata.get('acquisition_score')
            )
            incumbent_target_error = self._safe_float_or_none(
                prediction_metadata.get('incumbent_target_error_nm')
            )
            acquisition_mode = prediction_metadata.get('acquisition_mode')
            balanced_exploration_weight = self._safe_float_or_none(
                prediction_metadata.get('balanced_exploration_weight')
            )
            selected_normalized_recipe = prediction_metadata.get(
                'selected_normalized_recipe'
            )
            executed_normalized_recipe = prediction_metadata.get(
                'executed_normalized_recipe'
            )
            selected_physical_recipe = prediction_metadata.get(
                'selected_physical_recipe'
            )
            executed_physical_recipe = prediction_metadata.get(
                'executed_physical_recipe'
            )
            selected_controller_volume_balance = get_first_volume_balance(
                'selected_controller_volume_balances'
            )
            executed_controller_volume_balance = get_first_volume_balance(
                'executed_controller_volume_balances'
            )

            if acquisition_mode is not None:
                acquisition_mode = str(acquisition_mode)

            # Derive the predicted target error when older callers provide a
            # mean prediction but not the newer explicit audit field.
            if (
                predicted_target_error is None
                and predicted_mean is not None
            ):
                predicted_target_error = float(
                    abs(predicted_mean - target_lambda)
                )

            if qc_mean is None:
                target_error = None
                prediction_error = None
            else:
                target_error = float(abs(qc_mean - target_lambda))

                if predicted_mean is None:
                    prediction_error = None
                else:
                    prediction_error = float(qc_mean - predicted_mean)

            volume_balance = self._get_auto_recipe_volume_balance(recipe)
            selected_mask = prediction_metadata.get('selected_mask')

            row = {
                'experiment_name': self.rxn_sheet_name,
                'batch_number': int(batch_number),
                'reaction_number': int(self.auto_condition_counter),
                'condition_type': condition_type,
                'acquisition_mode': acquisition_mode,
                'acquisition_score': acquisition_score,
                'balanced_exploration_weight': (
                    balanced_exploration_weight
                ),
                'selected_mask': self._format_mask_for_report(selected_mask),
                'portfolio_selection_index': prediction_metadata.get(
                    'portfolio_selection_index'
                ),
                'portfolio_acquisition_modes': (
                    self._serialize_auto_audit_value(
                        prediction_metadata.get(
                            'portfolio_acquisition_modes'
                        )
                    )
                    if prediction_metadata.get(
                        'portfolio_acquisition_modes'
                    ) is not None
                    else None
                ),
                'portfolio_min_distance': self._safe_float_or_none(
                    prediction_metadata.get('portfolio_min_distance')
                ),
                'portfolio_nearest_distance': self._safe_float_or_none(
                    prediction_metadata.get('portfolio_nearest_distance')
                ),
                'optimizer_method': prediction_metadata.get(
                    'optimizer_method'
                ),
                'optimizer_success': prediction_metadata.get(
                    'optimizer_success'
                ),
                'optimizer_status': prediction_metadata.get(
                    'optimizer_status'
                ),
                'optimizer_message': prediction_metadata.get(
                    'optimizer_message'
                ),
                'active_variable_reagents': (
                    self._get_active_variable_reagents_from_mask(
                        selected_mask
                    )
                ),
                'target_lambda_max_nm': float(target_lambda),
                'predicted_target_error_nm': predicted_target_error,
                'predicted_lambda_mean_nm': predicted_mean,
                'predicted_lambda_std_nm': predicted_std,
                'incumbent_target_error_nm': incumbent_target_error,
                'selected_normalized_recipe': (
                    None
                    if selected_normalized_recipe is None
                    else self._serialize_auto_audit_value(
                        selected_normalized_recipe
                    )
                ),
                'executed_normalized_recipe': (
                    None
                    if executed_normalized_recipe is None
                    else self._serialize_auto_audit_value(
                        executed_normalized_recipe
                    )
                ),
                'selected_physical_recipe': (
                    None
                    if selected_physical_recipe is None
                    else self._serialize_auto_audit_value(
                        selected_physical_recipe
                    )
                ),
                'executed_physical_recipe': (
                    None
                    if executed_physical_recipe is None
                    else self._serialize_auto_audit_value(
                        executed_physical_recipe
                    )
                ),
                'optimizer_recipe_repaired': prediction_metadata.get(
                    'optimizer_recipe_repaired'
                ),
                'optimizer_recipe_repair_max_transfer_delta_uL': (
                    self._safe_float_or_none(
                        prediction_metadata.get(
                            'optimizer_recipe_repair_max_transfer_delta_uL'
                        )
                    )
                ),
                'optimizer_volume_balance': (
                    self._serialize_auto_audit_value(
                        prediction_metadata.get('optimizer_volume_balance')
                    )
                    if prediction_metadata.get('optimizer_volume_balance')
                    is not None
                    else None
                ),
                'selected_controller_volume_balance': (
                    self._serialize_auto_audit_value(
                        selected_controller_volume_balance
                    )
                    if selected_controller_volume_balance is not None
                    else None
                ),
                'executed_controller_volume_balance': (
                    self._serialize_auto_audit_value(
                        executed_controller_volume_balance
                    )
                    if executed_controller_volume_balance is not None
                    else None
                ),
                'mask_results': (
                    self._serialize_auto_audit_value(
                        prediction_metadata.get('mask_results')
                    )
                    if prediction_metadata.get('mask_results') is not None
                    else None
                ),
                'mask_result_count': int(
                    prediction_metadata.get('mask_result_count', 0)
                ),
                'feasible_mask_result_count': int(
                    prediction_metadata.get(
                        'feasible_mask_result_count',
                        0
                    )
                ),

                # Backward-compatible actual_lambda_* columns now represent
                # the QC-cleaned condition-level values used by Auto summaries.
                'actual_lambda_mean_nm': qc_mean,
                'actual_lambda_sd_nm': qc_sd,
                'actual_lambda_sem_nm': qc_sem,
                'actual_lambda_values_nm': replicate_qc['included_values'],

                # Raw, unmodified replicate summary.
                'actual_lambda_mean_raw_nm': raw_mean,
                'actual_lambda_sd_raw_nm': raw_sd,
                'actual_lambda_sem_raw_nm': raw_sem,
                'actual_lambda_values_raw_nm': replicate_qc['raw_values'],

                # Explicit QC-cleaned replicate summary.
                'actual_lambda_mean_qc_nm': qc_mean,
                'actual_lambda_sd_qc_nm': qc_sd,
                'actual_lambda_sem_qc_nm': qc_sem,
                'actual_lambda_values_qc_nm': replicate_qc['included_values'],

                # Replicate QC metadata.
                'n_replicates_total': replicate_qc['n_replicates_total'],
                'n_replicates_valid': replicate_qc['n_replicates_valid'],
                'n_replicates_used': replicate_qc['n_replicates_used'],
                'n_replicates_excluded': (
                    replicate_qc['n_replicates_excluded']
                ),
                'excluded_replicate_indices': (
                    replicate_qc['excluded_indices']
                ),
                'excluded_lambda_values_nm': (
                    replicate_qc['excluded_values']
                ),
                'replicate_qc_status': replicate_qc['qc_status'],
                'replicate_qc_reason': replicate_qc['qc_reason'],
                'replicate_outlier_threshold_nm': (
                    replicate_qc['replicate_outlier_threshold_nm']
                ),

                # Explicit GP model-training decision.
                'use_for_model_training': (
                    model_training_decision['use_for_model_training']
                ),
                'model_training_status': (
                    model_training_decision['model_training_status']
                ),
                'n_replicates_used_for_model_training': (
                    model_training_decision[
                        'n_replicates_used_for_model_training'
                    ]
                ),
                'model_training_reason': (
                    model_training_decision['model_training_reason']
                ),

                # Target decisions are stricter than GP training. These fields
                # make it explicit whether this aggregate may become the
                # target-EI incumbent or authorize a target-based stop.
                'eligible_for_target_incumbent': (
                    target_eligibility_decision[
                        'eligible_for_target_incumbent'
                    ]
                ),
                'eligible_for_target_stop': (
                    target_eligibility_decision[
                        'eligible_for_target_stop'
                    ]
                ),
                'target_eligibility_status': (
                    target_eligibility_decision[
                        'target_eligibility_status'
                    ]
                ),
                'target_eligibility_reason': (
                    target_eligibility_decision[
                        'target_eligibility_reason'
                    ]
                ),
                'replicate_sd_tolerance_nm': (
                    target_eligibility_decision[
                        'replicate_sd_tolerance_nm'
                    ]
                ),
                'n_finite_qc_replicates_for_target_validation': (
                    target_eligibility_decision[
                        'n_finite_qc_replicates_for_target_validation'
                    ]
                ),

                'target_error_nm': target_error,
                'prediction_error_nm': prediction_error,
                'fixed_volume_total_uL': volume_balance['fixed_volume_total'],
                'variable_volume_total_uL': (
                    volume_balance['variable_volume_total']
                ),
                'water_volume_uL': volume_balance['water_volume'],
                'total_volume_uL': volume_balance['total_volume'],
                'volume_feasible': volume_balance['volume_feasible'],
                'water_transfer_executable': (
                    volume_balance['water_transfer_executable']
                ),
                'variable_transfers_executable': (
                    volume_balance['variable_transfers_executable']
                ),
                'selected_fixed_volume_total_uL': (
                    None
                    if selected_controller_volume_balance is None
                    else selected_controller_volume_balance.get(
                        'fixed_volume_total'
                    )
                ),
                'selected_variable_volume_total_uL': (
                    None
                    if selected_controller_volume_balance is None
                    else selected_controller_volume_balance.get(
                        'variable_volume_total'
                    )
                ),
                'selected_water_volume_uL': (
                    None
                    if selected_controller_volume_balance is None
                    else selected_controller_volume_balance.get('water_volume')
                ),
                'selected_volume_feasible': (
                    None
                    if selected_controller_volume_balance is None
                    else selected_controller_volume_balance.get(
                        'volume_feasible'
                    )
                ),
                'executed_fixed_volume_total_uL': (
                    None
                    if executed_controller_volume_balance is None
                    else executed_controller_volume_balance.get(
                        'fixed_volume_total'
                    )
                ),
                'executed_variable_volume_total_uL': (
                    None
                    if executed_controller_volume_balance is None
                    else executed_controller_volume_balance.get(
                        'variable_volume_total'
                    )
                ),
                'executed_water_volume_uL': (
                    None
                    if executed_controller_volume_balance is None
                    else executed_controller_volume_balance.get('water_volume')
                ),
                'executed_volume_feasible': (
                    None
                    if executed_controller_volume_balance is None
                    else executed_controller_volume_balance.get(
                        'volume_feasible'
                    )
                ),
                'notes': prediction_metadata.get('notes', ''),
                'warnings': prediction_metadata.get('warnings', '')
            }

            for reagent_i, reagent_name in enumerate(self.variable_reagents):
                reagent_name = str(reagent_name)
                row[f'{reagent_name}_concentration'] = float(recipe[reagent_i])

                row[f'{reagent_name}_selected_normalized'] = (
                    get_recipe_component(
                        selected_normalized_recipe,
                        reagent_i
                    )
                )
                row[f'{reagent_name}_executed_normalized'] = (
                    get_recipe_component(
                        executed_normalized_recipe,
                        reagent_i
                    )
                )
                row[f'{reagent_name}_selected_concentration'] = (
                    get_recipe_component(
                        selected_physical_recipe,
                        reagent_i
                    )
                )
                row[f'{reagent_name}_executed_concentration'] = (
                    get_recipe_component(
                        executed_physical_recipe,
                        reagent_i
                    )
                )

                variable_volumes = volume_balance['variable_transfer_volumes']
                row[f'{reagent_name}_transfer_uL'] = float(
                    variable_volumes[reagent_name]
                )

                selected_variable_volumes = (
                    selected_controller_volume_balance.get(
                        'variable_transfer_volumes',
                        {}
                    )
                    if selected_controller_volume_balance is not None
                    else {}
                )
                executed_variable_volumes = (
                    executed_controller_volume_balance.get(
                        'variable_transfer_volumes',
                        {}
                    )
                    if executed_controller_volume_balance is not None
                    else {}
                )

                row[f'{reagent_name}_selected_transfer_uL'] = (
                    self._safe_float_or_none(
                        selected_variable_volumes.get(reagent_name)
                    )
                )
                row[f'{reagent_name}_executed_transfer_uL'] = (
                    self._safe_float_or_none(
                        executed_variable_volumes.get(reagent_name)
                    )
                )

            for rep_i in range(self.num_duplicates):
                col_name = f'actual_lambda_rep_{rep_i + 1}_nm'
                include_col_name = f'actual_lambda_rep_{rep_i + 1}_included_in_qc'

                raw_value = replicate_lambda_values[rep_i]

                row[col_name] = self._safe_float_or_none(raw_value)
                row[include_col_name] = (
                    rep_i in replicate_qc['included_indices']
                )

            self.auto_model_performance_rows.append(row)
            self.auto_condition_counter += 1

        self._update_auto_model_performance_closest_so_far()
    
    def _update_auto_model_performance_closest_so_far(self):
        '''
        Updates the closest-to-target-so-far flag for each condition-level row
        in the Auto model performance log.

        A row is marked True if it is the best observed condition at that point
        in the run sequence. Otherwise, it is marked False.

        This is useful for future progress plots and notebook reports because
        it identifies when Auto discovers a new best condition.

        params:
            None

        returns:
            None
        '''
        best_error_so_far = None

        for row in self.auto_model_performance_rows:
            target_error = row.get('target_error_nm')

            if target_error is None or pd.isna(target_error):
                row['closest_to_target_so_far'] = False
                continue

            if best_error_so_far is None or target_error < best_error_so_far:
                best_error_so_far = target_error
                row['closest_to_target_so_far'] = True
            else:
                row['closest_to_target_so_far'] = False

    def _get_best_qc_approved_target_error_nm(self):
        '''
        Returns the best replicate-validated condition-level target error.

        Target expected improvement needs an incumbent representing the best
        scientifically trusted result achieved so far. The Auto performance
        log contains one row per unique reaction condition and calculates
        target_error_nm from the QC-cleaned replicate aggregate. Incumbent
        eligibility is deliberately stricter than GP model-training
        eligibility: at least two QC-included finite replicates must agree
        within the configured replicate-SD tolerance.

        Individual replicate values are deliberately not inspected here. This
        prevents one unusually favorable well from setting an unrealistically
        strong incumbent when its condition-level replicate result was not
        approved for model training.

        params:
            None

        returns:
            float or None:
                Smallest finite, nonnegative QC-approved condition-level
                target error in nanometers, or None when none is available.
        '''
        eligible_target_errors_nm = []

        for row in self.auto_model_performance_rows:
            if (
                not row.get('use_for_model_training', False)
                or not row.get('eligible_for_target_incumbent', False)
            ):
                continue

            target_error_nm = row.get('target_error_nm')

            try:
                target_error_nm = float(target_error_nm)
            except (TypeError, ValueError):
                continue

            if (
                not math.isfinite(target_error_nm)
                or target_error_nm < 0.0
            ):
                continue

            eligible_target_errors_nm.append(target_error_nm)

        if len(eligible_target_errors_nm) == 0:
            return None

        return float(min(eligible_target_errors_nm))

    def _synchronize_target_ei_incumbent_from_performance(self, model):
        '''
        Synchronizes target EI with the best GP-approved condition result.

        This method must be called only after a successful GP initialization
        or update. That ordering guarantees that the incumbent and fitted GP
        describe the same accepted experimental history; a failed model update
        cannot leave target EI pointing at data the GP did not incorporate.

        Other acquisition modes do not use an incumbent and return immediately.

        params:
            OptimizationModel model:
                Active Auto optimization model.

        returns:
            float or None:
                Stored incumbent target error for target EI, otherwise None.
        '''
        acquisition_modes = getattr(
            model,
            'acquisition_modes',
            [model.acquisition_mode]
        )

        if 'target_ei' not in acquisition_modes:
            return None

        incumbent_target_error_nm = (
            self._get_best_qc_approved_target_error_nm()
        )

        if incumbent_target_error_nm is None:
            raise ValueError(
                "Target-EI cannot select a recipe because no replicate-"
                "validated condition-level target error is available after "
                "the GP model update. At least two finite QC-included "
                "replicates must agree within replicate_sd_tolerance_nm."
            )

        model.set_incumbent_target_error_nm(
            incumbent_target_error_nm
        )

        print(
            "<<controller>> target-EI incumbent condition-level error: "
            f"{incumbent_target_error_nm:.4f} nm"
        )

        return incumbent_target_error_nm
    
    def _export_auto_model_performance_log(self):
        '''
        Exports the condition-level Auto model performance log as a CSV.

        This CSV is the central reporting artifact for Auto mode. It stores
        one row per unique reaction condition and is intended to support future
        lambda-progress plots, all-batch comparison plots, notebook-ready run
        reports, and model-performance diagnostics.

        This does not replace experiment_data.csv. experiment_data.csv remains
        the row-per-well raw output and model-training audit file.

        params:
            None

        returns:
            str:
                Path to the exported Auto model performance log CSV.
        '''
        export_path = os.path.join(
            self.out_path,
            'pr_data',
            'auto_model_performance_log.csv'
        )

        performance_df = pd.DataFrame(self.auto_model_performance_rows)

        performance_df.to_csv(export_path, index=False)

        print(
            f"<<controller>> exported Auto model performance log to "
            f"{export_path}"
        )

        return export_path
    
    def _safe_auto_report_get(self, obj, key, default='not recorded'):
        '''
        Safely retrieves a value from a row-like object for Auto run reporting.

        This helper prevents report generation from failing if a field is
        missing, None, NaN, or unavailable in older output formats.

        params:
            obj:
                Row-like object, usually a pandas Series.

            str key:
                Column/key to retrieve.

            default:
                Value to return when the field is absent or invalid.

        returns:
            object:
                Retrieved value or default.
        '''
        try:
            value = obj.get(key, default)
        except Exception:
            value = default

        if value is None:
            return default

        try:
            if pd.isna(value):
                return default
        except Exception:
            pass

        return value

    def _safe_auto_report_numeric(self, series_or_value):
        '''
        Converts a report value or pandas Series to numeric where possible.

        Invalid values are coerced to NaN. This is used for summary statistics
        without risking report-generation failure.

        params:
            series_or_value:
                Value or pandas Series to convert.

        returns:
            pandas Series or numeric-like object:
                Numeric-converted value where possible.
        '''
        try:
            return pd.to_numeric(series_or_value, errors='coerce')
        except Exception:
            return series_or_value

    def _format_auto_report_value(self, value, suffix=''):
        '''
        Formats values for human-readable Auto run Markdown reports.

        Floats are rounded to three decimal places and trailing zeros are
        removed. Missing values are reported as "not recorded".

        params:
            value:
                Value to format.

            str suffix:
                Optional unit suffix, such as "nm" or "uL".

        returns:
            str:
                Formatted display value.
        '''
        if value is None:
            return 'not recorded'

        try:
            if pd.isna(value):
                return 'not recorded'
        except Exception:
            pass

        if isinstance(value, float):
            value_text = f'{value:.3f}'.rstrip('0').rstrip('.')
        else:
            value_text = str(value)

        if suffix and value_text != 'not recorded':
            return f'{value_text} {suffix}'

        return value_text

    def _count_auto_report_status(self, df, column, status):
        '''
        Counts condition-level rows matching a categorical status.

        params:
            pandas.DataFrame df:
                Auto model performance dataframe.

            str column:
                Column containing the categorical status.

            str status:
                Status value to count.

        returns:
            int:
                Number of matching rows.
        '''
        if df.empty or column not in df.columns:
            return 0

        try:
            return int((df[column] == status).sum())
        except Exception:
            return 0

    def _format_auto_report_condition_list(self, condition_numbers):
        '''
        Formats reaction condition numbers for compact Markdown text.

        params:
            list condition_numbers:
                Reaction condition numbers.

        returns:
            str:
                Compact condition list.
        '''
        if len(condition_numbers) == 0:
            return 'none'

        unique_condition_numbers = sorted(set(condition_numbers))

        if len(unique_condition_numbers) <= 8:
            return ', '.join([str(x) for x in unique_condition_numbers])

        first_values = ', '.join(
            [str(x) for x in unique_condition_numbers[:6]]
        )

        return (
            f'{len(unique_condition_numbers)} conditions '
            f'({first_values}, ...)'
        )

    def _auto_report_file_line(self, relative_path, label):
        '''
        Creates a Markdown bullet describing whether an expected output file is
        present.

        params:
            str relative_path:
                Path relative to self.out_path.

            str label:
                Human-readable file label.

        returns:
            str:
                Markdown bullet line.
        '''
        full_path = os.path.join(self.out_path, relative_path)
        exists_text = 'present' if os.path.exists(full_path) else 'not found'
        return f'- {label}: `{relative_path}` ({exists_text})'

    def _auto_report_not_applicable_file_line(
        self,
        relative_path,
        label,
        reason
    ):
        '''
        Creates a Markdown bullet for an output which does not apply to the
        current number of variable reagents.

        Reporting a dimension-specific visualization as ``not applicable``
        distinguishes an intentionally ungenerated plot from a plot that
        should have been written but is missing.
        '''
        return (
            f'- {label}: `{relative_path}` '
            f'(not applicable: {reason})'
        )
    
    def _summarize_auto_run_status_for_report(self):
        '''
        Summarizes terminal-output run status for the Auto Markdown report.

        This is report-only. It reads Debug/terminal_output.txt if present and
        attempts to distinguish true Auto run failure from post-success cleanup
        warnings.

        params:
            None

        returns:
            dict:
                Run-status summary fields for auto_run_report.md.
        '''
        terminal_path = os.path.join(
            self.out_path,
            'Debug',
            'terminal_output.txt'
        )

        terminal_text = ''
        terminal_output_present = False
        terminal_read_error = None

        try:
            if os.path.exists(terminal_path):
                terminal_output_present = True
                with open(
                    terminal_path,
                    'r',
                    encoding='utf-8',
                    errors='replace'
                ) as terminal_file:
                    terminal_text = terminal_file.read()
        except Exception as exc:
            terminal_read_error = str(exc)
            terminal_text = ''

        success_marker_found = 'Success!!!' in terminal_text

        if 'Exit due to max_iters' in terminal_text:
            exit_reason = 'max_iters reached'
        elif 'max_iters' in terminal_text:
            exit_reason = 'max_iters mentioned'
        elif success_marker_found:
            exit_reason = 'success marker reached'
        elif terminal_output_present:
            exit_reason = 'not identified from terminal output'
        else:
            exit_reason = 'terminal output not found'

        success_index = terminal_text.find('Success!!!')

        if success_index >= 0:
            pre_success_text = terminal_text[:success_index]
            post_success_text = terminal_text[success_index:]
        else:
            pre_success_text = terminal_text
            post_success_text = ''

        pre_success_traceback_found = 'Traceback' in pre_success_text

        pre_success_exception_found = (
            'Exception' in pre_success_text
            or 'exception' in pre_success_text
        )

        post_success_cleanup_warning_found = False

        if success_marker_found:
            post_success_cleanup_warning_found = (
                'Traceback' in post_success_text
                or 'Exception' in post_success_text
                or 'exception' in post_success_text
                or 'WARNING' in post_success_text
                or 'warning' in post_success_text
                or 'Eve' in post_success_text
                or 'serial' in post_success_text
                or 'teardown' in post_success_text
            )


        performance_log_path = os.path.join(
            self.out_path,
            'pr_data',
            'auto_model_performance_log.csv'
        )

        experiment_data_path = os.path.join(
            self.out_path,
            'pr_data',
            'experiment_data.csv'
        )

        final_progress_plot_path = os.path.join(
            self.out_path,
            'Plots',
            'lambda_progress_final.png'
        )

        final_replicate_plot_path = os.path.join(
            self.out_path,
            'Plots',
            'lambda_replicates_final.png'
        )

        if success_marker_found and not pre_success_traceback_found:
            completion_status = 'Success'
        elif pre_success_traceback_found or pre_success_exception_found:
            completion_status = 'Possible failure before success marker'
        elif terminal_output_present:
            completion_status = 'Unknown'
        else:
            completion_status = 'Unknown; terminal output not found'

        if terminal_read_error is not None:
            completion_status = 'Unknown; terminal output read error'

        return {
            'completion_status': completion_status,
            'exit_reason': exit_reason,
            'terminal_output_present': terminal_output_present,
            'terminal_read_error': terminal_read_error,
            'success_marker_found': success_marker_found,
            'pre_success_traceback_found': pre_success_traceback_found,
            'pre_success_exception_found': pre_success_exception_found,
            'post_success_cleanup_warning_found': (
                post_success_cleanup_warning_found
            ),
            'experiment_data_exported': os.path.exists(experiment_data_path),
            'performance_log_exported': os.path.exists(performance_log_path),
            'final_progress_plot_exported': os.path.exists(
                final_progress_plot_path
            ),
            'final_replicate_plot_exported': os.path.exists(
                final_replicate_plot_path
            )
        }
    
    def _build_auto_run_status_report_lines(self, run_status_summary):
        '''
        Builds the Auto Run Status Markdown section.

        This helper is used both when auto_run_report.md is first written and
        when the Run Status section is refreshed later after the terminal log
        has received the final success/shutdown output.

        params:
            dict run_status_summary:
                Output from _summarize_auto_run_status_for_report().

        returns:
            list:
                Markdown lines for the Run Status section.
        '''
        lines = []

        lines.append('## Run Status')
        lines.append('')

        run_status_rows = [
            [
                'Completion status',
                run_status_summary['completion_status']
            ],
            [
                'Exit reason',
                run_status_summary['exit_reason']
            ],
            [
                'Terminal output present',
                'yes' if run_status_summary[
                    'terminal_output_present'
                ] else 'no'
            ],
            [
                'Success marker found',
                'yes' if run_status_summary[
                    'success_marker_found'
                ] else 'no'
            ],
            [
                'Pre-success traceback detected',
                'yes' if run_status_summary[
                    'pre_success_traceback_found'
                ] else 'no'
            ],
            [
                'Pre-success exception detected',
                'yes' if run_status_summary[
                    'pre_success_exception_found'
                ] else 'no'
            ],
            [
                'Post-success cleanup warning detected',
                'yes' if run_status_summary[
                    'post_success_cleanup_warning_found'
                ] else 'no'
            ],
            [
                'Experiment data exported',
                'yes' if run_status_summary[
                    'experiment_data_exported'
                ] else 'no'
            ],
            [
                'Performance log exported',
                'yes' if run_status_summary[
                    'performance_log_exported'
                ] else 'no'
            ],
            [
                'Final progress plot exported',
                'yes' if run_status_summary[
                    'final_progress_plot_exported'
                ] else 'no'
            ],
            [
                'Final replicate plot exported',
                'yes' if run_status_summary[
                    'final_replicate_plot_exported'
                ] else 'no'
            ]
        ]

        lines.extend(
            self._build_padded_auto_report_markdown_table(
                ['Field', 'Value'],
                run_status_rows,
                alignments=['left', 'left']
            )
        )

        lines.append('')

        if run_status_summary['completion_status'] == 'Success':
            if run_status_summary[
                'post_success_cleanup_warning_found'
            ]:
                lines.append(
                    'The Auto run appears to have completed successfully. '
                    'A possible post-success cleanup warning was detected in '
                    'the terminal output, so review `Debug/terminal_output.txt` '
                    'if device shutdown behavior needs to be audited.'
                )
            else:
                lines.append(
                    'The Auto run appears to have completed successfully, and '
                    'no post-success cleanup warning was detected.'
                )
        else:
            lines.append(
                'The Auto run status could not be confirmed as a clean '
                'success from the terminal output. Review '
                '`Debug/terminal_output.txt` before treating this run as final.'
            )

        if run_status_summary['terminal_read_error'] is not None:
            lines.append('')
            lines.append(
                'Terminal-output read error: '
                f"`{run_status_summary['terminal_read_error']}`"
            )

        lines.append('')

        return lines

    def _refresh_auto_run_status_section_in_report(self):
        '''
        Refreshes only the Run Status section in auto_run_report.md.

        This is used after the final success marker and normal shutdown steps
        have written additional terminal output. It keeps the scientific report
        content unchanged while updating the Run Status section from the more
        complete terminal log.

        params:
            None

        returns:
            bool:
                True if the Run Status section was refreshed, otherwise False.
        '''
        report_path = os.path.join(
            self.out_path,
            'pr_data',
            'auto_run_report.md'
        )

        if not os.path.exists(report_path):
            print(
                '<<controller warning>> Auto run report status refresh skipped: '
                f'report not found at {report_path}'
            )
            return False

        try:
            with open(report_path, 'r', encoding='utf-8') as report_file:
                report_text = report_file.read()
        except Exception as exc:
            print(
                '<<controller warning>> Auto run report status refresh skipped: '
                f'could not read report: {exc}'
            )
            return False

        section_start = report_text.find('## Run Status')
        section_end = report_text.find('## Experiment Overview')

        if section_start < 0 or section_end < 0 or section_end <= section_start:
            print(
                '<<controller warning>> Auto run report status refresh skipped: '
                'could not locate Run Status section boundaries.'
            )
            return False

        run_status_summary = self._summarize_auto_run_status_for_report()

        refreshed_status_text = '\n'.join(
            self._build_auto_run_status_report_lines(run_status_summary)
        )

        updated_report_text = (
            report_text[:section_start]
            + refreshed_status_text
            + '\n'
            + report_text[section_end:]
        )

        try:
            with open(report_path, 'w', encoding='utf-8') as report_file:
                report_file.write(updated_report_text)
        except Exception as exc:
            print(
                '<<controller warning>> Auto run report status refresh failed: '
                f'could not write report: {exc}'
            )
            return False

        print(
            '<<controller>> refreshed Auto run report Run Status section '
            f'at {report_path}'
        )

        return True
    
    def _escape_auto_report_markdown_table_value(self, value):
        '''
        Escapes values for safe insertion into a Markdown table cell.

        This prevents pipe characters or newlines inside values from breaking
        Markdown table structure.

        params:
            value:
                Value to escape.

        returns:
            str:
                Markdown-safe table-cell text.
        '''
        value_text = str(value)
        value_text = value_text.replace('|', '\\|')
        value_text = value_text.replace('\n', ' ')
        value_text = value_text.replace('\r', ' ')
        return value_text

    def _format_auto_report_table_value(
        self,
        value,
        suffix='',
        missing_value='—'
    ):
        '''
        Formats a scalar value for compact Markdown tables.

        This is intentionally more compact than _format_auto_report_value().
        Missing values are represented with an em dash so table columns remain
        readable and visually compact.

        params:
            value:
                Value to format.

            str suffix:
                Optional unit suffix, such as "nm" or "uL".

            str missing_value:
                Display text for missing values.

        returns:
            str:
                Markdown-safe formatted table value.
        '''
        if value is None:
            return missing_value

        try:
            if pd.isna(value):
                return missing_value
        except Exception:
            pass

        if isinstance(value, float):
            value_text = f'{value:.3f}'.rstrip('0').rstrip('.')
        else:
            value_text = str(value)

        if value_text in ['not recorded', 'None', 'nan', 'NaN', '']:
            return missing_value

        if suffix and value_text != missing_value:
            value_text = f'{value_text} {suffix}'

        return self._escape_auto_report_markdown_table_value(value_text)

    def _format_auto_report_volume_summary(
        self,
        fixed_volume_uL,
        variable_volume_uL,
        water_volume_uL,
        volume_feasible
    ):
        '''
        Formats a compact volume-balance summary for the Markdown report.

        Full per-reagent and feasibility JSON remains in the performance CSV.
        The report deliberately presents only the totals needed for a quick
        human review, preventing the provenance table from becoming too wide.
        '''
        numeric_values = [
            self._safe_auto_report_numeric(value)
            for value in [
                fixed_volume_uL,
                variable_volume_uL,
                water_volume_uL
            ]
        ]

        if all(value is None for value in numeric_values):
            return '—'

        total_volume_uL = (
            None
            if any(value is None for value in numeric_values)
            else sum(numeric_values)
        )

        return '; '.join([
            'fixed=' + self._format_auto_report_table_value(
                numeric_values[0], suffix='uL'
            ),
            'variable=' + self._format_auto_report_table_value(
                numeric_values[1], suffix='uL'
            ),
            'water=' + self._format_auto_report_table_value(
                numeric_values[2], suffix='uL'
            ),
            'total=' + self._format_auto_report_table_value(
                total_volume_uL, suffix='uL'
            ),
            'feasible=' + self._format_auto_report_table_value(
                volume_feasible
            )
        ])

    def _format_auto_report_optimizer_status(
        self,
        optimizer_method,
        optimizer_success,
        optimizer_status,
        optimizer_message=None
    ):
        '''Formats the selected SciPy optimizer outcome for report tables.'''
        if (
            optimizer_method is None
            and optimizer_success is None
            and optimizer_status is None
        ):
            return '—'

        success_text = (
            'success'
            if optimizer_success is True
            else 'not successful'
            if optimizer_success is False
            else 'unknown success'
        )

        status_parts = [
            self._format_auto_report_table_value(optimizer_method),
            'status=' + self._format_auto_report_table_value(
                optimizer_status
            ),
            success_text
        ]

        # A success status is adequately described by its method and code. A
        # failure needs its SciPy message in the human-readable report so it
        # cannot be mistaken for a normal successful selection.
        if optimizer_success is False and optimizer_message:
            status_parts.append(
                'message=' + self._format_auto_report_table_value(
                    optimizer_message
                )
            )

        return '; '.join(status_parts)

    def _format_auto_report_replicate_list_value(
        self,
        value,
        missing_value='—'
    ):
        '''
        Formats replicate-list values for compact Markdown report tables.

        Converts code-like list strings such as "[683.0, 824.0, 695.0]" into
        cleaner report text such as "683, 824, 695".

        params:
            value:
                Replicate list value. May be a list, tuple, numpy array, or
                string representation of a list.

            str missing_value:
                Display text for missing values.

        returns:
            str:
                Markdown-safe compact replicate-list text.
        '''
        if value is None:
            return missing_value

        try:
            if pd.isna(value):
                return missing_value
        except Exception:
            pass

        parsed_value = value

        if isinstance(value, str):
            stripped_value = value.strip()

            if stripped_value in [
                '',
                'not recorded',
                'None',
                'nan',
                'NaN',
                '[]'
            ]:
                return missing_value

            if (
                stripped_value.startswith('[')
                and stripped_value.endswith(']')
            ):
                try:
                    import ast
                    parsed_value = ast.literal_eval(stripped_value)
                except Exception:
                    parsed_value = stripped_value

        if isinstance(parsed_value, np.ndarray):
            parsed_value = parsed_value.tolist()

        if isinstance(parsed_value, (list, tuple)):
            formatted_values = []

            for item in parsed_value:
                if item is None:
                    continue

                try:
                    if pd.isna(item):
                        continue
                except Exception:
                    pass

                if isinstance(item, float):
                    item_text = f'{item:.3f}'.rstrip('0').rstrip('.')
                else:
                    item_text = str(item)

                if item_text not in ['', 'None', 'nan', 'NaN']:
                    formatted_values.append(item_text)

            if len(formatted_values) == 0:
                return missing_value

            return self._escape_auto_report_markdown_table_value(
                ', '.join(formatted_values)
            )

        return self._escape_auto_report_markdown_table_value(parsed_value)

    def _build_padded_auto_report_markdown_table(
        self,
        headers,
        rows,
        alignments=None
    ):
        '''
        Builds a padded Markdown table so the raw Markdown source is readable.

        Markdown renderers do not require aligned pipe columns, but padded
        source tables are much easier to inspect in notebooks, text editors,
        and Git diffs.

        params:
            list headers:
                Table header labels.

            list rows:
                List of row lists. Each row must have the same number of cells
                as headers. Values are converted to strings.

            list alignments:
                Optional list of alignment strings for each column. Supported
                values are "left", "right", and "center".

        returns:
            list:
                Markdown table lines.
        '''
        if alignments is None:
            alignments = ['left'] * len(headers)

        normalized_rows = []

        for row in rows:
            normalized_row = []

            for value in row:
                normalized_row.append(str(value))

            normalized_rows.append(normalized_row)

        table_values = [
            [str(header) for header in headers]
        ] + normalized_rows

        column_widths = []

        for column_i in range(len(headers)):
            max_width = 0

            for row in table_values:
                if column_i < len(row):
                    max_width = max(max_width, len(str(row[column_i])))

            column_widths.append(max(max_width, 3))

        def _format_cell(value, width, alignment):
            value_text = str(value)

            if alignment == 'right':
                return value_text.rjust(width)

            if alignment == 'center':
                return value_text.center(width)

            return value_text.ljust(width)

        header_cells = []

        for column_i, header in enumerate(headers):
            header_cells.append(
                _format_cell(
                    header,
                    column_widths[column_i],
                    alignments[column_i]
                )
            )

        table_lines = [
            '| ' + ' | '.join(header_cells) + ' |'
        ]

        separator_cells = []

        for column_i, alignment in enumerate(alignments):
            width = column_widths[column_i]

            if alignment == 'right':
                separator_cells.append('-' * (width - 1) + ':')

            elif alignment == 'center':
                if width <= 3:
                    separator_cells.append(':-:')
                else:
                    separator_cells.append(
                        ':' + '-' * (width - 2) + ':'
                    )

            else:
                separator_cells.append('-' * width)

        table_lines.append(
            '| ' + ' | '.join(separator_cells) + ' |'
        )

        for row in normalized_rows:
            row_cells = []

            for column_i in range(len(headers)):
                if column_i < len(row):
                    value = row[column_i]
                else:
                    value = ''

                row_cells.append(
                    _format_cell(
                        value,
                        column_widths[column_i],
                        alignments[column_i]
                    )
                )

            table_lines.append(
                '| ' + ' | '.join(row_cells) + ' |'
            )

        return table_lines
    
    def _format_auto_design_axis_label(self, reagent_name):
        '''
        Formats reagent-axis labels for Auto design-space plots.

        This intentionally matches the existing 2D GPR plot convention:
        <reagent_name> (mM)
        '''
        return f"{str(reagent_name)} (mM)"

    def _get_auto_design_plot_font_sizes(self, scale_factor=1.25):
        '''
        Returns centralized font sizes for Auto design-space plots.

        The standard sizes are intended to remain readable when figures are
        inserted into slides while preserving smaller proportional text for
        dense pairwise and three-dimensional figures.

        params:
            float scale_factor:
                Multiplicative scale applied to the original design-plot font
                sizes.

        returns:
            dict:
                Font sizes for standard, compact, and 3D design-space plots.
        '''
        return {
            'title': 11.0 * scale_factor,
            'axis_label': 10.0 * scale_factor,
            'tick_label': 10.0 * scale_factor,
            'legend': 8.0 * scale_factor,
            'annotation': 7.0 * scale_factor,
            'compact_axis_label': 8.5 * scale_factor,
            'compact_tick_label': 8.0 * scale_factor,
            'three_d_axis_label': 9.0 * scale_factor,
            'three_d_tick_label': 9.0 * scale_factor
        }

    def _get_auto_design_bound_value(
        self,
        raw_bounds,
        dimension_index,
        reagent_name
    ):
        '''
        Returns one configured Auto concentration bound as a finite float.

        Auto currently stores min_conc and max_conc as ordered lists, but this
        helper also supports dictionaries and pandas Series so the plotting
        layer remains robust if the storage representation changes later.

        params:
            object raw_bounds:
                Configured minimum- or maximum-concentration collection.

            int dimension_index:
                Position of the reagent in the complete variable-reagent order.

            str reagent_name:
                Variable-reagent name associated with the requested bound.

        returns:
            float:
                Configured finite bound, or numpy.nan when unavailable.
        '''
        if raw_bounds is None:
            return np.nan

        try:
            if isinstance(raw_bounds, dict):
                if reagent_name in raw_bounds:
                    bound_value = raw_bounds[reagent_name]

                else:
                    bound_value = np.nan
                    reagent_name_string = str(reagent_name)

                    for bound_name, candidate_value in raw_bounds.items():
                        if str(bound_name) == reagent_name_string:
                            bound_value = candidate_value
                            break

            elif isinstance(raw_bounds, pd.Series):
                if reagent_name in raw_bounds.index:
                    bound_value = raw_bounds.loc[reagent_name]

                elif dimension_index < len(raw_bounds):
                    bound_value = raw_bounds.iloc[dimension_index]

                else:
                    return np.nan

            elif dimension_index < len(raw_bounds):
                bound_value = raw_bounds[dimension_index]

            else:
                return np.nan

            bound_value = float(bound_value)

            if np.isfinite(bound_value):
                return bound_value

        except Exception:
            pass

        return np.nan

    def _get_auto_design_executable_bounds(
        self,
        design_columns,
        reference_df=None
    ):
        '''
        Returns authoritative concentration bounds for Auto design-space axes.

        Configured self.min_conc and self.max_conc values are preferred because
        they define the executable reagent space searched by Auto. The complete
        run dataframe may be supplied as a defensive fallback if a configured
        bound is missing or invalid.

        The returned bounds are keyed by concentration-column name so every
        renderer can apply identical limits to seed-only and full-exploration
        figures.

        params:
            list design_columns:
                Variable-reagent concentration-column definitions.

            pandas.DataFrame or None reference_df:
                Preferably the complete condition-level run dataframe. It is
                used only when configured executable bounds are unavailable.

        returns:
            dict:
                Mapping of concentration-column name to a
                (minimum_concentration, maximum_concentration) tuple.
        '''
        executable_bounds = {}

        raw_minimum_bounds = getattr(
            self,
            'min_conc',
            None
        )

        raw_maximum_bounds = getattr(
            self,
            'max_conc',
            None
        )

        raw_variable_reagents = getattr(
            self,
            'variable_reagents',
            None
        )

        # self.variable_reagents is commonly a NumPy array returned by
        # pandas.unique(). Never use ``array or []`` here because NumPy arrays
        # with more than one element do not have a single truth value.
        if raw_variable_reagents is None:
            variable_reagents = []

        elif isinstance(raw_variable_reagents, str):
            variable_reagents = [
                raw_variable_reagents
            ]

        else:
            try:
                variable_reagents = [
                    str(reagent_name)
                    for reagent_name in list(
                        raw_variable_reagents
                    )
                ]

            except TypeError:
                variable_reagents = [
                    str(raw_variable_reagents)
                ]

        reagent_dimension_lookup = {
            reagent_name: dimension_index
            for dimension_index, reagent_name in enumerate(
                variable_reagents
            )
        }

        for local_dimension_index, design_column in enumerate(
            design_columns
        ):
            reagent_name = str(
                design_column['reagent_name']
            )

            column_name = design_column['column_name']

            complete_dimension_index = reagent_dimension_lookup.get(
                reagent_name,
                local_dimension_index
            )

            minimum_value = self._get_auto_design_bound_value(
                raw_bounds=raw_minimum_bounds,
                dimension_index=complete_dimension_index,
                reagent_name=reagent_name
            )

            maximum_value = self._get_auto_design_bound_value(
                raw_bounds=raw_maximum_bounds,
                dimension_index=complete_dimension_index,
                reagent_name=reagent_name
            )

            configured_bounds_are_valid = (
                np.isfinite(minimum_value)
                and np.isfinite(maximum_value)
                and maximum_value > minimum_value
            )

            if not configured_bounds_are_valid:
                finite_reference_values = np.asarray(
                    [],
                    dtype=float
                )

                if (
                    reference_df is not None
                    and column_name in reference_df.columns
                ):
                    reference_values = pd.to_numeric(
                        reference_df[column_name],
                        errors='coerce'
                    ).to_numpy(dtype=float)

                    finite_reference_values = reference_values[
                        np.isfinite(reference_values)
                    ]

                if finite_reference_values.size > 0:
                    minimum_value = float(
                        np.min(finite_reference_values)
                    )

                    maximum_value = float(
                        np.max(finite_reference_values)
                    )

            if (
                not np.isfinite(minimum_value)
                or not np.isfinite(maximum_value)
                or maximum_value <= minimum_value
            ):
                if np.isfinite(minimum_value):
                    center_value = float(minimum_value)

                elif np.isfinite(maximum_value):
                    center_value = float(maximum_value)

                else:
                    center_value = 0.0

                fallback_span = max(
                    abs(center_value) * 0.10,
                    1.0e-6
                )

                minimum_value = (
                    center_value - fallback_span / 2.0
                )

                maximum_value = (
                    center_value + fallback_span / 2.0
                )

            executable_bounds[column_name] = (
                float(minimum_value),
                float(maximum_value)
            )

        return executable_bounds

    def _expand_auto_design_limits_for_display(
        self,
        lower_limit,
        upper_limit,
        padding_fraction=0.03
    ):
        '''
        Adds a small proportional display margin around scientific axis bounds.

        This margin prevents points located exactly at executable boundaries
        from being visually clipped. Seed-only and full-exploration plots must
        receive the same underlying bounds and padding fraction.

        params:
            float lower_limit:
                Scientific lower concentration or projection limit.

            float upper_limit:
                Scientific upper concentration or projection limit.

            float padding_fraction:
                Fraction of the complete span added to each side.

        returns:
            tuple:
                Padded lower and upper display limits.
        '''
        lower_limit = float(lower_limit)
        upper_limit = float(upper_limit)

        if (
            not np.isfinite(lower_limit)
            or not np.isfinite(upper_limit)
            or upper_limit <= lower_limit
        ):
            return lower_limit, upper_limit

        padding_fraction = max(
            0.0,
            float(padding_fraction)
        )

        limit_span = upper_limit - lower_limit
        display_padding = limit_span * padding_fraction

        return (
            lower_limit - display_padding,
            upper_limit + display_padding
        )

    def _apply_auto_design_square_box_aspect(self, ax):
        '''
        Makes a two-dimensional scientific plotting panel physically square.

        This controls the shape of the plotting box without falsely requiring
        unlike reagent units or unlike numerical ranges to use equal data-unit
        scaling.
        '''
        set_box_aspect = getattr(
            ax,
            'set_box_aspect',
            None
        )

        if callable(set_box_aspect):
            set_box_aspect(1.0)

    def _apply_auto_design_cubic_box_aspect(self, ax):
        '''
        Gives a three-dimensional reagent-space plot equal physical x, y, and z
        box dimensions.
        '''
        set_box_aspect = getattr(
            ax,
            'set_box_aspect',
            None
        )

        if callable(set_box_aspect):
            set_box_aspect(
                (1.0, 1.0, 1.0)
            )
    
    def _apply_auto_design_plot_lab_frame_style(
        self,
        ax,
        grid_axis='both',
        compact=False
    ):
        '''
        Applies lab-standard styling to two-dimensional Auto design-space axes.

        Grid lines remain visible because they help communicate reagent-space
        position. All four axis spines are shown to create a complete boxed
        frame, and tick-label sizes use the centralized presentation-ready font
        settings.

        params:
            matplotlib.axes.Axes ax:
                Two-dimensional axis to style.

            str grid_axis:
                Matplotlib grid axis selection: x, y, or both.

            bool compact:
                Uses smaller tick text for dense pairwise projection figures.

        returns:
            None
        '''
        font_sizes = self._get_auto_design_plot_font_sizes()

        ax.grid(
            True,
            axis=grid_axis,
            linestyle=':',
            linewidth=0.6,
            alpha=0.35
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)
            spine.set_color('0.2')

        if compact:
            tick_label_size = font_sizes['compact_tick_label']
        else:
            tick_label_size = font_sizes['tick_label']

        ax.tick_params(
            axis='both',
            which='both',
            direction='out',
            top=False,
            right=False,
            labelsize=tick_label_size
        )

    def _apply_auto_design_plot_3d_lab_frame_style(self, ax):
        '''
        Applies lab-standard styling to three-dimensional Auto design-space axes.

        Three-dimensional Matplotlib axes use pane edges instead of ordinary
        top/right spines. This helper keeps the 3D grid visible, strengthens the
        surrounding pane edges, and applies presentation-ready tick-label sizes.

        params:
            mpl_toolkits.mplot3d.axes3d.Axes3D ax:
                Three-dimensional axis to style.

        returns:
            None
        '''
        font_sizes = self._get_auto_design_plot_font_sizes()

        ax.grid(True)

        for axis_name in ['x', 'y', 'z']:
            try:
                ax.tick_params(
                    axis=axis_name,
                    which='major',
                    labelsize=font_sizes['three_d_tick_label']
                )
            except Exception:
                pass

        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            try:
                axis.pane.set_visible(True)
                axis.pane.set_edgecolor('0.2')
                axis.pane.set_linewidth(0.9)
            except Exception:
                pass

            try:
                axis.line.set_color('0.2')
                axis.line.set_linewidth(0.9)
            except Exception:
                pass
    
    def _get_auto_design_concentration_column(self, reagent_name):
        '''
        Returns the expected Auto performance-log concentration column for a
        variable reagent.
        '''
        return f"{str(reagent_name)}_concentration"

    def _get_auto_design_columns(self, performance_df):
        '''
        Identifies available variable-reagent concentration columns for
        design-space plotting.

        This is read-only and follows self.variable_reagents order so labels
        remain consistent with existing 2D GPR plots.
        '''
        design_columns = []

        if performance_df is None or performance_df.empty:
            return design_columns

        variable_reagents = getattr(self, 'variable_reagents', [])

        if variable_reagents is None:
            variable_reagents = []

        for reagent_name in variable_reagents:
            column_name = self._get_auto_design_concentration_column(
                reagent_name
            )

            if column_name in performance_df.columns:
                design_columns.append(
                    {
                        'reagent_name': str(reagent_name),
                        'column_name': column_name
                    }
                )

        return design_columns

    def _get_auto_design_plot_dataframe(self):
        '''
        Builds a private plotting dataframe for Auto design-space visualization.

        This method copies self.auto_model_performance_rows and never mutates
        Auto-loop data, experiment_data, recipes, QC data, model-training data,
        or optimizer state.
        '''
        try:
            performance_df = pd.DataFrame(
                getattr(self, 'auto_model_performance_rows', [])
            ).copy(deep=True)
        except Exception:
            performance_df = pd.DataFrame()

        if performance_df.empty:
            return performance_df, []

        design_columns = self._get_auto_design_columns(performance_df)

        for design_column in design_columns:
            column_name = design_column['column_name']
            performance_df[column_name] = pd.to_numeric(
                performance_df[column_name],
                errors='coerce'
            )

        for optional_numeric_column in [
            'reaction_number',
            'batch_number',
            'target_error_nm',
            'actual_lambda_mean_nm'
        ]:
            if optional_numeric_column in performance_df.columns:
                performance_df[optional_numeric_column] = pd.to_numeric(
                    performance_df[optional_numeric_column],
                    errors='coerce'
                )

        design_column_names = [
            design_column['column_name']
            for design_column in design_columns
        ]

        if len(design_column_names) > 0:
            performance_df = performance_df.dropna(
                subset=design_column_names
            ).copy(deep=True)

        if 'reaction_number' in performance_df.columns:
            performance_df = performance_df.sort_values(
                'reaction_number'
            ).copy(deep=True)

        return performance_df, design_columns

    def _get_auto_design_best_condition_number(self, performance_df):
        '''
        Returns the reaction_number with the smallest target_error_nm, if
        available. Used only for plot highlighting.
        '''
        if (
            performance_df is None
            or performance_df.empty
            or 'target_error_nm' not in performance_df.columns
            or 'reaction_number' not in performance_df.columns
        ):
            return None

        try:
            target_errors = pd.to_numeric(
                performance_df['target_error_nm'],
                errors='coerce'
            )

            valid_target_errors = target_errors.dropna()

            if len(valid_target_errors) == 0:
                return None

            best_index = valid_target_errors.idxmin()
            best_condition_number = performance_df.loc[
                best_index,
                'reaction_number'
            ]

            if pd.isna(best_condition_number):
                return None

            return best_condition_number

        except Exception:
            return None

    def _save_auto_design_plot(self, fig, plot_filename):
        '''
        Saves an Auto design-space plot to self.plot_path.
        '''
        os.makedirs(self.plot_path, exist_ok=True)

        plot_path = os.path.join(
            self.plot_path,
            plot_filename
        )

        fig.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)

        print(
            "<<controller>> saved Auto design-space plot to "
            f"{plot_path}"
        )

        return plot_path

    def _plot_auto_design_grouped_points_2d(
        self,
        ax,
        plot_df,
        x_column,
        y_column,
        best_condition_number=None,
        annotate_points=True
    ):
        '''
        Draws seed, optimizer-selected, other, and best-condition points on a
        two-dimensional Auto design-space axis.

        This shared renderer is used by the standard 2D design plot, pairwise
        projections, and PCA projections. Condition-number annotations use the
        centralized design-plot font sizes for presentation readability.

        params:
            matplotlib.axes.Axes ax:
                Axis on which the points are drawn.

            pandas.DataFrame plot_df:
                Copied condition-level plotting dataframe.

            str x_column:
                Dataframe column plotted on the x-axis.

            str y_column:
                Dataframe column plotted on the y-axis.

            numeric or None best_condition_number:
                Reaction condition number highlighted with a red star. Pass
                None when generating a seed-only maximin design figure.

            bool annotate_points:
                Whether reaction condition numbers should be displayed beside
                the plotted points.

        returns:
            tuple:
                legend_handles:
                    Matplotlib handles for categories that were plotted.

                legend_labels:
                    Corresponding legend labels.
        '''
        legend_handles = []
        legend_labels = []

        font_sizes = self._get_auto_design_plot_font_sizes()

        condition_type_series = plot_df.get(
            'condition_type',
            pd.Series('', index=plot_df.index)
        ).fillna('').astype(str).str.strip().str.lower()
        
        seed_mask = condition_type_series == 'seed'
        optimizer_mask = condition_type_series == 'optimizer_selected'
        other_mask = ~(seed_mask | optimizer_mask)

        if seed_mask.any():
            seed_handle = ax.scatter(
                plot_df.loc[seed_mask, x_column],
                plot_df.loc[seed_mask, y_column],
                s=42,
                marker='o',
                facecolors='none',
                edgecolors='tab:blue',
                linewidths=1.2,
                alpha=0.95,
                label='Seed condition'
            )
            legend_handles.append(seed_handle)
            legend_labels.append('Seed condition')

        if optimizer_mask.any():
            optimizer_handle = ax.scatter(
                plot_df.loc[optimizer_mask, x_column],
                plot_df.loc[optimizer_mask, y_column],
                s=44,
                marker='s',
                facecolors='none',
                edgecolors='tab:orange',
                linewidths=1.2,
                alpha=0.95,
                label='Optimizer-selected'
            )
            legend_handles.append(optimizer_handle)
            legend_labels.append('Optimizer-selected')

        if other_mask.any():
            other_handle = ax.scatter(
                plot_df.loc[other_mask, x_column],
                plot_df.loc[other_mask, y_column],
                s=38,
                marker='^',
                facecolors='none',
                edgecolors='0.35',
                linewidths=1.1,
                alpha=0.85,
                label='Other condition'
            )
            legend_handles.append(other_handle)
            legend_labels.append('Other condition')

        if (
            best_condition_number is not None
            and 'reaction_number' in plot_df.columns
        ):
            try:
                best_mask = (
                    plot_df['reaction_number'].astype(float)
                    == float(best_condition_number)
                )
            except Exception:
                best_mask = pd.Series(False, index=plot_df.index)

            if best_mask.any():
                best_handle = ax.scatter(
                    plot_df.loc[best_mask, x_column],
                    plot_df.loc[best_mask, y_column],
                    s=115,
                    marker='*',
                    color='tab:red',
                    linewidths=0.9,
                    alpha=0.95,
                    label='Best observed condition'
                )
                legend_handles.append(best_handle)
                legend_labels.append('Best observed condition')

        if annotate_points and 'reaction_number' in plot_df.columns:
            for _, row in plot_df.iterrows():
                try:
                    ax.annotate(
                        str(int(row['reaction_number'])),
                        (row[x_column], row[y_column]),
                        xytext=(4, 4),
                        textcoords='offset points',
                        fontsize=font_sizes['annotation'],
                        alpha=0.8
                    )
                except Exception:
                    continue

        return legend_handles, legend_labels

    def _plot_initial_training_design_1d(
        self,
        plot_df,
        design_columns,
        plot_filename='initial_training_design_1d.png',
        plot_title='Auto Design-Space Exploration',
        include_best_condition=True,
        design_space_reference_df=None
    ):
        '''
        Generates a one-dimensional Auto reagent-design-space strip plot.

        The figure intentionally remains wider than it is tall because it
        represents one continuous reagent axis rather than a two-dimensional
        spatial relationship.

        Configured executable concentration limits are used whenever available.
        A complete-run reference dataframe may be supplied so seed-only and
        full-exploration figures use identical fallback limits if configured
        bounds are unavailable.

        params:
            pandas.DataFrame plot_df:
                Condition-level rows that should appear in this plot.

            list design_columns:
                Exactly one variable-reagent concentration-column definition.

            str plot_filename:
                Filename for the saved plot.

            str plot_title:
                Figure title displayed above the legend.

            bool include_best_condition:
                If True, highlights the condition with the smallest recorded
                target error. Set False for seed-only maximin figures.

            pandas.DataFrame or None design_space_reference_df:
                Complete condition-level dataframe used for authoritative
                fallback bounds. The plotted rows are not changed.

        returns:
            str or None:
                Saved plot path, or None when the inputs are unsuitable.
        '''
        if plot_df.empty or len(design_columns) != 1:
            return None

        if design_space_reference_df is None:
            design_space_reference_df = plot_df

        font_sizes = self._get_auto_design_plot_font_sizes()

        design_column = design_columns[0]
        reagent_name = design_column['reagent_name']
        x_column = design_column['column_name']

        executable_bounds = (
            self._get_auto_design_executable_bounds(
                design_columns=design_columns,
                reference_df=design_space_reference_df
            )
        )

        if x_column not in executable_bounds:
            return None

        x_minimum, x_maximum = executable_bounds[x_column]

        x_display_minimum, x_display_maximum = (
            self._expand_auto_design_limits_for_display(
                lower_limit=x_minimum,
                upper_limit=x_maximum,
                padding_fraction=0.03
            )
        )

        fig, ax = plt.subplots(
            figsize=(7.6, 3.4),
            dpi=300
        )

        condition_type_series = plot_df.get(
            'condition_type',
            pd.Series('', index=plot_df.index)
        ).fillna('').astype(str).str.strip().str.lower()

        seed_mask = condition_type_series == 'seed'
        optimizer_mask = condition_type_series == 'optimizer_selected'
        other_mask = ~(seed_mask | optimizer_mask)

        y_positions = pd.Series(
            0.0,
            index=plot_df.index,
            dtype=float
        )

        y_positions.loc[optimizer_mask] = 0.08
        y_positions.loc[other_mask] = -0.08

        legend_handles = []
        legend_labels = []

        if seed_mask.any():
            seed_handle = ax.scatter(
                plot_df.loc[seed_mask, x_column],
                y_positions.loc[seed_mask],
                s=42,
                marker='o',
                facecolors='none',
                edgecolors='tab:blue',
                linewidths=1.2,
                alpha=0.95,
                label='Seed condition',
                zorder=3
            )

            legend_handles.append(seed_handle)
            legend_labels.append('Seed condition')

        if optimizer_mask.any():
            optimizer_handle = ax.scatter(
                plot_df.loc[optimizer_mask, x_column],
                y_positions.loc[optimizer_mask],
                s=44,
                marker='s',
                facecolors='none',
                edgecolors='tab:orange',
                linewidths=1.2,
                alpha=0.95,
                label='Optimizer-selected',
                zorder=3
            )

            legend_handles.append(optimizer_handle)
            legend_labels.append('Optimizer-selected')

        if other_mask.any():
            other_handle = ax.scatter(
                plot_df.loc[other_mask, x_column],
                y_positions.loc[other_mask],
                s=38,
                marker='^',
                facecolors='none',
                edgecolors='0.35',
                linewidths=1.1,
                alpha=0.85,
                label='Other condition',
                zorder=3
            )

            legend_handles.append(other_handle)
            legend_labels.append('Other condition')

        best_condition_number = None

        if include_best_condition:
            best_condition_number = (
                self._get_auto_design_best_condition_number(
                    plot_df
                )
            )

        if (
            best_condition_number is not None
            and 'reaction_number' in plot_df.columns
        ):
            try:
                best_mask = (
                    plot_df['reaction_number'].astype(float)
                    == float(best_condition_number)
                )

            except Exception:
                best_mask = pd.Series(
                    False,
                    index=plot_df.index
                )

            if best_mask.any():
                best_handle = ax.scatter(
                    plot_df.loc[best_mask, x_column],
                    y_positions.loc[best_mask],
                    s=115,
                    marker='*',
                    color='tab:red',
                    linewidths=0.9,
                    alpha=0.95,
                    label='Best observed condition',
                    zorder=4
                )

                legend_handles.append(best_handle)
                legend_labels.append('Best observed condition')

        if 'reaction_number' in plot_df.columns and len(plot_df) <= 20:
            for row_index, row in plot_df.iterrows():
                try:
                    ax.annotate(
                        str(int(row['reaction_number'])),
                        (
                            row[x_column],
                            y_positions.loc[row_index]
                        ),
                        xytext=(4, 5),
                        textcoords='offset points',
                        fontsize=font_sizes['annotation'],
                        alpha=0.8
                    )

                except Exception:
                    continue

        ax.set_xlabel(
            self._format_auto_design_axis_label(
                reagent_name
            ),
            fontsize=font_sizes['axis_label']
        )

        ax.set_xlim(
            x_display_minimum,
            x_display_maximum
        )

        ax.set_ylim(
            -0.19,
            0.19
        )

        ax.set_yticks([])

        ax.axhline(
            y=0.0,
            color='0.7',
            linewidth=0.8,
            zorder=1
        )

        self._apply_auto_design_plot_lab_frame_style(
            ax=ax,
            grid_axis='x',
            compact=False
        )

        # The frame helper may update tick formatting, so explicitly preserve
        # the intentionally blank categorical y-axis afterward.
        ax.set_yticks([])

        fig.suptitle(
            plot_title,
            fontsize=font_sizes['title'],
            fontweight='normal',
            y=0.965
        )

        if len(legend_handles) > 0:
            fig.legend(
                legend_handles,
                legend_labels,
                loc='upper center',
                bbox_to_anchor=(0.5, 0.84),
                ncol=min(len(legend_handles), 4),
                frameon=False,
                fontsize=font_sizes['legend'],
                handlelength=1.2,
                handletextpad=0.45,
                columnspacing=1.0
            )

        fig.subplots_adjust(
            left=0.09,
            right=0.97,
            bottom=0.25,
            top=0.67
        )

        return self._save_auto_design_plot(
            fig,
            plot_filename
        )

    def _plot_initial_training_design_2d(
        self,
        plot_df,
        design_columns,
        plot_filename='initial_training_design_2d.png',
        plot_title='Auto Design-Space Exploration',
        include_best_condition=True,
        design_space_reference_df=None
    ):
        '''
        Generates a square two-dimensional Auto reagent-design-space plot.

        The plotting panel is physically square so equal numerical reagent
        ranges receive equal visual treatment. Unlike-reagent numerical ranges
        are not forced to use equal data-unit scaling.

        Configured executable concentration limits are used whenever available.
        A complete-run reference dataframe may be supplied so seed-only and
        full-exploration figures use identical fallback limits if configured
        bounds are unavailable.

        params:
            pandas.DataFrame plot_df:
                Condition-level rows that should appear in this plot.

            list design_columns:
                Exactly two variable-reagent concentration-column definitions.

            str plot_filename:
                Filename for the saved plot.

            str plot_title:
                Figure title displayed above the legend.

            bool include_best_condition:
                If True, highlights the condition with the smallest recorded
                target error. Set False for seed-only maximin figures.

            pandas.DataFrame or None design_space_reference_df:
                Complete condition-level dataframe used for authoritative
                fallback bounds. The plotted rows are not changed.

        returns:
            str or None:
                Saved plot path, or None when the inputs are unsuitable.
        '''
        if plot_df.empty or len(design_columns) != 2:
            return None

        if design_space_reference_df is None:
            design_space_reference_df = plot_df

        font_sizes = self._get_auto_design_plot_font_sizes()

        x_design = design_columns[0]
        y_design = design_columns[1]

        x_column = x_design['column_name']
        y_column = y_design['column_name']

        executable_bounds = (
            self._get_auto_design_executable_bounds(
                design_columns=design_columns,
                reference_df=design_space_reference_df
            )
        )

        if (
            x_column not in executable_bounds
            or y_column not in executable_bounds
        ):
            return None

        x_minimum, x_maximum = executable_bounds[x_column]
        y_minimum, y_maximum = executable_bounds[y_column]

        x_display_minimum, x_display_maximum = (
            self._expand_auto_design_limits_for_display(
                lower_limit=x_minimum,
                upper_limit=x_maximum,
                padding_fraction=0.03
            )
        )

        y_display_minimum, y_display_maximum = (
            self._expand_auto_design_limits_for_display(
                lower_limit=y_minimum,
                upper_limit=y_maximum,
                padding_fraction=0.03
            )
        )

        fig, ax = plt.subplots(
            figsize=(6.4, 6.4),
            dpi=300
        )

        best_condition_number = None

        if include_best_condition:
            best_condition_number = (
                self._get_auto_design_best_condition_number(
                    plot_df
                )
            )

        legend_handles, legend_labels = (
            self._plot_auto_design_grouped_points_2d(
                ax=ax,
                plot_df=plot_df,
                x_column=x_column,
                y_column=y_column,
                best_condition_number=best_condition_number,
                annotate_points=(len(plot_df) <= 20)
            )
        )

        ax.set_xlabel(
            self._format_auto_design_axis_label(
                x_design['reagent_name']
            ),
            fontsize=font_sizes['axis_label']
        )

        ax.set_ylabel(
            self._format_auto_design_axis_label(
                y_design['reagent_name']
            ),
            fontsize=font_sizes['axis_label']
        )

        ax.set_xlim(
            x_display_minimum,
            x_display_maximum
        )

        ax.set_ylim(
            y_display_minimum,
            y_display_maximum
        )

        self._apply_auto_design_plot_lab_frame_style(
            ax=ax,
            grid_axis='both',
            compact=False
        )

        self._apply_auto_design_square_box_aspect(ax)

        fig.suptitle(
            plot_title,
            fontsize=font_sizes['title'],
            fontweight='normal',
            y=0.975
        )

        if len(legend_handles) > 0:
            fig.legend(
                legend_handles,
                legend_labels,
                loc='upper center',
                bbox_to_anchor=(0.5, 0.905),
                ncol=min(len(legend_handles), 4),
                frameon=False,
                fontsize=font_sizes['legend'],
                handlelength=1.2,
                handletextpad=0.45,
                columnspacing=1.0
            )

            top_margin = 0.78

        else:
            top_margin = 0.85

        fig.subplots_adjust(
            left=0.16,
            right=0.96,
            bottom=0.14,
            top=top_margin
        )

        return self._save_auto_design_plot(
            fig,
            plot_filename
        )

    def _plot_initial_training_design_3d(
        self,
        plot_df,
        design_columns,
        plot_filename='initial_training_design_3d.png',
        plot_title='Auto Design-Space Exploration',
        include_best_condition=True,
        design_space_reference_df=None
    ):
        '''
        Generates a three-dimensional Auto reagent-design-space scatter plot.

        The scientific plotting volume is physically cubic so the x, y, and z
        reagent dimensions receive equal visual treatment. This does not force
        different reagent concentration ranges to use identical data-unit
        scaling.

        Configured executable concentration limits are authoritative whenever
        available. A complete-run reference dataframe may be supplied so the
        seed-only and full-exploration figures use identical fallback limits.

        The title appears above the shared legend. Three-dimensional grid
        planes, pane boundaries, and axis lines remain visible because they
        communicate condition position within reagent space.

        params:
            pandas.DataFrame plot_df:
                Condition-level rows that should appear in this plot.

            list design_columns:
                Exactly three variable-reagent concentration-column
                definitions.

            str plot_filename:
                Filename for the saved plot.

            str plot_title:
                Figure title displayed above the legend.

            bool include_best_condition:
                If True, highlights the condition with the smallest recorded
                target error. Set False for seed-only maximin figures.

            pandas.DataFrame or None design_space_reference_df:
                Complete condition-level dataframe used for authoritative
                fallback bounds. The plotted rows are not changed.

        returns:
            str or None:
                Saved plot path, or None when the inputs are unsuitable.
        '''
        if plot_df.empty or len(design_columns) != 3:
            return None

        if design_space_reference_df is None:
            design_space_reference_df = plot_df

        font_sizes = self._get_auto_design_plot_font_sizes()

        x_design = design_columns[0]
        y_design = design_columns[1]
        z_design = design_columns[2]

        x_column = x_design['column_name']
        y_column = y_design['column_name']
        z_column = z_design['column_name']

        executable_bounds = (
            self._get_auto_design_executable_bounds(
                design_columns=design_columns,
                reference_df=design_space_reference_df
            )
        )

        required_columns = [
            x_column,
            y_column,
            z_column
        ]

        if any(
            column_name not in executable_bounds
            for column_name in required_columns
        ):
            return None

        x_minimum, x_maximum = executable_bounds[x_column]
        y_minimum, y_maximum = executable_bounds[y_column]
        z_minimum, z_maximum = executable_bounds[z_column]

        x_display_minimum, x_display_maximum = (
            self._expand_auto_design_limits_for_display(
                lower_limit=x_minimum,
                upper_limit=x_maximum,
                padding_fraction=0.03
            )
        )

        y_display_minimum, y_display_maximum = (
            self._expand_auto_design_limits_for_display(
                lower_limit=y_minimum,
                upper_limit=y_maximum,
                padding_fraction=0.03
            )
        )

        z_display_minimum, z_display_maximum = (
            self._expand_auto_design_limits_for_display(
                lower_limit=z_minimum,
                upper_limit=z_maximum,
                padding_fraction=0.03
            )
        )

        fig = plt.figure(
            figsize=(7.2, 6.4),
            dpi=300
        )

        ax = fig.add_subplot(
            111,
            projection='3d'
        )

        condition_type_series = plot_df.get(
            'condition_type',
            pd.Series('', index=plot_df.index)
        ).fillna('').astype(str).str.strip().str.lower()

        seed_mask = condition_type_series == 'seed'
        optimizer_mask = (
            condition_type_series == 'optimizer_selected'
        )
        other_mask = ~(
            seed_mask | optimizer_mask
        )

        # The best condition is a presentation category of its own.  Do not
        # also draw its seed/optimizer marker at the identical 3D coordinate:
        # Matplotlib's depth sorting can otherwise place that marker over the
        # red star even when the star is created later.  Excluding the point
        # from its underlying category changes neither the plotted data nor
        # the best-condition calculation; it only makes the highlighted point
        # visually unambiguous.
        best_condition_number = None
        best_mask = pd.Series(
            False,
            index=plot_df.index
        )

        if include_best_condition:
            best_condition_number = (
                self._get_auto_design_best_condition_number(
                    plot_df
                )
            )

        if (
            best_condition_number is not None
            and 'reaction_number' in plot_df.columns
        ):
            try:
                best_mask = (
                    plot_df['reaction_number'].astype(float)
                    == float(best_condition_number)
                )
            except Exception:
                best_mask = pd.Series(
                    False,
                    index=plot_df.index
                )

        display_seed_mask = seed_mask & ~best_mask
        display_optimizer_mask = optimizer_mask & ~best_mask
        display_other_mask = other_mask & ~best_mask

        legend_handles = []
        legend_labels = []

        if display_seed_mask.any():
            seed_handle = ax.scatter(
                plot_df.loc[display_seed_mask, x_column],
                plot_df.loc[display_seed_mask, y_column],
                plot_df.loc[display_seed_mask, z_column],
                s=46,
                marker='o',
                facecolors='none',
                edgecolors='tab:blue',
                linewidths=1.2,
                alpha=0.95,
                label='Seed condition',
                depthshade=False
            )

            legend_handles.append(seed_handle)
            legend_labels.append('Seed condition')

        if display_optimizer_mask.any():
            optimizer_handle = ax.scatter(
                plot_df.loc[display_optimizer_mask, x_column],
                plot_df.loc[display_optimizer_mask, y_column],
                plot_df.loc[display_optimizer_mask, z_column],
                s=48,
                marker='s',
                facecolors='none',
                edgecolors='tab:orange',
                linewidths=1.2,
                alpha=0.95,
                label='Optimizer-selected',
                depthshade=False
            )

            legend_handles.append(optimizer_handle)
            legend_labels.append('Optimizer-selected')

        if display_other_mask.any():
            other_handle = ax.scatter(
                plot_df.loc[display_other_mask, x_column],
                plot_df.loc[display_other_mask, y_column],
                plot_df.loc[display_other_mask, z_column],
                s=42,
                marker='^',
                facecolors='none',
                edgecolors='0.35',
                linewidths=1.1,
                alpha=0.85,
                label='Other condition',
                depthshade=False
            )

            legend_handles.append(other_handle)
            legend_labels.append('Other condition')

        if best_mask.any():
            best_handle = ax.scatter(
                plot_df.loc[best_mask, x_column],
                plot_df.loc[best_mask, y_column],
                plot_df.loc[best_mask, z_column],
                s=125,
                marker='*',
                color='tab:red',
                linewidths=0.9,
                alpha=0.95,
                label='Best observed condition',
                depthshade=False
            )

            legend_handles.append(best_handle)
            legend_labels.append(
                'Best observed condition'
            )

        if (
            'reaction_number' in plot_df.columns
            and len(plot_df) <= 20
        ):
            for _, row in plot_df.iterrows():
                try:
                    ax.text(
                        row[x_column],
                        row[y_column],
                        row[z_column],
                        str(int(row['reaction_number'])),
                        fontsize=font_sizes['annotation'],
                        alpha=0.8
                    )

                except Exception:
                    continue

        ax.set_xlim(
            x_display_minimum,
            x_display_maximum
        )

        ax.set_ylim(
            y_display_minimum,
            y_display_maximum
        )

        ax.set_zlim(
            z_display_minimum,
            z_display_maximum
        )

        ax.set_xlabel(
            self._format_auto_design_axis_label(
                x_design['reagent_name']
            ),
            fontsize=font_sizes['three_d_axis_label'],
            labelpad=10
        )

        ax.set_ylabel(
            self._format_auto_design_axis_label(
                y_design['reagent_name']
            ),
            fontsize=font_sizes['three_d_axis_label'],
            labelpad=10
        )

        ax.set_zlabel(
            self._format_auto_design_axis_label(
                z_design['reagent_name']
            ),
            fontsize=font_sizes['three_d_axis_label'],
            labelpad=10
        )

        self._apply_auto_design_plot_3d_lab_frame_style(
            ax
        )

        self._apply_auto_design_cubic_box_aspect(
            ax
        )

        # Use a stable viewing angle that exposes all three reagent axes while
        # preserving the underlying coordinate representation.
        ax.view_init(
            elev=22,
            azim=-55
        )

        fig.suptitle(
            plot_title,
            fontsize=font_sizes['title'],
            fontweight='normal',
            y=0.975
        )

        if len(legend_handles) > 0:
            fig.legend(
                legend_handles,
                legend_labels,
                loc='upper center',
                bbox_to_anchor=(0.5, 0.905),
                ncol=min(
                    len(legend_handles),
                    4
                ),
                frameon=False,
                fontsize=font_sizes['legend'],
                handlelength=1.2,
                handletextpad=0.45,
                columnspacing=1.0
            )

            top_margin = 0.80

        else:
            top_margin = 0.87

        fig.subplots_adjust(
            left=0.02,
            right=0.96,
            bottom=0.04,
            top=top_margin
        )

        return self._save_auto_design_plot(
            fig,
            plot_filename
        )

    def _plot_initial_training_design_pairwise(
        self,
        plot_df,
        design_columns,
        plot_filename='initial_training_design_pairwise.png',
        plot_title='Auto Design-Space Exploration: Pairwise Projections',
        include_best_condition=True,
        compact=False,
        max_compact_dimensions=6,
        design_space_reference_df=None
    ):
        '''
        Generates square pairwise projections of Auto reagent-design space.

        Each active subplot represents one two-reagent projection and uses a
        physically square plotting box. The numerical data-unit scales are not
        forced to be equal when the two reagents have different executable
        concentration ranges.

        Configured executable concentration limits are authoritative whenever
        available. A complete-run reference dataframe may be supplied so the
        seed-only and full-exploration figures use identical fallback limits.

        For three through six variable reagents, every two-reagent combination
        is displayed. For seven or more variable reagents, compact=True limits
        the figure to the first max_compact_dimensions reagents in the
        established variable-reagent order.

        Every active subplot retains horizontal and vertical design-space grid
        lines and a complete boxed frame. The figure title appears above the
        shared legend.

        params:
            pandas.DataFrame plot_df:
                Condition-level rows that should appear in the figure.

            list design_columns:
                Variable-reagent concentration-column definitions.

            str plot_filename:
                Filename for the saved plot.

            str plot_title:
                Main figure title displayed above the shared legend.

            bool include_best_condition:
                If True, highlights the condition with the smallest recorded
                target error. Set False for seed-only maximin figures.

            bool compact:
                If True, limits the number of reagent dimensions displayed.

            int max_compact_dimensions:
                Maximum number of reagent dimensions shown when compact=True.

            pandas.DataFrame or None design_space_reference_df:
                Complete condition-level dataframe used for authoritative
                fallback bounds. The plotted rows are not changed.

        returns:
            str or None:
                Saved plot path, or None when the inputs are unsuitable.
        '''
        if plot_df.empty or len(design_columns) < 2:
            return None

        if design_space_reference_df is None:
            design_space_reference_df = plot_df

        font_sizes = self._get_auto_design_plot_font_sizes()

        plot_design_columns = list(design_columns)

        if compact and len(plot_design_columns) > max_compact_dimensions:
            plot_design_columns = plot_design_columns[
                :max_compact_dimensions
            ]

        n_dimensions = len(plot_design_columns)

        if n_dimensions < 2:
            return None

        executable_bounds = (
            self._get_auto_design_executable_bounds(
                design_columns=plot_design_columns,
                reference_df=design_space_reference_df
            )
        )

        display_limits_by_column = {}

        for design_column in plot_design_columns:
            column_name = design_column['column_name']

            if column_name not in executable_bounds:
                return None

            lower_bound, upper_bound = executable_bounds[
                column_name
            ]

            display_limits_by_column[column_name] = (
                self._expand_auto_design_limits_for_display(
                    lower_limit=lower_bound,
                    upper_limit=upper_bound,
                    padding_fraction=0.03
                )
            )

        pairs = []

        for x_index in range(n_dimensions):
            for y_index in range(
                x_index + 1,
                n_dimensions
            ):
                pairs.append(
                    (x_index, y_index)
                )

        n_pairs = len(pairs)

        if n_pairs == 0:
            return None

        if n_pairs <= 3:
            n_cols = n_pairs

        elif n_pairs <= 6:
            n_cols = 3

        else:
            n_cols = 4

        n_rows = int(
            np.ceil(n_pairs / n_cols)
        )

        # The subplot boxes themselves are made square below. These figure
        # dimensions provide enough surrounding room for enlarged labels,
        # ticks, titles, and the shared legend without stretching the scientific
        # plotting panels.
        panel_width = 3.9
        panel_height = 3.9
        figure_header_height = 0.75

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(
                panel_width * n_cols,
                panel_height * n_rows + figure_header_height
            ),
            dpi=300,
            squeeze=False
        )

        axes = np.asarray(
            axes
        ).reshape(-1)

        best_condition_number = None

        if include_best_condition:
            best_condition_number = (
                self._get_auto_design_best_condition_number(
                    plot_df
                )
            )

        annotate_points = (
            len(plot_df) <= 12
            and n_pairs <= 6
        )

        # Dense multi-panel figures use the centralized compact font tier.
        # Three-panel figures retain the standard axis-label size.
        use_compact_style = (
            compact
            or n_pairs > 3
        )

        if use_compact_style:
            axis_label_size = (
                font_sizes['compact_axis_label']
            )

        else:
            axis_label_size = (
                font_sizes['axis_label']
            )

        final_legend_handles = []
        final_legend_labels = []

        for pair_number, (
            x_index,
            y_index
        ) in enumerate(pairs):
            ax = axes[pair_number]

            x_design = plot_design_columns[
                x_index
            ]

            y_design = plot_design_columns[
                y_index
            ]

            x_column = x_design[
                'column_name'
            ]

            y_column = y_design[
                'column_name'
            ]

            legend_handles, legend_labels = (
                self._plot_auto_design_grouped_points_2d(
                    ax=ax,
                    plot_df=plot_df,
                    x_column=x_column,
                    y_column=y_column,
                    best_condition_number=(
                        best_condition_number
                    ),
                    annotate_points=annotate_points
                )
            )

            if len(final_legend_handles) == 0:
                final_legend_handles = (
                    legend_handles
                )

                final_legend_labels = (
                    legend_labels
                )

            x_display_minimum, x_display_maximum = (
                display_limits_by_column[x_column]
            )

            y_display_minimum, y_display_maximum = (
                display_limits_by_column[y_column]
            )

            ax.set_xlim(
                x_display_minimum,
                x_display_maximum
            )

            ax.set_ylim(
                y_display_minimum,
                y_display_maximum
            )

            ax.set_xlabel(
                self._format_auto_design_axis_label(
                    x_design['reagent_name']
                ),
                fontsize=axis_label_size
            )

            ax.set_ylabel(
                self._format_auto_design_axis_label(
                    y_design['reagent_name']
                ),
                fontsize=axis_label_size
            )

            self._apply_auto_design_plot_lab_frame_style(
                ax=ax,
                grid_axis='both',
                compact=use_compact_style
            )

            self._apply_auto_design_square_box_aspect(
                ax
            )

        for empty_ax in axes[n_pairs:]:
            empty_ax.set_visible(False)

        display_title = plot_title

        if (
            compact
            and len(design_columns) > len(
                plot_design_columns
            )
        ):
            display_title = (
                f'{plot_title} '
                f'({len(plot_design_columns)} of '
                f'{len(design_columns)} variables shown)'
            )

        fig.suptitle(
            display_title,
            fontsize=font_sizes['title'],
            fontweight='normal',
            y=0.985
        )

        if len(final_legend_handles) > 0:
            fig.legend(
                final_legend_handles,
                final_legend_labels,
                loc='upper center',
                bbox_to_anchor=(0.5, 0.948),
                ncol=min(
                    len(final_legend_handles),
                    4
                ),
                frameon=False,
                fontsize=font_sizes['legend'],
                handlelength=1.2,
                handletextpad=0.45,
                columnspacing=1.0
            )

            top_margin = 0.875

        else:
            top_margin = 0.92

        fig.subplots_adjust(
            left=0.07,
            right=0.98,
            bottom=0.075,
            top=top_margin,
            hspace=0.50,
            wspace=0.42
        )

        return self._save_auto_design_plot(
            fig,
            plot_filename
        )

    def _plot_initial_training_design_parallel_coordinates(
        self,
        plot_df,
        design_columns,
        plot_filename=(
            'initial_training_design_parallel_coordinates.png'
        ),
        plot_title=(
            'Auto Design-Space Exploration: Parallel Coordinates'
        ),
        include_best_condition=True,
        design_space_reference_df=None
    ):
        '''
        Generates a parallel-coordinate summary of higher-dimensional Auto
        reagent-design space.

        The figure intentionally remains wide because each vertical axis
        represents a separate reagent dimension. Forcing this visualization
        into a square plotting area would reduce label readability and distort
        its intended multidimensional comparison.

        Every reagent concentration is normalized against the authoritative
        executable concentration range used by Auto. Configured min_conc and
        max_conc values are preferred. A complete-run reference dataframe may
        be supplied so seed-only and full-exploration figures use identical
        defensive fallback bounds.

        A normalized value of:

            0.0 represents the executable minimum concentration.
            1.0 represents the executable maximum concentration.

        Normalization is display-only and does not alter recipes, optimizer
        inputs, model-training data, QC decisions, or logged concentrations.

        The title is displayed above the legend. Horizontal normalized-value
        guides, vertical reagent guides, and the complete boxed plotting frame
        remain visible.

        params:
            pandas.DataFrame plot_df:
                Condition-level rows that should appear in the figure.

            list design_columns:
                Variable-reagent concentration-column definitions.

            str plot_filename:
                Filename for the saved plot.

            str plot_title:
                Figure title displayed above the legend.

            bool include_best_condition:
                If True, highlights the condition with the smallest recorded
                target error. Set False for seed-only maximin figures.

            pandas.DataFrame or None design_space_reference_df:
                Complete condition-level dataframe used for authoritative
                fallback bounds. The plotted rows are not changed.

        returns:
            str or None:
                Saved plot path, or None when the inputs are unsuitable.
        '''
        if plot_df.empty or len(design_columns) < 2:
            return None

        if design_space_reference_df is None:
            design_space_reference_df = plot_df

        font_sizes = self._get_auto_design_plot_font_sizes()

        column_names = [
            design_column['column_name']
            for design_column in design_columns
        ]

        reagent_names = [
            str(design_column['reagent_name'])
            for design_column in design_columns
        ]

        missing_plot_columns = [
            column_name
            for column_name in column_names
            if column_name not in plot_df.columns
        ]

        if len(missing_plot_columns) > 0:
            return None

        display_df = plot_df.copy(deep=True)

        executable_bounds = (
            self._get_auto_design_executable_bounds(
                design_columns=design_columns,
                reference_df=design_space_reference_df
            )
        )

        normalized_values = []

        for column_name in column_names:
            if column_name not in executable_bounds:
                return None

            minimum_value, maximum_value = (
                executable_bounds[column_name]
            )

            concentration_values = pd.to_numeric(
                display_df[column_name],
                errors='coerce'
            ).to_numpy(dtype=float)

            concentration_span = (
                maximum_value - minimum_value
            )

            if (
                not np.isfinite(concentration_span)
                or concentration_span <= 0
            ):
                return None

            normalized_column_values = (
                concentration_values - minimum_value
            ) / concentration_span

            normalized_values.append(
                normalized_column_values
            )

        normalized_array = np.vstack(
            normalized_values
        ).T

        if normalized_array.size == 0:
            return None

        fig, ax = plt.subplots(
            figsize=(
                max(
                    8.4,
                    len(column_names) * 1.15
                ),
                5.4
            ),
            dpi=300
        )

        x_positions = np.arange(
            len(column_names)
        )

        condition_type_series = display_df.get(
            'condition_type',
            pd.Series('', index=display_df.index)
        ).fillna('').astype(str).str.strip().str.lower()

        best_condition_number = None

        if include_best_condition:
            best_condition_number = (
                self._get_auto_design_best_condition_number(
                    display_df
                )
            )

        seed_present = False
        optimizer_present = False
        other_present = False
        best_present = False

        for row_position, (
            _,
            row
        ) in enumerate(display_df.iterrows()):
            if row_position >= normalized_array.shape[0]:
                continue

            y_values = normalized_array[
                row_position,
                :
            ]

            if np.isnan(y_values).all():
                continue

            condition_type = condition_type_series.iloc[
                row_position
            ]

            if condition_type == 'seed':
                line_color = 'tab:blue'
                line_alpha = 0.58
                line_width = 1.2
                seed_present = True

            elif condition_type == 'optimizer_selected':
                line_color = 'tab:orange'
                line_alpha = 0.74
                line_width = 1.3
                optimizer_present = True

            else:
                line_color = '0.45'
                line_alpha = 0.48
                line_width = 1.1
                other_present = True

            is_best_condition = False

            if (
                include_best_condition
                and best_condition_number is not None
                and 'reaction_number' in display_df.columns
            ):
                try:
                    is_best_condition = (
                        float(row['reaction_number'])
                        == float(best_condition_number)
                    )

                except Exception:
                    is_best_condition = False

            if is_best_condition:
                line_color = 'tab:red'
                line_alpha = 0.98
                line_width = 2.3
                best_present = True

            ax.plot(
                x_positions,
                y_values,
                color=line_color,
                alpha=line_alpha,
                linewidth=line_width,
                marker='o',
                markersize=3.2,
                zorder=4 if is_best_condition else 3
            )

            if (
                'reaction_number' in display_df.columns
                and len(display_df) <= 20
                and np.isfinite(y_values[-1])
            ):
                try:
                    ax.text(
                        x_positions[-1] + 0.05,
                        y_values[-1],
                        str(int(row['reaction_number'])),
                        fontsize=font_sizes['annotation'],
                        alpha=0.75,
                        va='center'
                    )

                except Exception:
                    pass

        ax.set_xticks(
            x_positions
        )

        ax.set_xticklabels(
            reagent_names,
            rotation=35,
            ha='right',
            fontsize=font_sizes['compact_axis_label']
        )

        ax.set_ylabel(
            (
                'Normalized concentration within '
                'executable reagent range'
            ),
            fontsize=font_sizes['axis_label']
        )

        ax.set_xlim(
            -0.15,
            len(column_names) - 1 + 0.35
        )

        # A small margin outside 0-1 prevents markers at exact executable
        # boundaries from being clipped while retaining the scientific meaning
        # of the normalized concentration scale.
        ax.set_ylim(
            -0.05,
            1.05
        )

        ax.set_yticks(
            np.linspace(
                0.0,
                1.0,
                6
            )
        )

        self._apply_auto_design_plot_lab_frame_style(
            ax=ax,
            grid_axis='both',
            compact=True
        )

        legend_handles = []
        legend_labels = []

        if seed_present:
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color='tab:blue',
                    linewidth=1.5,
                    marker='o',
                    markersize=4.0
                )
            )

            legend_labels.append(
                'Seed condition'
            )

        if optimizer_present:
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color='tab:orange',
                    linewidth=1.5,
                    marker='o',
                    markersize=4.0
                )
            )

            legend_labels.append(
                'Optimizer-selected'
            )

        if other_present:
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color='0.45',
                    linewidth=1.4,
                    marker='o',
                    markersize=4.0
                )
            )

            legend_labels.append(
                'Other condition'
            )

        if best_present:
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color='tab:red',
                    linewidth=2.3,
                    marker='o',
                    markersize=4.0
                )
            )

            legend_labels.append(
                'Best observed condition'
            )

        fig.suptitle(
            plot_title,
            fontsize=font_sizes['title'],
            fontweight='normal',
            y=0.975
        )

        if len(legend_handles) > 0:
            fig.legend(
                legend_handles,
                legend_labels,
                loc='upper center',
                bbox_to_anchor=(0.5, 0.895),
                ncol=min(
                    len(legend_handles),
                    4
                ),
                frameon=False,
                fontsize=font_sizes['legend'],
                handlelength=1.4,
                handletextpad=0.5,
                columnspacing=1.0
            )

            top_margin = 0.77

        else:
            top_margin = 0.84

        fig.subplots_adjust(
            left=0.11,
            right=0.94,
            bottom=0.25,
            top=top_margin
        )

        return self._save_auto_design_plot(
            fig,
            plot_filename
        )

    def _plot_initial_training_design_pca(
        self,
        plot_df,
        design_columns,
        plot_filename='initial_training_design_pca.png',
        plot_title='Auto Design-Space Exploration: PCA Projection',
        include_best_condition=True,
        projection_reference_df=None
    ):
        '''
        Generates a square two-component PCA projection of higher-dimensional
        Auto reagent-design space using NumPy singular-value decomposition.

        The caller may provide either the complete condition-level dataframe or
        a filtered seed-only dataframe. projection_reference_df establishes the
        scaling, center, component directions, and displayed PC1/PC2 limits.

        Passing the complete run as projection_reference_df for both seed-only
        and full-exploration figures ensures that identical conditions retain:

            - identical PCA coordinates,
            - identical x- and y-axis limits,
            - identical visual positions within the plotting panel.

        Concentrations are scaled against configured executable reagent bounds
        whenever those bounds are valid. The complete reference dataframe is
        used only as a defensive fallback. Scaling and PCA are display-only and
        do not modify recipes, optimizer inputs, model-training data, or logged
        concentrations.

        The scientific plotting panel is physically square. PC1 and PC2 are not
        forced to use equal numerical data-unit scales because their projected
        ranges may legitimately differ.

        params:
            pandas.DataFrame plot_df:
                Condition-level rows that should appear in this plot.

            list design_columns:
                Variable-reagent concentration-column definitions.

            str plot_filename:
                Filename for the saved plot.

            str plot_title:
                Figure title displayed above the legend.

            bool include_best_condition:
                If True, highlights the condition with the smallest recorded
                target error. Set False for seed-only maximin figures.

            pandas.DataFrame or None projection_reference_df:
                Complete condition-level dataframe used to establish executable
                scaling, PCA coordinates, and common display limits.

        returns:
            str or None:
                Saved plot path, or None when the inputs are unsuitable.
        '''
        if plot_df.empty or len(design_columns) < 2:
            return None

        if projection_reference_df is None:
            projection_reference_df = plot_df

        if projection_reference_df.empty:
            return None

        font_sizes = self._get_auto_design_plot_font_sizes()

        column_names = [
            design_column['column_name']
            for design_column in design_columns
        ]

        missing_display_columns = [
            column_name
            for column_name in column_names
            if column_name not in plot_df.columns
        ]

        missing_reference_columns = [
            column_name
            for column_name in column_names
            if column_name not in projection_reference_df.columns
        ]

        if (
            len(missing_display_columns) > 0
            or len(missing_reference_columns) > 0
        ):
            return None

        display_numeric_df = plot_df[
            column_names
        ].apply(
            pd.to_numeric,
            errors='coerce'
        )

        reference_numeric_df = projection_reference_df[
            column_names
        ].apply(
            pd.to_numeric,
            errors='coerce'
        )

        display_matrix = display_numeric_df.to_numpy(
            dtype=float
        )

        reference_matrix = reference_numeric_df.to_numpy(
            dtype=float
        )

        display_valid_mask = np.isfinite(
            display_matrix
        ).all(axis=1)

        reference_valid_mask = np.isfinite(
            reference_matrix
        ).all(axis=1)

        display_matrix = display_matrix[
            display_valid_mask,
            :
        ]

        reference_matrix = reference_matrix[
            reference_valid_mask,
            :
        ]

        display_plot_df = plot_df.loc[
            display_valid_mask
        ].copy(deep=True)

        if (
            display_matrix.shape[0] == 0
            or reference_matrix.shape[0] < 2
            or reference_matrix.shape[1] < 2
        ):
            return None

        executable_bounds = (
            self._get_auto_design_executable_bounds(
                design_columns=design_columns,
                reference_df=projection_reference_df
            )
        )

        scaling_minimums = []
        scaling_maximums = []

        for column_name in column_names:
            if column_name not in executable_bounds:
                return None

            minimum_value, maximum_value = (
                executable_bounds[column_name]
            )

            scaling_minimums.append(
                minimum_value
            )

            scaling_maximums.append(
                maximum_value
            )

        scaling_minimums = np.asarray(
            scaling_minimums,
            dtype=float
        )

        scaling_maximums = np.asarray(
            scaling_maximums,
            dtype=float
        )

        scaling_ranges = (
            scaling_maximums - scaling_minimums
        )

        scaling_ranges[
            ~np.isfinite(scaling_ranges)
            | (scaling_ranges <= 0)
        ] = 1.0

        reference_scaled = (
            reference_matrix - scaling_minimums
        ) / scaling_ranges

        display_scaled = (
            display_matrix - scaling_minimums
        ) / scaling_ranges

        reference_center = np.mean(
            reference_scaled,
            axis=0
        )

        reference_centered = (
            reference_scaled - reference_center
        )

        display_centered = (
            display_scaled - reference_center
        )

        try:
            _, singular_values, component_matrix = np.linalg.svd(
                reference_centered,
                full_matrices=False
            )

        except np.linalg.LinAlgError:
            return None

        if component_matrix.shape[0] < 2:
            return None

        reference_scores = (
            reference_centered @ component_matrix.T
        )

        display_scores = (
            display_centered @ component_matrix.T
        )

        if (
            reference_scores.shape[1] < 2
            or display_scores.shape[1] < 2
        ):
            return None

        variance_values = singular_values ** 2

        variance_total = float(
            np.sum(variance_values)
        )

        if variance_total > 0:
            explained_variance = (
                variance_values / variance_total
            )

        else:
            explained_variance = np.zeros_like(
                variance_values
            )

        pc1_variance_percent = (
            100.0 * float(explained_variance[0])
            if len(explained_variance) > 0
            else 0.0
        )

        pc2_variance_percent = (
            100.0 * float(explained_variance[1])
            if len(explained_variance) > 1
            else 0.0
        )

        def _get_common_projection_display_limits(
            reference_values
        ):
            '''
            Returns stable shared display limits for one PCA component.
            '''
            reference_values = np.asarray(
                reference_values,
                dtype=float
            )

            finite_values = reference_values[
                np.isfinite(reference_values)
            ]

            if finite_values.size == 0:
                return -0.5, 0.5

            lower_limit = float(
                np.min(finite_values)
            )

            upper_limit = float(
                np.max(finite_values)
            )

            if upper_limit > lower_limit:
                return (
                    self._expand_auto_design_limits_for_display(
                        lower_limit=lower_limit,
                        upper_limit=upper_limit,
                        padding_fraction=0.05
                    )
                )

            center_value = float(
                lower_limit
            )

            fallback_half_span = max(
                abs(center_value) * 0.05,
                0.05
            )

            return (
                center_value - fallback_half_span,
                center_value + fallback_half_span
            )

        pc1_display_minimum, pc1_display_maximum = (
            _get_common_projection_display_limits(
                reference_scores[:, 0]
            )
        )

        pc2_display_minimum, pc2_display_maximum = (
            _get_common_projection_display_limits(
                reference_scores[:, 1]
            )
        )

        pc1_column = '_auto_design_pc1'
        pc2_column = '_auto_design_pc2'

        display_plot_df[pc1_column] = (
            display_scores[:, 0]
        )

        display_plot_df[pc2_column] = (
            display_scores[:, 1]
        )

        fig, ax = plt.subplots(
            figsize=(6.4, 6.4),
            dpi=300
        )

        best_condition_number = None

        if include_best_condition:
            best_condition_number = (
                self._get_auto_design_best_condition_number(
                    display_plot_df
                )
            )

        legend_handles, legend_labels = (
            self._plot_auto_design_grouped_points_2d(
                ax=ax,
                plot_df=display_plot_df,
                x_column=pc1_column,
                y_column=pc2_column,
                best_condition_number=best_condition_number,
                annotate_points=(
                    len(display_plot_df) <= 20
                )
            )
        )

        ax.set_xlim(
            pc1_display_minimum,
            pc1_display_maximum
        )

        ax.set_ylim(
            pc2_display_minimum,
            pc2_display_maximum
        )

        ax.set_xlabel(
            (
                f'Principal component 1 '
                f'({pc1_variance_percent:.1f}% variance)'
            ),
            fontsize=font_sizes['axis_label']
        )

        ax.set_ylabel(
            (
                f'Principal component 2 '
                f'({pc2_variance_percent:.1f}% variance)'
            ),
            fontsize=font_sizes['axis_label']
        )

        self._apply_auto_design_plot_lab_frame_style(
            ax=ax,
            grid_axis='both',
            compact=False
        )

        self._apply_auto_design_square_box_aspect(
            ax
        )

        fig.suptitle(
            plot_title,
            fontsize=font_sizes['title'],
            fontweight='normal',
            y=0.975
        )

        if len(legend_handles) > 0:
            fig.legend(
                legend_handles,
                legend_labels,
                loc='upper center',
                bbox_to_anchor=(0.5, 0.905),
                ncol=min(
                    len(legend_handles),
                    4
                ),
                frameon=False,
                fontsize=font_sizes['legend'],
                handlelength=1.2,
                handletextpad=0.45,
                columnspacing=1.0
            )

            top_margin = 0.78

        else:
            top_margin = 0.85

        fig.subplots_adjust(
            left=0.15,
            right=0.96,
            bottom=0.14,
            top=top_margin
        )

        return self._save_auto_design_plot(
            fig,
            plot_filename
        )

    def _plot_initial_training_designs_after_run(self):
        '''
        Generates dimension-aware Auto reagent-design plots after an Auto run.

        Two complementary plot groups are generated:

            Initial maximin seed design:
                Includes only rows whose condition_type is seed.
                Optimizer-selected conditions and the best-observed-condition
                marker are intentionally omitted.

            Auto design-space exploration:
                Includes every available condition-level row, distinguishes
                seed and optimizer-selected conditions, and highlights the
                best observed condition when that information is available.

        Both members of each seed/exploration pair use the complete run as
        their scientific reference space. Configured executable min_conc and
        max_conc values remain authoritative whenever available.

        This ensures that:

            - seed-only and exploration reagent axes use identical limits,
            - seed coverage is not exaggerated by independent autoscaling,
            - pairwise projections use the same reagent limits wherever a
              reagent appears,
            - parallel-coordinate normalization is identical between figures,
            - PCA figures use the same scaling, center, component directions,
              and displayed PC limits.

        Plot geometry is selected according to scientific purpose:

            - 2D and pairwise panels are physically square,
            - 3D reagent space is physically cubic,
            - PCA uses a square plotting panel,
            - 1D remains a wide strip plot,
            - parallel coordinates remain wide for multidimensional clarity.

        This method is report/plot-only. It reads copies of
        self.auto_model_performance_rows and does not modify recipes, QC,
        model-training data, optimizer behavior, CSV logs, or robot execution.

        Automatic output behavior:

            0 variables:
                No design plots; warning only.

            1 variable:
                initial_maximin_seed_design_1d.png
                auto_design_space_exploration_1d.png

            2 variables:
                initial_maximin_seed_design_2d.png
                auto_design_space_exploration_2d.png

            3 variables:
                initial_maximin_seed_design_pairwise.png
                auto_design_space_exploration_pairwise.png
                initial_maximin_seed_design_3d.png
                auto_design_space_exploration_3d.png

            4-6 variables:
                initial_maximin_seed_design_pairwise.png
                auto_design_space_exploration_pairwise.png
                initial_maximin_seed_design_parallel_coordinates.png
                auto_design_space_exploration_parallel_coordinates.png

            7+ variables:
                initial_maximin_seed_design_pairwise_compact.png
                auto_design_space_exploration_pairwise_compact.png
                initial_maximin_seed_design_parallel_coordinates.png
                auto_design_space_exploration_parallel_coordinates.png
                initial_maximin_seed_design_pca.png
                auto_design_space_exploration_pca.png

        returns:
            list:
                Successfully generated plot paths.
        '''
        generated_plot_paths = []

        plot_df, design_columns = (
            self._get_auto_design_plot_dataframe()
        )

        if plot_df.empty:
            print(
                "<<controller warning>> skipping Auto design-space plots "
                "because no condition-level rows were available"
            )
            return generated_plot_paths

        n_design_dimensions = len(
            design_columns
        )

        if n_design_dimensions == 0:
            print(
                "<<controller warning>> skipping Auto design-space plots "
                "because no variable reagent concentration columns were "
                "detected"
            )
            return generated_plot_paths

        condition_type_series = plot_df.get(
            'condition_type',
            pd.Series('', index=plot_df.index)
        ).fillna('').astype(str).str.strip().str.lower()

        seed_plot_df = plot_df.loc[
            condition_type_series == 'seed'
        ].copy(deep=True)

        if seed_plot_df.empty:
            print(
                "<<controller warning>> no condition_type=seed rows were "
                "available; seed-only maximin plots will be skipped, but "
                "full Auto design-space exploration plots will still be "
                "generated"
            )

        def _try_design_plot(
            plot_function,
            plot_description
        ):
            '''
            Executes one report-only design plot without allowing a plotting
            error to interrupt Auto completion or other plot exports.
            '''
            try:
                plot_path = plot_function()

                if plot_path is not None:
                    generated_plot_paths.append(
                        plot_path
                    )

            except Exception as exc:
                print(
                    f"<<controller warning>> failed to generate "
                    f"{plot_description}; continuing Auto mode. "
                    f"Error: {exc}"
                )

        def _generate_seed_and_exploration_plots(
            plot_method,
            seed_filename,
            exploration_filename,
            seed_title,
            exploration_title,
            plot_description,
            shared_kwargs=None
        ):
            '''
            Calls one generalized renderer for both the seed-only dataframe and
            the complete condition-level dataframe.

            shared_kwargs contains the complete-run reference dataframe and any
            renderer-specific settings that must remain identical between the
            two exported figures.
            '''
            if shared_kwargs is None:
                shared_kwargs = {}

            if not seed_plot_df.empty:
                seed_call_kwargs = dict(
                    shared_kwargs
                )

                seed_call_kwargs.update(
                    {
                        'plot_df': seed_plot_df,
                        'design_columns': design_columns,
                        'plot_filename': seed_filename,
                        'plot_title': seed_title,
                        'include_best_condition': False
                    }
                )

                _try_design_plot(
                    lambda call_kwargs=seed_call_kwargs: (
                        plot_method(**call_kwargs)
                    ),
                    (
                        f"initial maximin seed-design "
                        f"{plot_description}"
                    )
                )

            exploration_call_kwargs = dict(
                shared_kwargs
            )

            exploration_call_kwargs.update(
                {
                    'plot_df': plot_df,
                    'design_columns': design_columns,
                    'plot_filename': exploration_filename,
                    'plot_title': exploration_title,
                    'include_best_condition': True
                }
            )

            _try_design_plot(
                lambda call_kwargs=exploration_call_kwargs: (
                    plot_method(**call_kwargs)
                ),
                (
                    f"Auto design-space exploration "
                    f"{plot_description}"
                )
            )

        common_design_space_reference = {
            'design_space_reference_df': plot_df
        }

        if n_design_dimensions == 1:
            _generate_seed_and_exploration_plots(
                plot_method=(
                    self._plot_initial_training_design_1d
                ),
                seed_filename=(
                    'initial_maximin_seed_design_1d.png'
                ),
                exploration_filename=(
                    'auto_design_space_exploration_1d.png'
                ),
                seed_title=(
                    'Initial Maximin Seed Design: 1D Reagent Space'
                ),
                exploration_title=(
                    'Auto Design-Space Exploration: 1D Reagent Space'
                ),
                plot_description='1D reagent-space plot',
                shared_kwargs=common_design_space_reference
            )

        elif n_design_dimensions == 2:
            _generate_seed_and_exploration_plots(
                plot_method=(
                    self._plot_initial_training_design_2d
                ),
                seed_filename=(
                    'initial_maximin_seed_design_2d.png'
                ),
                exploration_filename=(
                    'auto_design_space_exploration_2d.png'
                ),
                seed_title=(
                    'Initial Maximin Seed Design: 2D Reagent Space'
                ),
                exploration_title=(
                    'Auto Design-Space Exploration: 2D Reagent Space'
                ),
                plot_description='2D reagent-space plot',
                shared_kwargs=common_design_space_reference
            )

        elif n_design_dimensions == 3:
            pairwise_shared_kwargs = {
                'compact': False,
                'design_space_reference_df': plot_df
            }

            _generate_seed_and_exploration_plots(
                plot_method=(
                    self._plot_initial_training_design_pairwise
                ),
                seed_filename=(
                    'initial_maximin_seed_design_pairwise.png'
                ),
                exploration_filename=(
                    'auto_design_space_exploration_pairwise.png'
                ),
                seed_title=(
                    'Initial Maximin Seed Design: Pairwise Projections'
                ),
                exploration_title=(
                    'Auto Design-Space Exploration: Pairwise Projections'
                ),
                plot_description='pairwise projection plot',
                shared_kwargs=pairwise_shared_kwargs
            )

            _generate_seed_and_exploration_plots(
                plot_method=(
                    self._plot_initial_training_design_3d
                ),
                seed_filename=(
                    'initial_maximin_seed_design_3d.png'
                ),
                exploration_filename=(
                    'auto_design_space_exploration_3d.png'
                ),
                seed_title=(
                    'Initial Maximin Seed Design: 3D Reagent Space'
                ),
                exploration_title=(
                    'Auto Design-Space Exploration: 3D Reagent Space'
                ),
                plot_description='3D reagent-space plot',
                shared_kwargs=common_design_space_reference
            )

        elif n_design_dimensions <= 6:
            pairwise_shared_kwargs = {
                'compact': False,
                'design_space_reference_df': plot_df
            }

            _generate_seed_and_exploration_plots(
                plot_method=(
                    self._plot_initial_training_design_pairwise
                ),
                seed_filename=(
                    'initial_maximin_seed_design_pairwise.png'
                ),
                exploration_filename=(
                    'auto_design_space_exploration_pairwise.png'
                ),
                seed_title=(
                    'Initial Maximin Seed Design: Pairwise Projections'
                ),
                exploration_title=(
                    'Auto Design-Space Exploration: Pairwise Projections'
                ),
                plot_description='pairwise projection plot',
                shared_kwargs=pairwise_shared_kwargs
            )

            _generate_seed_and_exploration_plots(
                plot_method=(
                    self._plot_initial_training_design_parallel_coordinates
                ),
                seed_filename=(
                    'initial_maximin_seed_design_'
                    'parallel_coordinates.png'
                ),
                exploration_filename=(
                    'auto_design_space_exploration_'
                    'parallel_coordinates.png'
                ),
                seed_title=(
                    'Initial Maximin Seed Design: Parallel Coordinates'
                ),
                exploration_title=(
                    'Auto Design-Space Exploration: Parallel Coordinates'
                ),
                plot_description='parallel-coordinate plot',
                shared_kwargs=common_design_space_reference
            )

        else:
            compact_pairwise_shared_kwargs = {
                'compact': True,
                'design_space_reference_df': plot_df
            }

            _generate_seed_and_exploration_plots(
                plot_method=(
                    self._plot_initial_training_design_pairwise
                ),
                seed_filename=(
                    'initial_maximin_seed_design_'
                    'pairwise_compact.png'
                ),
                exploration_filename=(
                    'auto_design_space_exploration_'
                    'pairwise_compact.png'
                ),
                seed_title=(
                    'Initial Maximin Seed Design: '
                    'Compact Pairwise Projections'
                ),
                exploration_title=(
                    'Auto Design-Space Exploration: '
                    'Compact Pairwise Projections'
                ),
                plot_description='compact pairwise projection plot',
                shared_kwargs=compact_pairwise_shared_kwargs
            )

            _generate_seed_and_exploration_plots(
                plot_method=(
                    self._plot_initial_training_design_parallel_coordinates
                ),
                seed_filename=(
                    'initial_maximin_seed_design_'
                    'parallel_coordinates.png'
                ),
                exploration_filename=(
                    'auto_design_space_exploration_'
                    'parallel_coordinates.png'
                ),
                seed_title=(
                    'Initial Maximin Seed Design: Parallel Coordinates'
                ),
                exploration_title=(
                    'Auto Design-Space Exploration: Parallel Coordinates'
                ),
                plot_description='parallel-coordinate plot',
                shared_kwargs=common_design_space_reference
            )

            _generate_seed_and_exploration_plots(
                plot_method=(
                    self._plot_initial_training_design_pca
                ),
                seed_filename=(
                    'initial_maximin_seed_design_pca.png'
                ),
                exploration_filename=(
                    'auto_design_space_exploration_pca.png'
                ),
                seed_title=(
                    'Initial Maximin Seed Design: PCA Projection'
                ),
                exploration_title=(
                    'Auto Design-Space Exploration: PCA Projection'
                ),
                plot_description='PCA projection plot',
                shared_kwargs={
                    'projection_reference_df': plot_df
                }
            )

        if len(generated_plot_paths) == 0:
            print(
                "<<controller warning>> Auto design-space plotting completed "
                "without generating any plot files"
            )

        else:
            print(
                "<<controller>> generated Auto design-space plots: "
                + ", ".join(
                    generated_plot_paths
                )
            )

        return generated_plot_paths
    
    def _auto_report_plot_markdown_if_exists(
        self,
        plot_filename,
        title,
        caption=None
    ):
        '''
        Returns Markdown lines embedding a plot if the plot file exists.

        params:
            str plot_filename:
                Filename inside the Plots directory.

            str title:
                Section title for the plot.

            str caption:
                Optional explanatory caption.

        returns:
            list:
                Markdown lines. Empty list if the plot does not exist.
        '''
        plot_path = os.path.join(
            self.plot_path,
            plot_filename
        )

        if not os.path.exists(plot_path):
            return []

        lines = []

        lines.append(f'### {title}')
        lines.append('')

        if caption is not None:
            lines.append(caption)
            lines.append('')

        lines.append(
            f'![{title}](../Plots/{plot_filename})'
        )
        lines.append('')

        return lines
    
    def _write_auto_run_report(self):
        '''
        Writes a human-readable Markdown report for the completed Auto mode run.

        This v2 report is intended to function as an automated scientific
        notebook write-up. It summarizes the run, identifies the best condition,
        reports replicate QC behavior, interprets model prediction performance,
        summarizes recipe/volume feasibility, includes a compact condition
        table, and embeds references to final plots.

        This method is report-only. It does not change optimizer behavior, model
        training, recipe generation, plotting, robot actions, or raw data export.

        params:
            None

        returns:
            str:
                Path to the exported Auto run report Markdown file.
        '''
        report_dir = os.path.join(self.out_path, 'pr_data')
        os.makedirs(report_dir, exist_ok=True)

        report_path = os.path.join(report_dir, 'auto_run_report.md')

        n_conditions = len(getattr(self, 'auto_model_performance_rows', []))

        run_status_summary = self._summarize_auto_run_status_for_report()

        try:
            performance_df = pd.DataFrame(self.auto_model_performance_rows)
        except Exception:
            performance_df = pd.DataFrame()

        robo_params = getattr(self, 'robo_params', {})

        experiment_name = getattr(
            self,
            'rxn_sheet_name',
            getattr(self, 'experiment_name', None)
        )

        target_lambda_max_nm = robo_params.get(
            'target',
            robo_params.get('target_lambda_max_nm', None)
        )

        initial_data = robo_params.get('initial_data', None)
        max_iterations = robo_params.get('max_iterations', None)
        num_duplicates = robo_params.get('num_duplicates', None)
        allow_true_zero = robo_params.get('allow_true_zero', None)
        pi_legacy_tare_offset_g = robo_params.get(
            'pi_legacy_tare_offset_g',
            0.0
        )
        acquisition_mode = robo_params.get('acquisition_mode', 'exploit')
        acquisition_modes = list(
            robo_params.get('acquisition_modes', [acquisition_mode])
        )
        using_acquisition_portfolio = bool(
            robo_params.get(
                'using_acquisition_portfolio',
                len(acquisition_modes) > 1
            )
        )
        portfolio_min_distance = robo_params.get(
            'portfolio_min_distance',
            None
        )
        balanced_exploration_weight = robo_params.get(
            'balanced_exploration_weight',
            1.0
        )
        replicate_outlier_threshold_nm = robo_params.get(
            'replicate_outlier_threshold_nm',
            50.0
        )
        replicate_sd_tolerance_nm = robo_params.get(
            'replicate_sd_tolerance_nm',
            25.0
        )

        acquisition_objective_descriptions = {
            'exploit': (
                'Select the feasible recipe whose GP-predicted mean λmax is '
                'closest to the requested target.'
            ),
            'explore': (
                'Select the feasible recipe with the greatest GP predictive '
                'standard deviation.'
            ),
            'balanced': (
                'Trade absolute predicted target error against weighted GP '
                'predictive uncertainty using a target-aware straddle score.'
            ),
            'target_ei': (
                'Maximize the expected reduction in the best QC-approved '
                'and replicate-validated condition-level absolute target error '
                'achieved so far.'
            )
        }
        acquisition_objective_description = (
            acquisition_objective_descriptions.get(
                acquisition_mode,
                'Use the configured GP-guided acquisition score.'
            )
        )

        seed_conditions = self._count_auto_report_status(
            performance_df,
            'condition_type',
            'seed'
        )

        optimizer_conditions = self._count_auto_report_status(
            performance_df,
            'condition_type',
            'optimizer_selected'
        )

        qc_passed = self._count_auto_report_status(
            performance_df,
            'replicate_qc_status',
            'passed'
        )

        qc_not_applied = self._count_auto_report_status(
            performance_df,
            'replicate_qc_status',
            'not_applied'
        )

        qc_excluded = self._count_auto_report_status(
            performance_df,
            'replicate_qc_status',
            'excluded_replicate'
        )

        qc_flagged = self._count_auto_report_status(
            performance_df,
            'replicate_qc_status',
            'flagged_not_excluded'
        )

        conditions_used_all = self._count_auto_report_status(
            performance_df,
            'model_training_status',
            'used_all_valid_replicates'
        )

        conditions_used_qc_only = self._count_auto_report_status(
            performance_df,
            'model_training_status',
            'used_qc_included_only'
        )

        conditions_used_flagged = self._count_auto_report_status(
            performance_df,
            'model_training_status',
            'used_flagged_condition'
        )

        best_condition_row = None

        if (
            not performance_df.empty
            and 'target_error_nm' in performance_df.columns
        ):
            try:
                target_error_series = self._safe_auto_report_numeric(
                    performance_df['target_error_nm']
                )
                valid_target_errors = target_error_series.dropna()

                if len(valid_target_errors) > 0:
                    best_index = valid_target_errors.idxmin()
                    best_condition_row = performance_df.loc[best_index]
            except Exception:
                best_condition_row = None

        prediction_rows = 0
        prediction_error_mean = None
        prediction_error_median = None
        prediction_abs_error_mean = None
        prediction_abs_error_median = None
        prediction_bias_text = (
            'Prediction-bias interpretation was unavailable because no '
            'pre-experiment prediction errors were recorded.'
        )

        if (
            not performance_df.empty
            and 'prediction_error_nm' in performance_df.columns
        ):
            try:
                prediction_errors = self._safe_auto_report_numeric(
                    performance_df['prediction_error_nm']
                ).dropna()

                prediction_rows = int(len(prediction_errors))

                if prediction_rows > 0:
                    prediction_error_mean = float(prediction_errors.mean())
                    prediction_error_median = float(prediction_errors.median())
                    prediction_abs_error_mean = float(
                        prediction_errors.abs().mean()
                    )
                    prediction_abs_error_median = float(
                        prediction_errors.abs().median()
                    )

                    if prediction_error_mean > 0:
                        prediction_bias_text = (
                            'On average, observed λmax values were higher than '
                            'the pre-experiment GP predictions, indicating '
                            'model underprediction during the optimizer-selected '
                            'portion of this run.'
                        )
                    elif prediction_error_mean < 0:
                        prediction_bias_text = (
                            'On average, observed λmax values were lower than '
                            'the pre-experiment GP predictions, indicating '
                            'model overprediction during the optimizer-selected '
                            'portion of this run.'
                        )
                    else:
                        prediction_bias_text = (
                            'The mean signed prediction error was approximately '
                            'zero, suggesting no directional prediction bias in '
                            'the recorded optimizer-selected conditions.'
                        )
            except Exception:
                prediction_rows = 0
                prediction_error_mean = None
                prediction_error_median = None
                prediction_abs_error_mean = None
                prediction_abs_error_median = None

        target_error_mean = None
        target_error_median = None

        if (
            not performance_df.empty
            and 'target_error_nm' in performance_df.columns
        ):
            try:
                target_errors = self._safe_auto_report_numeric(
                    performance_df['target_error_nm']
                ).dropna()

                if len(target_errors) > 0:
                    target_error_mean = float(target_errors.mean())
                    target_error_median = float(target_errors.median())
            except Exception:
                target_error_mean = None
                target_error_median = None

        volume_infeasible_count = 0
        water_not_executable_count = 0

        if not performance_df.empty:
            if 'volume_feasible' in performance_df.columns:
                try:
                    volume_infeasible_count = int(
                        (performance_df['volume_feasible'] == False).sum()
                    )
                except Exception:
                    volume_infeasible_count = 0

            if 'water_transfer_executable' in performance_df.columns:
                try:
                    water_not_executable_count = int(
                        (
                            performance_df['water_transfer_executable']
                            == False
                        ).sum()
                    )
                except Exception:
                    water_not_executable_count = 0

        water_min = None
        water_max = None
        variable_volume_min = None
        variable_volume_max = None
        total_volume_min = None
        total_volume_max = None

        if not performance_df.empty:
            if 'water_volume_uL' in performance_df.columns:
                try:
                    water_values = self._safe_auto_report_numeric(
                        performance_df['water_volume_uL']
                    ).dropna()

                    if len(water_values) > 0:
                        water_min = float(water_values.min())
                        water_max = float(water_values.max())
                except Exception:
                    water_min = None
                    water_max = None

            if 'variable_volume_total_uL' in performance_df.columns:
                try:
                    variable_values = self._safe_auto_report_numeric(
                        performance_df['variable_volume_total_uL']
                    ).dropna()

                    if len(variable_values) > 0:
                        variable_volume_min = float(variable_values.min())
                        variable_volume_max = float(variable_values.max())
                except Exception:
                    variable_volume_min = None
                    variable_volume_max = None

            if 'total_volume_uL' in performance_df.columns:
                try:
                    total_values = self._safe_auto_report_numeric(
                        performance_df['total_volume_uL']
                    ).dropna()

                    if len(total_values) > 0:
                        total_volume_min = float(total_values.min())
                        total_volume_max = float(total_values.max())
                except Exception:
                    total_volume_min = None
                    total_volume_max = None

        best_reaction_number = 'not recorded'
        best_batch_number = 'not recorded'
        best_condition_type = 'not recorded'
        best_lambda_mean = None
        best_target_error = None
        best_qc_status = 'not recorded'
        best_target_eligible = False
        best_target_eligibility_status = 'not recorded'

        if best_condition_row is not None:
            best_reaction_number = self._safe_auto_report_get(
                best_condition_row,
                'reaction_number'
            )
            best_batch_number = self._safe_auto_report_get(
                best_condition_row,
                'batch_number'
            )
            best_condition_type = self._safe_auto_report_get(
                best_condition_row,
                'condition_type'
            )
            best_lambda_mean = self._safe_auto_report_get(
                best_condition_row,
                'actual_lambda_mean_nm',
                None
            )
            best_target_error = self._safe_auto_report_get(
                best_condition_row,
                'target_error_nm',
                None
            )
            best_qc_status = self._safe_auto_report_get(
                best_condition_row,
                'replicate_qc_status'
            )
            best_target_eligible = (
                self._safe_auto_report_get(
                    best_condition_row,
                    'eligible_for_target_incumbent',
                    False
                ) == True
            )
            best_target_eligibility_status = self._safe_auto_report_get(
                best_condition_row,
                'target_eligibility_status'
            )

        if best_condition_row is None:
            executive_summary = (
                f'Auto mode completed {n_conditions} condition-level rows, but '
                'no best condition could be identified because no valid '
                '`target_error_nm` values were available.'
            )
        else:
            executive_summary = (
                f'Auto mode completed {n_conditions} unique reaction '
                f'conditions, including {seed_conditions} seed condition(s) '
                f'and {optimizer_conditions} optimizer-selected condition(s). '
                f'The target λmax was '
                f'{self._format_auto_report_value(target_lambda_max_nm, "nm")}. '
                f'The closest observed QC-cleaned condition was condition '
                f'{best_reaction_number}, with mean λmax '
                f'{self._format_auto_report_value(best_lambda_mean, "nm")} '
                f'and target error '
                f'{self._format_auto_report_value(best_target_error, "nm")}. '
                f'Target-decision eligible: {best_target_eligible}. '
                f'Replicate QC excluded clear outlier replicate(s) in '
                f'{qc_excluded} condition(s) and flagged {qc_flagged} '
                f'ambiguous condition(s) without automatic exclusion. '
                f'Volume-infeasible condition rows: '
                f'{volume_infeasible_count}; water-transfer-not-executable '
                f'condition rows: {water_not_executable_count}.'
            )

        if best_condition_row is None:
            best_interpretation = (
                'No best-condition interpretation is available because no '
                'valid target-error values were recorded.'
            )
        elif best_condition_type == 'optimizer_selected':
            best_interpretation = (
                'The best observed condition was optimizer-selected rather '
                'than part of the initial seed design, suggesting that the '
                'GP-guided target optimizer identified a condition at least as '
                'close to the target as the initial design during this run.'
            )
        elif best_condition_type == 'seed':
            best_interpretation = (
                'The best observed condition came from the initial seed design. '
                'In this run, later optimizer-selected conditions did not '
                'surpass the best seed condition with respect to absolute '
                'target error.'
            )
        else:
            best_interpretation = (
                'The best observed condition was identified, but its condition '
                'type was not recorded clearly enough for interpretation.'
            )

        if best_target_error is not None:
            try:
                if float(best_target_error) <= 10.0:
                    if best_target_eligible:
                        best_interpretation += (
                            ' The best condition was within 10 nm of the target '
                            'and passed replicate validation, so it is a '
                            'validated target hit for incumbent and stopping '
                            'decisions.'
                        )
                    else:
                        best_interpretation += (
                            ' The best condition was numerically within 10 nm '
                            'of the target, but it did not pass replicate '
                            'validation and therefore cannot establish a '
                            'target-EI incumbent or authorize a target-based '
                            'stop.'
                        )
                elif float(best_target_error) <= 25.0:
                    best_interpretation += (
                        ' The best condition was within 25 nm of the target, '
                        'indicating close approach to the requested wavelength.'
                    )
                else:
                    best_interpretation += (
                        ' The best condition approached the requested target '
                        'but did not reach a close-target threshold in this run.'
                    )
            except Exception:
                pass

        if qc_excluded == 0 and qc_flagged == 0:
            qc_interpretation = (
                'Replicate QC did not identify excluded or flagged conditions. '
                'The replicate data appear internally consistent under the '
                'current QC threshold.'
            )
        elif qc_excluded > 0 and qc_flagged == 0:
            qc_interpretation = (
                f'Replicate QC excluded clear isolated outlier replicate(s) in '
                f'{qc_excluded} condition(s). No ambiguous flagged conditions '
                f'were recorded. This suggests the QC system removed isolated '
                f'outliers while otherwise preserving condition-level data.'
            )
        elif qc_excluded == 0 and qc_flagged > 0:
            qc_interpretation = (
                f'Replicate QC flagged {qc_flagged} condition(s) but did not '
                f'exclude replicates. This indicates ambiguous replicate '
                f'variability where no clear two-against-one outlier pattern '
                f'was detected, so the data-preserving policy retained the '
                f'condition(s) for model training.'
            )
        else:
            qc_interpretation = (
                f'Replicate QC excluded clear isolated outlier replicate(s) in '
                f'{qc_excluded} condition(s) and flagged {qc_flagged} '
                f'ambiguous condition(s) without exclusion. This indicates '
                f'measurable replicate variability, while preserving ambiguous '
                f'data and excluding only clearer isolated outliers.'
            )

        acquisition_audit_table_lines = []

        if (
            performance_df.empty
            or 'condition_type' not in performance_df.columns
        ):
            acquisition_audit_df = pd.DataFrame()
        else:
            acquisition_audit_df = performance_df[
                performance_df['condition_type'] == 'optimizer_selected'
            ]

        if acquisition_audit_df.empty:
            acquisition_audit_table_lines.append(
                'No optimizer-selected conditions were available for the '
                'acquisition audit trail.'
            )
        else:
            acquisition_audit_headers = [
                'Condition',
                'Batch',
                'Mode',
                'Balanced weight',
                'Score',
                'Predicted target error',
                'Predicted λmax',
                'GP SD',
                'Incumbent target error',
                'Selected mask',
                'SciPy result',
                'Recipe repaired',
                'Masks evaluated'
            ]
            acquisition_audit_alignments = [
                'right',
                'right',
                'left',
                'right',
                'right',
                'right',
                'right',
                'right',
                'right',
                'left',
                'left',
                'left',
                'right'
            ]
            acquisition_audit_rows = []

            for _, row in acquisition_audit_df.sort_values(
                'reaction_number'
            ).iterrows():
                acquisition_audit_rows.append(
                    [
                        self._format_auto_report_table_value(
                            self._safe_auto_report_get(
                                row,
                                'reaction_number',
                                None
                            )
                        ),
                        self._format_auto_report_table_value(
                            self._safe_auto_report_get(
                                row,
                                'batch_number',
                                None
                            )
                        ),
                        self._format_auto_report_table_value(
                            self._safe_auto_report_get(
                                row,
                                'acquisition_mode',
                                None
                            )
                        ),
                        self._format_auto_report_table_value(
                            self._safe_auto_report_get(
                                row,
                                'balanced_exploration_weight',
                                None
                            )
                        ),
                        self._format_auto_report_table_value(
                            self._safe_auto_report_get(
                                row,
                                'acquisition_score',
                                None
                            )
                        ),
                        self._format_auto_report_table_value(
                            self._safe_auto_report_get(
                                row,
                                'predicted_target_error_nm',
                                None
                            ),
                            suffix='nm'
                        ),
                        self._format_auto_report_table_value(
                            self._safe_auto_report_get(
                                row,
                                'predicted_lambda_mean_nm',
                                None
                            ),
                            suffix='nm'
                        ),
                        self._format_auto_report_table_value(
                            self._safe_auto_report_get(
                                row,
                                'predicted_lambda_std_nm',
                                None
                            ),
                            suffix='nm'
                        ),
                        self._format_auto_report_table_value(
                            self._safe_auto_report_get(
                                row,
                                'incumbent_target_error_nm',
                                None
                            ),
                            suffix='nm'
                        ),
                        self._format_auto_report_table_value(
                            self._safe_auto_report_get(
                                row,
                                'selected_mask',
                                None
                            )
                        ),
                        self._format_auto_report_optimizer_status(
                            self._safe_auto_report_get(
                                row,
                                'optimizer_method',
                                None
                            ),
                            self._safe_auto_report_get(
                                row,
                                'optimizer_success',
                                None
                            ),
                            self._safe_auto_report_get(
                                row,
                                'optimizer_status',
                                None
                            ),
                            self._safe_auto_report_get(
                                row,
                                'optimizer_message',
                                None
                            )
                        ),
                        self._format_auto_report_table_value(
                            self._safe_auto_report_get(
                                row,
                                'optimizer_recipe_repaired',
                                None
                            )
                        ),
                        self._format_auto_report_table_value(
                            self._safe_auto_report_get(
                                row,
                                'mask_result_count',
                                None
                            )
                        )
                    ]
                )

            acquisition_audit_table_lines.extend(
                self._build_padded_auto_report_markdown_table(
                    headers=acquisition_audit_headers,
                    rows=acquisition_audit_rows,
                    alignments=acquisition_audit_alignments
                )
            )

        acquisition_provenance_table_lines = []

        if acquisition_audit_df.empty:
            acquisition_provenance_table_lines.append(
                'No optimizer-selected recipe provenance was available.'
            )
        else:
            provenance_headers = [
                'Condition',
                'Selected normalized recipe',
                'Executed normalized recipe',
                'Selected physical recipe',
                'Executed physical recipe',
                'Selected volume summary',
                'Executed volume summary'
            ]
            provenance_alignments = [
                'right',
                'left',
                'left',
                'left',
                'left',
                'left',
                'left'
            ]
            provenance_rows = []

            for _, row in acquisition_audit_df.sort_values(
                'reaction_number'
            ).iterrows():
                provenance_rows.append([
                    self._format_auto_report_table_value(
                        self._safe_auto_report_get(
                            row,
                            'reaction_number',
                            None
                        )
                    ),
                    self._format_auto_report_table_value(
                        self._safe_auto_report_get(
                            row,
                            'selected_normalized_recipe',
                            None
                        )
                    ),
                    self._format_auto_report_table_value(
                        self._safe_auto_report_get(
                            row,
                            'executed_normalized_recipe',
                            None
                        )
                    ),
                    self._format_auto_report_table_value(
                        self._safe_auto_report_get(
                            row,
                            'selected_physical_recipe',
                            None
                        )
                    ),
                    self._format_auto_report_table_value(
                        self._safe_auto_report_get(
                            row,
                            'executed_physical_recipe',
                            None
                        )
                    ),
                    self._format_auto_report_volume_summary(
                        self._safe_auto_report_get(
                            row,
                            'selected_fixed_volume_total_uL',
                            None
                        ),
                        self._safe_auto_report_get(
                            row,
                            'selected_variable_volume_total_uL',
                            None
                        ),
                        self._safe_auto_report_get(
                            row,
                            'selected_water_volume_uL',
                            None
                        ),
                        self._safe_auto_report_get(
                            row,
                            'selected_volume_feasible',
                            None
                        )
                    ),
                    self._format_auto_report_volume_summary(
                        self._safe_auto_report_get(
                            row,
                            'executed_fixed_volume_total_uL',
                            None
                        ),
                        self._safe_auto_report_get(
                            row,
                            'executed_variable_volume_total_uL',
                            None
                        ),
                        self._safe_auto_report_get(
                            row,
                            'executed_water_volume_uL',
                            None
                        ),
                        self._safe_auto_report_get(
                            row,
                            'executed_volume_feasible',
                            None
                        )
                    )
                ])

            acquisition_provenance_table_lines.extend(
                self._build_padded_auto_report_markdown_table(
                    headers=provenance_headers,
                    rows=provenance_rows,
                    alignments=provenance_alignments
                )
            )

        condition_table_lines = []

        if performance_df.empty:
            condition_table_lines.append(
                'No condition-level rows were available for tabulation.'
            )
        else:
            condition_table_headers = [
                'Condition',
                'Batch',
                'Type',
                'Predicted λmax',
                'GP SD',
                'Raw λmax values',
                'QC-used λmax values',
                'Mean λmax',
                'Target error',
                'QC status',
                'Target eligible',
                'Target eligibility status'
            ]

            condition_table_alignments = [
                'right',
                'right',
                'left',
                'right',
                'right',
                'left',
                'left',
                'right',
                'right',
                'left',
                'left',
                'left'
            ]

            condition_table_rows = []

            for _, row in performance_df.sort_values(
                'reaction_number'
            ).iterrows():
                condition_table_rows.append(
                    [
                        self._format_auto_report_table_value(
                            self._safe_auto_report_get(
                                row,
                                'reaction_number',
                                None
                            )
                        ),
                        self._format_auto_report_table_value(
                            self._safe_auto_report_get(
                                row,
                                'batch_number',
                                None
                            )
                        ),
                        self._format_auto_report_table_value(
                            self._safe_auto_report_get(
                                row,
                                'condition_type',
                                None
                            )
                        ),
                        self._format_auto_report_table_value(
                            self._safe_auto_report_get(
                                row,
                                'predicted_lambda_mean_nm',
                                None
                            ),
                            suffix='nm'
                        ),
                        self._format_auto_report_table_value(
                            self._safe_auto_report_get(
                                row,
                                'predicted_lambda_std_nm',
                                None
                            ),
                            suffix='nm'
                        ),
                        self._format_auto_report_replicate_list_value(
                            self._safe_auto_report_get(
                                row,
                                'actual_lambda_values_raw_nm',
                                None
                            )
                        ),
                        self._format_auto_report_replicate_list_value(
                            self._safe_auto_report_get(
                                row,
                                'actual_lambda_values_nm',
                                None
                            )
                        ),
                        self._format_auto_report_table_value(
                            self._safe_auto_report_get(
                                row,
                                'actual_lambda_mean_nm',
                                None
                            ),
                            suffix='nm'
                        ),
                        self._format_auto_report_table_value(
                            self._safe_auto_report_get(
                                row,
                                'target_error_nm',
                                None
                            ),
                            suffix='nm'
                        ),
                        self._format_auto_report_table_value(
                            self._safe_auto_report_get(
                                row,
                                'replicate_qc_status',
                                None
                            )
                        ),
                        self._format_auto_report_table_value(
                            self._safe_auto_report_get(
                                row,
                                'eligible_for_target_incumbent',
                                None
                            )
                        ),
                        self._format_auto_report_table_value(
                            self._safe_auto_report_get(
                                row,
                                'target_eligibility_status',
                                None
                            )
                        )
                    ]
                )

            condition_table_lines.extend(
                self._build_padded_auto_report_markdown_table(
                    headers=condition_table_headers,
                    rows=condition_table_rows,
                    alignments=condition_table_alignments
                )
            )

        lines = []

        lines.append('# Auto Mode Run Report')
        lines.append('')
        lines.append('## Executive Scientific Summary')
        lines.append('')
        lines.append(executive_summary)
        lines.append('')

        lines.extend(
            self._build_auto_run_status_report_lines(run_status_summary)
        )

        lines.append('## Experiment Overview')
        lines.append('')
        lines.append(f'- Experiment name: `{experiment_name}`')
        lines.append(f'- Experiment output path: `{self.out_path}`')
        lines.append(f'- Total condition-level rows: {n_conditions}')
        lines.append(f'- Seed conditions: {seed_conditions}')
        lines.append(
            f'- Optimizer-selected conditions: {optimizer_conditions}'
        )
        lines.append('')
        lines.append('## Auto Settings')
        lines.append('')
        lines.append(
            f'- Target λmax: '
            f'{self._format_auto_report_value(target_lambda_max_nm, "nm")}'
        )
        lines.append(
            f'- Initial seed conditions requested: '
            f'{self._format_auto_report_value(initial_data)}'
        )
        lines.append(
            f'- Maximum optimizer iterations requested: '
            f'{self._format_auto_report_value(max_iterations)}'
        )
        lines.append(
            f'- Replicates / duplicates per condition: '
            f'{self._format_auto_report_value(num_duplicates)}'
        )
        lines.append(
            f'- True-zero mixed masks allowed: '
            f'{self._format_auto_report_value(allow_true_zero)}'
        )
        lines.append(
            f'- Raspberry Pi legacy tare payload offset: '
            f'{self._format_auto_report_value(pi_legacy_tare_offset_g, "g")}'
        )
        lines.append(
            f'- Acquisition mode(s): '
            f'`{self._format_auto_report_value(";".join(acquisition_modes))}`'
        )

        if using_acquisition_portfolio:
            lines.append(
                '- Portfolio selection order resolves only near-duplicate '
                'candidates; every member used the same pre-batch GP and '
                'target-EI incumbent.'
            )
            lines.append(
                f'- Portfolio minimum normalized RMS distance: '
                f'{self._format_auto_report_value(portfolio_min_distance)}'
            )

        if 'balanced' in acquisition_modes:
            lines.append(
                f'- Balanced exploration weight: '
                f'{self._format_auto_report_value(balanced_exploration_weight)}'
            )

        lines.append(
            f'- Replicate outlier threshold: '
            f'{self._format_auto_report_value(replicate_outlier_threshold_nm, "nm")}'
        )
        lines.append(
            f'- Replicate SD tolerance for incumbents/stopping: '
            f'{self._format_auto_report_value(replicate_sd_tolerance_nm, "nm")}'
        )
        lines.append('')
        lines.append('## Optimization Objective')
        lines.append('')
        if using_acquisition_portfolio:
            lines.append(
                'Auto mode used the ordered target-aware acquisition portfolio '
                f'`{";".join(acquisition_modes)}`. Each mode selected one '
                'physically feasible, portfolio-distinct condition before the '
                'batch was measured or the GP was updated.'
            )
        else:
            lines.append(
                f'Auto mode used the `{acquisition_mode}` target-aware '
                f'acquisition mode. {acquisition_objective_description}'
            )
        lines.append('')
        lines.append(
            'All acquisition modes are converted to minimization scores and '
            'use the same reagent-mask, exact-zero, executable-transfer, water '
            'top-off, and overflow feasibility rules. Lower recorded scores '
            'are preferred within a mode. Score units depend on the mode: '
            '`exploit` uses nm²; `explore`, `balanced`, and `target_ei` use nm.'
        )
        lines.append('')
        lines.append('## Acquisition Audit Trail')
        lines.append('')
        lines.append(
            'Each optimizer-selected row records the acquisition decision '
            'before the experiment ran: canonical mode, minimized score, '
            'predicted target error, GP mean and standard deviation, target-EI '
            'incumbent when applicable, balanced weight when used, selected '
            'reagent mask, mask count, and repair status.'
        )
        lines.append('')
        lines.extend(acquisition_audit_table_lines)
        lines.append('')
        lines.append('### Optimizer Recipe Execution Provenance')
        lines.append('')
        lines.append(
            'The selected and executed representations below are captured '
            'before measurement. A controlled run may proceed only when the '
            'controller reports `optimizer_recipe_repaired = False`; otherwise '
            'the batch stops before robot commands are created.'
        )
        lines.append('')
        lines.extend(acquisition_provenance_table_lines)
        lines.append('')
        lines.append('## Best Condition Found')
        lines.append('')

        if best_condition_row is None:
            lines.append(
                'No best condition could be identified because no valid '
                '`target_error_nm` values were available in the condition-level '
                'performance rows.'
            )
            lines.append('')
        else:
            lines.append(
                'The best observed condition is defined as the condition with '
                'the smallest absolute QC-cleaned target error recorded in '
                '`auto_model_performance_log.csv`.'
            )
            lines.append('')
            lines.append(
                f'- Reaction condition number: {best_reaction_number}'
            )
            lines.append(f'- Batch number: {best_batch_number}')
            lines.append(f'- Condition type: {best_condition_type}')
            lines.append(
                f'- Target λmax: '
                f'{self._format_auto_report_value(target_lambda_max_nm, "nm")}'
            )
            lines.append(
                f'- Observed QC-cleaned mean λmax: '
                f'{self._format_auto_report_value(best_lambda_mean, "nm")}'
            )
            lines.append(
                f'- Target error: '
                f'{self._format_auto_report_value(best_target_error, "nm")}'
            )
            lines.append(f'- Replicate QC status: {best_qc_status}')
            lines.append(
                f'- Eligible for target incumbent/stopping: '
                f'{best_target_eligible}'
            )
            lines.append(
                f'- Target eligibility status: '
                f'{best_target_eligibility_status}'
            )
            lines.append('')

        lines.append('### Best Condition Interpretation')
        lines.append('')
        lines.append(best_interpretation)
        lines.append('')
        lines.append('## Compact Condition Table')
        lines.append('')
        lines.extend(condition_table_lines)
        lines.append('')
        lines.append('## Replicate QC Summary')
        lines.append('')
        lines.append(
            'Replicate QC is conservative and data-preserving. Raw replicate '
            'values are preserved in the performance log. Clear isolated '
            'outliers may be excluded from QC-cleaned condition summaries and '
            'model training, but ambiguous noisy conditions are flagged rather '
            'than automatically removed.'
        )
        lines.append('')
        lines.append(f'- QC passed conditions: {qc_passed}')
        lines.append(f'- QC not applied conditions: {qc_not_applied}')
        lines.append(f'- Conditions with excluded replicate(s): {qc_excluded}')
        lines.append(
            f'- Flagged but not excluded conditions: {qc_flagged}'
        )
        lines.append(
            f'- Conditions using all valid replicates for model training: '
            f'{conditions_used_all}'
        )
        lines.append(
            f'- Conditions using QC-included replicates only for model '
            f'training: {conditions_used_qc_only}'
        )
        lines.append(
            f'- Flagged conditions retained for model training: '
            f'{conditions_used_flagged}'
        )
        lines.append('')
        lines.append('### QC Interpretation')
        lines.append('')
        lines.append(qc_interpretation)
        lines.append('')

        if (
            not performance_df.empty
            and 'replicate_qc_status' in performance_df.columns
        ):
            qc_detail_df = performance_df[
                performance_df['replicate_qc_status'].isin(
                    ['excluded_replicate', 'flagged_not_excluded']
                )
            ]

            if len(qc_detail_df) > 0:
                lines.append('### QC Details')
                lines.append('')

                for _, row in qc_detail_df.iterrows():
                    reaction_number = self._safe_auto_report_get(
                        row,
                        'reaction_number'
                    )
                    qc_status = self._safe_auto_report_get(
                        row,
                        'replicate_qc_status'
                    )
                    qc_reason = self._safe_auto_report_get(
                        row,
                        'replicate_qc_reason'
                    )
                    raw_values = self._safe_auto_report_get(
                        row,
                        'actual_lambda_values_raw_nm'
                    )
                    qc_values = self._safe_auto_report_get(
                        row,
                        'actual_lambda_values_nm'
                    )
                    excluded_values = self._safe_auto_report_get(
                        row,
                        'excluded_lambda_values_nm'
                    )
                    model_status = self._safe_auto_report_get(
                        row,
                        'model_training_status'
                    )

                    lines.append(f'- Condition {reaction_number}:')
                    lines.append(f'  - QC status: {qc_status}')
                    lines.append(f'  - Raw values: {raw_values}')
                    lines.append(f'  - QC-included values: {qc_values}')
                    lines.append(f'  - Excluded values: {excluded_values}')
                    lines.append(f'  - QC reason: {qc_reason}')
                    lines.append(f'  - Model-training status: {model_status}')

                lines.append('')

        lines.append('## Model Prediction Performance')
        lines.append('')

        if prediction_rows == 0:
            lines.append(
                'No pre-experiment prediction-error summary is available. '
                'This is expected for seed-only runs or runs where prediction '
                'columns were not populated before the model observed the '
                'experimental outcome.'
            )
            lines.append('')
        else:
            lines.append(
                'Prediction errors are signed values calculated only where '
                'pre-experiment model predictions were recorded before the '
                'experimental result was added back into training.'
            )
            lines.append('')
            lines.append(
                f'- Conditions with signed prediction-error values: '
                f'{prediction_rows}'
            )
            lines.append(
                f'- Mean signed prediction error: '
                f'{self._format_auto_report_value(prediction_error_mean, "nm")}'
            )
            lines.append(
                f'- Median signed prediction error: '
                f'{self._format_auto_report_value(prediction_error_median, "nm")}'
            )
            lines.append(
                f'- Mean absolute prediction error: '
                f'{self._format_auto_report_value(prediction_abs_error_mean, "nm")}'
            )
            lines.append(
                f'- Median absolute prediction error: '
                f'{self._format_auto_report_value(prediction_abs_error_median, "nm")}'
            )
            lines.append(
                f'- Mean target error across conditions: '
                f'{self._format_auto_report_value(target_error_mean, "nm")}'
            )
            lines.append(
                f'- Median target error across conditions: '
                f'{self._format_auto_report_value(target_error_median, "nm")}'
            )
            lines.append('')
            lines.append('### Prediction Interpretation')
            lines.append('')
            lines.append(prediction_bias_text)
            lines.append('')

        lines.append('## Recipe and Volume Feasibility Summary')
        lines.append('')
        lines.append(
            'Auto mode preserves raw recipe and volume information in the '
            'condition-level performance log. Final controller-side feasibility '
            'checks remain the authoritative guardrail for physical execution.'
        )
        lines.append('')
        lines.append(f'- Volume-infeasible condition rows: {volume_infeasible_count}')
        lines.append(
            f'- Water-transfer-not-executable condition rows: '
            f'{water_not_executable_count}'
        )
        lines.append(
            f'- Water top-off range: '
            f'{self._format_auto_report_value(water_min, "uL")} to '
            f'{self._format_auto_report_value(water_max, "uL")}'
        )
        lines.append(
            f'- Variable reagent volume range: '
            f'{self._format_auto_report_value(variable_volume_min, "uL")} to '
            f'{self._format_auto_report_value(variable_volume_max, "uL")}'
        )
        lines.append(
            f'- Final total volume range: '
            f'{self._format_auto_report_value(total_volume_min, "uL")} to '
            f'{self._format_auto_report_value(total_volume_max, "uL")}'
        )
        lines.append('')
        lines.append('## Key Plots')
        lines.append('')
        lines.append('### Final λmax Progress Plot')
        lines.append('')
        lines.append('![Final λmax Progress](../Plots/lambda_progress_final.png)')
        lines.append('')
        lines.append('### Final Replicate Diagnostic Plot')
        lines.append('')
        lines.append(
            '![Final Replicate Diagnostic](../Plots/lambda_replicates_final.png)'
        )
        lines.append('')

        if len(getattr(self, 'variable_reagents', [])) == 3:
            lines.append('### Final Conditional GP Slice Atlases')
            lines.append('')
            lines.append(
                'These maps are conditional two-reagent slices through the '
                'three-reagent fitted GP. Each panel holds its third reagent '
                'at the renderer-reported observed reference condition '
                '(best QC-approved when available).'
            )
            lines.append('')
            for plot_filename, plot_title, plot_caption in (
                (
                    'gpr_3d_mean_orthogonal_slices_final.png',
                    'Conditional GP Mean Slices',
                    'Predicted lambda-max slices, with the target contour '
                    'and physically infeasible regions shown explicitly.'
                ),
                (
                    'gpr_3d_uncertainty_orthogonal_slices_final.png',
                    'Conditional GP Uncertainty Slices',
                    'Predictive GP standard-deviation slices on a shared '
                    'nanometer scale.'
                ),
                (
                    'gpr_3d_target_probability_orthogonal_slices_final.png',
                    'Conditional Target-Tolerance Probability Slices',
                    'Probability of falling within the same target tolerance '
                    'used by controller stopping.'
                )
            ):
                lines.extend(
                    self._auto_report_plot_markdown_if_exists(
                        plot_filename=plot_filename,
                        title=plot_title,
                        caption=plot_caption
                    )
                )

        seed_design_plot_lines = []
        exploration_design_plot_lines = []

        seed_design_plot_lines.extend(
            self._auto_report_plot_markdown_if_exists(
                plot_filename='initial_maximin_seed_design_1d.png',
                title='Initial Maximin Seed Design: 1D Reagent Space',
                caption=(
                    'This figure shows only the initial maximin seed conditions '
                    'in one-dimensional executable reagent space. Optimizer-'
                    'selected conditions and the best-observed-condition marker '
                    'are intentionally omitted so the initial design coverage '
                    'can be evaluated independently.'
                )
            )
        )

        seed_design_plot_lines.extend(
            self._auto_report_plot_markdown_if_exists(
                plot_filename='initial_maximin_seed_design_2d.png',
                title='Initial Maximin Seed Design: 2D Reagent Space',
                caption=(
                    'This figure shows only the initial maximin seed conditions '
                    'across the two executable reagent-concentration axes. It '
                    'documents the coverage of the design used to initialize '
                    'the Auto model before Bayesian optimization began.'
                )
            )
        )

        seed_design_plot_lines.extend(
            self._auto_report_plot_markdown_if_exists(
                plot_filename='initial_maximin_seed_design_pairwise.png',
                title='Initial Maximin Seed Design: Pairwise Projections',
                caption=(
                    'These pairwise projections show only the initial maximin '
                    'seed conditions across every displayed two-reagent '
                    'combination. They provide the primary static assessment '
                    'of initial coverage for three- through six-dimensional '
                    'reagent spaces.'
                )
            )
        )

        seed_design_plot_lines.extend(
            self._auto_report_plot_markdown_if_exists(
                plot_filename='initial_maximin_seed_design_3d.png',
                title='Initial Maximin Seed Design: 3D Reagent Space',
                caption=(
                    'For a three-variable Auto run, this figure shows the full '
                    'three-dimensional position of only the initial maximin '
                    'seed conditions before optimizer-selected experiments '
                    'were added.'
                )
            )
        )

        seed_design_plot_lines.extend(
            self._auto_report_plot_markdown_if_exists(
                plot_filename=(
                    'initial_maximin_seed_design_'
                    'parallel_coordinates.png'
                ),
                title=(
                    'Initial Maximin Seed Design: Parallel Coordinates'
                ),
                caption=(
                    'This parallel-coordinate figure shows only the initial '
                    'maximin seed conditions across the higher-dimensional '
                    'reagent space. Concentrations are normalized for display '
                    'against executable reagent ranges; raw concentrations are '
                    'not modified.'
                )
            )
        )

        seed_design_plot_lines.extend(
            self._auto_report_plot_markdown_if_exists(
                plot_filename=(
                    'initial_maximin_seed_design_'
                    'pairwise_compact.png'
                ),
                title=(
                    'Initial Maximin Seed Design: '
                    'Compact Pairwise Projections'
                ),
                caption=(
                    'For a high-dimensional Auto run, this compact figure '
                    'shows only the initial maximin seed conditions across a '
                    'limited set of reagent-pair projections so the report '
                    'remains readable.'
                )
            )
        )

        seed_design_plot_lines.extend(
            self._auto_report_plot_markdown_if_exists(
                plot_filename='initial_maximin_seed_design_pca.png',
                title='Initial Maximin Seed Design: PCA Projection',
                caption=(
                    'This PCA figure shows only the initial maximin seed '
                    'conditions in a two-component summary of the higher-'
                    'dimensional design space. PCA axes are reduced-dimensional '
                    'coordinates and are not physical reagent axes.'
                )
            )
        )

        exploration_design_plot_lines.extend(
            self._auto_report_plot_markdown_if_exists(
                plot_filename='auto_design_space_exploration_1d.png',
                title='Auto Design-Space Exploration: 1D Reagent Space',
                caption=(
                    'This figure shows where Auto began with the initial seed '
                    'conditions and where it subsequently sampled optimizer-'
                    'selected conditions in one-dimensional reagent space. The '
                    'best observed condition is highlighted when available.'
                )
            )
        )

        exploration_design_plot_lines.extend(
            self._auto_report_plot_markdown_if_exists(
                plot_filename='auto_design_space_exploration_2d.png',
                title='Auto Design-Space Exploration: 2D Reagent Space',
                caption=(
                    'This figure distinguishes the initial seed conditions '
                    'from optimizer-selected conditions across the two '
                    'executable reagent-concentration axes. The best observed '
                    'condition is highlighted when available.'
                )
            )
        )

        exploration_design_plot_lines.extend(
            self._auto_report_plot_markdown_if_exists(
                plot_filename=(
                    'auto_design_space_exploration_pairwise.png'
                ),
                title=(
                    'Auto Design-Space Exploration: Pairwise Projections'
                ),
                caption=(
                    'These pairwise projections show how the optimizer extended '
                    'the initial seed design across every displayed two-reagent '
                    'combination. Seed and optimizer-selected conditions are '
                    'shown separately, with the best observed condition '
                    'highlighted when available.'
                )
            )
        )

        exploration_design_plot_lines.extend(
            self._auto_report_plot_markdown_if_exists(
                plot_filename='auto_design_space_exploration_3d.png',
                title='Auto Design-Space Exploration: 3D Reagent Space',
                caption=(
                    'For a three-variable Auto run, this figure shows the full '
                    'three-dimensional relationship between the initial seed '
                    'conditions and the optimizer-selected conditions.'
                )
            )
        )

        exploration_design_plot_lines.extend(
            self._auto_report_plot_markdown_if_exists(
                plot_filename=(
                    'auto_design_space_exploration_'
                    'parallel_coordinates.png'
                ),
                title=(
                    'Auto Design-Space Exploration: Parallel Coordinates'
                ),
                caption=(
                    'This parallel-coordinate figure summarizes the complete '
                    'higher-dimensional Auto trajectory, distinguishing the '
                    'initial seed design from optimizer-selected conditions. '
                    'Normalization is display-only and does not alter raw '
                    'concentrations.'
                )
            )
        )

        exploration_design_plot_lines.extend(
            self._auto_report_plot_markdown_if_exists(
                plot_filename=(
                    'auto_design_space_exploration_'
                    'pairwise_compact.png'
                ),
                title=(
                    'Auto Design-Space Exploration: '
                    'Compact Pairwise Projections'
                ),
                caption=(
                    'For a high-dimensional Auto run, this compact figure '
                    'shows seed and optimizer-selected conditions across a '
                    'limited set of reagent-pair projections so the report '
                    'remains readable.'
                )
            )
        )

        exploration_design_plot_lines.extend(
            self._auto_report_plot_markdown_if_exists(
                plot_filename='auto_design_space_exploration_pca.png',
                title='Auto Design-Space Exploration: PCA Projection',
                caption=(
                    'This PCA figure summarizes the complete higher-dimensional '
                    'Auto trajectory in two components. The seed-only and full-'
                    'exploration PCA figures use the same reference projection '
                    'so identical seed conditions retain identical coordinates. '
                    'PCA axes are not physical reagent axes.'
                )
            )
        )

        if len(seed_design_plot_lines) > 0:
            lines.append('## Initial Maximin Seed Design')
            lines.append('')
            lines.append(
                'These figures isolate the initial maximin seed conditions used '
                'to initialize the Auto model. They show the starting coverage '
                'of executable reagent-design space before the optimizer '
                'selected any additional experiments.'
            )
            lines.append('')
            lines.extend(seed_design_plot_lines)

        if len(exploration_design_plot_lines) > 0:
            lines.append('## Auto Design-Space Exploration')
            lines.append('')
            lines.append(
                'These figures show how Auto explored reagent-design space '
                'after initialization. Initial seed conditions and optimizer-'
                'selected conditions are displayed as separate categories, and '
                'the best observed condition is highlighted when available.'
            )
            lines.append('')
            lines.extend(exploration_design_plot_lines)

        lines.append('## Generated Files')
        lines.append('')

        lines.append(
            self._auto_report_file_line(
                os.path.join(
                    'pr_data',
                    'experiment_data.csv'
                ),
                'Raw well-level experiment data'
            )
        )

        lines.append(
            self._auto_report_file_line(
                os.path.join(
                    'pr_data',
                    'auto_model_performance_log.csv'
                ),
                'Condition-level Auto performance log'
            )
        )

        lines.append(
            '- Human-readable Auto run report: '
            '`pr_data/auto_run_report.md` (present)'
        )

        lines.append(
            self._auto_report_file_line(
                os.path.join(
                    'Plots',
                    'lambda_progress_final.png'
                ),
                'Final condition-level lambda progress plot'
            )
        )

        lines.append(
            self._auto_report_file_line(
                os.path.join(
                    'Plots',
                    'lambda_replicates_final.png'
                ),
                'Final replicate-level lambda diagnostic plot'
            )
        )

        # Design-space renderers are deliberately dimension-specific.  Record
        # unavailable renderers as not applicable rather than as missing files
        # so a two-variable report does not resemble a plotting failure.
        n_design_dimensions = len(self.variable_reagents)
        design_plot_file_entries = [
            (
                'initial_maximin_seed_design_1d.png',
                'Initial maximin seed design 1D reagent-space plot',
                n_design_dimensions == 1
            ),
            (
                'auto_design_space_exploration_1d.png',
                'Auto design-space exploration 1D reagent-space plot',
                n_design_dimensions == 1
            ),
            (
                'initial_maximin_seed_design_2d.png',
                'Initial maximin seed design 2D reagent-space plot',
                n_design_dimensions == 2
            ),
            (
                'auto_design_space_exploration_2d.png',
                'Auto design-space exploration 2D reagent-space plot',
                n_design_dimensions == 2
            ),
            (
                'initial_maximin_seed_design_pairwise.png',
                'Initial maximin seed design pairwise projection plot',
                3 <= n_design_dimensions <= 6
            ),
            (
                'auto_design_space_exploration_pairwise.png',
                'Auto design-space exploration pairwise projection plot',
                3 <= n_design_dimensions <= 6
            ),
            (
                'initial_maximin_seed_design_3d.png',
                'Initial maximin seed design 3D reagent-space plot',
                n_design_dimensions == 3
            ),
            (
                'auto_design_space_exploration_3d.png',
                'Auto design-space exploration 3D reagent-space plot',
                n_design_dimensions == 3
            ),
            (
                'initial_maximin_seed_design_parallel_coordinates.png',
                'Initial maximin seed design parallel-coordinate plot',
                n_design_dimensions >= 4
            ),
            (
                'auto_design_space_exploration_parallel_coordinates.png',
                'Auto design-space exploration parallel-coordinate plot',
                n_design_dimensions >= 4
            ),
            (
                'initial_maximin_seed_design_pairwise_compact.png',
                'Initial maximin seed design compact pairwise plot',
                n_design_dimensions > 6
            ),
            (
                'auto_design_space_exploration_pairwise_compact.png',
                'Auto design-space exploration compact pairwise plot',
                n_design_dimensions > 6
            ),
            (
                'initial_maximin_seed_design_pca.png',
                'Initial maximin seed design PCA projection plot',
                n_design_dimensions > 6
            ),
            (
                'auto_design_space_exploration_pca.png',
                'Auto design-space exploration PCA projection plot',
                n_design_dimensions > 6
            )
        ]

        for (
            plot_filename,
            plot_description,
            plot_is_applicable
        ) in design_plot_file_entries:
            relative_plot_path = os.path.join('Plots', plot_filename)

            if plot_is_applicable:
                lines.append(
                    self._auto_report_file_line(
                        relative_plot_path,
                        plot_description
                    )
                )
            else:
                lines.append(
                    self._auto_report_not_applicable_file_line(
                        relative_plot_path,
                        plot_description,
                        (
                            f'{n_design_dimensions}-variable Auto run'
                        )
                    )
                )
        lines.append(
            self._auto_report_file_line(
                os.path.join('Debug', 'terminal_output.txt'),
                'Captured terminal output'
            )
        )
        lines.append('')
        lines.append('## Notes and Warnings')
        lines.append('')

        if volume_infeasible_count == 0 and water_not_executable_count == 0:
            lines.append(
                '- No volume feasibility or water-transfer execution problems '
                'were detected in the condition-level performance rows.'
            )
        else:
            lines.append(
                '- One or more volume or water-transfer feasibility warnings '
                'were detected. Review `auto_model_performance_log.csv` before '
                'interpreting the run.'
            )

        if qc_excluded > 0:
            lines.append(
                '- At least one replicate was excluded by QC. Excluded values '
                'are preserved in the raw/QC columns and visualized in the '
                'replicate diagnostic plot.'
            )

        if qc_flagged > 0:
            lines.append(
                '- At least one condition was flagged but not excluded. These '
                'conditions are intentionally retained for model training under '
                'the current data-preserving policy.'
            )

        if prediction_rows == 0:
            lines.append(
                '- Prediction-performance metrics may be unavailable for seed '
                'conditions or early-stop runs without optimizer-selected '
                'conditions.'
            )

        lines.append('')
        lines.append('## Conclusion')
        lines.append('')

        if best_condition_row is None:
            lines.append(
                'The Auto mode run report was generated, but no best condition '
                'could be identified from target-error values. Review the raw '
                'experiment data and condition-level performance log.'
            )
        else:
            lines.append(
                f'The closest observed condition to the requested target was '
                f'condition {best_reaction_number}, with QC-cleaned mean λmax '
                f'{self._format_auto_report_value(best_lambda_mean, "nm")} '
                f'and target error '
                f'{self._format_auto_report_value(best_target_error, "nm")}. '
                f'{best_interpretation}'
            )

        lines.append('')
        lines.append('---')
        lines.append('')
        lines.append(
            'Report generated automatically by OT2Control Auto mode from '
            '`auto_model_performance_log.csv`.'
        )
        lines.append('')

        with open(report_path, 'w', encoding='utf-8') as report_file:
            report_file.write('\n'.join(lines))

        print(
            f"<<controller>> exported Auto run report to "
            f"{report_path}"
        )

        return report_path
    
    def _get_auto_target_tolerance_nm(self):
        '''
        Returns the condition-level target tolerance shared by Auto decisions.

        The same tolerance controls controller-owned target stopping and the
        three-variable visualization probability maps.  Keeping this lookup
        central prevents a plot from silently describing a different success
        region than the one Auto actually uses.
        '''
        tolerance_nm = float(
            self.robo_params.get('target_tolerance_nm', 10.0)
        )

        if not math.isfinite(tolerance_nm) or tolerance_nm < 0.0:
            raise ValueError(
                "target_tolerance_nm must be a finite, nonnegative number. "
                f"Received: {tolerance_nm!r}."
            )

        return tolerance_nm

    def _calculate_auto_target_probability(
        self,
        predicted_mean_nm,
        predicted_std_nm,
        target_nm,
        tolerance_nm
    ):
        '''
        Calculates P(|lambda max - target| <= tolerance) under the GP normal
        predictive distribution.

        A zero standard deviation uses its deterministic limit rather than a
        division by zero.  The method is vectorized with NumPy and uses only
        math.erf, keeping it compatible with the project's established Python
        and SciPy environments without introducing a new dependency.
        '''
        predicted_mean_nm = np.asarray(predicted_mean_nm, dtype=float)
        predicted_std_nm = np.asarray(predicted_std_nm, dtype=float)

        if (
            predicted_mean_nm.shape != predicted_std_nm.shape
            or not np.all(np.isfinite(predicted_mean_nm))
            or not np.all(np.isfinite(predicted_std_nm))
            or np.any(predicted_std_nm < 0.0)
        ):
            raise ValueError(
                "Target-probability plotting requires finite mean and "
                "nonnegative standard-deviation arrays of equal shape."
            )

        target_nm = float(target_nm)
        tolerance_nm = float(tolerance_nm)

        if (
            not math.isfinite(target_nm)
            or not math.isfinite(tolerance_nm)
            or tolerance_nm < 0.0
        ):
            raise ValueError(
                "Target probability requires a finite target and a finite, "
                "nonnegative tolerance."
            )

        probability = np.zeros(predicted_mean_nm.shape, dtype=float)
        deterministic = predicted_std_nm <= 1e-12
        probability[deterministic] = (
            np.abs(predicted_mean_nm[deterministic] - target_nm)
            <= tolerance_nm
        ).astype(float)

        stochastic = ~deterministic
        if np.any(stochastic):
            standard_deviation = predicted_std_nm[stochastic]
            lower_z = (
                target_nm - tolerance_nm - predicted_mean_nm[stochastic]
            ) / standard_deviation
            upper_z = (
                target_nm + tolerance_nm - predicted_mean_nm[stochastic]
            ) / standard_deviation
            normal_cdf = np.vectorize(
                lambda value: 0.5 * (
                    1.0 + math.erf(value / math.sqrt(2.0))
                ),
                otypes=[float]
            )
            probability[stochastic] = (
                normal_cdf(upper_z) - normal_cdf(lower_z)
            )

        return np.clip(probability, 0.0, 1.0)

    def plot_3D_GPR_orthogonal_slices(
        self,
        model,
        batch_number=None,
        final_snapshot=False,
        grid_size=100
    ):
        '''
        Saves conditional two-dimensional GP slice atlases for exactly three
        variable reagents.

        Each panel varies a pair of reagents over their physical concentration
        ranges while holding the third reagent at one documented reference
        recipe.  The three companion atlases show predicted lambda max,
        predictive GP standard deviation, and the probability of landing
        within Auto's actual condition-level target tolerance.  Feasibility is
        evaluated by OptimizationModel's authoritative volume-balance logic,
        so the plotting layer cannot introduce a competing transfer rule.

        In addition to the three compact atlases, this method saves separate
        mean and uncertainty feasibility-overlay versions. Those diagnostic
        figures use the same gray exclusion fill, water boundaries, 5 uL
        reagent boundaries, target contour, and figure-level legend convention
        as the established two-variable GP feasibility heatmaps.

        This renderer is observational only: it never updates the GP, changes
        acquisition state, repairs recipes, or touches robot-facing data.
        '''
        generated_plot_paths = []
        reagent_names = [
            str(reagent_name)
            for reagent_name in list(
                getattr(self, 'variable_reagents', [])
            )
        ]

        if len(reagent_names) != 3:
            print(
                "<<controller>> skipping orthogonal GP slice plots because "
                "there are not exactly three variable reagents"
            )
            return generated_plot_paths

        if model is None:
            raise ValueError(
                "3D GP slice plots require an initialized OptimizationModel."
            )

        try:
            grid_size = int(grid_size)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("3D GP slice grid_size must be an integer.")

        if grid_size < 21:
            raise ValueError(
                "3D GP slice grid_size must be at least 21 for a useful "
                "conditional map."
            )

        predict_batch = getattr(
            model,
            'predict_lambda_distribution_nm_batch',
            None
        )
        balance_for_plotting = getattr(
            model,
            'get_candidate_volume_balance_for_plotting',
            None
        )

        if not callable(predict_batch) or not callable(balance_for_plotting):
            raise AttributeError(
                "3D GP slice plots require the read-only batch-prediction "
                "and feasibility helpers supplied by OptimizationModel."
            )

        bounds = []
        for reagent_index, reagent_name in enumerate(reagent_names):
            lower_bound = self._get_auto_design_bound_value(
                self.min_conc,
                reagent_index,
                reagent_name
            )
            upper_bound = self._get_auto_design_bound_value(
                self.max_conc,
                reagent_index,
                reagent_name
            )

            if (
                not np.isfinite(lower_bound)
                or not np.isfinite(upper_bound)
                or upper_bound <= lower_bound
            ):
                raise ValueError(
                    "3D GP slice plots require finite increasing "
                    f"concentration bounds for {reagent_name}."
                )

            bounds.append((float(lower_bound), float(upper_bound)))

        target_nm = float(self.getModelInfo()['target'])
        tolerance_nm = self._get_auto_target_tolerance_nm()
        font_sizes = self._get_auto_design_plot_font_sizes()

        # Prefer the best QC-approved condition because it is the most useful
        # scientific operating point.  The first complete finite observation
        # is a deterministic fallback for early debug runs without an eligible
        # target incumbent.
        reference_recipe = None
        reference_label = None
        ranked_rows = []
        for row in getattr(self, 'auto_model_performance_rows', []):
            values = []
            for reagent_name in reagent_names:
                try:
                    value = float(row.get(f'{reagent_name}_concentration'))
                except (TypeError, ValueError):
                    value = np.nan
                values.append(value)

            if not np.all(np.isfinite(values)):
                continue

            try:
                target_error = float(row.get('target_error_nm'))
            except (TypeError, ValueError):
                target_error = np.inf

            is_qc_approved = bool(
                row.get('use_for_model_training', False)
                and row.get('eligible_for_target_incumbent', False)
                and math.isfinite(target_error)
            )
            ranked_rows.append((not is_qc_approved, target_error, values))

        if len(ranked_rows) > 0:
            ranked_rows.sort(key=lambda item: (item[0], item[1]))
            reference_recipe = np.asarray(ranked_rows[0][2], dtype=float)
            if not ranked_rows[0][0]:
                reference_label = 'best QC-approved observed condition'
            else:
                reference_label = 'first available observed condition'

        if reference_recipe is None:
            raise ValueError(
                "3D GP slice plots require at least one complete observed "
                "condition to define a scientifically interpretable slice."
            )

        reference_normalized = np.asarray([
            (reference_recipe[index] - bounds[index][0])
            / (bounds[index][1] - bounds[index][0])
            for index in range(3)
        ], dtype=float)

        if not np.all(
            np.isfinite(reference_normalized)
            & (reference_normalized >= -1e-9)
            & (reference_normalized <= 1.0 + 1e-9)
        ):
            raise ValueError(
                "The selected 3D slice reference recipe lies outside the "
                "configured Auto concentration bounds."
            )

        # A measured-condition overlay is deliberately restricted to rows that
        # were accepted into model training.  That communicates the data the
        # fitted GP actually represents without treating QC-excluded values as
        # equivalent observations.
        observed_conditions = []
        for row in getattr(self, 'auto_model_performance_rows', []):
            if not row.get('use_for_model_training', False):
                continue

            observed_recipe = []
            for reagent_name in reagent_names:
                try:
                    concentration = float(
                        row.get(f'{reagent_name}_concentration')
                    )
                except (TypeError, ValueError):
                    concentration = np.nan
                observed_recipe.append(concentration)

            if not np.all(np.isfinite(observed_recipe)):
                continue

            normalized_recipe = np.asarray([
                (observed_recipe[index] - bounds[index][0])
                / (bounds[index][1] - bounds[index][0])
                for index in range(3)
            ], dtype=float)

            if not np.all(np.isfinite(normalized_recipe)):
                continue

            observed_conditions.append({
                'physical_recipe': np.asarray(observed_recipe, dtype=float),
                'normalized_recipe': normalized_recipe
            })

        # Match the established two-variable GP field density. Physical
        # feasibility is evaluated separately below on a denser grid, exactly
        # as the 2D overlay plots do, so diagnostic boundaries are smooth
        # without changing the ordinary GP heatmap pixels.
        normalized_axis_values = np.linspace(0.0, 1.0, grid_size)
        feasibility_grid_size = max(401, grid_size)
        feasibility_axis_values = np.linspace(
            0.0,
            1.0,
            feasibility_grid_size
        )
        slice_half_width = 0.5 / float(grid_size - 1)
        orientations = ((0, 1, 2), (0, 2, 1), (1, 2, 0))
        panel_data = []

        def _evaluate_slice_feasibility(recipes, grid_shape):
            '''Evaluates raw physical feasibility without repairing near zero.'''
            feasible = np.zeros(recipes.shape[0], dtype=bool)
            water_volume = np.full(recipes.shape[0], np.nan, dtype=float)
            transfer_volume_by_reagent = {
                reagent_name: np.full(
                    recipes.shape[0],
                    np.nan,
                    dtype=float
                )
                for reagent_name in reagent_names
            }

            for point_index, recipe in enumerate(recipes):
                balance = balance_for_plotting(recipe)
                feasible[point_index] = bool(balance['volume_feasible'])
                water_volume[point_index] = float(balance['water_volume'])

                for reagent_name in reagent_names:
                    transfer_volume_by_reagent[reagent_name][
                        point_index
                    ] = float(
                        balance['variable_transfer_volumes'][reagent_name]
                    )

            return {
                'feasible': feasible.reshape(grid_shape),
                'water_volume_uL': water_volume.reshape(grid_shape),
                'transfer_volume_uL_by_reagent': {
                    reagent_name: transfer_volume_grid.reshape(grid_shape)
                    for reagent_name, transfer_volume_grid
                    in transfer_volume_by_reagent.items()
                }
            }

        for x_index, y_index, fixed_index in orientations:
            x_normalized, y_normalized = np.meshgrid(
                normalized_axis_values,
                normalized_axis_values,
                indexing='xy'
            )
            recipes = np.tile(
                reference_normalized,
                (x_normalized.size, 1)
            )
            recipes[:, x_index] = x_normalized.ravel(order='C')
            recipes[:, y_index] = y_normalized.ravel(order='C')

            predicted_mean, predicted_std = predict_batch(recipes)
            predicted_mean = np.asarray(predicted_mean, dtype=float).reshape(
                x_normalized.shape
            )
            predicted_std = np.asarray(predicted_std, dtype=float).reshape(
                x_normalized.shape
            )

            ordinary_feasibility = _evaluate_slice_feasibility(
                recipes,
                x_normalized.shape
            )

            feasibility_x_normalized, feasibility_y_normalized = (
                np.meshgrid(
                    feasibility_axis_values,
                    feasibility_axis_values,
                    indexing='xy'
                )
            )
            feasibility_recipes = np.tile(
                reference_normalized,
                (feasibility_x_normalized.size, 1)
            )
            feasibility_recipes[:, x_index] = (
                feasibility_x_normalized.ravel(order='C')
            )
            feasibility_recipes[:, y_index] = (
                feasibility_y_normalized.ravel(order='C')
            )
            dense_feasibility = _evaluate_slice_feasibility(
                feasibility_recipes,
                feasibility_x_normalized.shape
            )

            panel_data.append({
                'x_index': x_index,
                'y_index': y_index,
                'fixed_index': fixed_index,
                'x_physical': (
                    bounds[x_index][0]
                    + x_normalized * (bounds[x_index][1] - bounds[x_index][0])
                ),
                'y_physical': (
                    bounds[y_index][0]
                    + y_normalized * (bounds[y_index][1] - bounds[y_index][0])
                ),
                'mean_nm': predicted_mean,
                'std_nm': predicted_std,
                'probability': self._calculate_auto_target_probability(
                    predicted_mean,
                    predicted_std,
                    target_nm,
                    tolerance_nm
                ),
                'feasible': ordinary_feasibility['feasible'],
                'feasibility_x_physical': (
                    bounds[x_index][0]
                    + feasibility_x_normalized * (
                        bounds[x_index][1] - bounds[x_index][0]
                    )
                ),
                'feasibility_y_physical': (
                    bounds[y_index][0]
                    + feasibility_y_normalized * (
                        bounds[y_index][1] - bounds[y_index][0]
                    )
                ),
                'feasibility_mask': dense_feasibility['feasible'],
                'feasibility_water_volume_uL': (
                    dense_feasibility['water_volume_uL']
                ),
                'feasibility_transfer_volume_uL_by_reagent': (
                    dense_feasibility['transfer_volume_uL_by_reagent']
                )
            })

        feasible_mean_chunks = [
            panel['mean_nm'][panel['feasible']]
            for panel in panel_data
            if np.any(panel['feasible'])
        ]
        feasible_std_chunks = [
            panel['std_nm'][panel['feasible']]
            for panel in panel_data
            if np.any(panel['feasible'])
        ]

        if len(feasible_mean_chunks) == 0 or len(feasible_std_chunks) == 0:
            raise ValueError(
                "No physically executable points were available for the 3D "
                "conditional GP slices."
            )

        feasible_mean_values = np.concatenate(feasible_mean_chunks)
        feasible_std_values = np.concatenate(feasible_std_chunks)

        mean_norm = plt.Normalize(
            vmin=float(np.min(feasible_mean_values)),
            vmax=float(np.max(feasible_mean_values))
        )
        max_std_nm = max(float(np.max(feasible_std_values)), 1.0)

        field_definitions = (
            (
                'mean',
                'Predicted $\\lambda_{max}$ (nm)',
                lambda panel: panel['mean_nm'],
                'inferno',
                mean_norm,
                f'Conditional GP mean; target = {target_nm:g} nm'
            ),
            (
                'uncertainty',
                'Predictive GP SD (nm)',
                lambda panel: panel['std_nm'],
                'viridis',
                plt.Normalize(vmin=0.0, vmax=max_std_nm),
                'Conditional GP predictive uncertainty'
            ),
            (
                'target_probability',
                'Probability of meeting target criterion',
                lambda panel: panel['probability'],
                'cividis',
                plt.Normalize(vmin=0.0, vmax=1.0),
                (
                    'GP probability of meeting the controller stopping '
                    f'criterion (target ± {tolerance_nm:g} nm)'
                )
            )
        )

        def _grid_spans_contour_level(grid, level):
            '''Returns whether a finite grid crosses one contour level.'''
            finite_values = np.asarray(grid, dtype=float)
            finite_values = finite_values[np.isfinite(finite_values)]

            return (
                finite_values.size > 0
                and np.min(finite_values) < level
                and np.max(finite_values) > level
            )

        def _draw_slice_feasibility_overlay(
            axis,
            panel,
            field_name
        ):
            '''Draws the 2D-style physical-boundary overlay for one slice.'''
            axis.contourf(
                panel['feasibility_x_physical'],
                panel['feasibility_y_physical'],
                (~panel['feasibility_mask']).astype(float),
                levels=[0.5, 1.5],
                colors=['0.70'],
                alpha=0.55,
                antialiased=True,
                corner_mask=False,
                zorder=2
            )

            water_volume_grid = panel['feasibility_water_volume_uL']
            if _grid_spans_contour_level(water_volume_grid, 0.0):
                axis.contour(
                    panel['feasibility_x_physical'],
                    panel['feasibility_y_physical'],
                    water_volume_grid,
                    levels=[0.0],
                    colors=['#0072B2'],
                    linewidths=1.35,
                    linestyles='solid',
                    zorder=4
                )

            if _grid_spans_contour_level(water_volume_grid, 5.0):
                axis.contour(
                    panel['feasibility_x_physical'],
                    panel['feasibility_y_physical'],
                    water_volume_grid,
                    levels=[5.0],
                    colors=['#D55E00'],
                    linewidths=1.35,
                    linestyles='dashed',
                    zorder=4
                )

            reagent_boundary_colors = (
                '#009E73',
                '#CC79A7',
                '#E69F00'
            )
            for reagent_index, reagent_name in enumerate(reagent_names):
                transfer_volume_grid = panel[
                    'feasibility_transfer_volume_uL_by_reagent'
                ][reagent_name]

                if _grid_spans_contour_level(transfer_volume_grid, 5.0):
                    axis.contour(
                        panel['feasibility_x_physical'],
                        panel['feasibility_y_physical'],
                        transfer_volume_grid,
                        levels=[5.0],
                        colors=[reagent_boundary_colors[reagent_index]],
                        linewidths=1.15,
                        linestyles='dotted',
                        zorder=4
                    )

            if (
                field_name == 'mean'
                and _grid_spans_contour_level(
                    panel['mean_nm'],
                    target_nm
                )
            ):
                axis.contour(
                    panel['x_physical'],
                    panel['y_physical'],
                    panel['mean_nm'],
                    levels=[target_nm],
                    colors=['#000000'],
                    linewidths=1.1,
                    linestyles='dashdot',
                    zorder=5
                )

        def _build_slice_feasibility_legend(
            axis,
            field_name
        ):
            '''Builds one figure-level legend using the established 2D style.'''
            legend_handles = [
                mpatches.Patch(
                    facecolor='0.70',
                    alpha=0.55,
                    label=(
                        'Excluded: overflow or non-executable transfer'
                    )
                )
            ]

            water_grids = [
                panel['feasibility_water_volume_uL']
                for panel in panel_data
            ]
            if any(
                _grid_spans_contour_level(water_grid, 0.0)
                for water_grid in water_grids
            ):
                water_zero_handle, = axis.plot(
                    [], [], color='#0072B2', linewidth=1.35,
                    label='Water = 0 uL boundary'
                )
                legend_handles.append(water_zero_handle)

            if any(
                _grid_spans_contour_level(water_grid, 5.0)
                for water_grid in water_grids
            ):
                water_minimum_handle, = axis.plot(
                    [], [], color='#D55E00', linewidth=1.35,
                    linestyle='dashed', label='Water = 5 uL boundary'
                )
                legend_handles.append(water_minimum_handle)

            reagent_boundary_colors = (
                '#009E73',
                '#CC79A7',
                '#E69F00'
            )
            for reagent_index, reagent_name in enumerate(reagent_names):
                if any(
                    _grid_spans_contour_level(
                        panel['feasibility_transfer_volume_uL_by_reagent'][
                            reagent_name
                        ],
                        5.0
                    )
                    for panel in panel_data
                ):
                    reagent_minimum_handle, = axis.plot(
                        [], [],
                        color=reagent_boundary_colors[reagent_index],
                        linewidth=1.15,
                        linestyle='dotted',
                        label=f'{reagent_name} = 5 uL boundary'
                    )
                    legend_handles.append(reagent_minimum_handle)

            if field_name == 'mean' and any(
                _grid_spans_contour_level(panel['mean_nm'], target_nm)
                for panel in panel_data
            ):
                target_handle, = axis.plot(
                    [], [], color='#000000', linewidth=1.1,
                    linestyle='dashdot',
                    label=f'Target = {target_nm:.0f} nm'
                )
                legend_handles.append(target_handle)

            return legend_handles

        def _format_slice_heatmap_axis(axis):
            '''Applies the grid-free, square 2D GP heatmap frame style.'''
            # These are heatmaps, rather than point-based design-space plots.
            # The established 2D GP figures deliberately omit grid lines so
            # they cannot be mistaken for another measured-data layer.
            axis.grid(False)
            axis.tick_params(
                axis='both',
                which='both',
                direction='out',
                top=False,
                right=False,
                width=0.9,
                labelsize=font_sizes['tick_label']
            )

            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.9)
                spine.set_color('0.2')

            self._apply_auto_design_square_box_aspect(axis)

        def _center_slice_axes_and_colorbar(axes, colorbar):
            '''Centers the complete three-panel heatmap group in its figure.'''
            # set_box_aspect resolves its final active positions during the
            # draw pass. Resolve that geometry before calculating the shift,
            # otherwise the later draw would undo the apparent centering.
            axes[0].figure.canvas.draw()
            all_axes = list(axes) + [colorbar.ax]
            group_left = min(axis.get_position().x0 for axis in all_axes)
            group_right = max(axis.get_position().x1 for axis in all_axes)
            horizontal_shift = 0.5 - (group_left + group_right) / 2.0

            for axis in all_axes:
                position = axis.get_position()
                axis.set_position([
                    position.x0 + horizontal_shift,
                    position.y0,
                    position.width,
                    position.height
                ])

        def _is_reference_observation(observation):
            '''Returns whether one GP-training row is the slice reference.'''
            return bool(np.allclose(
                observation['physical_recipe'],
                reference_recipe,
                rtol=0.0,
                atol=1.0e-12
            ))

        has_visible_near_slice_training_condition = any(
            abs(
                observation['normalized_recipe'][panel['fixed_index']]
                - reference_normalized[panel['fixed_index']]
            ) <= slice_half_width
            and not _is_reference_observation(observation)
            for panel in panel_data
            for observation in observed_conditions
        )

        def _build_slice_observation_legend(axis):
            '''Builds only the marker entries that can be seen in the atlas.'''
            legend_handles = []

            if has_visible_near_slice_training_condition:
                training_handle, = axis.plot(
                    [], [],
                    marker='o',
                    markersize=6,
                    markerfacecolor='white',
                    markeredgecolor='#202020',
                    markeredgewidth=0.8,
                    linestyle='None',
                    label='Near-slice GP-training condition'
                )
                legend_handles.append(training_handle)

            reference_label_for_legend = (
                'Slice reference (best QC-approved condition)'
                if reference_label == 'best QC-approved observed condition'
                else 'Slice reference (first observed condition)'
            )
            reference_handle, = axis.plot(
                [], [],
                marker='*',
                markersize=10,
                markerfacecolor='#f2c14e',
                markeredgecolor='#1a1a1a',
                markeredgewidth=0.8,
                linestyle='None',
                label=reference_label_for_legend
            )
            legend_handles.append(reference_handle)

            return legend_handles

        if final_snapshot:
            final_suffix = 'final'
        else:
            if batch_number is None:
                batch_number = getattr(self, 'batch_num', 0)
            final_suffix = f'after_batch_{int(batch_number)}'

        for field_name, colorbar_label, value_getter, colormap, norm, title in (
            field_definitions
        ):
            if field_name == 'target_probability':
                maximum_probability = max(
                    float(np.max(panel['probability']))
                    for panel in panel_data
                )
                title = (
                    f'{title}; maximum = {maximum_probability:.3f}'
                )

            # Long reagent names require more physical separation than a
            # compact three-panel atlas provides.  This wider canvas retains
            # square data panels and a centered colorbar while preventing a
            # panel's y-axis title from encroaching on its neighbor.
            figure, axes = plt.subplots(1, 3, figsize=(18.2, 5.8), dpi=300)
            # This file enables Matplotlib's global auto-layout setting. The
            # slice atlas uses explicit panel and colorbar placement instead,
            # so disable auto-layout for this figure before centering it.
            figure.set_tight_layout(False)
            # Center the complete axis-and-colorbar group beneath the title
            # band.  The small right margin mirrors the left margin instead
            # of leaving an unused white strip beside the colorbar.
            figure.subplots_adjust(
                left=0.07,
                right=0.92,
                bottom=0.16,
                top=0.76,
                wspace=0.38
            )
            image = None

            for axis, panel in zip(axes, panel_data):
                # Match the ordinary 2D heatmaps: the primary figure shows
                # the complete configured GP domain without physical shading.
                # The separate diagnostic figure below owns every feasibility
                # overlay, boundary, and target-contour annotation.
                field_values = value_getter(panel)
                image = axis.pcolormesh(
                    panel['x_physical'],
                    panel['y_physical'],
                    field_values,
                    shading='auto',
                    cmap=colormap,
                    norm=norm
                )

                x_index = panel['x_index']
                y_index = panel['y_index']
                fixed_index = panel['fixed_index']
                for observation in observed_conditions:
                    if abs(
                        observation['normalized_recipe'][fixed_index]
                        - reference_normalized[fixed_index]
                    ) > slice_half_width:
                        continue

                    axis.scatter(
                        observation['physical_recipe'][x_index],
                        observation['physical_recipe'][y_index],
                        marker='o',
                        s=35,
                        facecolors='white',
                        edgecolors='#202020',
                        linewidths=0.8,
                        zorder=4
                    )

                axis.scatter(
                    reference_recipe[x_index],
                    reference_recipe[y_index],
                    marker='*',
                    s=105,
                    facecolors='#f2c14e',
                    edgecolors='#1a1a1a',
                    linewidths=0.8,
                    zorder=5
                )
                axis.set_xlabel(
                    self._format_auto_design_axis_label(reagent_names[x_index]),
                    fontsize=font_sizes['axis_label']
                )
                axis.set_ylabel(
                    self._format_auto_design_axis_label(reagent_names[y_index]),
                    fontsize=font_sizes['axis_label']
                )
                axis.set_title(
                    f'Hold {reagent_names[fixed_index]} = '
                    f'{reference_recipe[fixed_index]:.4g} mM',
                    fontsize=font_sizes['axis_label'],
                    pad=9
                )
                _format_slice_heatmap_axis(axis)
                # pcolormesh treats these coordinates as cell centers.  Clip
                # the displayed span back to the configured concentration
                # endpoints so the 3D slice atlas uses the same true design
                # bounds as the 2D heatmaps.
                axis.set_xlim(
                    float(np.min(panel['x_physical'])),
                    float(np.max(panel['x_physical']))
                )
                axis.set_ylim(
                    float(np.min(panel['y_physical'])),
                    float(np.max(panel['y_physical']))
                )

            colorbar = figure.colorbar(
                image,
                ax=list(axes),
                shrink=0.91,
                pad=0.02
            )
            colorbar.set_label(
                colorbar_label,
                fontsize=font_sizes['axis_label']
            )
            colorbar.ax.tick_params(
                labelsize=font_sizes['tick_label'],
                width=0.9
            )
            _center_slice_axes_and_colorbar(axes, colorbar)
            figure.suptitle(
                title,
                fontsize=font_sizes['title'],
                fontweight='normal',
                y=0.965
            )
            observation_legend_handles = _build_slice_observation_legend(
                axes[0]
            )
            figure.legend(
                observation_legend_handles,
                [handle.get_label() for handle in observation_legend_handles],
                loc='upper center',
                bbox_to_anchor=(0.5, 0.91),
                ncol=len(observation_legend_handles),
                frameon=False,
                fontsize=font_sizes['legend'],
                handlelength=1.2,
                handletextpad=0.45,
                columnspacing=1.0
            )
            output_path = os.path.join(
                self.plot_path,
                f'gpr_3d_{field_name}_orthogonal_slices_{final_suffix}.png'
            )
            # Keep these multi-panel scientific figures at print resolution
            # so they can be placed directly in posters and presentations.
            figure.savefig(output_path, dpi=300)
            plt.close(figure)
            generated_plot_paths.append(output_path)

        # Preserve the compact original atlases above, then emit separate
        # feasibility-overlay versions matching the established 2D diagnostic
        # convention. Target probability does not have a direct 2D analogue,
        # so only mean and uncertainty receive these extra physical-boundary
        # figures.
        for (
            field_name,
            colorbar_label,
            value_getter,
            colormap,
            norm,
            title
        ) in field_definitions[:2]:
            figure, axes = plt.subplots(1, 3, figsize=(18.2, 6.6), dpi=300)
            # Keep the manually centered panel-plus-colorbar group intact at
            # save time rather than allowing global auto-layout to move it.
            figure.set_tight_layout(False)
            figure.subplots_adjust(
                left=0.07,
                right=0.92,
                bottom=0.15,
                top=0.74,
                wspace=0.38
            )
            image = None

            for axis, panel in zip(axes, panel_data):
                # Deliberately leave the GP field unmasked in the diagnostic
                # version, as the existing 2D feasibility plots do. The gray
                # overlay and labeled boundaries then make the physical region
                # explicit without changing the GP's configured domain.
                image = axis.pcolormesh(
                    panel['x_physical'],
                    panel['y_physical'],
                    value_getter(panel),
                    shading='auto',
                    cmap=colormap,
                    norm=norm
                )
                _draw_slice_feasibility_overlay(
                    axis,
                    panel,
                    field_name
                )

                x_index = panel['x_index']
                y_index = panel['y_index']
                fixed_index = panel['fixed_index']
                for observation in observed_conditions:
                    if abs(
                        observation['normalized_recipe'][fixed_index]
                        - reference_normalized[fixed_index]
                    ) > slice_half_width:
                        continue

                    axis.scatter(
                        observation['physical_recipe'][x_index],
                        observation['physical_recipe'][y_index],
                        marker='o',
                        s=35,
                        facecolors='white',
                        edgecolors='#202020',
                        linewidths=0.8,
                        zorder=6
                    )

                axis.scatter(
                    reference_recipe[x_index],
                    reference_recipe[y_index],
                    marker='*',
                    s=105,
                    facecolors='#f2c14e',
                    edgecolors='#1a1a1a',
                    linewidths=0.8,
                    zorder=7
                )
                axis.set_xlabel(
                    self._format_auto_design_axis_label(reagent_names[x_index]),
                    fontsize=font_sizes['axis_label']
                )
                axis.set_ylabel(
                    self._format_auto_design_axis_label(reagent_names[y_index]),
                    fontsize=font_sizes['axis_label']
                )
                axis.set_title(
                    f'Hold {reagent_names[fixed_index]} = '
                    f'{reference_recipe[fixed_index]:.4g} mM',
                    fontsize=font_sizes['axis_label'],
                    pad=9
                )
                _format_slice_heatmap_axis(axis)
                # The field uses the 100 x 100 ordinary GP grid, while the
                # physical boundaries use a separate 401 x 401 grid.  Exact
                # endpoint limits prevent a half-cell display fringe between
                # those dense overlays and the heatmap border.
                axis.set_xlim(
                    float(np.min(panel['x_physical'])),
                    float(np.max(panel['x_physical']))
                )
                axis.set_ylim(
                    float(np.min(panel['y_physical'])),
                    float(np.max(panel['y_physical']))
                )

            colorbar = figure.colorbar(
                image,
                ax=list(axes),
                shrink=0.91,
                pad=0.02
            )
            colorbar.set_label(
                colorbar_label,
                fontsize=font_sizes['axis_label']
            )
            colorbar.ax.tick_params(
                labelsize=font_sizes['tick_label'],
                width=0.9
            )
            _center_slice_axes_and_colorbar(axes, colorbar)
            figure.suptitle(
                f'{title}: physical feasibility overlay',
                fontsize=font_sizes['title'],
                fontweight='normal',
                y=0.97
            )
            feasibility_legend_handles = _build_slice_feasibility_legend(
                axes[0],
                field_name
            )
            feasibility_legend_handles.extend(
                _build_slice_observation_legend(axes[0])
            )
            figure.legend(
                feasibility_legend_handles,
                [handle.get_label() for handle in feasibility_legend_handles],
                loc='upper center',
                bbox_to_anchor=(0.5, 0.91),
                ncol=3,
                frameon=False,
                fontsize=font_sizes['legend'],
                handlelength=1.7,
                columnspacing=0.9
            )
            output_path = os.path.join(
                self.plot_path,
                f'gpr_3d_{field_name}_orthogonal_slices_feasibility_'
                f'{final_suffix}.png'
            )
            # Save the overlay counterpart at the same print resolution as
            # its ordinary GP slice atlas.
            figure.savefig(output_path, dpi=300)
            plt.close(figure)
            generated_plot_paths.append(output_path)

        return generated_plot_paths

    def _update_auto_quit_from_condition_level_performance(
        self,
        model,
        batch_number
    ):
        '''
        Updates the Auto quit flag using condition-level duplicate statistics.

        The optimizer receives physical replicate-well results, so raw
        optimizer-side target stopping can stop too early if one random
        duplicate happens to land near the target. This controller-side rule
        evaluates the duplicate-aggregated condition mean and replicate
        variability instead.

        params:
            OptimizationModel model:
                Auto optimizer object whose quit flag should be updated.

            int batch_number:
                Completed batch number to evaluate for stopping.
        '''
        if len(self.auto_model_performance_rows) == 0:
            return

        performance_df = pd.DataFrame(self.auto_model_performance_rows)

        batch_df = performance_df[
            performance_df['batch_number'] == batch_number
        ].copy()

        if batch_df.empty:
            print(
                "<<controller warning>> could not evaluate condition-level "
                f"Auto stop rule because batch {batch_number} has no "
                "performance rows"
            )
            return

        # target_tolerance_nm is Header-configurable. Replicate consistency
        # remains an independent conservative safeguard (25 nm by default).
        target_tolerance_nm = self._get_auto_target_tolerance_nm()
        replicate_sd_tolerance_nm = (
            self._get_auto_replicate_sd_tolerance_nm()
        )

        target_error_values = pd.to_numeric(
            batch_df['target_error_nm'],
            errors='coerce'
        )

        replicate_sd_values = pd.to_numeric(
            batch_df['actual_lambda_sd_nm'],
            errors='coerce'
        )

        target_hit = target_error_values <= target_tolerance_nm

        # A target hit must be explicitly authorized by the same
        # replicate-validation policy used for target-EI incumbents. Missing
        # eligibility fields fail closed. The additional count/SD checks make
        # the stop rule robust when reading partially populated or externally
        # edited performance rows.
        if 'eligible_for_target_stop' in batch_df.columns:
            target_stop_eligible = (
                batch_df['eligible_for_target_stop'] == True
            )
        else:
            target_stop_eligible = pd.Series(
                False,
                index=batch_df.index,
                dtype=bool
            )

        if (
            'n_finite_qc_replicates_for_target_validation'
            in batch_df.columns
        ):
            finite_replicate_counts = pd.to_numeric(
                batch_df[
                    'n_finite_qc_replicates_for_target_validation'
                ],
                errors='coerce'
            )
        else:
            finite_replicate_counts = pd.Series(
                float('nan'),
                index=batch_df.index,
                dtype=float
            )

        replicate_consistent = (
            replicate_sd_values.notna()
            & np.isfinite(replicate_sd_values)
            & (replicate_sd_values <= replicate_sd_tolerance_nm)
            & (finite_replicate_counts >= 2)
        )

        validated_hit = (
            target_hit
            & target_stop_eligible
            & replicate_consistent
        )

        if validated_hit.any():
            best_hit_row = batch_df.loc[
                target_error_values[validated_hit].idxmin()
            ]

            model.quit = True

            print(
                "<<controller>> Exit due to validated condition-level target "
                "hit"
            )
            print(
                "<<controller>> stopping condition: "
                f"mean lambda max = "
                f"{best_hit_row['actual_lambda_mean_nm']:.4f} nm, "
                f"target error = {best_hit_row['target_error_nm']:.4f} nm, "
                f"replicate SD = "
                f"{best_hit_row['actual_lambda_sd_nm']:.4f} nm"
            )

            return

        if model.curr_iter >= model.max_iters:
            model.quit = True
            print("<<controller>> Exit due to max_iters")
            return

        model.quit = False

        print(
            "<<controller>> continuing Auto: no validated condition-level "
            "target hit"
        )

        finite_target_errors = target_error_values[
            target_error_values.notna()
            & np.isfinite(target_error_values)
        ]

        if finite_target_errors.empty:
            print(
                "<<controller warning>> latest batch has no finite "
                "condition-level target error to summarize"
            )
            return

        best_row_index = finite_target_errors.idxmin()
        best_row = batch_df.loc[best_row_index]

        print(
            "<<controller>> best condition in latest batch: "
            f"mean lambda max = {best_row['actual_lambda_mean_nm']:.4f} nm, "
            f"target error = {best_row['target_error_nm']:.4f} nm, "
            f"replicate SD = {best_row['actual_lambda_sd_nm']:.4f} nm"
        )
    
    def _clip_errorbars_to_lambda_display_window(
        self,
        means,
        errors,
        condition_numbers,
        y_min_nm=300.0,
        y_max_nm=1000.0,
        lower_cap_visibility_pad_nm=2.0,
        upper_cap_visibility_pad_nm=2.0
    ):
        '''
        Clips vertical error bars to the displayed lambda-max window.

        This is display-only. Raw SEM/SD values remain unchanged in the Auto
        performance log. Error bars are allowed to span nearly the full visible
        lambda-max display range. If a raw error bar would extend below or above
        that display window, the displayed bar is clipped slightly inside the
        window and the condition number is returned for plot annotation.

        The lower and upper cap visibility pads are intentionally configurable.
        They keep clipped caps just inside the boxed plot frame while preserving
        the fixed displayed lambda-max window.

        params:
            array-like means:
                Mean lambda values for plotted points.

            array-like errors:
                Raw SEM or GP predictive SD values.

            array-like condition_numbers:
                Reaction condition numbers corresponding to the plotted points.

            float y_min_nm:
                Lower displayed lambda-max bound.

            float y_max_nm:
                Upper displayed lambda-max bound.

            float lower_cap_visibility_pad_nm:
                Display-only padding used to keep lower clipped error-bar caps
                visible just above the lower axis limit.

            float upper_cap_visibility_pad_nm:
                Display-only padding used to keep upper clipped error-bar caps
                visible just below the upper axis limit.

        returns:
            tuple:
                display_yerr:
                    2 x N numpy array of asymmetric display error bars suitable
                    for matplotlib yerr.

                clipped_condition_numbers:
                    List of condition numbers where the raw error bar exceeded
                    the display window.
        '''
        means = np.array(means, dtype=float)
        errors = np.array(errors, dtype=float)
        condition_numbers = np.array(condition_numbers, dtype=float)

        errors = np.nan_to_num(
            errors,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        raw_lower_bounds = means - errors
        raw_upper_bounds = means + errors

        clipped_low_mask = raw_lower_bounds <= y_min_nm
        clipped_high_mask = raw_upper_bounds >= y_max_nm

        clipped_mask = clipped_low_mask | clipped_high_mask

        lower_clip_boundary = y_min_nm + lower_cap_visibility_pad_nm
        upper_clip_boundary = y_max_nm - upper_cap_visibility_pad_nm

        if lower_clip_boundary >= upper_clip_boundary:
            lower_clip_boundary = y_min_nm
            upper_clip_boundary = y_max_nm

        lower_available_to_axis = np.maximum(means - y_min_nm, 0.0)
        upper_available_to_axis = np.maximum(y_max_nm - means, 0.0)

        lower_available_to_visible_cap = np.maximum(
            means - lower_clip_boundary,
            0.0
        )

        upper_available_to_visible_cap = np.maximum(
            upper_clip_boundary - means,
            0.0
        )

        lower_display_errors = np.minimum(errors, lower_available_to_axis)
        upper_display_errors = np.minimum(errors, upper_available_to_axis)

        lower_display_errors = np.where(
            clipped_low_mask,
            lower_available_to_visible_cap,
            lower_display_errors
        )

        upper_display_errors = np.where(
            clipped_high_mask,
            upper_available_to_visible_cap,
            upper_display_errors
        )

        clipped_condition_numbers = condition_numbers[
            clipped_mask
        ].astype(int).tolist()

        display_yerr = np.vstack(
            [
                lower_display_errors,
                upper_display_errors
            ]
        )

        return display_yerr, clipped_condition_numbers
    
    def _generate_auto_plot_suite(
        self,
        stage,
        model=None,
        batch_number=None
    ):
        '''
        Generates the Auto plots appropriate to one lifecycle stage.

        This is the central coordinator for automatic Auto plotting. The
        spreadsheet selects a general auto_plot_profile; this method determines
        which outputs are scientifically appropriate for the current run stage
        and number of variable reagents.

        Supported stages:

            after_measurement:
                Runs after measured lambda-max results and replicate QC have
                been recorded for a completed batch.

                standard:
                    Generates cumulative lambda progress and replicate plots.
                    Portfolio runs additionally generate a companion trace
                    with acquisition-mode-specific marker styles.

                final_only or off:
                    Generates nothing.

            after_model_update:
                Runs after the GP model has incorporated the completed batch.

                standard:
                    Refreshes the fitted two-dimensional GP prediction grid and
                    generates prediction and uncertainty heatmaps when there
                    are exactly two variable reagents. For exactly three
                    reagents, generates conditional mean, uncertainty, and
                    target-tolerance probability slice atlases instead.

                final_only or off:
                    Generates nothing.

            final:
                Runs after CSV exports are complete and all batches have
                finished.

                standard:
                    Generates final lambda plots, dimension-aware seed and
                    exploration plots, and the final Auto report.

                final_only:
                    Generates the same final outputs. Two-variable runs first
                    refresh the fitted GP grid for one final heatmap pair;
                    three-variable runs generate one final conditional-slice
                    atlas set because per-batch plotting was suppressed.

                off:
                    Generates no automatic plots or report.

        Plotting and report failures are isolated by output type. A failure in
        one diagnostic does not prevent the remaining outputs from being
        attempted and does not alter recipes, optimizer data, QC decisions,
        robot execution, or CSV exports.

        params:
            str stage:
                One of after_measurement, after_model_update, or final.

            OptimizationModel or None model:
                Current Auto optimization model. Required for GP heatmaps or
                three-variable conditional slice atlases.

            int or None batch_number:
                Completed batch represented by the generated outputs.

        returns:
            list:
                Paths of successfully generated plots and reports.
        '''
        generated_output_paths = []

        normalized_stage = str(
            stage
        ).strip().lower().replace(
            '-',
            '_'
        ).replace(
            ' ',
            '_'
        )

        valid_stages = {
            'after_measurement',
            'after_model_update',
            'final'
        }

        if normalized_stage not in valid_stages:
            raise ValueError(
                "Auto plot-suite stage must be one of: "
                "after_measurement, after_model_update, or final. "
                f"Received: {stage!r}."
            )

        plot_profile = str(
            getattr(
                self,
                'robo_params',
                {}
            ).get(
                'auto_plot_profile',
                'standard'
            )
        ).strip().lower()

        if plot_profile not in {
            'standard',
            'final_only',
            'off'
        }:
            raise ValueError(
                "Auto plot profile must be standard, final_only, or off. "
                f"Received: {plot_profile!r}."
            )

        if plot_profile == 'off':
            return generated_output_paths

        if (
            plot_profile == 'final_only'
            and normalized_stage != 'final'
        ):
            return generated_output_paths

        if batch_number is None:
            if normalized_stage == 'final':
                batch_number = (
                    int(
                        getattr(
                            self,
                            'batch_num',
                            0
                        )
                    )
                    - 1
                )

            else:
                batch_number = getattr(
                    self,
                    'batch_num',
                    None
                )

        try:
            completed_batch_number = int(
                batch_number
            )

        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                "Auto plot-suite batch_number must identify a completed "
                f"integer batch. Received: {batch_number!r}."
            )

        if completed_batch_number < 0:
            raise ValueError(
                "Auto plot-suite batch_number cannot be negative. "
                f"Received: {completed_batch_number}."
            )

        def _record_generated_paths(result):
            '''
            Adds path-like output values to generated_output_paths.
            '''
            if result is None:
                return

            if isinstance(
                result,
                (str, os.PathLike)
            ):
                result_path = os.fspath(
                    result
                )

                if result_path not in generated_output_paths:
                    generated_output_paths.append(
                        result_path
                    )

                return

            if isinstance(
                result,
                (list, tuple, set)
            ):
                for item in result:
                    _record_generated_paths(
                        item
                    )

        def _run_output_step(
            output_description,
            output_function
        ):
            '''
            Runs one nonfatal plot/report step and records returned paths.
            '''
            try:
                output_result = output_function()

                _record_generated_paths(
                    output_result
                )

            except Exception as exc:
                print(
                    f"<<controller warning>> failed to generate "
                    f"{output_description} during Auto plot stage "
                    f"{normalized_stage}; continuing Auto mode. "
                    f"Error: {exc}"
                )

        def _refresh_and_plot_2d_gp(
            optimization_model,
            plot_batch_number
        ):
            '''
            Refreshes the controller-facing GP grids from the current fitted
            model, then generates the paired prediction and uncertainty plots.

            The refresh must happen after update_experiment_data() so a plot
            titled "After Batch N" actually includes Batch N in the fitted GP.
            '''
            if optimization_model is None:
                print(
                    "<<controller warning>> skipping automatic 2D GP plots "
                    "because no optimization model was supplied"
                )
                return []

            refresh_method = getattr(
                optimization_model,
                'refresh_prediction_grid_for_plotting',
                None
            )

            if not callable(refresh_method):
                raise AttributeError(
                    "OptimizationModel does not provide "
                    "refresh_prediction_grid_for_plotting(). Update "
                    "optimizers.py before using automatic GP plotting."
                )

            refresh_method()

            return self.plot_2D_GPR(
                model=optimization_model,
                batch_number=plot_batch_number
            )

        def _plot_3d_gp_slices(
            optimization_model,
            plot_batch_number,
            final_snapshot=False
        ):
            '''Generates read-only conditional GP slices from the fitted GP.'''
            if optimization_model is None:
                print(
                    "<<controller warning>> skipping automatic 3D GP slice "
                    "plots because no optimization model was supplied"
                )
                return []

            return self.plot_3D_GPR_orthogonal_slices(
                model=optimization_model,
                batch_number=plot_batch_number,
                final_snapshot=final_snapshot
            )

        if normalized_stage == 'after_measurement':
            _run_output_step(
                (
                    f"lambda progress plot through batch "
                    f"{completed_batch_number}"
                ),
                lambda: self._plot_lambda_progress_after_batch(
                    completed_batch_number
                )
            )

            if len(getattr(model, 'acquisition_modes', [])) > 1:
                _run_output_step(
                    (
                        f"portfolio acquisition trace through batch "
                        f"{completed_batch_number}"
                    ),
                    lambda: self._plot_auto_portfolio_trace_after_batch(
                        completed_batch_number
                    )
                )

            _run_output_step(
                (
                    f"lambda replicate plot through batch "
                    f"{completed_batch_number}"
                ),
                lambda: (
                    self._plot_lambda_replicate_progress_after_batch(
                        completed_batch_number
                    )
                )
            )

        elif normalized_stage == 'after_model_update':
            n_variable_reagents = len(
                getattr(
                    self,
                    'variable_reagents',
                    []
                )
            )

            if n_variable_reagents == 2:
                _run_output_step(
                    (
                        f"2D GP prediction and uncertainty plots after "
                        f"batch {completed_batch_number}"
                    ),
                    lambda: _refresh_and_plot_2d_gp(
                        optimization_model=model,
                        plot_batch_number=completed_batch_number
                    )
                )

            elif n_variable_reagents == 3:
                _run_output_step(
                    (
                        f"3D conditional GP slice atlases after batch "
                        f"{completed_batch_number}"
                    ),
                    lambda: _plot_3d_gp_slices(
                        optimization_model=model,
                        plot_batch_number=completed_batch_number
                    )
                )

        elif normalized_stage == 'final':
            # final_only suppresses all per-batch GP plots, so create one final
            # fitted-model snapshot here when exactly two variables are used.
            if (
                plot_profile == 'final_only'
                and len(
                    getattr(
                        self,
                        'variable_reagents',
                        []
                    )
                ) == 2
            ):
                _run_output_step(
                    'final 2D GP prediction and uncertainty plots',
                    lambda: _refresh_and_plot_2d_gp(
                        optimization_model=model,
                        plot_batch_number=completed_batch_number
                    )
                )

            if len(
                getattr(
                    self,
                    'variable_reagents',
                    []
                )
            ) == 3:
                _run_output_step(
                    'final 3D conditional GP slice atlases',
                    lambda: _plot_3d_gp_slices(
                        optimization_model=model,
                        plot_batch_number=completed_batch_number,
                        final_snapshot=True
                    )
                )

            _run_output_step(
                'final lambda progress plot',
                lambda: self._plot_lambda_progress_after_batch(
                    completed_batch_number,
                    plot_filename='lambda_progress_final.png',
                    plot_title=(
                        rf'Final Auto $\lambda_{{\max}}$ Progress'
                    )
                )
            )

            _run_output_step(
                'final lambda replicate plot',
                lambda: (
                    self._plot_lambda_replicate_progress_after_batch(
                        completed_batch_number,
                        plot_filename='lambda_replicates_final.png',
                        plot_title=(
                            rf'Final Auto Replicate '
                            rf'$\lambda_{{\max}}$ Values'
                        )
                    )
                )
            )

            if len(getattr(model, 'acquisition_modes', [])) > 1:
                _run_output_step(
                    'final portfolio acquisition trace',
                    lambda: self._plot_auto_portfolio_trace_after_batch(
                        completed_batch_number,
                        plot_filename='acquisition_portfolio_trace_final.png',
                        plot_title='Final Auto Acquisition Portfolio Trace'
                    )
                )

            _run_output_step(
                'final dimension-aware Auto design-space plots',
                self._plot_initial_training_designs_after_run
            )

            # Generate the report last so it can detect and embed every final
            # plot that was successfully written.
            _run_output_step(
                'final Auto run report',
                self._write_auto_run_report
            )

        if len(generated_output_paths) > 0:
            terminal_verbosity = self.robo_params.get(
                'auto_terminal_verbosity',
                'standard'
            )
            if terminal_verbosity == 'diagnostic':
                print(
                    f"<<controller diagnostic>> Auto plot stage "
                    f"{normalized_stage} generated: "
                    + ", ".join(generated_output_paths)
                )
            elif terminal_verbosity == 'standard':
                print(
                    f"<<controller>> Auto plot stage {normalized_stage} "
                    f"generated {len(generated_output_paths)} artifact(s)"
                )

        return generated_output_paths
    
    def _apply_auto_lambda_plot_lab_frame_style(self, ax):
        '''
        Applies lab-standard axis styling to Auto lambda plots.

        This styling removes background grid lines while preserving explicitly
        drawn scientific reference lines such as the target lambda line. It also
        makes all four plot spines visible so exported plots have a complete
        boxed frame.

        params:
            matplotlib.axes.Axes ax:
                Axis object to style.

        returns:
            None
        '''
        ax.grid(False)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)
            spine.set_color('0.2')

        font_sizes = self._get_auto_lambda_plot_font_sizes()

        ax.tick_params(
            axis='both',
            which='both',
            direction='out',
            top=False,
            right=False,
            labelsize=font_sizes['tick_label']
        )
    
    def _get_auto_lambda_plot_font_sizes(self, scale_factor=1.25):
        '''
        Returns centralized font sizes for Auto lambda progress and replicate
        diagnostic plots.

        The scale factor makes the exported plots more readable in slides while
        keeping the proportions controlled and consistent between companion
        lambda plots.

        params:
            float scale_factor:
                Multiplicative font-size scale applied to the previous base
                Auto lambda plot font sizes.

        returns:
            dict:
                Font-size values for titles, labels, ticks, legends, and footer
                annotations.
        '''
        return {
            'title': 11.0 * scale_factor,
            'axis_label': 10.0 * scale_factor,
            'tick_label': 10.0 * scale_factor,
            'legend': 8.0 * scale_factor,
            'footer': 7.3 * scale_factor
        }
    
    def _plot_lambda_progress_after_batch(
        self,
        batch_number,
        plot_filename=None,
        plot_title=None,
        y_axis_mode='robust',
        errorbar_display_cap_nm=75.0,
        y_display_min_nm=300.0,
        y_display_max_nm=1000.0
    ):
        '''
        Generates a cumulative lambda max progress plot after a completed Auto
        batch.

        This plot is dimension-agnostic. It does not plot reagent-space
        coordinates, so it can be used for any number of variable reagents.

        Experimental results are plotted as condition-level lambda max means
        with SEM error bars across duplicate wells. Model predictions, when
        available, are plotted as pre-experiment GP-predicted lambda max means
        with GP predictive standard deviation error bars.

        Actual and model points are plotted at the same reaction condition
        number when both are available. This keeps the visual meaning clear:
        both values refer to the same recipe condition.

        Error bars are displayed within the lambda-max scan/display window,
        default 300-1000 nm. Raw SEM/SD values remain unchanged in the Auto
        performance log. If an error bar would extend outside the display
        window, it is clipped at the display boundary and the condition number
        is noted in the plot annotation.

        params:
            int batch_number:
                Highest completed batch number to include in the cumulative
                plot.

            str plot_filename:
                Optional filename for the saved plot.

            str plot_title:
                Optional title for the saved plot.

            str y_axis_mode:
                Retained for backward compatibility. The plot now uses a fixed
                lambda display window defined by y_display_min_nm and
                y_display_max_nm.

            float errorbar_display_cap_nm:
                Retained for backward compatibility. The previous arbitrary
                fixed-size display cap is no longer used.

            float y_display_min_nm:
                Lower displayed lambda-max bound.

            float y_display_max_nm:
                Upper displayed lambda-max bound.

        returns:
            str or None:
                Path to the saved plot, or None if there are no rows to plot.
        '''
        if len(self.auto_model_performance_rows) == 0:
            print(
                "<<controller warning>> skipping lambda progress plot because "
                "auto_model_performance_rows is empty"
            )
            return None

        performance_df = pd.DataFrame(self.auto_model_performance_rows)

        performance_df = performance_df[
            performance_df['batch_number'] <= batch_number
        ].copy()

        if performance_df.empty:
            print(
                "<<controller warning>> skipping lambda progress plot because "
                f"there are no performance rows through batch {batch_number}"
            )
            return None

        performance_df = performance_df.sort_values('reaction_number')

        performance_df['reaction_number'] = performance_df[
            'reaction_number'
        ].astype(float)

        has_prediction = performance_df[
            'predicted_lambda_mean_nm'
        ].notna()

        prediction_df = performance_df[has_prediction].copy()

        actual_x_values = performance_df['reaction_number'].to_numpy(
            dtype=float
        )

        actual_means = pd.to_numeric(
            performance_df['actual_lambda_mean_nm'],
            errors='coerce'
        ).to_numpy(dtype=float)

        actual_sems = pd.to_numeric(
            performance_df['actual_lambda_sem_nm'],
            errors='coerce'
        ).fillna(0.0).to_numpy(dtype=float)

        actual_display_yerr, actual_clipped_condition_numbers = (
            self._clip_errorbars_to_lambda_display_window(
                means=actual_means,
                errors=actual_sems,
                condition_numbers=actual_x_values,
                y_min_nm=y_display_min_nm,
                y_max_nm=y_display_max_nm
            )
        )

        target_lambda = float(performance_df['target_lambda_max_nm'].iloc[0])

        observed_label = 'Observed mean ± SEM'
        prediction_label = 'GP prediction ± SD'

        actual_color = 'tab:blue'
        model_color = 'tab:orange'
        target_color = '0.25'

        font_sizes = self._get_auto_lambda_plot_font_sizes()

        fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=300)
        fig.set_tight_layout(False)

        target_handle = ax.axhline(
            target_lambda,
            color=target_color,
            linestyle='--',
            linewidth=1.1,
            alpha=0.8,
            zorder=1,
            label=f'Target = {target_lambda:.0f} nm'
        )

        prediction_handle = None
        prediction_clipped_condition_numbers = []

        if not prediction_df.empty:
            pred_x_values = prediction_df[
                'reaction_number'
            ].to_numpy(dtype=float)

            predicted_means = pd.to_numeric(
                prediction_df['predicted_lambda_mean_nm'],
                errors='coerce'
            ).to_numpy(dtype=float)

            predicted_stds = pd.to_numeric(
                prediction_df['predicted_lambda_std_nm'],
                errors='coerce'
            ).fillna(0.0).to_numpy(dtype=float)

            predicted_display_yerr, prediction_clipped_condition_numbers = (
                self._clip_errorbars_to_lambda_display_window(
                    means=predicted_means,
                    errors=predicted_stds,
                    condition_numbers=pred_x_values,
                    y_min_nm=y_display_min_nm,
                    y_max_nm=y_display_max_nm
                )
            )

            prediction_handle = ax.errorbar(
                pred_x_values,
                predicted_means,
                yerr=predicted_display_yerr,
                fmt='s',
                color=model_color,
                ecolor=model_color,
                markerfacecolor='none',
                markeredgecolor=model_color,
                markeredgewidth=1.2,
                elinewidth=1.0,
                capsize=4,
                markersize=4.0,
                barsabove=True,
                alpha=0.75,
                zorder=2,
                label=prediction_label
            )

        observed_handle = ax.errorbar(
            actual_x_values,
            actual_means,
            yerr=actual_display_yerr,
            fmt='o',
            color=actual_color,
            ecolor=actual_color,
            markerfacecolor='none',
            markeredgecolor=actual_color,
            markeredgewidth=1.2,
            elinewidth=1.0,
            capsize=4,
            markersize=4.0,
            barsabove=True,
            alpha=0.95,
            zorder=3,
            label=observed_label
        )

        integer_ticks = performance_df['reaction_number'].astype(int).to_list()
        ax.set_xticks(integer_ticks)

        ax.set_xlabel(
            'Reaction condition number',
            fontsize=font_sizes['axis_label'],
            labelpad=5
        )

        ax.set_ylabel(
            r'$\lambda_{\max}$ (nm)',
            fontsize=font_sizes['axis_label']
        )

        if plot_title is None:
            plot_title = (
                rf'Auto $\lambda_{{\max}}$ Progress After Batch '
                f'{batch_number}'
            )

        fig.suptitle(
            plot_title,
            fontsize=font_sizes['title'],
            fontweight='normal',
            y=0.965
        )

        ax.set_ylim(y_display_min_nm, y_display_max_nm)

        self._apply_auto_lambda_plot_lab_frame_style(ax)

        legend_handles = [observed_handle]
        legend_labels = [observed_label]

        if prediction_handle is not None:
            legend_handles.append(prediction_handle)
            legend_labels.append(prediction_label)

        legend_handles.append(target_handle)
        legend_labels.append(f'Target = {target_lambda:.0f} nm')

        ax.legend(
            legend_handles,
            legend_labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 1.015),
            ncol=min(len(legend_handles), 4),
            frameon=False,
            fontsize=font_sizes['legend'],
            handlelength=1.2,
            handletextpad=0.45,
            columnspacing=0.9,
            borderaxespad=0.0
        )

        def _format_clipped_condition_list(condition_numbers):
            '''
            Formats clipped reaction condition numbers for a compact plot note.
            '''
            if len(condition_numbers) == 0:
                return 'none'

            unique_condition_numbers = sorted(set(condition_numbers))

            if len(unique_condition_numbers) <= 8:
                return ", ".join(
                    [str(x) for x in unique_condition_numbers]
                )

            first_values = ", ".join(
                [str(x) for x in unique_condition_numbers[:6]]
            )

            return (
                f"{len(unique_condition_numbers)} conditions "
                f"({first_values}, ...)"
            )

        display_note_parts = []

        if len(prediction_clipped_condition_numbers) > 0:
            display_note_parts.append(
                "GP SD clipped at conditions "
                + _format_clipped_condition_list(
                    prediction_clipped_condition_numbers
                )
            )

        if len(actual_clipped_condition_numbers) > 0:
            display_note_parts.append(
                "SEM clipped at conditions "
                + _format_clipped_condition_list(
                    actual_clipped_condition_numbers
                )
            )

        if len(display_note_parts) > 0:
            display_note = (
                f"Display window: {y_display_min_nm:.0f}–"
                f"{y_display_max_nm:.0f} nm | "
                + " | ".join(display_note_parts)
            )

            fig.text(
                0.5,
                0.035,
                display_note,
                ha='center',
                va='center',
                fontsize=font_sizes['footer'],
                color='0.35'
            )

            bottom_margin = 0.17
        else:
            bottom_margin = 0.14

        fig.subplots_adjust(
            left=0.12,
            right=0.97,
            bottom=bottom_margin,
            top=0.86
        )

        if plot_filename is None:
            plot_filename = f'lambda_progress_after_batch_{batch_number}.png'

        plot_path = os.path.join(
            self.plot_path,
            plot_filename
        )

        fig.savefig(plot_path)
        plt.close(fig)

        print(
            f"<<controller>> saved lambda progress plot to {plot_path}"
        )

        return plot_path
    
    def _plot_lambda_replicate_progress_after_batch(
        self,
        batch_number,
        plot_filename=None,
        plot_title=None,
        y_axis_mode='robust',
        errorbar_display_cap_nm=75.0,
        y_display_min_nm=300.0,
        y_display_max_nm=1000.0
    ):
        '''
        Generates a cumulative replicate-level lambda max progress plot after a
        completed Auto batch.

        This companion plot preserves the GP prediction display from the main
        lambda progress plot, but replaces the observed condition mean +/- SEM
        with individual replicate lambda max points.

        QC-included replicate values are plotted separately from QC-excluded
        replicate values. Raw replicate values are not altered. GP predictive
        SD error bars are displayed within the lambda-max scan/display window,
        default 300-1000 nm. If a GP SD error bar would extend outside the
        display window, it is clipped at the display boundary and the condition
        number is noted in the plot annotation.

        params:
            int batch_number:
                Highest completed batch number to include in the cumulative
                plot.

            str plot_filename:
                Optional output filename. If None, a batch-specific filename is
                generated automatically.

            str plot_title:
                Optional plot title.

            str y_axis_mode:
                Retained for backward compatibility. The plot now uses a fixed
                lambda display window defined by y_display_min_nm and
                y_display_max_nm.

            float errorbar_display_cap_nm:
                Retained for backward compatibility. The previous arbitrary
                fixed-size display cap is no longer used.

            float y_display_min_nm:
                Lower displayed lambda-max bound.

            float y_display_max_nm:
                Upper displayed lambda-max bound.

        returns:
            str or None:
                Path to the saved plot, or None if there are no rows to plot.
        '''
        if len(self.auto_model_performance_rows) == 0:
            print(
                "<<controller warning>> skipping lambda replicate progress "
                "plot because auto_model_performance_rows is empty"
            )
            return None

        performance_df = pd.DataFrame(self.auto_model_performance_rows)

        performance_df = performance_df[
            performance_df['batch_number'] <= batch_number
        ].copy()

        if performance_df.empty:
            print(
                "<<controller warning>> skipping lambda replicate progress "
                f"plot because there are no performance rows through batch "
                f"{batch_number}"
            )
            return None

        performance_df = performance_df.sort_values('reaction_number')

        performance_df['reaction_number'] = performance_df[
            'reaction_number'
        ].astype(float)

        target_lambda = float(performance_df['target_lambda_max_nm'].iloc[0])

        has_prediction = performance_df[
            'predicted_lambda_mean_nm'
        ].notna()

        prediction_df = performance_df[has_prediction].copy()

        replicate_value_columns = [
            col for col in performance_df.columns
            if (
                col.startswith('actual_lambda_rep_')
                and col.endswith('_nm')
            )
        ]

        def _replicate_number_from_column(column_name):
            '''
            Extracts the replicate number from columns named like:
            actual_lambda_rep_1_nm
            '''
            return int(
                column_name.replace('actual_lambda_rep_', '').replace(
                    '_nm',
                    ''
                )
            )

        replicate_value_columns = sorted(
            replicate_value_columns,
            key=_replicate_number_from_column
        )

        if len(replicate_value_columns) == 0:
            print(
                "<<controller warning>> skipping lambda replicate progress "
                "plot because no actual_lambda_rep_*_nm columns were found"
            )
            return None

        included_x_values = []
        included_y_values = []
        excluded_x_values = []
        excluded_y_values = []

        # Plot all replicate values directly above the same reaction condition.
        # Do not horizontally jitter/offset replicate points; overlapping points
        # intentionally indicate identical or very similar replicate lambda max
        # values at the same condition.
        replicate_offsets = [0.0] * len(replicate_value_columns)

        def _value_is_true(value):
            '''
            Converts bool/string/numeric QC inclusion values into a boolean.
            Missing values default to True so older rows without QC flags still
            plot as included observations.
            '''
            if value is None or pd.isna(value):
                return True

            if isinstance(value, str):
                return value.strip().lower() in [
                    'true',
                    '1',
                    'yes',
                    'y'
                ]

            return bool(value)

        for _, row in performance_df.iterrows():
            reaction_number = float(row['reaction_number'])

            for rep_i, value_column in enumerate(replicate_value_columns):
                if value_column not in row.index:
                    continue

                replicate_value = row[value_column]

                if replicate_value is None or pd.isna(replicate_value):
                    continue

                replicate_number = _replicate_number_from_column(value_column)

                included_column = (
                    f'actual_lambda_rep_{replicate_number}_included_in_qc'
                )

                included_in_qc = True

                if included_column in row.index:
                    included_in_qc = _value_is_true(row[included_column])

                x_value = reaction_number + replicate_offsets[rep_i]
                y_value = float(replicate_value)

                if included_in_qc:
                    included_x_values.append(x_value)
                    included_y_values.append(y_value)
                else:
                    excluded_x_values.append(x_value)
                    excluded_y_values.append(y_value)

        if (
            len(included_y_values) == 0
            and len(excluded_y_values) == 0
        ):
            print(
                "<<controller warning>> skipping lambda replicate progress "
                "plot because no replicate lambda max values were available"
            )
            return None

        target_color = '0.25'
        included_color = 'tab:blue'
        excluded_color = 'tab:red'
        model_color = 'tab:orange'

        font_sizes = self._get_auto_lambda_plot_font_sizes()

        fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=300)
        fig.set_tight_layout(False)

        target_handle = ax.axhline(
            target_lambda,
            color=target_color,
            linestyle='--',
            linewidth=1.1,
            alpha=0.8,
            zorder=1,
            label=f'Target = {target_lambda:.0f} nm'
        )

        prediction_handle = None
        prediction_clipped_condition_numbers = []

        if not prediction_df.empty:
            pred_x_values = prediction_df[
                'reaction_number'
            ].to_numpy(dtype=float)

            predicted_means = pd.to_numeric(
                prediction_df['predicted_lambda_mean_nm'],
                errors='coerce'
            ).to_numpy(dtype=float)

            predicted_stds = pd.to_numeric(
                prediction_df['predicted_lambda_std_nm'],
                errors='coerce'
            ).fillna(0.0).to_numpy(dtype=float)

            predicted_display_yerr, prediction_clipped_condition_numbers = (
                self._clip_errorbars_to_lambda_display_window(
                    means=predicted_means,
                    errors=predicted_stds,
                    condition_numbers=pred_x_values,
                    y_min_nm=y_display_min_nm,
                    y_max_nm=y_display_max_nm
                )
            )

            prediction_handle = ax.errorbar(
                pred_x_values,
                predicted_means,
                yerr=predicted_display_yerr,
                fmt='s',
                color=model_color,
                ecolor=model_color,
                markerfacecolor='none',
                markeredgecolor=model_color,
                markeredgewidth=1.2,
                elinewidth=1.0,
                capsize=4,
                markersize=4.0,
                barsabove=True,
                alpha=0.75,
                zorder=2,
                label='GP prediction ± SD'
            )

        included_handle = None
        excluded_handle = None

        if len(included_y_values) > 0:
            included_handle = ax.scatter(
                included_x_values,
                included_y_values,
                s=22,
                marker='o',
                facecolors='none',
                edgecolors=included_color,
                linewidths=1.0,
                alpha=0.95,
                zorder=4,
                label='QC-included replicate'
            )

        if len(excluded_y_values) > 0:
            excluded_handle = ax.scatter(
                excluded_x_values,
                excluded_y_values,
                s=30,
                marker='x',
                color=excluded_color,
                linewidths=1.2,
                alpha=0.95,
                zorder=5,
                label='QC-excluded replicate'
            )

        integer_ticks = performance_df['reaction_number'].astype(int).to_list()
        ax.set_xticks(integer_ticks)

        ax.set_xlabel(
            'Reaction condition number',
            fontsize=font_sizes['axis_label'],
            labelpad=5
        )

        ax.set_ylabel(
            r'$\lambda_{\max}$ (nm)',
            fontsize=font_sizes['axis_label']
        )

        if plot_title is None:
            plot_title = (
                rf'Auto Replicate $\lambda_{{\max}}$ Values After Batch '
                f'{batch_number}'
            )

        fig.suptitle(
            plot_title,
            fontsize=font_sizes['title'],
            fontweight='normal',
            y=0.965
        )

        ax.set_ylim(y_display_min_nm, y_display_max_nm)

        self._apply_auto_lambda_plot_lab_frame_style(ax)

        legend_handles = []
        legend_labels = []

        if included_handle is not None:
            legend_handles.append(included_handle)
            legend_labels.append('QC-included replicate')

        if excluded_handle is not None:
            legend_handles.append(excluded_handle)
            legend_labels.append('QC-excluded replicate')

        if prediction_handle is not None:
            legend_handles.append(prediction_handle)
            legend_labels.append('GP prediction ± SD')

        legend_handles.append(target_handle)
        legend_labels.append(f'Target = {target_lambda:.0f} nm')

        ax.legend(
            legend_handles,
            legend_labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 1.015),
            ncol=min(len(legend_handles), 4),
            frameon=False,
            fontsize=font_sizes['legend'],
            handlelength=1.2,
            handletextpad=0.45,
            columnspacing=0.9,
            borderaxespad=0.0
        )

        def _format_condition_list(condition_numbers):
            '''
            Formats reaction condition numbers for a compact plot note.
            '''
            if len(condition_numbers) == 0:
                return 'none'

            unique_condition_numbers = sorted(set(condition_numbers))

            if len(unique_condition_numbers) <= 8:
                return ", ".join(
                    [str(x) for x in unique_condition_numbers]
                )

            first_values = ", ".join(
                [str(x) for x in unique_condition_numbers[:6]]
            )

            return (
                f"{len(unique_condition_numbers)} conditions "
                f"({first_values}, ...)"
            )

        excluded_condition_numbers = performance_df.loc[
            performance_df.get(
                'n_replicates_excluded',
                pd.Series(0, index=performance_df.index)
            ).fillna(0).astype(float) > 0,
            'reaction_number'
        ].astype(int).to_list()

        flagged_condition_numbers = performance_df.loc[
            performance_df.get(
                'replicate_qc_status',
                pd.Series('', index=performance_df.index)
            ).fillna('').astype(str) == 'flagged_not_excluded',
            'reaction_number'
        ].astype(int).to_list()

        note_parts = []

        if len(prediction_clipped_condition_numbers) > 0:
            note_parts.append(
                "GP SD clipped at conditions "
                + _format_condition_list(
                    prediction_clipped_condition_numbers
                )
            )

        if len(excluded_condition_numbers) > 0:
            note_parts.append(
                "QC exclusions at conditions "
                + _format_condition_list(excluded_condition_numbers)
            )

        if len(flagged_condition_numbers) > 0:
            note_parts.append(
                "QC flagged-not-excluded at conditions "
                + _format_condition_list(flagged_condition_numbers)
            )

        if len(note_parts) > 0:
            plot_note = (
                f"Display window: {y_display_min_nm:.0f}–"
                f"{y_display_max_nm:.0f} nm | "
                + " | ".join(note_parts)
            )

            fig.text(
                0.5,
                0.035,
                plot_note,
                ha='center',
                va='center',
                fontsize=font_sizes['footer'],
                color='0.35'
            )

            bottom_margin = 0.17
        else:
            bottom_margin = 0.12

        fig.subplots_adjust(
            left=0.12,
            right=0.97,
            bottom=bottom_margin,
            top=0.86
        )

        if plot_filename is None:
            plot_filename = (
                f'lambda_replicates_after_batch_{batch_number}.png'
            )

        plot_path = os.path.join(
            self.plot_path,
            plot_filename
        )

        fig.savefig(plot_path)
        plt.close(fig)

        print(
            "<<controller>> saved lambda replicate progress plot to "
            f"{plot_path}"
        )

        return plot_path

    def _plot_auto_portfolio_trace_after_batch(
        self,
        batch_number,
        plot_filename=None,
        plot_title=None,
        y_display_min_nm=300.0,
        y_display_max_nm=1000.0
    ):
        '''
        Generates a portfolio-specific companion plot without changing the
        established lambda progress or replicate diagnostic figures.

        Each acquisition mode receives a fixed, colorblind-friendly color and
        marker shape. Filled markers represent observed condition means in
        the upper panel and QC-included individual replicates in the lower
        panel. Hollow markers represent pre-experiment GP predictions; red
        outlines in the replicate panel identify QC-excluded observations.

        The visualization is intentionally a trace of the ordered portfolio,
        rather than a separate GP plot per mode. Every portfolio member is
        selected from the same pre-batch GP and all QC-approved observations
        update that one shared model after the batch is complete.
        '''
        if len(self.auto_model_performance_rows) == 0:
            return None

        performance_df = pd.DataFrame(self.auto_model_performance_rows)
        performance_df = performance_df[
            performance_df['batch_number'] <= batch_number
        ].copy()

        if performance_df.empty:
            return None

        performance_df = performance_df.sort_values('reaction_number')
        performance_df['reaction_number'] = performance_df[
            'reaction_number'
        ].astype(float)

        mode_styles = {
            'seed': {
                'label': 'Seed condition',
                'color': '0.30',
                'marker': 'o'
            },
            'exploit': {
                'label': 'Exploit',
                'color': '#0072B2',
                'marker': 's'
            },
            'explore': {
                'label': 'Explore',
                'color': '#009E73',
                'marker': '^'
            },
            'balanced': {
                'label': 'Balanced',
                'color': '#E69F00',
                'marker': 'D'
            },
            'target_ei': {
                'label': 'Target EI',
                'color': '#CC79A7',
                'marker': 'P'
            },
            'other': {
                'label': 'Other optimizer mode',
                'color': '#56B4E9',
                'marker': 'v'
            }
        }
        style_order = [
            'seed',
            'exploit',
            'explore',
            'balanced',
            'target_ei',
            'other'
        ]

        def _get_style_key(row):
            condition_type = str(row.get('condition_type', '')).strip().lower()

            if condition_type == 'seed':
                return 'seed'

            acquisition_mode = row.get('acquisition_mode')

            if acquisition_mode is None or pd.isna(acquisition_mode):
                return 'other'

            acquisition_mode = str(acquisition_mode).strip().lower()

            if acquisition_mode in mode_styles:
                return acquisition_mode

            return 'other'

        performance_df['_portfolio_style_key'] = performance_df.apply(
            _get_style_key,
            axis=1
        )

        active_style_keys = [
            style_key
            for style_key in style_order
            if style_key in set(performance_df['_portfolio_style_key'])
        ]

        target_lambda = float(
            performance_df['target_lambda_max_nm'].iloc[0]
        )
        font_sizes = self._get_auto_lambda_plot_font_sizes()
        fig, (summary_ax, replicate_ax) = plt.subplots(
            2,
            1,
            figsize=(8.4, 7.3),
            dpi=300,
            sharex=True
        )
        fig.set_tight_layout(False)

        target_handle = summary_ax.axhline(
            target_lambda,
            color='0.25',
            linestyle='--',
            linewidth=1.1,
            alpha=0.8,
            zorder=1
        )
        replicate_ax.axhline(
            target_lambda,
            color='0.25',
            linestyle='--',
            linewidth=1.1,
            alpha=0.8,
            zorder=1
        )

        for style_key in active_style_keys:
            style = mode_styles[style_key]
            mode_df = performance_df[
                performance_df['_portfolio_style_key'] == style_key
            ].copy()
            x_values = mode_df['reaction_number'].to_numpy(dtype=float)
            actual_means = pd.to_numeric(
                mode_df['actual_lambda_mean_nm'],
                errors='coerce'
            ).to_numpy(dtype=float)
            actual_sems = pd.to_numeric(
                mode_df['actual_lambda_sem_nm'],
                errors='coerce'
            ).fillna(0.0).to_numpy(dtype=float)
            actual_yerr, _ = self._clip_errorbars_to_lambda_display_window(
                means=actual_means,
                errors=actual_sems,
                condition_numbers=x_values,
                y_min_nm=y_display_min_nm,
                y_max_nm=y_display_max_nm
            )

            summary_ax.errorbar(
                x_values,
                actual_means,
                yerr=actual_yerr,
                fmt=style['marker'],
                color=style['color'],
                ecolor=style['color'],
                markerfacecolor=style['color'],
                markeredgecolor=style['color'],
                markeredgewidth=1.0,
                elinewidth=1.0,
                capsize=4,
                markersize=5.0,
                barsabove=True,
                alpha=0.92,
                zorder=4
            )

            prediction_df = mode_df[
                pd.to_numeric(
                    mode_df['predicted_lambda_mean_nm'],
                    errors='coerce'
                ).notna()
            ].copy()

            if not prediction_df.empty:
                prediction_x = prediction_df['reaction_number'].to_numpy(
                    dtype=float
                )
                prediction_means = pd.to_numeric(
                    prediction_df['predicted_lambda_mean_nm'],
                    errors='coerce'
                ).to_numpy(dtype=float)
                prediction_stds = pd.to_numeric(
                    prediction_df['predicted_lambda_std_nm'],
                    errors='coerce'
                ).fillna(0.0).to_numpy(dtype=float)
                prediction_yerr, _ = (
                    self._clip_errorbars_to_lambda_display_window(
                        means=prediction_means,
                        errors=prediction_stds,
                        condition_numbers=prediction_x,
                        y_min_nm=y_display_min_nm,
                        y_max_nm=y_display_max_nm
                    )
                )

                summary_ax.errorbar(
                    prediction_x,
                    prediction_means,
                    yerr=prediction_yerr,
                    fmt=style['marker'],
                    color=style['color'],
                    ecolor=style['color'],
                    markerfacecolor='none',
                    markeredgecolor=style['color'],
                    markeredgewidth=1.2,
                    elinewidth=1.0,
                    capsize=4,
                    markersize=5.8,
                    barsabove=True,
                    alpha=0.78,
                    zorder=3
                )

        replicate_value_columns = sorted(
            [
                column
                for column in performance_df.columns
                if (
                    column.startswith('actual_lambda_rep_')
                    and column.endswith('_nm')
                )
            ],
            key=lambda column: int(
                column.replace('actual_lambda_rep_', '').replace('_nm', '')
            )
        )

        has_qc_exclusion = False

        for _, row in performance_df.iterrows():
            style = mode_styles[row['_portfolio_style_key']]
            reaction_number = float(row['reaction_number'])

            for value_column in replicate_value_columns:
                replicate_value = row.get(value_column)

                if replicate_value is None or pd.isna(replicate_value):
                    continue

                replicate_number = int(
                    value_column.replace('actual_lambda_rep_', '').replace(
                        '_nm',
                        ''
                    )
                )
                qc_column = (
                    f'actual_lambda_rep_{replicate_number}_included_in_qc'
                )
                qc_value = row.get(qc_column, True)

                if isinstance(qc_value, str):
                    included_in_qc = qc_value.strip().lower() in [
                        'true',
                        '1',
                        'yes',
                        'y'
                    ]
                elif qc_value is None or pd.isna(qc_value):
                    included_in_qc = True
                else:
                    included_in_qc = bool(qc_value)

                if included_in_qc:
                    replicate_ax.scatter(
                        [reaction_number],
                        [float(replicate_value)],
                        s=30,
                        marker=style['marker'],
                        facecolors=style['color'],
                        edgecolors=style['color'],
                        linewidths=0.9,
                        alpha=0.9,
                        zorder=4
                    )
                else:
                    has_qc_exclusion = True
                    replicate_ax.scatter(
                        [reaction_number],
                        [float(replicate_value)],
                        s=38,
                        marker=style['marker'],
                        facecolors='none',
                        edgecolors='#D55E00',
                        linewidths=1.3,
                        alpha=0.95,
                        zorder=5
                    )

        summary_ax.set_ylabel(
            r'Condition mean $\lambda_{\max}$ (nm)',
            fontsize=font_sizes['axis_label']
        )
        replicate_ax.set_ylabel(
            r'Replicate $\lambda_{\max}$ (nm)',
            fontsize=font_sizes['axis_label']
        )
        replicate_ax.set_xlabel(
            'Reaction condition number',
            fontsize=font_sizes['axis_label'],
            labelpad=5
        )

        for axis in (summary_ax, replicate_ax):
            axis.set_ylim(y_display_min_nm, y_display_max_nm)
            self._apply_auto_lambda_plot_lab_frame_style(axis)

        integer_ticks = performance_df['reaction_number'].astype(int).to_list()
        replicate_ax.set_xticks(integer_ticks)

        if plot_title is None:
            plot_title = (
                f'Auto Acquisition Portfolio Trace After Batch {batch_number}'
            )

        fig.suptitle(
            plot_title,
            fontsize=font_sizes['title'],
            fontweight='normal',
            y=0.985
        )

        mode_handles = [
            Line2D(
                [0],
                [0],
                marker=mode_styles[style_key]['marker'],
                color=mode_styles[style_key]['color'],
                markerfacecolor=mode_styles[style_key]['color'],
                markersize=6,
                linewidth=0,
                label=mode_styles[style_key]['label']
            )
            for style_key in active_style_keys
        ]
        semantic_handles = [
            Line2D(
                [0],
                [0],
                marker='o',
                color='0.20',
                markerfacecolor='0.20',
                markersize=5,
                linewidth=0,
                label='Filled: observed / QC-included'
            ),
            Line2D(
                [0],
                [0],
                marker='o',
                color='0.20',
                markerfacecolor='none',
                markersize=5,
                linewidth=0,
                label='Hollow: GP prediction'
            )
        ]

        if has_qc_exclusion:
            semantic_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    color='#D55E00',
                    markerfacecolor='none',
                    markersize=5,
                    linewidth=0,
                    label='Red outline: QC-excluded'
                )
            )

        target_legend_handle = Line2D(
            [0],
            [0],
            color=target_handle.get_color(),
            linestyle='--',
            linewidth=1.1,
            label=f'Target = {target_lambda:.0f} nm'
        )
        legend_handles = mode_handles + semantic_handles + [
            target_legend_handle
        ]

        fig.legend(
            legend_handles,
            [handle.get_label() for handle in legend_handles],
            loc='upper center',
            bbox_to_anchor=(0.5, 0.955),
            ncol=4,
            frameon=False,
            fontsize=font_sizes['legend'],
            handlelength=1.1,
            handletextpad=0.4,
            columnspacing=0.8
        )
        fig.text(
            0.5,
            0.02,
            'Mode markers identify the selection rule; all modes share one '
            'pre-batch GP and jointly update it after QC.',
            ha='center',
            va='center',
            fontsize=font_sizes['footer'],
            color='0.35'
        )
        fig.subplots_adjust(
            left=0.16,
            right=0.97,
            bottom=0.10,
            top=0.80,
            hspace=0.18
        )

        if plot_filename is None:
            plot_filename = (
                f'acquisition_portfolio_trace_after_batch_{batch_number}.png'
            )

        plot_path = os.path.join(self.plot_path, plot_filename)
        fig.savefig(plot_path)
        plt.close(fig)
        print(
            "<<controller>> saved acquisition portfolio trace to "
            f"{plot_path}"
        )

        return plot_path
    
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

    def Normalize_Denormalize_Recipes(self, X, normalize_flag=True):
        '''
        Handles normalization and denormalization of different reagents on the deck
        Input: A set of recipes in a 2D numpy array
        Output: A set of normalize/denormalized (converted based on flag) recipes in a 2D numpy array  
        '''
        
        # Normalization is the conversion of concentration to 0-1 range
        if normalize_flag:
            operation = 'divide'
        
        # Denormalization is the conversion of 0-1 range to concentrations
        else:
            operation = 'multiply'
        
        
        # Fore every recipe (set of reagent concentrations) go through each reagent and normalize/denormalize specific min and max concentrations
        for row in X:
            for i in range(len(row)):
                if operation == 'multiply':
                    row[i] = (self.max_conc[i]-self.min_conc[i]) * row[i] + self.min_conc[i]
                else:
                    row[i] = (row[i]-self.min_conc[i]) / (self.max_conc[i]-self.min_conc[i])
        
        return X

    def _build_auto_optimizer_selection_metadata(
        self,
        model,
        selected_normalized_recipes,
        selected_physical_recipes,
        executed_normalized_recipes,
        executed_physical_recipes,
        optimizer_recipe_repaired,
        repair_max_transfer_delta_uL,
        selected_volume_balances,
        executed_volume_balances
    ):
        '''
        Builds immutable selection/execution provenance for one Auto batch.

        The GP prediction and acquisition values are captured on the optimizer
        before any experiment is run. Recipe arrays and volume balances are
        copied into plain Python structures so later model updates or in-place
        normalization cannot alter the selection-time record.
        '''
        acquisition_mode = getattr(
            model,
            'last_optimizer_acquisition_mode',
            model.acquisition_mode
        )

        balanced_exploration_weight = None
        if acquisition_mode == 'balanced':
            balanced_exploration_weight = getattr(
                model,
                'last_optimizer_balanced_exploration_weight',
                getattr(model, 'balanced_exploration_weight', None)
            )

        mask_results = getattr(model, 'last_mask_results', None)

        if mask_results is None:
            mask_result_count = 0
            feasible_mask_result_count = 0
        else:
            mask_result_count = len(mask_results)
            feasible_mask_result_count = sum(
                1
                for result in mask_results
                if (
                    result.get('x_full') is not None
                    and result.get('volume_balance') is not None
                    and result['volume_balance'].get(
                        'volume_feasible',
                        False
                    )
                )
            )

        selected_mask = getattr(model, 'last_selected_mask', None)
        selected_mask_for_audit = (
            None
            if selected_mask is None
            else np.asarray(
                selected_mask
            ).astype(int).reshape(-1).tolist()
        )

        return {
            'acquisition_mode': acquisition_mode,
            'acquisition_score': getattr(
                model,
                'last_optimizer_acquisition_score',
                None
            ),
            'balanced_exploration_weight': (
                balanced_exploration_weight
            ),
            'selected_mask': selected_mask_for_audit,
            'optimizer_method': getattr(
                model,
                'last_optimizer_method',
                None
            ),
            'optimizer_success': getattr(
                model,
                'last_optimizer_success',
                None
            ),
            'optimizer_status': getattr(
                model,
                'last_optimizer_status',
                None
            ),
            'optimizer_message': getattr(
                model,
                'last_optimizer_message',
                None
            ),
            'predicted_target_error_nm': getattr(
                model,
                'last_optimizer_predicted_target_error_nm',
                None
            ),
            'predicted_lambda_mean_nm': getattr(
                model,
                'last_optimizer_predicted_lambda_mean_nm',
                getattr(model, 'last_optimizer_predicted_lambda_max', None)
            ),
            'predicted_lambda_std_nm': getattr(
                model,
                'last_optimizer_predicted_lambda_std_nm',
                None
            ),
            'incumbent_target_error_nm': getattr(
                model,
                'last_optimizer_incumbent_target_error_nm',
                None
            ),
            'selected_normalized_recipe': np.asarray(
                selected_normalized_recipes,
                dtype=float
            ).tolist(),
            'executed_normalized_recipe': np.asarray(
                executed_normalized_recipes,
                dtype=float
            ).tolist(),
            'selected_physical_recipe': np.asarray(
                selected_physical_recipes,
                dtype=float
            ).tolist(),
            'executed_physical_recipe': np.asarray(
                executed_physical_recipes,
                dtype=float
            ).tolist(),
            'optimizer_recipe_repaired': bool(
                optimizer_recipe_repaired
            ),
            'optimizer_recipe_repair_max_transfer_delta_uL': float(
                repair_max_transfer_delta_uL
            ),
            'optimizer_volume_balance': dict(
                copy.deepcopy(
                    getattr(
                        model,
                        'last_optimizer_volume_balance',
                        {}
                    ) or {}
                )
            ),
            'selected_controller_volume_balances': [
                copy.deepcopy(volume_balance)
                for volume_balance in selected_volume_balances
            ],
            'executed_controller_volume_balances': [
                copy.deepcopy(volume_balance)
                for volume_balance in executed_volume_balances
            ],
            'mask_results': copy.deepcopy(list(mask_results or [])),
            'mask_result_count': int(mask_result_count),
            'feasible_mask_result_count': int(
                feasible_mask_result_count
            ),
            'notes': (
                'Optimizer-selected recipe; acquisition prediction and '
                'recipe provenance captured before experiment execution.'
            )
        }

    def _prepare_auto_optimizer_recipe_for_execution(
        self,
        model,
        normalized_recipes,
        batch_label,
        additional_selection_metadata=None
    ):
        '''
        Converts and validates an optimizer proposal before robot preparation.

        Optimizer-selected recipes must already satisfy every executable
        transfer invariant enforced by the controller. The controller still
        applies its true-zero rule as an independent verification, but any
        resulting transfer change is treated as an optimizer/controller
        contract violation and stops the run before wells or robot commands are
        created. Initial maximin seed repair remains intentionally unchanged.

        A recipe-design CSV is exported before a repair mismatch raises so the
        failed proposal and controller-prepared counterpart remain available
        for diagnosis.

        returns:
            tuple(np.ndarray, dict):
                Executable physical-space recipes and immutable selection
                metadata when the optimizer proposal passes unchanged.
        '''
        # Source-level unit tests and downstream audit utilities may call this
        # helper without a fully initialized Controller. Presentation defaults
        # to standard in that case, while recipe safety remains unchanged.
        terminal_verbosity = getattr(
            self,
            'robo_params',
            {}
        ).get(
            'auto_terminal_verbosity',
            'standard'
        )

        selected_normalized_recipes = np.array(
            normalized_recipes,
            dtype=float,
            copy=True
        )

        if selected_normalized_recipes.ndim == 1:
            selected_normalized_recipes = (
                selected_normalized_recipes.reshape(1, -1)
            )

        if (
            selected_normalized_recipes.ndim != 2
            or selected_normalized_recipes.shape[1]
            != len(self.variable_reagents)
        ):
            raise ValueError(
                "Optimizer-selected normalized recipes must be a "
                "two-dimensional array with one column per variable reagent. "
                f"Received shape {selected_normalized_recipes.shape}."
            )

        if selected_normalized_recipes.shape[0] != 1:
            raise ValueError(
                "Auto optimizer selection must contain exactly one recipe per "
                "iteration because the active OptimizationModel/controller "
                "contract uses batch_size=1. Received "
                f"{selected_normalized_recipes.shape[0]} recipes."
            )

        if not np.all(np.isfinite(selected_normalized_recipes)):
            raise ValueError(
                "Optimizer-selected normalized recipes contain non-finite "
                "values and will not be executed."
            )

        normalized_bound_tolerance = 1e-9
        if (
            np.any(
                selected_normalized_recipes
                < -normalized_bound_tolerance
            )
            or np.any(
                selected_normalized_recipes
                > 1.0 + normalized_bound_tolerance
            )
        ):
            raise ValueError(
                "Optimizer-selected normalized recipes contain values outside "
                "the allowed [0, 1] model space and will not be executed. "
                f"Received: {selected_normalized_recipes}."
            )

        # Normalize_Denormalize_Recipes mutates its argument. Always pass a
        # copy so the selection-time normalized proposal remains immutable.
        selected_physical_recipes = self.Normalize_Denormalize_Recipes(
            selected_normalized_recipes.copy(),
            normalize_flag=False
        )

        executed_physical_recipes = (
            self._apply_true_zero_transfer_rule_to_recipes(
                selected_physical_recipes.copy()
            )
        )

        executed_normalized_recipes = self.Normalize_Denormalize_Recipes(
            executed_physical_recipes.copy(),
            normalize_flag=True
        )

        selected_volume_balances = [
            self._get_auto_recipe_volume_balance(recipe)
            for recipe in selected_physical_recipes
        ]
        executed_volume_balances = [
            self._get_auto_recipe_volume_balance(recipe)
            for recipe in executed_physical_recipes
        ]

        transfer_deltas_uL = []

        for selected_balance, executed_balance in zip(
            selected_volume_balances,
            executed_volume_balances
        ):
            for reagent_name in self.variable_reagents:
                transfer_deltas_uL.append(
                    abs(
                        float(
                            selected_balance['variable_transfer_volumes'][
                                reagent_name
                            ]
                        )
                        - float(
                            executed_balance['variable_transfer_volumes'][
                                reagent_name
                            ]
                        )
                    )
                )

        repair_max_transfer_delta_uL = max(
            transfer_deltas_uL,
            default=0.0
        )
        optimizer_recipe_repaired = (
            repair_max_transfer_delta_uL > 1e-9
        )

        metadata = self._build_auto_optimizer_selection_metadata(
            model=model,
            selected_normalized_recipes=selected_normalized_recipes,
            selected_physical_recipes=selected_physical_recipes,
            executed_normalized_recipes=executed_normalized_recipes,
            executed_physical_recipes=executed_physical_recipes,
            optimizer_recipe_repaired=optimizer_recipe_repaired,
            repair_max_transfer_delta_uL=(
                repair_max_transfer_delta_uL
            ),
            selected_volume_balances=selected_volume_balances,
            executed_volume_balances=executed_volume_balances
        )

        if additional_selection_metadata is not None:
            metadata.update(
                copy.deepcopy(additional_selection_metadata)
            )

        # Retain the controller-side audit on the model even when execution is
        # blocked, which makes the failure inspectable in terminal/debug tests.
        model.last_controller_selection_metadata = copy.deepcopy(metadata)

        # terminal_output.txt is the first artifact reviewed during a
        # human-supervised dry debug. The complete selected-versus-prepared
        # provenance is valuable for diagnostic runs, while standard terminal
        # output remains concise and directs the user to the persistent CSV
        # and report artifacts for the same immutable audit record.
        terminal_provenance = {
            'acquisition_mode': metadata['acquisition_mode'],
            'selected_mask': metadata['selected_mask'],
            'selected_normalized_recipe': (
                metadata['selected_normalized_recipe']
            ),
            'executed_normalized_recipe': (
                metadata['executed_normalized_recipe']
            ),
            'selected_physical_recipe': (
                metadata['selected_physical_recipe']
            ),
            'executed_physical_recipe': (
                metadata['executed_physical_recipe']
            ),
            'optimizer_recipe_repaired': (
                metadata['optimizer_recipe_repaired']
            ),
            'optimizer_recipe_repair_max_transfer_delta_uL': (
                metadata[
                    'optimizer_recipe_repair_max_transfer_delta_uL'
                ]
            ),
            'selected_controller_volume_balances': (
                metadata['selected_controller_volume_balances']
            ),
            'executed_controller_volume_balances': (
                metadata['executed_controller_volume_balances']
            )
        }
        if terminal_verbosity == 'diagnostic':
            print(
                "<<controller diagnostic>> optimizer selection/execution "
                "provenance: "
                + self._serialize_auto_audit_value(terminal_provenance)
            )

        self._export_auto_batch_recipe_design(
            repaired_recipes=executed_physical_recipes,
            original_recipes=selected_physical_recipes,
            batch_label=batch_label,
            selection_metadata=metadata
        )

        if optimizer_recipe_repaired:
            print(
                "<<controller error>> optimizer-selected physical recipe: "
                f"{selected_physical_recipes}"
            )
            print(
                "<<controller error>> controller-prepared physical recipe: "
                f"{executed_physical_recipes}"
            )
            raise RuntimeError(
                "Optimizer/controller executable-recipe invariant failed: "
                "the controller true-zero rule changed an optimizer-selected "
                "transfer by as much as "
                f"{repair_max_transfer_delta_uL:.12g} uL. The batch was "
                "stopped before wells or robot commands were created. Review "
                f"the recipe-design audit for {batch_label}."
            )

        self._validate_auto_recipe_volume_feasibility(
            executed_physical_recipes,
            context_label=f"model-suggested {batch_label}"
        )

        if terminal_verbosity != 'essential':
            print(
                "<<controller>> optimizer/controller recipe invariant "
                "passed; no transfer repair was required"
            )

        return executed_physical_recipes, metadata

    def _prepare_auto_portfolio_recipes_for_execution(
        self,
        model,
        selection_records,
        batch_label
    ):
        '''
        Validates an ordered portfolio through the established single-recipe
        controller contract, then combines its executable conditions.

        Each record is restored onto the model only long enough to reuse the
        pre-execution provenance, exact-zero, and volume validation path. The
        model is not fitted or otherwise updated here, so every selection
        remains tied to the same pre-batch GP snapshot.
        '''
        selection_records = list(selection_records)

        if len(selection_records) == 0:
            raise ValueError(
                "Cannot prepare an empty acquisition portfolio."
            )

        original_mode = model.acquisition_mode
        prepared_recipes = []
        metadata_records = []

        try:
            for selection_record in selection_records:
                acquisition_mode = selection_record['acquisition_mode']
                model.acquisition_mode = acquisition_mode

                # _prepare_auto_optimizer_recipe_for_execution intentionally
                # consumes the established last_optimizer_* audit interface.
                # Restore a per-mode immutable snapshot instead of allowing
                # the final portfolio mode to overwrite earlier provenance.
                model.last_optimizer_acquisition_mode = acquisition_mode
                model.last_optimizer_acquisition_score = selection_record.get(
                    'acquisition_score'
                )
                model.last_optimizer_balanced_exploration_weight = (
                    selection_record.get('balanced_exploration_weight')
                )
                model.last_selected_mask = selection_record.get(
                    'selected_mask'
                )
                model.last_optimizer_method = selection_record.get(
                    'optimizer_method'
                )
                model.last_optimizer_success = selection_record.get(
                    'optimizer_success'
                )
                model.last_optimizer_status = selection_record.get(
                    'optimizer_status'
                )
                model.last_optimizer_message = selection_record.get(
                    'optimizer_message'
                )
                model.last_optimizer_predicted_target_error_nm = (
                    selection_record.get('predicted_target_error_nm')
                )
                model.last_optimizer_predicted_lambda_mean_nm = (
                    selection_record.get('predicted_lambda_mean_nm')
                )
                model.last_optimizer_predicted_lambda_std_nm = (
                    selection_record.get('predicted_lambda_std_nm')
                )
                model.last_optimizer_incumbent_target_error_nm = (
                    selection_record.get('incumbent_target_error_nm')
                )
                model.last_optimizer_volume_balance = copy.deepcopy(
                    selection_record.get('optimizer_volume_balance')
                )
                model.last_mask_results = copy.deepcopy(
                    selection_record.get('mask_results', [])
                )

                record_label = (
                    f"{batch_label}_{selection_record['portfolio_selection_index']}"
                    f"_{acquisition_mode}"
                )
                prepared_recipe, metadata = (
                    self._prepare_auto_optimizer_recipe_for_execution(
                        model=model,
                        normalized_recipes=np.asarray(
                            selection_record['normalized_recipe'],
                            dtype=float
                        ).reshape(1, -1),
                        batch_label=record_label,
                        additional_selection_metadata={
                            'portfolio_selection_index': (
                                selection_record[
                                    'portfolio_selection_index'
                                ]
                            ),
                            'portfolio_acquisition_modes': list(
                                selection_record[
                                    'portfolio_acquisition_modes'
                                ]
                            ),
                            'portfolio_min_distance': selection_record[
                                'portfolio_min_distance'
                            ],
                            'portfolio_nearest_distance': (
                                selection_record.get(
                                    'portfolio_nearest_distance'
                                )
                            )
                        }
                    )
                )
                prepared_recipes.append(prepared_recipe[0])
                metadata_records.append(metadata)
        finally:
            model.acquisition_mode = original_mode

        model.last_controller_selection_metadata = copy.deepcopy(
            metadata_records
        )

        return np.asarray(prepared_recipes, dtype=float), metadata_records

    @terminal_output_capture_guard
    @error_exit
    def _run(self, port, simulate, model, no_pr):
        '''
        private function to run
        '''
        
        # Helper functions for normalization and denormalization
        def normalize(x, min_val, max_val):
           return (x - min_val) / (max_val - min_val)

        def denormalize(x_normalized, min_val, max_val):
            return x_normalized * (max_val - min_val) + min_val
        
        self.batch_num = 0 #used internally for unique filenames

        self.last_auto_source_volume_audit = []

        self.well_count = 0 #used internally for unique wellnames
        self.create_connection(simulate, no_pr, port)
        # Begin optimization
        print('<<controller>> executing batch {}'.format(self.batch_num))

        # Generate initial data which is a list of recipes (normalized)
        print("<<controller>> generating maximin Latin hypercube initial design")
        X_initial = model.generate_initial_design()
        if (
            self.robo_params.get('auto_terminal_verbosity', 'standard')
            == 'diagnostic'
        ):
            print(f"<<controller diagnostic>> normalized seed design: {X_initial}")

        # The list of recipes is denormalized with different maximums for each reagent
        X_Initial_Denormalized = self.Normalize_Denormalize_Recipes(X_initial, normalize_flag=False)

        # Keep a copy of the original denormalized design so the debug export
        # can show exactly what true-zero repair changed.
        X_Initial_Denormalized_before_repair = X_Initial_Denormalized.copy()

        # Apply Auto true-zero transfer behavior before duplicating/running
        # recipes. This ensures the model is later trained on the same
        # physically executable recipes that the robot actually ran.
        X_Initial_Denormalized = self._apply_true_zero_transfer_rule_to_recipes(
            X_Initial_Denormalized
        )

        self._validate_auto_recipe_volume_feasibility(
            X_Initial_Denormalized,
            context_label=f"initial seed batch {self.batch_num}"
        )

        self._export_auto_batch_recipe_design(
            repaired_recipes=X_Initial_Denormalized,
            original_recipes=X_Initial_Denormalized_before_repair,
            batch_label=f"batch_{self.batch_num}"
        )

        if (
            self.robo_params.get('auto_terminal_verbosity', 'standard')
            == 'diagnostic'
        ):
            print(
                "<<controller diagnostic>> physical seed design: "
                f"{X_Initial_Denormalized}"
            )

        # Duplicate each unique recipe according to the Header num_duplicates setting.
        recipes = self.duplicate_list_elements(X_Initial_Denormalized, self.num_duplicates)

        print(f"<<controller>> preparing {recipes.shape[0]} recipe wells with {recipes.shape[1]} variable reagents")

        # Generate wellnames for this batch
        wellnames = [self._generate_wellname() for i in range(recipes.shape[0])]
        
        # Plan and execute a reaction according to the Header num_duplicates setting.
        self._create_samples(wellnames, recipes, model)

        # Pull in the scan data
        filenames = self.rxn_df[
                (self.rxn_df['op'] == 'scan') |
                (self.rxn_df['op'] == 'scan_until_complete')
                ].reset_index()

        last_filename = filenames.loc[filenames['index'].idxmax(),'scan_filename']
        scan_data = self._get_sample_data(wellnames, last_filename)

        # Helper function that extracts lambda maxes from scan data
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

        # Lambda maxes are Y_intial
        Y_initial = find_max(scan_data)
        if (
            self.robo_params.get('auto_terminal_verbosity', 'standard')
            == 'diagnostic'
        ):
            print(f"<<controller diagnostic>> seed lambda maxima: {Y_initial}")

        self._append_auto_model_performance_rows(
            unique_recipes=X_Initial_Denormalized,
            lambda_max_values=Y_initial,
            condition_type='seed',
            batch_number=self.batch_num,
            prediction_metadata={
                'notes': (
                    'Initial seed design; no pre-experiment GP prediction '
                    'available.'
                )
            }
        )
        
        self._generate_auto_plot_suite(
            stage='after_measurement',
            model=model,
            batch_number=self.batch_num
        )

        # Build QC-filtered seed data for GP model training. Raw replicate
        # results remain preserved in experiment_data.csv and in the Auto
        # performance log, but excluded replicate outliers are not used as model
        # knowledge.
        qc_initial_recipes, qc_initial_lambda_values = (
            self._build_auto_qc_model_training_data(
                X_Initial_Denormalized,
                Y_initial
            )
        )

        # Normalize the QC-filtered lambda maxes to pass to the GP model.
        qc_Y_initial_normalized = normalize(
            np.array(qc_initial_lambda_values),
            300,
            900
        ).reshape(-1, 1)

        # Normalize the QC-filtered recipe concentrations to pass to the GP
        # model. Use a copy because Normalize_Denormalize_Recipes mutates its
        # input.
        qc_X_initial_normalized = self.Normalize_Denormalize_Recipes(
            qc_initial_recipes.copy(),
            normalize_flag=True
        )

        # Create the model with QC-filtered initial data.
        model.initialize_optimizer(
            qc_X_initial_normalized,
            qc_Y_initial_normalized
        )

        # Synchronize target EI only after the seed observations have been
        # successfully incorporated into the fitted GP. The incumbent comes
        # from the same QC-approved condition-level performance history.
        self._synchronize_target_ei_incumbent_from_performance(model)

        # The initial seed observations are now incorporated into the fitted
        # GP. Generate scientifically current model plots for batch 0 when the
        # selected Auto plot profile requests per-batch outputs.
        self._generate_auto_plot_suite(
            stage='after_model_update',
            model=model,
            batch_number=self.batch_num
        )

        # Evaluate whether the initial seed batch already contains a validated
        # condition-level target hit. This uses duplicate-aggregated lambda max
        # statistics rather than any single physical replicate well.
        self._update_auto_quit_from_condition_level_performance(
            model,
            self.batch_num
        )

        if (
            self.robo_params.get('auto_terminal_verbosity', 'standard')
            == 'diagnostic'
        ):
            print(f"<<controller diagnostic>> cumulative model X: {model.optimizer.X}")
            print(f"<<controller diagnostic>> cumulative model Y: {model.optimizer.Y}")

        # Update data on the controller side (this function updates the df that is exported to pr_data called self.experiment_data)
        self._update_experiment_data(recipes, Y_initial)

        # Intial data is considered the zeroith batch 
        self.batch_num += 1

        # Enter iterative while loop now until max_iters is hit or close to the target
        while not model.quit:

            # Portfolio members are selected from the same fitted GP snapshot.
            # The controller deliberately updates the GP only after every
            # member has been measured and QC-filtered below.
            acquisition_modes = getattr(
                model,
                'acquisition_modes',
                [model.acquisition_mode]
            )

            print("<<controller>> selecting next reaction from updated model")

            if len(acquisition_modes) == 1:
                X_new = model.getNextReaction()
                if (
                    self.robo_params.get(
                        'auto_terminal_verbosity',
                        'standard'
                    )
                    == 'diagnostic'
                ):
                    print(
                        f'<<controller diagnostic>> normalized proposal for '
                        f'batch {self.batch_num}: {X_new}'
                    )

                (
                    X_new_Denormalized,
                    optimizer_prediction_metadata
                ) = self._prepare_auto_optimizer_recipe_for_execution(
                    model=model,
                    normalized_recipes=X_new,
                    batch_label=f"batch_{self.batch_num}"
                )
            else:
                selection_records = model.getNextPortfolio(
                    acquisition_modes=acquisition_modes,
                    portfolio_min_distance=getattr(
                        model,
                        'portfolio_min_distance',
                        0.05
                    )
                )
                print(
                    f'<<controller>> executing portfolio batch '
                    f'{self.batch_num}: ' + ';'.join(acquisition_modes)
                )

                (
                    X_new_Denormalized,
                    optimizer_prediction_metadata
                ) = self._prepare_auto_portfolio_recipes_for_execution(
                    model=model,
                    selection_records=selection_records,
                    batch_label=f"batch_{self.batch_num}"
                )
            
            # Duplicate the repaired recipe to create replicate wells.
            recipes = self.duplicate_list_elements(X_new_Denormalized, self.num_duplicates)
            
            print(f"<<controller>> preparing {recipes.shape[0]} recipe wells with {recipes.shape[1]} variable reagents")
            
            # Run the experiments
            wellnames = [self._generate_wellname() for i in range(recipes.shape[0])]
            self._create_samples(wellnames, recipes, model)
            
            # Pull in the scan data
            filenames = self.rxn_df[
                    (self.rxn_df['op'] == 'scan') |
                    (self.rxn_df['op'] == 'scan_until_complete')
                    ].reset_index()
            last_filename = filenames.loc[filenames['index'].idxmax(),'scan_filename']
            scan_data = self._get_sample_data(wellnames, last_filename) 
            
            # Y_new is lambda maxes from the new recipe
            Y_new = find_max(scan_data)
            if (
                self.robo_params.get(
                    'auto_terminal_verbosity',
                    'standard'
                )
                == 'diagnostic'
            ):
                print(
                    f"<<controller diagnostic>> batch {self.batch_num} "
                    f"lambda maxima: {Y_new}"
                )

            self._append_auto_model_performance_rows(
                unique_recipes=X_new_Denormalized,
                lambda_max_values=Y_new,
                condition_type='optimizer_selected',
                batch_number=self.batch_num,
                prediction_metadata=optimizer_prediction_metadata
            )

            self._generate_auto_plot_suite(
                stage='after_measurement',
                model=model,
                batch_number=self.batch_num
            )

            # Build QC-filtered optimizer-batch data for GP model training.
            # Raw replicate results remain preserved in experiment_data.csv and
            # in the Auto performance log, but excluded replicate outliers are
            # not used as model knowledge.
            qc_new_recipes, qc_new_lambda_values = (
                self._build_auto_qc_model_training_data(
                    X_new_Denormalized,
                    Y_new
                )
            )

            # Normalize the QC-filtered lambda maxes and recipes to pass to the
            # GP model. Use a copy because Normalize_Denormalize_Recipes mutates
            # its input.
            qc_Y_new_normalized = normalize(
                np.array(qc_new_lambda_values),
                300,
                900
            ).reshape(-1, 1)

            qc_X_new_normalized = self.Normalize_Denormalize_Recipes(
                qc_new_recipes.copy(),
                normalize_flag=True
            )

            # Update the model with only QC-included replicate observations.
            model.update_experiment_data(
                np.vstack((model.optimizer.X, qc_X_new_normalized)),
                np.vstack((model.optimizer.Y, qc_Y_new_normalized)),
                qc_X_new_normalized,
                qc_Y_new_normalized
            )

            # Keep target EI aligned with the newly fitted GP only after its
            # QC-filtered batch update succeeds.
            self._synchronize_target_ei_incumbent_from_performance(model)

            # The completed batch is now part of the fitted GP. The plot-suite
            # coordinator refreshes the 2D prediction and uncertainty grids
            # before saving heatmaps, so "After Batch N" truly includes Batch N.
            self._generate_auto_plot_suite(
                stage='after_model_update',
                model=model,
                batch_number=self.batch_num
            )

            # Override optimizer-side quit behavior with the scientifically
            # correct condition-level duplicate rule. This prevents Auto from
            # stopping just because one physical replicate randomly hits the
            # target while its duplicate does not.
            self._update_auto_quit_from_condition_level_performance(
                model,
                self.batch_num
            )

            # Add new denormalized data to the controller experiment_data
            self._update_experiment_data(recipes, Y_new, axis=0) 
            self.batch_num += 1    
            
        # Save the row-per-well experiment data used for raw output and model
        # audit. This remains separate from the condition-level Auto
        # performance log.
        self.experiment_data.to_csv(
            f'{os.path.join(self.out_path, "pr_data")}/experiment_data.csv',
            index=False
        )

        # Save the row-per-condition Auto performance log used for reporting,
        # plotting, and future notebook-ready summaries.
        self._export_auto_model_performance_log()

        # Generate the final output suite only after both core CSV exports are
        # complete. The coordinator applies the selected Auto plot profile,
        # generates dimension-appropriate plots, and writes the report last.
        self._generate_auto_plot_suite(
            stage='final',
            model=model,
            batch_number=self.batch_num - 1
        )

        print(
            "<<controller>> Auto run completed; condition-level results, "
            "recipe audits, and configured output artifacts were exported."
        )

        self.close_connection()
        self.pr.shutdown()

        if (
            self.robo_params.get(
                'auto_plot_profile',
                'standard'
            )
            != 'off'
        ):
            self._refresh_auto_run_status_section_in_report()

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

    def _create_samples(self, wellnames, recipes, model=None):
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
        # Retain the controller-side physical recipe check before allocating
        # product wells. The subsequent source-volume preflight uses the fully
        # resolved protocol dataframe, after the robot has reported its
        # mass-derived source inventory but before liquid handling begins.
        self._validate_auto_recipe_volume_feasibility(
            recipes,
            context_label=(
                f"Auto batch {getattr(self, 'batch_num', 'unknown')} "
                f"({len(wellnames)} physical wells)"
            )
        )

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
                print(f"<<controller>> building protocol dataframe for {len(wellnames)} wells")
                
                self.rxn_df = self._build_rxn_df(wellnames, recipes)
                self._insert_tot_vol_transfer()

                #print('trying to build df.')
                #print(f'products list:{self._products}')
                #print(f'rxn_df: {self.rxn_df}')

                if self.tot_vols:
                    # Ignore tiny negative floating-point artifacts, but still
                    # catch real negative water/top-off volumes.
                    if (self.rxn_df.loc[0,self._products] < -1e-9).any():
                        raise NotImplementedError('A product overflowed it\'s container using the most concentrated solutions on the deck. Future iterations will ask Mark to add a more concentrated solution')
                print("<<controller>> protocol dataframe built successfully")
                successful_build = True
            except ConversionError as e:
                # A legacy conversion recovery may create and execute a new
                # dilution protocol. That liquid would not be part of a
                # successfully constructed, source-preflighted batch, so
                # source-volume protection must not permit this unaccounted
                # physical path.
                # Legacy sheets keep their established recovery behavior.
                if (
                    self.robo_params.get(
                        'auto_source_volume_check',
                        'off'
                    )
                    == 'required'
                ):
                    raise ValueError(
                        "Auto source-volume protection stopped before an "
                        "unplanned dilution/recovery could consume liquid. "
                        "Correct the conversion problem or explicitly revise "
                        "the declared source-volume plan before retrying."
                    ) from e

                self._handle_conversion_err(e)

        self._preflight_auto_source_volumes(
            self.rxn_df,
            context_label=(
                f"Auto batch {getattr(self, 'batch_num', 'unknown')} "
                f"({len(wellnames)} physical wells)"
            )
        )

        self.execute_protocol_df(model)



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

    def _get_96_well_plate_order(self):
        '''
        Gets the well order used by the robot for a standard 96-well plate.

        The robot fills wells top-to-bottom within a column, then moves
        left-to-right across columns. For example:
            A1, B1, C1, ..., H1, A2, B2, ..., H12

        returns:
            list:
                Ordered list of 96-well plate positions.
        '''
        return [
            f"{row}{col}"
            for col in range(1, 13)
            for row in ["A", "B", "C", "D", "E", "F", "G", "H"]
        ]       
    
    def _count_available_96_well_plate_wells(self, starting_well):
        '''
        Counts how many wells are available on a 96-well plate from the selected
        starting well through the end of the plate.

        params:
            str starting_well:
                User-selected starting well, such as A1, A4, or E6.

        returns:
            int:
                Number of wells available from starting_well to H12, following
                the robot's top-to-bottom, left-to-right well order.
        '''
        plate_order = self._get_96_well_plate_order()
        starting_well = str(starting_well).strip().upper()

        if starting_well not in plate_order:
            raise ValueError(
                f"Starting well {starting_well} is not valid for a 96-well plate. "
                f"Expected a well like A1 through H12."
            )

        first_index = plate_order.index(starting_well)

        return len(plate_order) - first_index
    
    def _check_auto_well_capacity(self, model):
        '''
        Checks whether the Auto run could require more product wells than are
        available from the selected starting well on the 96-well plate.

        This check is intentionally conservative. The run may stop early if the
        model reaches the target, but the check assumes the maximum possible run:
            initial_data * num_duplicates
            + max_iterations * model.batch_size * num_duplicates

        params:
            OptimizationModel model:
                The Auto optimization model. Used for batch_size.

        Postconditions:
            - Prints the maximum number of wells the Auto run may require.
            - Prints the number of wells available from the selected starting well.
            - Raises an error before the run begins if there are not enough wells.
        '''
        initial_data = int(self.getModelInfo()["initial_data"])
        max_iterations = int(self.getModelInfo()["max_iterations"])
        batch_size = int(model.batch_size)

        initial_wells = initial_data * self.num_duplicates
        iteration_wells = max_iterations * batch_size * self.num_duplicates
        required_wells = initial_wells + iteration_wells

        starting_well = self.robo_params.get('platereader_input_first_usable')

        if not starting_well:
            raise ValueError(
                "Could not find the original plate-reader starting well for the Auto "
                "capacity check. Please confirm the deck_positions sheet specifies "
                "one first usable platereader well."
            )

        available_wells = self._count_available_96_well_plate_wells(starting_well)

        print("<<controller>> checking Auto well capacity")
        print(f"<<controller>> selected starting well: {starting_well}")
        print(
            f"<<controller>> Auto run may require up to {required_wells} wells "
            f"({initial_wells} initial + {iteration_wells} iterative)"
        )
        print(f"<<controller>> {available_wells} wells available from {starting_well} to H12")

        if required_wells > available_wells:
            raise Exception(
                f"Auto run requires up to {required_wells} wells, but only "
                f"{available_wells} wells are available from starting well {starting_well}. "
                f"Choose an earlier starting well, reduce initial_data, reduce max_iterations, "
                f"or reduce the number of replicates."
            )

        print("<<controller>> Auto well capacity check passed")

    def _count_available_pipette_tips_by_size(self):
        '''
        Counts pipette tips available from the configured deck positions.

        This uses the labware dataframe built from the deck_positions sheet.
        For each configured pipette tip rack, it counts from the rack's
        first_usable position through the end of the 96-position rack.

        returns:
            dict:
                Available pipette tips keyed by pipette size.
                Example:
                    {20.0: 96, 300.0: 192}
        '''
        labware_df = self.robo_params['labware_df']
        rack_order = self._get_96_well_plate_order()

        available_counts = {
            20.0: 0,
            300.0: 0
        }

        tip_rack_rows = labware_df.loc[
            labware_df['name'].astype(str).str.contains('tip_rack', case=False, na=False)
        ]

        for _, row in tip_rack_rows.iterrows():
            rack_name = str(row['name'])
            first_usable = str(row['first_usable']).strip().upper()

            if '20' in rack_name:
                pipette_size = 20.0
            elif '300' in rack_name:
                pipette_size = 300.0
            else:
                continue

            if first_usable == '' or first_usable.lower() == 'nan':
                raise ValueError(
                    f"Missing first_usable pipette tip position for {rack_name}. "
                    f"Please specify a starting pipette tip such as A1."
                )

            if first_usable not in rack_order:
                raise ValueError(
                    f"Starting pipette tip position {first_usable} for {rack_name} "
                    f"is not valid. Expected a position from A1 through H12."
                )

            first_index = rack_order.index(first_usable)
            available_counts[pipette_size] += len(rack_order) - first_index

        return available_counts
    
    def _count_non_water_reagent_groups_for_tip_estimate(self):
        '''
        Counts unique non-water reagent groups available on the deck.

        This is used for a conservative Auto pipette tip estimate. The estimate
        assumes each non-water reagent group may require pipette tip changes
        during each Auto batch.

        returns:
            int:
                Number of unique non-water reagent groups.
        '''
        reagent_df = self.robo_params['reagent_df']

        reagent_names = []

        for reagent_container_name in reagent_df.index:
            reagent_container_name = str(reagent_container_name)

            # Reagent containers are named like silver_nitrateC0.375.
            # Split at the concentration marker C to recover the base reagent name.
            if 'C' in reagent_container_name:
                base_name = reagent_container_name.split('C')[0]
            else:
                base_name = reagent_container_name

            if base_name != 'Water':
                reagent_names.append(base_name)

        return len(set(reagent_names))
    
    def _estimate_max_auto_pipette_tips_needed(self, model):
        '''
        Conservatively estimates the maximum pipette tips needed for the full
        Auto run.

        This estimate is intentionally conservative because future Auto recipes
        are not known before the run begins. It assumes each non-water reagent
        group may require both pipette sizes during each batch.

        The count includes startup pipette tips picked up during robot
        initialization.

        params:
            OptimizationModel model:
                The Auto optimization model. Used for max iteration structure.

        returns:
            dict:
                Estimated maximum pipette tips needed, keyed by pipette size.
        '''
        max_iterations = int(self.getModelInfo()["max_iterations"])

        # One initial seed batch plus up to max_iterations model-suggested batches.
        total_batches = 1 + max_iterations

        non_water_reagent_groups = self._count_non_water_reagent_groups_for_tip_estimate()

        # Conservative estimate:
        # For each pipette size, assume each non-water reagent group may require
        # one pipette tip per batch. This includes startup behavior in the total
        # upper-bound count and intentionally overestimates rather than risking
        # a false pass.
        estimated_counts = {
            20.0: max(1, total_batches * non_water_reagent_groups),
            300.0: max(1, total_batches * non_water_reagent_groups)
        }

        return estimated_counts
    
    def _check_auto_pipette_tip_capacity(self, model):
        '''
        Checks whether the configured deck positions provide enough pipette tips
        for the planned Auto run.

        This prevents the robot from starting a closed-loop Auto experiment that
        may later fail because the mapped pipette tip racks do not contain enough
        available pipette tips.

        Because the estimate is conservative, the user is allowed to override the
        warning and continue. However, overriding does not make unmapped pipette
        tip positions available to the robot.

        params:
            OptimizationModel model:
                The Auto optimization model.

        Postconditions:
            - Prints estimated pipette tips needed.
            - Prints configured pipette tips available.
            - Passes automatically if configured capacity is sufficient.
            - Warns and prompts the user if estimated use exceeds configured capacity.
            - Stops before the robot starts if the user does not accept the warning.
        '''
        needed_counts = self._estimate_max_auto_pipette_tips_needed(model)
        available_counts = self._count_available_pipette_tips_by_size()

        print("<<controller>> checking Auto pipette tip capacity")
        print("<<controller>> estimated maximum pipette tips needed:")
        print(f"<<controller>>   20 uL pipette tips: {needed_counts.get(20.0, 0)}")
        print(f"<<controller>>   300 uL pipette tips: {needed_counts.get(300.0, 0)}")
        print("<<controller>> configured pipette tips available from deck:")
        print(f"<<controller>>   20 uL pipette tips: {available_counts.get(20.0, 0)}")
        print(f"<<controller>>   300 uL pipette tips: {available_counts.get(300.0, 0)}")

        exceeded_capacity = False

        for pipette_size in [20.0, 300.0]:
            needed = needed_counts.get(pipette_size, 0)
            available = available_counts.get(pipette_size, 0)

            if needed > available:
                exceeded_capacity = True
                print(
                    f"<<controller>> WARNING: estimated {int(pipette_size)} uL "
                    f"pipette tip use exceeds configured capacity."
                )
                print(
                    f"<<controller>>   estimated needed: {needed} "
                    f"{int(pipette_size)} uL pipette tips"
                )
                print(
                    f"<<controller>>   configured available: {available} "
                    f"{int(pipette_size)} uL pipette tips"
                )

        if exceeded_capacity:
            print(
                "<<controller>> The robot may run out of mapped pipette tips during this run."
            )
            print(
                "<<controller>> Physically adding pipette tips mid-run may not help unless "
                "those positions are mapped and available to the robot."
            )
            print(
                "<<controller>> Consider adding another pipette tip rack to the deck sheet, "
                "choosing an earlier first usable pipette tip, reducing max_iterations, "
                "or reducing replicates."
            )

            confirm = input("Continue anyway? [yn] ").lower()

            if confirm != 'y':
                raise RuntimeError(
                    "Auto run stopped because configured pipette tip capacity was not confirmed."
                )

            print("<<controller>> Auto pipette tip capacity warning accepted; continuing")
        else:
            print("<<controller>> Auto pipette tip capacity check passed")

    def _apply_true_zero_transfer_rule_to_volume(self, volume):
        '''
        Applies the Auto true-zero transfer rule to one transfer volume.

        Auto mode should allow true 0 uL transfers, but nonzero transfers below
        the robot's reliable minimum should be mapped to an executable value.

        Rule:
            0 uL stays 0
            0 < volume < 2.5 uL maps to 0 uL
            2.5 <= volume < 5.0 uL maps to 5.0 uL
            volume >= 5.0 uL is unchanged

        A small tolerance is used at the 0, 2.5, and 5.0 uL boundaries so
        floating-point artifacts do not send mathematically equivalent values
        to the wrong side of a threshold.

        params:
            float volume:
                Transfer volume in uL.

        returns:
            float:
                Repaired transfer volume in uL.
        '''
        volume = float(volume)
        boundary_tol = 1e-9

        if math.isclose(volume, 0.0, rel_tol=0, abs_tol=boundary_tol):
            return 0.0

        # Values clearly below the midpoint round down to true zero.
        # The tolerance prevents values like 2.4999999999999996 from being
        # treated differently from 2.5 due only to floating-point artifacts.
        if volume < 2.5 - boundary_tol:
            return 0.0

        # Values from the midpoint up to just below the minimum transfer round
        # up to 5 uL. Values effectively equal to 5 uL are left unchanged below.
        if volume < 5.0 - boundary_tol:
            return 5.0

        return volume
    
    def _apply_true_zero_transfer_rule_to_recipes(self, recipes):
        '''
        Applies the Auto true-zero transfer rule to denormalized recipe
        concentrations.

        Recipes are stored as target concentrations, but the robot executes
        transfer volumes. This function converts each recipe concentration to
        its corresponding transfer volume, applies the true-zero transfer rule,
        and then converts the repaired volume back to concentration.

        This ensures the robot, model updates, and exported experiment data all
        use the same physically executable recipe.

        params:
            np.ndarray recipes:
                Denormalized recipe concentrations with shape:
                    n_recipes x n_variable_reagents

        returns:
            np.ndarray:
                Repaired denormalized recipe concentrations with the same shape
                as recipes.
        '''
        repaired_recipes = np.array(recipes, dtype=float, copy=True)

        for recipe_i in range(repaired_recipes.shape[0]):
            for reagent_i, reagent_name in enumerate(self.variable_reagents):
                target_conc = float(repaired_recipes[recipe_i, reagent_i])

                if math.isclose(target_conc, 0.0, rel_tol=0, abs_tol=1e-12):
                    repaired_recipes[recipe_i, reagent_i] = 0.0
                    continue

                stock_conc = self._get_variable_reagent_stock_conc(reagent_name)

                if math.isclose(stock_conc, 0.0, rel_tol=0, abs_tol=1e-12):
                    raise ValueError(
                        f"Cannot apply true-zero transfer rule for {reagent_name}: "
                        "stock concentration is 0."
                    )

                total_volume = float(self.template_meta['tot_vol'])

                # _convert_conc_to_vol() effectively uses:
                # transfer_volume = target_concentration * total_volume / stock_concentration
                transfer_volume = target_conc * total_volume / stock_conc

                repaired_volume = self._apply_true_zero_transfer_rule_to_volume(
                    transfer_volume
                )

                # Only print when the true-zero rule actually changes the
                # planned transfer. This keeps normal output clean while making
                # recipe repairs visible during Auto runs.
                if not math.isclose(repaired_volume, transfer_volume, rel_tol=0, abs_tol=1e-9):
                    print(
                        f"<<controller>> true-zero adjusted {reagent_name}: "
                        f"{transfer_volume:.4f} uL -> {repaired_volume:.4f} uL"
                    )

                repaired_conc = repaired_volume * stock_conc / total_volume
                repaired_recipes[recipe_i, reagent_i] = repaired_conc

        return repaired_recipes

    def _get_variable_reagent_stock_conc(self, reagent_name):
        '''
        Gets the stock concentration currently available on the deck for a
        variable reagent.

        Reagent containers are indexed by names like:
            silver_nitrateC0.375
            potassium_bromideC0.01

        This helper matches the base reagent name before the concentration
        marker and returns the deck concentration from reagent_df.

        params:
            str reagent_name:
                Base reagent name, such as 'silver_nitrate'.

        returns:
            float:
                Stock concentration of the reagent on the deck.
        '''
        reagent_df = self.robo_params['reagent_df']

        matching_concs = []

        for reagent_container_name in reagent_df.index:
            reagent_container_name = str(reagent_container_name)

            if 'C' in reagent_container_name:
                base_name = reagent_container_name.split('C')[0]
            else:
                base_name = reagent_container_name

            if base_name == reagent_name:
                matching_concs.append(float(reagent_df.loc[reagent_container_name, 'conc']))

        if len(matching_concs) == 0:
            raise ValueError(
                f"Could not find stock concentration for variable reagent "
                f"{reagent_name} in reagent_df."
            )

        if len(matching_concs) > 1:
            raise ValueError(
                f"Found multiple stock concentrations for variable reagent "
                f"{reagent_name}: {matching_concs}. The true-zero debug export "
                f"currently expects one stock concentration per variable reagent."
            )

        return matching_concs[0]

    def _get_fixed_reagent_volumes(self):
        '''
        Gets the fixed reagent transfer volumes used in each Auto reaction.

        Fixed reagent volumes come from transfer rows in the original reaction
        template. This helper intentionally uses the stored template dataframe
        and stored fixed reagent list instead of the live self.rxn_df, because
        self.rxn_df is replaced during Auto runs with generated protocol rows.

        A fixed reagent should have one clear nonzero template transfer volume.
        If the fixed volume cannot be found, is blank, or is ambiguous, this
        helper raises an error instead of silently returning an incomplete or
        incorrect volume balance.

        returns:
            dict:
                Fixed reagent names as keys and fixed transfer volumes in uL
                as values.
        '''
        fixed_reagent_volumes = {}

        # Use the original input/template dataframe if it exists. During Auto
        # execution, self.rxn_df is replaced with generated protocol rows, so
        # relying on the live dataframe can accidentally classify Water or
        # variable reagents as fixed reagents after batch 0.
        template_df = getattr(self, 'rxn_df_template', self.rxn_df)

        # Use the stored fixed reagent list if available. This keeps fixed
        # reagent identity locked to the original template instead of
        # recalculating it from the generated protocol dataframe.
        if hasattr(self, 'fixed_reagents'):
            fixed_reagents = self.fixed_reagents
        else:
            fixed_reagents = template_df.loc[
                template_df['conc'].notna() & (template_df['op'] == 'transfer'),
                'reagent'
            ].unique()

        candidate_volume_columns = []

        # Prefer the explicit Template column if present.
        if 'Template' in template_df.columns:
            candidate_volume_columns.append('Template')

        # Also inspect product/template columns if they are still available,
        # while avoiding duplicates if Template is already included.
        for product_col in getattr(self, '_products', []):
            if product_col in template_df.columns and product_col not in candidate_volume_columns:
                candidate_volume_columns.append(product_col)

        if not candidate_volume_columns:
            raise ValueError(
                "Could not identify any candidate volume columns for fixed "
                "reagent parsing in the original reaction template."
            )

        for reagent in fixed_reagents:
            matching_rows = template_df[
                (template_df['op'] == 'transfer') &
                (template_df['reagent'] == reagent)
            ]

            if matching_rows.empty:
                raise ValueError(
                    f"Could not find a transfer row for fixed reagent {reagent} "
                    "in the original reaction template."
                )

            candidate_volumes = []

            for _, row in matching_rows.iterrows():
                for volume_col in candidate_volume_columns:
                    volume = pd.to_numeric(row.get(volume_col), errors='coerce')

                    if pd.isna(volume):
                        continue

                    volume = float(volume)

                    if math.isclose(volume, 0.0, rel_tol=0, abs_tol=1e-9):
                        continue

                    candidate_volumes.append(volume)

            if len(candidate_volumes) == 0:
                raise ValueError(
                    f"Could not find a nonzero fixed transfer volume for "
                    f"{reagent} in the original reaction template."
                )

            unique_volumes = []

            for volume in candidate_volumes:
                if not any(
                    math.isclose(volume, existing, rel_tol=0, abs_tol=1e-9)
                    for existing in unique_volumes
                ):
                    unique_volumes.append(volume)

            if len(unique_volumes) > 1:
                raise ValueError(
                    f"Found multiple possible fixed transfer volumes for "
                    f"{reagent}: {unique_volumes}. The fixed reagent volume "
                    "must be unambiguous for Auto volume-feasibility checks."
                )

            fixed_reagent_volumes[reagent] = float(unique_volumes[0])

        return fixed_reagent_volumes

    def _get_fixed_reagent_volume_total(self):
        '''
        Gets the total volume occupied by fixed reagents in each Auto reaction.

        returns:
            float:
                Total fixed reagent volume in uL.
        '''
        fixed_reagent_volumes = self._get_fixed_reagent_volumes()

        return float(sum(fixed_reagent_volumes.values()))
    
    def _get_variable_transfer_volumes_for_recipe(self, recipe):
        '''
        Converts one denormalized Auto recipe into variable reagent transfer
        volumes.

        Auto recipes are stored as target concentrations. This helper converts
        those target concentrations into the physical transfer volumes required
        to make each concentration in the final reaction volume.

        params:
            np.ndarray recipe:
                One denormalized recipe row with one concentration per variable
                reagent, ordered the same way as self.variable_reagents.

        returns:
            dict:
                Variable reagent names as keys and transfer volumes in uL as
                values.
        '''
        recipe = np.asarray(recipe, dtype=float).reshape(-1)

        if recipe.shape[0] != len(self.variable_reagents):
            raise ValueError(
                "Recipe length does not match number of variable reagents. "
                f"Recipe has {recipe.shape[0]} values, but Auto expected "
                f"{len(self.variable_reagents)} variable reagents."
            )

        total_volume = float(self.template_meta['tot_vol'])
        variable_transfer_volumes = {}

        for reagent_i, reagent_name in enumerate(self.variable_reagents):
            stock_conc = self._get_variable_reagent_stock_conc(reagent_name)

            if math.isclose(stock_conc, 0.0, rel_tol=0, abs_tol=1e-12):
                raise ValueError(
                    f"Cannot calculate transfer volume for {reagent_name}: "
                    "stock concentration is 0."
                )

            target_conc = float(recipe[reagent_i])
            transfer_volume = target_conc * total_volume / stock_conc

            variable_transfer_volumes[reagent_name] = float(transfer_volume)

        return variable_transfer_volumes
    
    def _get_auto_recipe_volume_balance(self, recipe):
        '''
        Calculates the full volume balance for one denormalized Auto recipe.

        Auto recipes are stored as target concentrations for variable reagents.
        This helper converts those concentrations into transfer volumes, adds
        the fixed reagent volume, calculates the water top-off volume, and
        determines whether the recipe physically fits in the final reaction
        volume.

        Water is treated as the filler that occupies any remaining volume.
        Variable reagent volumes are not rescaled or redistributed.

        A recipe is volume-feasible only if:
            1. each variable transfer is exactly 0 uL or at least 5 uL
            2. fixed + variable volumes do not exceed the final reaction volume
            3. water top-off is either exactly 0 uL or at least 5 uL

        This prevents recipes with non-executable variable or water transfers
        from being run at unintended concentrations or underfilled volumes.

        params:
            np.ndarray recipe:
                One denormalized recipe row with one concentration per variable
                reagent, ordered the same way as self.variable_reagents.

        returns:
            dict:
                Volume-balance information for the recipe.
        '''
        total_volume = float(self.template_meta['tot_vol'])

        fixed_transfer_volumes = self._get_fixed_reagent_volumes()
        fixed_volume_total = float(sum(fixed_transfer_volumes.values()))

        variable_transfer_volumes = self._get_variable_transfer_volumes_for_recipe(
            recipe
        )
        variable_volume_total = float(sum(variable_transfer_volumes.values()))

        variable_transfer_executable_by_reagent = {
            reagent_name: bool(
                math.isclose(
                    transfer_volume,
                    0.0,
                    rel_tol=0,
                    abs_tol=1e-9
                )
                or transfer_volume >= 5.0 - 1e-9
            )
            for reagent_name, transfer_volume
            in variable_transfer_volumes.items()
        }
        variable_transfers_executable = all(
            variable_transfer_executable_by_reagent.values()
        )

        volume_before_water = fixed_volume_total + variable_volume_total
        water_volume = total_volume - volume_before_water

        volume_tol = 1e-9

        # Treat tiny floating-point artifacts around zero as exactly zero water.
        if math.isclose(water_volume, 0.0, rel_tol=0, abs_tol=volume_tol):
            water_volume = 0.0

        volume_does_not_overflow = water_volume >= -volume_tol

        # Water top-off must be executable. If water is needed, it must be at
        # least 5 uL. Otherwise, skipping the water would leave the final well
        # volume below the intended total volume and change the effective
        # concentrations/scans.
        water_transfer_executable = (
            math.isclose(water_volume, 0.0, rel_tol=0, abs_tol=volume_tol)
            or water_volume >= 5.0 - volume_tol
        )

        volume_feasible = (
            variable_transfers_executable
            and volume_does_not_overflow
            and water_transfer_executable
        )

        return {
            'total_volume': total_volume,
            'fixed_transfer_volumes': fixed_transfer_volumes,
            'fixed_volume_total': fixed_volume_total,
            'variable_transfer_volumes': variable_transfer_volumes,
            'variable_transfer_executable_by_reagent': (
                variable_transfer_executable_by_reagent
            ),
            'variable_transfers_executable': bool(
                variable_transfers_executable
            ),
            'variable_volume_total': variable_volume_total,
            'volume_before_water': volume_before_water,
            'water_volume': float(water_volume),
            'volume_does_not_overflow': bool(volume_does_not_overflow),
            'water_transfer_executable': bool(water_transfer_executable),
            'volume_feasible': bool(volume_feasible)
        }
    
    def _validate_auto_recipe_volume_feasibility(self, recipes, context_label='Auto batch'):
        '''
        Validates that repaired Auto recipes are physically executable before
        they are used to build robot transfers.

        This is a controller-side safety check. The optimizer should avoid
        infeasible recipes, but the controller is the final authority before
        robot execution.

        A recipe is considered volume-feasible only if:
            1. every variable transfer is exactly 0 uL or at least 5 uL
            2. fixed + variable reagent volumes do not exceed the final
               reaction volume
            3. required water top-off is either exactly 0 uL or at least 5 uL

        If any repaired recipe is not physically executable, the run is stopped
        before robot commands are created.

        params:
            np.ndarray recipes:
                Repaired denormalized recipe concentrations with shape:
                    n_recipes x n_variable_reagents

            str context_label:
                Human-readable label used in error messages.

        raises:
            ValueError:
                If any recipe is not physically volume-feasible.
        '''
        recipes = np.asarray(recipes, dtype=float)

        if recipes.ndim == 1:
            recipes = recipes.reshape(1, -1)

        invalid_messages = []

        for recipe_i, recipe in enumerate(recipes):
            volume_balance = self._get_auto_recipe_volume_balance(recipe)

            if not volume_balance['volume_feasible']:
                overflow_volume = max(
                    0.0,
                    -1.0 * volume_balance['water_volume']
                )

                invalid_messages.append(
                    f"{context_label}, recipe index {recipe_i}: "
                    f"fixed volume = {volume_balance['fixed_volume_total']:.4f} uL, "
                    f"variable volume = {volume_balance['variable_volume_total']:.4f} uL, "
                    f"variable transfers executable = "
                    f"{volume_balance['variable_transfers_executable']}, "
                    f"total before water = {volume_balance['volume_before_water']:.4f} uL, "
                    f"allowed total = {volume_balance['total_volume']:.4f} uL, "
                    f"water top-off = {volume_balance['water_volume']:.4f} uL, "
                    f"does not overflow = {volume_balance['volume_does_not_overflow']}, "
                    f"water executable = {volume_balance['water_transfer_executable']}, "
                    f"overflow = {overflow_volume:.4f} uL"
                )

        if invalid_messages:
            raise ValueError(
                "Auto recipe volume feasibility check failed. These recipes "
                "are not physically executable and will not be executed. "
                "A recipe may fail because it contains a non-executable 0-5 uL "
                "variable transfer, overfills the final reaction volume, or "
                "requires a non-executable 0-5 uL water top-off transfer:\n"
                + "\n".join(invalid_messages)
            )

    def _get_auto_batch_source_volume_requirements(self, rxn_df):
        '''
        Builds the ordered source-withdrawal plan for a fully constructed Auto
        protocol dataframe.

        This intentionally runs after _build_rxn_df() has selected the exact
        stock container for every transfer and after _insert_tot_vol_transfer()
        has added Water. The preflight therefore audits the same named sources
        and volumes that execute_protocol_df() would send to the robot.

        The controller's existing ``loc_req`` response exposes each resolved
        source group's *aggregate* aspiratable volume. It does not expose the
        volumes of individual same-stock tubes inside a robot MultiContainer.
        Consequently, this plan supports an aggregate inventory check only;
        the robot remains responsible for its established live tube-switching
        behavior when an individual tube becomes insufficient.
        '''
        if not isinstance(rxn_df, pd.DataFrame):
            raise ValueError(
                "Auto source-volume preflight requires a constructed "
                "protocol dataframe."
            )

        source_plan = []
        transfer_rows = rxn_df.loc[rxn_df['op'] == 'transfer']

        for _, transfer_row in transfer_rows.iterrows():
            source_name = transfer_row.get('chemical_name')

            if pd.isna(source_name) or not str(source_name).strip():
                raise ValueError(
                    "Auto source-volume preflight found a transfer row "
                    "without a resolved chemical_name source."
                )

            transfer_volumes = pd.to_numeric(
                transfer_row[self._products],
                errors='coerce'
            ).fillna(0.0)

            if (transfer_volumes < -1e-9).any():
                raise ValueError(
                    "Auto source-volume preflight found a negative transfer "
                    f"volume for source {source_name}."
                )

            ordered_transfer_volumes = [
                self._round_transfer_volume(float(transfer_volume))
                for transfer_volume in transfer_volumes
                if transfer_volume > 1e-9
            ]

            if ordered_transfer_volumes:
                source_plan.append((
                    str(source_name),
                    ordered_transfer_volumes
                ))

        return source_plan

    def _export_auto_source_volume_audit(self, audit_rows):
        '''Writes append-only per-batch source-volume preflight results.'''
        debug_path = getattr(self, 'debug_path', None)

        if debug_path is None:
            return

        os.makedirs(debug_path, exist_ok=True)
        export_path = os.path.join(
            debug_path,
            'auto_source_volume_audit.csv'
        )
        audit_df = pd.DataFrame(audit_rows)
        write_header = not os.path.exists(export_path)

        audit_df.to_csv(
            export_path,
            mode='a',
            header=write_header,
            index=False
        )

    def _preflight_auto_source_volumes(self, rxn_df, context_label):
        '''
        Fail-closed aggregate source-inventory check immediately before Auto
        liquid handling.

        This uses the controller's normal, already-supported ``loc_req``
        cache populated by _build_rxn_df(). The cache reflects the robot's
        current mass-derived source volumes and tracked prior withdrawals, but
        reports a same-stock MultiContainer as one aggregate source group.
        It therefore blocks batches whose total planned use plus reserve
        exceeds that group's available aspiratable volume. It deliberately
        does not claim to simulate individual tube allocation, pipette
        substeps, or fallback switching; those remain robot-runtime behavior
        and require no new robot-side code.

        returns:
            list or None:
                Passed audit rows when protection is required, otherwise None
                for backward-compatible legacy worksheets.
        '''
        if (
            self.robo_params.get('auto_source_volume_check', 'off')
            != 'required'
        ):
            return None

        source_plan = self._get_auto_batch_source_volume_requirements(rxn_df)
        reserve_volume_uL = float(
            self.robo_params.get('auto_source_reserve_volume_uL', 0.0)
        )
        planned_usage_by_source = {}

        for source_name, transfer_volumes in source_plan:
            planned_usage_by_source[source_name] = (
                planned_usage_by_source.get(source_name, 0.0)
                + sum(transfer_volumes)
            )

        failures = []
        audit_rows = []

        for source_name in sorted(planned_usage_by_source):
            planned_usage_uL = planned_usage_by_source[source_name]
            source_entry = self._cached_reader_locs.get(source_name)

            if source_entry is None:
                failures.append(
                    '{}: no current source-volume record was returned by the '
                    'robot location query.'.format(source_name)
                )
                continue

            # ChemCacheEntry mirrors the Raspberry Pi's ``loc_resp`` payload,
            # whose fourth volume field is the liquid that remains aspiratable
            # after the robot's dead-volume allowance.
            available_aspirable_uL = float(source_entry.aspirable_vol)
            required_aspirable_uL = (
                planned_usage_uL + reserve_volume_uL
            )
            remaining_aspirable_uL = (
                available_aspirable_uL - planned_usage_uL
            )
            passed = (
                available_aspirable_uL >= required_aspirable_uL - 1e-9
            )

            audit_rows.append({
                'source_chemical_name': source_name,
                'source_loc': source_entry.loc,
                'source_deck_pos': source_entry.deck_pos,
                'robot_reported_current_volume_uL': float(source_entry.vol),
                'robot_reported_aspirable_volume_uL': available_aspirable_uL,
                'planned_usage_uL': planned_usage_uL,
                'source_group_reserve_uL': reserve_volume_uL,
                'required_aspirable_volume_uL': required_aspirable_uL,
                'remaining_aspirable_volume_uL': remaining_aspirable_uL,
                'preflight_scope': 'aggregate_source_group',
                'batch_number': getattr(self, 'batch_num', None),
                'context_label': context_label,
                'preflight_passed': passed
            })

            if not passed:
                failures.append(
                    '{}: planned use {:.4f} uL plus reserve {:.4f} uL '
                    'requires {:.4f} uL, but the robot reports only {:.4f} '
                    'uL aspiratable for this source group.'.format(
                        source_name,
                        planned_usage_uL,
                        reserve_volume_uL,
                        required_aspirable_uL,
                        available_aspirable_uL
                    )
                )

        overall_passed = len(failures) == 0
        self.last_auto_source_volume_audit = copy.deepcopy(audit_rows)
        self._export_auto_source_volume_audit(audit_rows)

        if not overall_passed:
            raise ValueError(
                "Auto aggregate source-volume preflight failed before liquid "
                "transfers were sent:\n" + '\n'.join(failures)
            )

        print(
            "<<controller>> Auto source-volume preflight passed for "
            f"{context_label}: aggregate source inventory is sufficient."
        )

        if (
            self.robo_params.get('auto_terminal_verbosity', 'standard')
            == 'diagnostic'
        ):
            for audit_row in audit_rows:
                print(
                    "<<controller diagnostic>> source-volume "
                    f"{audit_row['source_chemical_name']}: "
                    f"planned {audit_row['planned_usage_uL']:.4f} uL, "
                    f"remaining "
                    f"{audit_row['remaining_aspirable_volume_uL']:.4f} uL"
                )

        return audit_rows

    def _export_auto_batch_recipe_design(
        self,
        repaired_recipes,
        original_recipes=None,
        batch_label=None,
        selection_metadata=None
    ):
        '''
        Exports the unique Auto recipe design for a batch before replicate wells
        are created.

        This file is intended for debugging and auditability. It records:
            - original recipe concentrations suggested by the model/design
            - repaired recipe concentrations after true-zero transfer repair
            - original and repaired normalized model-space values
            - original and repaired transfer volumes
            - whether each reagent was adjusted by the true-zero repair rule
            - whether each repaired transfer is true-zero valid
            - simple spacing metrics for checking maximin design quality
            - fixed reagent volume, variable reagent volume, and water top-off
              volume for each repaired recipe
            - whether each repaired recipe avoids overflow, requires executable water
              top-off, and is physically volume-feasible

        params:
            np.ndarray repaired_recipes:
                Repaired denormalized recipe concentrations with shape:
                    n_unique_recipes x n_variable_reagents

            np.ndarray original_recipes:
                Optional original denormalized recipe concentrations before
                true-zero repair. If not provided, repaired_recipes are used as
                the original values.

            str batch_label:
                Optional label for the exported file name. If not provided,
                the current self.batch_num is used.

            dict selection_metadata:
                Optional optimizer selection provenance. Seed-design callers
                may omit it. Optimizer-selected batches use it to persist the
                acquisition decision beside the selected/prepared recipe delta.
        '''
        repaired_recipes = np.array(repaired_recipes, dtype=float, copy=True)

        if repaired_recipes.ndim == 1:
            repaired_recipes = repaired_recipes.reshape(1, -1)

        if original_recipes is None:
            original_recipes = repaired_recipes.copy()
        else:
            original_recipes = np.array(original_recipes, dtype=float, copy=True)

            if original_recipes.ndim == 1:
                original_recipes = original_recipes.reshape(1, -1)

        if original_recipes.shape != repaired_recipes.shape:
            raise ValueError(
                "original_recipes and repaired_recipes must have the same shape "
                "for Auto recipe design export."
            )

        if batch_label is None:
            batch_label = f"batch_{self.batch_num}"

        total_volume = float(self.template_meta['tot_vol'])

        original_normalized = self.Normalize_Denormalize_Recipes(
            original_recipes.copy(),
            normalize_flag=True
        )

        repaired_normalized = self.Normalize_Denormalize_Recipes(
            repaired_recipes.copy(),
            normalize_flag=True
        )
        
        export_df = pd.DataFrame({
            'batch_num': [self.batch_num] * repaired_recipes.shape[0],
            'recipe_index': range(repaired_recipes.shape[0])
        })

        if selection_metadata is not None:
            repeated_value_count = repaired_recipes.shape[0]
            selection_scalar_fields = (
                'acquisition_mode',
                'acquisition_score',
                'balanced_exploration_weight',
                'predicted_target_error_nm',
                'predicted_lambda_mean_nm',
                'predicted_lambda_std_nm',
                'incumbent_target_error_nm',
                'optimizer_recipe_repaired',
                'optimizer_recipe_repair_max_transfer_delta_uL',
                'mask_result_count',
                'feasible_mask_result_count',
                'portfolio_selection_index',
                'portfolio_min_distance',
                'portfolio_nearest_distance'
            )

            for field_name in selection_scalar_fields:
                export_df[field_name] = [
                    selection_metadata.get(field_name)
                ] * repeated_value_count

            export_df['selected_mask'] = [
                self._serialize_auto_audit_value(
                    selection_metadata.get('selected_mask')
                )
            ] * repeated_value_count

            export_df['portfolio_acquisition_modes'] = [
                self._serialize_auto_audit_value(
                    selection_metadata.get(
                        'portfolio_acquisition_modes'
                    )
                )
            ] * repeated_value_count

            export_df['optimizer_volume_balance'] = [
                self._serialize_auto_audit_value(
                    selection_metadata.get('optimizer_volume_balance')
                )
            ] * repeated_value_count

            export_df['selected_controller_volume_balances'] = [
                self._serialize_auto_audit_value(
                    selection_metadata.get(
                        'selected_controller_volume_balances'
                    )
                )
            ] * repeated_value_count

            export_df['executed_controller_volume_balances'] = [
                self._serialize_auto_audit_value(
                    selection_metadata.get(
                        'executed_controller_volume_balances'
                    )
                )
            ] * repeated_value_count

            export_df['mask_results'] = [
                self._serialize_auto_audit_value(
                    selection_metadata.get('mask_results')
                )
            ] * repeated_value_count

        for reagent_i, reagent_name in enumerate(self.variable_reagents):
            stock_conc = self._get_variable_reagent_stock_conc(reagent_name)

            original_transfer = original_recipes[:, reagent_i] * total_volume / stock_conc
            repaired_transfer = repaired_recipes[:, reagent_i] * total_volume / stock_conc

            export_df[f'{reagent_name}_original_concentration'] = original_recipes[:, reagent_i]
            export_df[f'{reagent_name}_repaired_concentration'] = repaired_recipes[:, reagent_i]

            export_df[f'{reagent_name}_original_normalized'] = original_normalized[:, reagent_i]
            export_df[f'{reagent_name}_repaired_normalized'] = repaired_normalized[:, reagent_i]

            export_df[f'{reagent_name}_original_transfer_uL'] = original_transfer
            export_df[f'{reagent_name}_repaired_transfer_uL'] = repaired_transfer

            export_df[f'{reagent_name}_true_zero_adjusted'] = ~np.isclose(
                original_transfer,
                repaired_transfer,
                rtol=0,
                atol=1e-9
            )

            # After true-zero repair, every transfer should be either effectively
            # zero or at least the minimum allowed transfer volume.
            export_df[f'{reagent_name}_true_zero_valid'] = (
                np.isclose(repaired_transfer, 0.0, rtol=0, atol=1e-9) |
                (repaired_transfer >= 5.0 - 1e-9)
            )

        # Spacing metrics are calculated in repaired normalized model space,
        # because these are the actual design points used by the model/robot.
        if repaired_normalized.shape[0] < 2:
            nearest_neighbor_distances = np.zeros(repaired_normalized.shape[0])
            batch_min_pairwise_distance = 0.0
        else:
            nearest_neighbor_distances = []

            for i in range(repaired_normalized.shape[0]):
                distances = []

                for j in range(repaired_normalized.shape[0]):
                    if i == j:
                        continue

                    distances.append(
                        np.linalg.norm(repaired_normalized[i] - repaired_normalized[j])
                    )

                nearest_neighbor_distances.append(min(distances))

            nearest_neighbor_distances = np.asarray(nearest_neighbor_distances, dtype=float)
            batch_min_pairwise_distance = float(nearest_neighbor_distances.min())

        export_df['nearest_neighbor_distance_repaired_normalized'] = nearest_neighbor_distances
        export_df['batch_min_pairwise_distance_repaired_normalized'] = batch_min_pairwise_distance

        fixed_volume_totals = []
        variable_volume_totals = []
        water_volumes = []
        volume_before_water_values = []
        variable_transfers_executable_values = []
        volume_does_not_overflow_values = []
        water_transfer_executable_values = []
        volume_feasible_values = []

        for recipe in repaired_recipes:
            volume_balance = self._get_auto_recipe_volume_balance(recipe)

            fixed_volume_totals.append(volume_balance['fixed_volume_total'])
            variable_volume_totals.append(volume_balance['variable_volume_total'])
            water_volumes.append(volume_balance['water_volume'])
            volume_before_water_values.append(volume_balance['volume_before_water'])
            variable_transfers_executable_values.append(
                volume_balance['variable_transfers_executable']
            )
            volume_does_not_overflow_values.append(volume_balance['volume_does_not_overflow'])
            water_transfer_executable_values.append(volume_balance['water_transfer_executable'])
            volume_feasible_values.append(volume_balance['volume_feasible'])

        export_df['fixed_volume_total_uL'] = fixed_volume_totals
        export_df['variable_volume_total_uL'] = variable_volume_totals
        export_df['water_volume_uL'] = water_volumes
        export_df['volume_before_water_uL'] = volume_before_water_values
        export_df['variable_transfers_executable'] = (
            variable_transfers_executable_values
        )
        export_df['volume_does_not_overflow'] = volume_does_not_overflow_values
        export_df['water_transfer_executable'] = water_transfer_executable_values
        export_df['volume_feasible'] = volume_feasible_values

        auto_recipe_design_debug_path = os.path.join(
            self.debug_path,
            'auto_recipe_design'
        )

        os.makedirs(auto_recipe_design_debug_path, exist_ok=True)

        export_path = os.path.join(
            auto_recipe_design_debug_path,
            f'auto_recipe_design_{batch_label}.csv'
        )

        export_df.to_csv(export_path, index=False)
        print(f"<<controller>> exported Auto recipe design to {export_path}")
    
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
        rxn_df = self.rxn_df_template.copy() #starting point. still neeeds products
       

        recipe_df = pd.DataFrame(recipes, index=wellnames, columns=self.reagent_order)
        


        n_wellnames = np.array(wellnames)
        #n_wellnames_reshaped = n_wellnames.reshape(2,2)
        #n_reagent_order = self.reagent_order.reshape(2,2)
        #n_recipes = recipes.reshape(2,2)
        
        

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
        
        # Naming of output scan file:
        #   RTG_004_auto_scan-0.csv
        #   RTG_004      = self.rxn_sheet_name from the reaction/sheet name
        #   auto_scan    = scan_filename from the spreadsheet template
        #   0            = self.batch_num from the Auto loop
        rxn_df['scan_filename'] = rxn_df['scan_filename'].apply(lambda x: np.nan if pd.isna(x) 
                else "{}_{}-{}".format(self.rxn_sheet_name, x, self.batch_num))
        rxn_df['plot_filename'] = rxn_df['plot_filename'].apply(lambda x: np.nan if pd.isna(x) 
                else "{}-{}".format(x, self.batch_num))
        rxn_df.drop(columns='Template',inplace=True) #no longer need template
        return rxn_df

    def run_all_checks(self): 
        found_errors = super().run_all_checks()
        found_errors = max(found_errors,self.check_conc())
        if found_errors == 0:
            print("<<controller>> spreadsheet/setup prechecks passed")
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
                if re.fullmatch(e.reagent+r'C\d*\.\d*', key)]
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
                print("<<controller>> building protocol dataframe")
                self.rxn_df = self._convert_conc_to_vol(self.rxn_df,self._products)
                self._insert_tot_vol_transfer()
                if self.tot_vols: #has at least one element
                    if (self.rxn_df.loc[0,self._products] < 0).any():
                        raise NotImplementedError("A product overflowed it's container using the most concentrated solutions on the deck. Future iterations will ask Mark to add a more concentrated solution")
                successful_build = True
                print("<<controller>> protocol dataframe built successfully")
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
            print("<<controller>> spreadsheet/setup prechecks passed")
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
                    while not bool(re.match(r'\D\d',line)) and line != '':
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
#         df['num'] = df['col'].str.extract(r'(\\d+)').astype(int)
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
