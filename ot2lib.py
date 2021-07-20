from abc import ABC
from abc import abstractmethod
from collections import defaultdict
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

from Armchair.armchair import Armchair
import Armchair.armchair as armchair


#VISUALIZATION
def df_popout(df):
    '''
    Neat trick for viewing df as html in browser
    With some minor tweaks
    Credit to Stack Overflow
    @author Shovalt
    '''
    with NamedTemporaryFile(delete=False, suffix='.html') as f:
        html_str = df.to_html()
        f.write(str.encode(html_str,'ascii'))
    webbrowser.open(f.name)

#CLIENT

class ProtocolExecutor(): 
#this has two keys, 'deck_pos' and 'loc'. They map to the plate reader and the loc on that plate
#reader given a regular loc for a 96well plate.
#Please do not read this. paste it into a nice json viewer.
    PLATEREADER_INDEX_TRANSLATOR = bidict({'A1': ('E1', 'platereader4'), 'A2': ('D1', 'platereader4'), 'A3': ('C1', 'platereader4'), 'A4': ('B1', 'platereader4'), 'A5': ('A1', 'platereader4'), 'A12': ('A1', 'platereader7'), 'A11': ('B1', 'platereader7'), 'A10': ('C1', 'platereader7'), 'A9': ('D1', 'platereader7'), 'A8': ('E1', 'platereader7'), 'A7': ('F1', 'platereader7'), 'A6': ('G1', 'platereader7'), 'B1': ('E2', 'platereader4'), 'B2': ('D2', 'platereader4'), 'B3': ('C2', 'platereader4'), 'B4': ('B2', 'platereader4'), 'B5': ('A2', 'platereader4'), 'B6': ('G2', 'platereader7'), 'B7': ('F2', 'platereader7'), 'B8': ('E2', 'platereader7'), 'B9': ('D2', 'platereader7'), 'B10': ('C2', 'platereader7'), 'B11': ('B2', 'platereader7'), 'B12': ('A2', 'platereader7'), 'C1': ('E3', 'platereader4'), 'C2': ('D3', 'platereader4'), 'C3': ('C3', 'platereader4'), 'C4': ('B3', 'platereader4'), 'C5': ('A3', 'platereader4'), 'C6': ('G3', 'platereader7'), 'C7': ('F3', 'platereader7'), 'C8': ('E3', 'platereader7'), 'C9': ('D3', 'platereader7'), 'C10': ('C3', 'platereader7'), 'C11': ('B3', 'platereader7'), 'C12': ('A3', 'platereader7'), 'D1': ('E4', 'platereader4'), 'D2': ('D4', 'platereader4'), 'D3': ('C4', 'platereader4'), 'D4': ('B4', 'platereader4'), 'D5': ('A4', 'platereader4'), 'D6': ('G4', 'platereader7'), 'D7': ('F4', 'platereader7'), 'D8': ('E4', 'platereader7'), 'D9': ('D4', 'platereader7'), 'D10': ('C4', 'platereader7'), 'D11': ('B4', 'platereader7'), 'D12': ('A4', 'platereader7'), 'E1': ('E5', 'platereader4'), 'E2': ('D5', 'platereader4'), 'E3': ('C5', 'platereader4'), 'E4': ('B5', 'platereader4'), 'E5': ('A5', 'platereader4'), 'E6': ('G5', 'platereader7'), 'E7': ('F5', 'platereader7'), 'E8': ('E5', 'platereader7'), 'E9': ('D5', 'platereader7'), 'E10': ('C5', 'platereader7'), 'E11': ('B5', 'platereader7'), 'E12': ('A5', 'platereader7'), 'F1': ('E6', 'platereader4'), 'F2': ('D6', 'platereader4'), 'F3': ('C6', 'platereader4'), 'F4': ('B6', 'platereader4'), 'F5': ('A6', 'platereader4'), 'F6': ('G6', 'platereader7'), 'F7': ('F6', 'platereader7'), 'F8': ('E6', 'platereader7'), 'F9': ('D6', 'platereader7'), 'F10': ('C6', 'platereader7'), 'F11': ('B6', 'platereader7'), 'F12': ('A6', 'platereader7'), 'G1': ('E7', 'platereader4'), 'G2': ('D7', 'platereader4'), 'G3': ('C7', 'platereader4'), 'G4': ('B7', 'platereader4'), 'G5': ('A7', 'platereader4'), 'G6': ('G7', 'platereader7'), 'G7': ('F7', 'platereader7'), 'G8': ('E7', 'platereader7'), 'G9': ('D7', 'platereader7'), 'G10': ('C7', 'platereader7'), 'G11': ('B7', 'platereader7'), 'G12': ('A7', 'platereader7'), 'H1': ('E8', 'platereader4'), 'H2': ('D8', 'platereader4'), 'H3': ('C8', 'platereader4'), 'H4': ('B8', 'platereader4'), 'H5': ('A8', 'platereader4'), 'H6': ('G8', 'platereader7'), 'H7': ('F8', 'platereader7'), 'H8': ('E8', 'platereader7'), 'H9': ('D8', 'platereader7'), 'H10': ('C8', 'platereader7'), 'H11': ('B8', 'platereader7'), 'H12': ('A8', 'platereader7')})
    '''
    class to execute a protocol from the docs
    ATTRIBUTES:
        armchair.Armchair portal: the Armchair object to ship files across
        rxn_sheet_name: the name of the reaction sheet
        str cache_path: path to a directory for all cache files
        bool use_cache: read from cache if possible
        df rxn_df: the reaction df. Not passed in, but created in init
        str my_ip: the ip of this controller
        str server_ip: the ip of the server. This is modified for simulation, but returned to 
          original state at the end of simulation
        dict<str:object> robo_params: convenient place for the parameters for the robot
            TODO update this documentation
            df reagent_df
            dict instruments
            df labware_df
            df product_df
        bool simulate: whether a simulation is being run or not. False by default. changed true 
          temporarily when simulating
        int buff_size: this is the size of the buffer between Armchair commands. It's size
          corresponds to the number of commands you want to pile up in the socket buffer.
          Really more for developers

    CONSTANTS:
        dict<dict<str:str>> PLATEREADER_INDEX_TRANSLATOR: used to translate from locs on wellplate
          to locs on the opentrons object. Use a json viewer for more structural info
    '''
    def __init__(self, rxn_sheet_name, my_ip, server_ip, buff_size=4, use_cache=False, out_path='Eve_Files', cache_path='Cache'):
        '''
        Note that init does not initialize the portal. This must be done explicitly or by calling
        a run function that creates a portal. The portal is not passed to init because although
        the code must not use more than one portal at a time, the portal may change over the 
        lifetime of the class
        '''
        #set according to input
        self.cache_path = cache_path
        self.use_cache = use_cache
        self._make_out_dirs(out_path)
        self.my_ip = my_ip
        self.server_ip = server_ip
        self.buff_size=4
        self.simulate = False #by default will be changed if a simulation is run
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
        self.rxn_df = self._load_rxn_df(input_data)
        self._query_reagents(wks_key, credentials)
        raw_reagent_df = self._download_reagent_data(wks_key, credentials)#will be replaced soon
        #with a parsed reagent_df. This is exactly as is pulled from gsheets
        empty_containers = self._get_empty_containers(raw_reagent_df)
        products_to_labware = self._get_products_to_labware(input_data)
        self.robo_params['reagent_df'] = self._parse_raw_reagent_df(raw_reagent_df)
        self.robo_params['instruments'] = self._get_instrument_dict(deck_data)
        self.robo_params['labware_df'] = self._get_labware_df(deck_data, empty_containers)
        self.robo_params['product_df'] = self._get_product_df(products_to_labware)
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
        eve_thread = threading.Thread(target=launch_eve_server, kwargs={'my_ip':'','barrier':b})
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
        
    def _run(self, port, simulate):
        '''
        Returns:
            bool: True if all tests were passed
        '''
        #create a connection
        sock = socket.socket(socket.AF_INET)
        sock.connect((self.server_ip, port))
        buffered_sock = BufferedSocket(sock, timeout=None)
        print("<<controller>> connected")
        self.portal = Armchair(buffered_sock,'controller','Armchair_Logs', buffsize=1)

        self.init_robot(simulate)
        self.execute_protocol_df()
        self.close_connection()
        return
        

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
            while not wks_key and i < len(name_key_pairs):
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
        rxn_df.rename({'operation':'op', 'dilution concentration':'dilution_conc','concentration (mM)':'conc', 'reagent (must be uniquely named)':'reagent', 'pause time (s)':'pause_time', 'comments (e.g. new bottle)':'comments'}, axis=1, inplace=True)
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
        if pd.isnull(row['reagent']):
            #this must not be a transfer. this operation has no chemical name
            return np.nan
        elif pd.isnull(row['conc']):
            #this uses a chemical, but the chemical doesn't have a concentration (probably a mix)
            return row['reagent'].replace(' ', '_')
        else:
            #this uses a chemical with a conc. Probably a stock solution
            return "{}C{}".format(row['reagent'], row['conc']).replace(' ', '_')
        return pd.Series(new_cols)

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
                int index: deck_pos
                str position: the position of the empty container on the labware
        returns:
            df:
              str name: the common name of the labware (made unique 
            TODO update this documentation
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
        #TODO assert (usable_rows.shape[0] == 1), "too many first wells specified for platereader"
        #check if that works
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
        reagent_df[['loc', 'deck_pos', 'mass', 'comments']] = ''
        if not self.use_cache:
            d2g.upload(reagent_df.reset_index(),spreadsheet_key,wks_name = 'reagent_info', row_names=False , credentials = credentials)

    def _download_reagent_data(self, spreadsheet_key, credentials):
        '''
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
        return reagent_info

    def _get_empty_containers(self, raw_reagent_df):
        '''
        only one line, but there's a lot going on. extracts the empty lines from the raw_reagent_df
        params:
            df raw_reagent_df: as in reagent_info of excel
        returns:
            df empty_containers:
                INDEX:
                int deck_pos: the position on the deck
                COLS:
                str loc: location on the labware
        '''
        return raw_reagent_df.loc['empty' == raw_reagent_df.index].set_index('deck_pos').drop(columns=['conc', 'mass'])
    
    def _parse_raw_reagent_df(self, raw_reagent_df):
        '''
        parses the raw_reagent_df into final form for reagent_df
        params:
            df raw_reagent_df: as in excel
        returns:
            df reagent_df: empties ignored, columns with correct types
        '''
        reagent_df = raw_reagent_df.drop(['empty'], errors='ignore') # incase not on axis
        try:
            reagent_df = reagent_df.astype({'conc':float,'deck_pos':int,'mass':float})
        except ValueError as e:
            raise ValueError("Your reagent info could not be parsed. Likely you left out a required field, or you did not specify a concentration on the input sheet")
        return reagent_df

    def _get_product_df(self, products_to_labware):
        '''
        Creates a df to be used by robot to initialize containers for the products it will make
        params:
            df products_to_labware: as passed to init_robot
        returns:
            df products:
                INDEX:
                str chemical_name: the name of this rxn
                COLS:
                str labware: the labware to put this rxn in or None if no preference
                float max_vol: the maximum volume that will ever ocupy this container
        TODO great place to catch not enough liquid errors
        '''
        products = products_to_labware.keys()
        max_vols = [self._get_rxn_max_vol(product, products) for product in products]
        product_df = pd.DataFrame(products_to_labware, index=['labware','container']).T
        product_df['max_vol'] = max_vols
        return product_df

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
        #TODO handle dilutions into
        #current solution is to assume a solution is never aspirated during a dilution which
        #will assume larger than necessary volumes
        vol_change_rows = self.rxn_df.loc[self.rxn_df['op'].apply(lambda x: x in ['transfer','dilution'])]
        aspirations = vol_change_rows['chemical_name'] == name
        max_vol = 0
        current_vol = 0
        for i, is_aspiration in aspirations.iteritems():
            if is_aspiration and self.rxn_df.loc[i,'op'] == 'transfer':
                #This is a row where we're transfering from this well
                current_vol -= self.rxn_df.loc[i, products].sum()
            else:
                current_vol += self.rxn_df.loc[i,name]
                max_vol = max(max_vol, current_vol)
        return max_vol

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
                self.send_transfer_command(row,i)
            elif row['op'] == 'pause':
                cid = self.portal.send_pack('pause',row['pause_time'])
            elif row['op'] == 'stop':
                #read through the inflight packets
                self.portal.send_pack('stop')
                self._stop(i)
            elif row['op'] == 'scan':
                #TODO implement scans
                pass
            elif row['op'] == 'dilution':
                #TODO implement dilutions
                self.send_dilution_commands(row, i)
            else:
                raise Exception('invalid operation {}'.format(row['op']))

    def send_dilution_commands(self,row,i):
        '''
        used to execute a dilution. This is analogous to microcode. This function will send two
          commands. Water is always added first.
            transfer: transfer water into the container
            transfer: transfer reagent into the container
        params:
            pd.Series row: a row of self.rxn_df
            int i: index of this row
        returns:
            int: the cid of this command
        Preconditions:
            The buffer has room for at least one command
        Postconditions:
            Two transfer commands have been sent to the robot to: 1) add water. 2) add reagent.
            Will block on ready if the buffer is filled
        '''
        water_transfer_row, reagent_transfer_row = self._get_dilution_transfer_rows(row)
        self.send_transfer_command(water_transfer_row, i)
        self.send_transfer_command(reagent_transfer_row, i)

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
        input("stopped on line {} of protocol. Please press enter to continue execution".format(i+1))
        self.portal.send_pack('continue')

    def send_transfer_command(self, row, i):
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

    def _make_out_dirs(self, out_path):
        '''
        params:
            str out_path: the path for all files output by controller
        Postconditions:
            All paths used by this class have been initialized if they were not before
            They are not overwritten if they already exist
        '''
        self.eve_files_path = os.path.join(out_path, 'Eve_Files')
        self.debug_path = os.path.join(out_path, 'Debug')
        paths = [out_path, self.eve_files_path, self.debug_path]
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    def close_connection(self):
        '''
        runs through closing procedure with robot
        params:
            TODO update
        Postconditions:
            Log files have been written to self.out_path
            Connection has been closed
        '''
        print('<<controller>> initializing breakdown')
        self.portal.send_pack('close')
        #server will initiate file transfer
        pack_type, cid, arguments = self.portal.recv_pack()
        while pack_type == 'ready':
            #spin through all the queued ready packets
            pack_type, cid, arguments = self.portal.recv_pack()
        assert(pack_type == 'sending_files')
        port = arguments[0]
        filenames = arguments[1]
        #TODO race condition exists here. ConnectionRefusedError
        sock = socket.socket(socket.AF_INET)
        sock.connect((self.server_ip, port))
        buffered_sock = BufferedSocket(sock,maxsize=4e9) #file better not be bigger than 4GB
        for filename in filenames:
            with open(os.path.join(self.eve_files_path,filename), 'wb') as write_file:
                data = buffered_sock.recv_until(armchair.FTP_EOF)
                write_file.write(data)
        self.translate_wellmap()
        print('<<controller>> files recieved')
        sock.close()
        #server should now send a close command
        pack_type, cid, arguments = self.portal.recv_pack()
        assert(pack_type == 'close')
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

    def error_handler(self):
        pass
    
    def init_robot(self, simulate):
        '''
        this does the dirty work of sending accumulated params over network to the robot
        Postconditions:
            robot has been initialized with necessary params
        '''
        #send robot data to initialize itself
        cid = self.portal.send_pack('init', simulate, 
                self.robo_params['using_temp_ctrl'], self.robo_params['temp'],
                self.robo_params['labware_df'].to_dict(), self.robo_params['instruments'],
                self.robo_params['reagent_df'].to_dict(), self.my_ip)
    
        #send robot data to initialize empty product containers. Because we know things like total
        #vol and desired labware, this makes sense for a planned experiment
        self.portal.send_pack('init_containers', self.robo_params['product_df'].to_dict())

    
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
            if r['op'] == 'transfer' and not r['reagent']:
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
        loc_pos_empty_pairs = pd.Series(loc_pos_empty_pairs)
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
        sbs =self._get_vol_lab_cont_sbs()
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
        if row['loc_t'] != 'any' and not row['loc'] == row['loc_t']:
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
                INDEX
                chemical_name: the containers name
                COLS: symmetric. Theoretical are suffixed _t
                str deck_pos: position on deck
                float vol: the volume in the container
                list<tuple<str, float>> history: the chem_name paired with the amount or
                  keyword 'aspirate' and vol
        '''
        #copy the locals cause we're changing them
        labware_df = self.robo_params['labware_df'].set_index('name').rename(index={'platereader7':'platereader','platereader4':'platereader'}) #converting to dict like
        product_df = self.robo_params['product_df'].copy()
        reagent_df = self.robo_params['reagent_df'].copy()
        def get_deck_pos(labware):
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
        reagent_df['deck_pos'] = reagent_df['deck_pos'].apply(lambda x: [x])
        reagent_df['vol'] = 'any' #I'm not checking this because it's harder to check, and works fine
        reagent_df['container'] = 'any' #actually fixed, but checked by combo deck_pos and loc
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
        dispenses = self.rxn_df[name].sum()
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

#SERVER
#CONTAINERS
class Container(ABC):
    """
    
    Abstract container class to be overwritten for well, tube, etc.
    ABSTRACT ATTRIBUTES:
        str name: the common name we use to refer to this container
        float vol: the volume of the liquid in this container in uL
        int deck_pos: the position on the deck
        str loc: a location on the deck_pos object (e.g. 'A5')
        float conc: the concentration of the substance
        float disp_height: the height to dispense at
        float asp_height: the height to aspirate from
        list<tup<timestamp, str, float> history: the history of this container. Contents:
          timestamp timestamp: the time of the addition/removal
          str chem_name: the name of the chemical added or blank if aspiration
          float vol: the volume of chemical added/removed
    CONSTANTS:
        float DEAD_VOL: the volume at which this 
        float MIN_HEIGHT: the minimum height at which to pipette from 
    ABSTRACT METHODS:
        _update_height void: updates self.height to height at which to pipet (a bit below water line)
    IMPLEMENTED METHODS:
        update_vol(float del_vol) void: updates the volume upon an aspiration
    """

    def __init__(self, name, deck_pos, loc, vol=0,  conc=1):
        self.name = name
        self.deck_pos = deck_pos
        self.loc = loc
        self.vol = vol
        self._update_height()
        self.conc = conc
        self.history = []
        if vol:
            #create an entry with yourself as first
            self.history.append((datetime.now().strftime('%d-%b-%Y %H:%M:%S:%f'), name, vol))

    DEAD_VOL = 0
    MIN_HEIGHT = 0

    @abstractmethod
    def _update_height(self):
        pass

    def update_vol(self, del_vol,name=''):
        '''
        params:
            float del_vol: the change in volume. -vol is an aspiration
            str name: the thing coming in if it is a dispense
        Postconditions:
            the volume has been adjusted
            height has been adjusted
            the history has been updated
        '''
        #if you are dispersing without specifying the name of incoming chemical, complain
        assert ((del_vol < 0) or (name and del_vol > 0)), 'Developer Error: dispensing without \
                specifying src'
        self.history.append((datetime.now().strftime('%d-%b-%Y %H:%M:%S:%f'), name, del_vol))
        self.vol = self.vol + del_vol
        self._update_height()

    @property
    def disp_height(self):
        pass

    @property
    def asp_height(self):
        pass

    @property
    def aspiratible_vol(self):
        return self.vol - self.DEAD_VOL

        
class Tube20000uL(Container):
    """
    Spcific tube with measurements taken to provide implementations of abstract methods
    INHERITED ATTRIBUTES
        str name, float vol, int deck_pos, str loc, float disp_height, float asp_height
    OVERRIDDEN CONSTANTS:
        float DEAD_VOL: the volume at which this 
    INHERITED METHODS
        _update_height void, update_vol(float del_vol) void,
    """

    DEAD_VOL = 2000
    MIN_HEIGHT = 4

    def __init__(self, name, deck_pos, loc, mass=6.6699, conc=1):
        '''
        mass is defaulted to the avg_mass so that there is nothing in the container
        '''
        density_water_25C = 0.9970479 # g/mL
        avg_tube_mass15 = 6.6699 # grams
        self.mass = mass - avg_tube_mass15 # N = 1 (in grams) 
        assert (self.mass >= -1e-9),'the mass you entered for {} is less than the mass of the tube it\'s in.'.format(name)
        vol = (self.mass / density_water_25C) * 1000 # converts mL to uL
        super().__init__(name, deck_pos, loc, vol, conc)
       # 15mm diameter for 15 ml tube  -5: Five mL mark is 19 mm high for the base/noncylindrical protion of tube 

    def _update_height(self):
        diameter_15 = 14.0 # mm (V1 number = 14.4504)
        height_bottom_cylinder = 30.5  #mm
        height = ((self.vol - self.DEAD_VOL)/(math.pi*(diameter_15/2)**2))+height_bottom_cylinder
        self.height = height if height > height_bottom_cylinder else self.MIN_HEIGHT

    @property
    def disp_height(self):
        return self.height + 10 #mm

    @property
    def asp_height(self):
        tip_depth = 5
        return self.height - tip_depth
            
class Tube50000uL(Container):
    """
    Spcific tube with measurements taken to provide implementations of abstract methods
    INHERITED ATTRIBUTES
        str name, float vol, int deck_pos, str loc, float disp_height, float asp_height
    INHERITED METHODS
        _update_height void, update_vol(float del_vol) void,
    """

    DEAD_VOL = 5000
    MIN_HEIGHT = 4

    def __init__(self, name, deck_pos, loc, mass=13.3950, conc=1):
        density_water_25C = 0.9970479 # g/mL
        avg_tube_mass50 = 13.3950 # grams
        self.mass = mass - avg_tube_mass50 # N = 1 (in grams) 
        assert (self.mass >= -1e-9),'the mass you entered for {} is less than the mass of the tube it\'s in.'.format(name)
        vol = (self.mass / density_water_25C) * 1000 # converts mL to uL
        super().__init__(name, deck_pos, loc, vol, conc)
       # 15mm diameter for 15 ml tube  -5: Five mL mark is 19 mm high for the base/noncylindrical protion of tube 
        
    def _update_height(self):
        diameter_50 = 26.50 # mm (V1 number = 26.7586)
        height_bottom_cylinder = 21 #mm
        height = ((self.vol - self.DEAD_VOL)/(math.pi*(diameter_50/2)**2)) + height_bottom_cylinder
        self.height = height if height > height_bottom_cylinder else self.MIN_HEIGHT

    @property
    def disp_height(self):
        return self.height + 10 #mm

    @property
    def asp_height(self):
        tip_depth = 5
        return self.height - tip_depth

class Tube2000uL(Container):
    """
    2000uL tube with measurements taken to provide implementations of abstract methods
    INHERITED ATTRIBUTES
         str name, float vol, int deck_pos, str loc, float disp_height, float asp_height
    INHERITED METHODS
        _update_height void, update_vol(float del_vol) void,
    """

    DEAD_VOL = 250 #uL
    MIN_HEIGHT = 4

    def __init__(self, name, deck_pos, loc, mass=1.4, conc=2):
        density_water_4C = 0.9998395 # g/mL
        avg_tube_mass2 =  1.4        # grams
        self.mass = mass - avg_tube_mass2 # N = 1 (in grams) 
        assert (self.mass >= -1e-9),'the mass you entered for {} is less than the mass of the tube it\'s in.'.format(name)
        vol = (self.mass / density_water_4C) * 1000 # converts mL to uL
        super().__init__(name, deck_pos, loc, vol, conc)
           
    def _update_height(self):
        diameter_2 = 8.30 # mm
        height_bottom_cylinder = 10.5 #mm
        height = ((self.vol - self.DEAD_VOL)/(math.pi*(diameter_2/2)**2)) + height_bottom_cylinder
        self.height = height if height > height_bottom_cylinder else self.MIN_HEIGHT

    @property
    def disp_height(self):
        return self.height + 10 #mm

    @property
    def asp_height(self):
        tip_depth = 4.5 # mm
        return self.height - tip_depth

class Well96(Container):
    """
        a well in a 96 well plate
        INHERITED ATTRIBUTES
             str name, float vol, int deck_pos, str loc, float disp_height, float asp_height
        INHERITED CONSTANTS
            int DEAD_VOL TODO update DEAD_VOL from experimentation
        INHERITED METHODS
            _update_height void, update_vol(float del_vol) void,
    """

    MIN_HEIGHT = 1

    def __init__(self, name, deck_pos, loc, vol=0, conc=1):
        #vol is defaulted here because the well will probably start without anything in it
        super().__init__(name, deck_pos, loc, vol, conc)
           
    def _update_height(self):
        #this method is not needed for a well of such small size because we always aspirate
        #and dispense at the same heights
        self.height = None

    @property
    def disp_height(self):
        return 10 #mm

    @property
    def asp_height(self):
        return self.MIN_HEIGHT

#LABWARE
#TODO Labware was built with the assumption that once you ask for a bit of labware, you will use it
#if you don't want to pop, we must allow that functionality
class Labware(ABC):
    '''
    The opentrons labware class is lacking in some regards. It does not appear to have
    a method for removing tubes from the labware, which is what I need to do, hence this
    wrapper class to hold opentrons labware objects
    Note that tipracks are not included. The way we access them is normal enough that opentrons
    API does everything we need for them
    ATTRIBUTES:
        Opentrons.Labware labware: the opentrons object
        bool full: True if there are no more empty containers
        int deck_pos: to map back to deck position
        str name: the name associated with this labware
    CONSTANTS
        list<str> CONTAINERS_SERVICED: the container types on this labware
    ABSTRACT METHODS:
        get_container_type(loc) str: returns the type of container at that location
        pop_next_well(vol=None) str: returns the index of the next available well
          If there are no available wells of the volume requested, return None
    '''

    CONTAINERS_SERVICED = []

    def __init__(self, labware, deck_pos):
        self.labware = labware
        self.full = False
        self.deck_pos = deck_pos

    @abstractmethod
    def pop_next_well(self, vol=None, container_type=None):
        '''
        returns the next available well
        Or returns None if there is no next availible well with specified volume
        container_type takes precedence over volume, but you shouldn't need to call it with both
        '''
        pass

    @abstractmethod
    def get_container_type(self, loc):
        '''
        params:
            str loc: the location on the labware. e.g. A1
        returns:
            str the type of container class
        '''
        pass

    def get_well(self,loc):
        '''
        params:
            str loc: the location on the labaware e.g. A1
        returns:
            the opentrons well object at that location
        '''
        return self.labware.wells_by_name()[loc]

    @property
    def name(self):
        return self.labware.name

class TubeHolder(Labware):
    #TODO better documentation of labware subclasses
    '''
    Subclass of Labware object that may not have all containers filled, and allows for diff
    sized containers
    INHERITED METHODS:
        pop_next_well(vol=None) str: Note vol is should be provided here, otherwise a random size
          will be chosen
        get_container_type(loc) str
    INHERITED_ATTRIBUTES:
        Opentrons.Labware labware, bool full, int deck_pos, str name
    OVERRIDEN CONSTANTS
        list<str> CONTAINERS_SERVICED
    ATTRIBUTES:
        list<str> empty_tubes: contains locs of the empty tubes. Necessary because the user may
          not put tubes into every slot. Sorted order smallest tube to largest
    '''

    CONTAINERS_SERVICED = ['Tube50000uL', 'Tube20000uL', 'Tube2000uL']

    def __init__(self, labware, empty_tubes, deck_pos):
        super().__init__(labware,deck_pos)
        #We create a dictionary of tubes with the container as the key and a list as the 
        #value. The list contains all tubes that fit that volume range
        self.empty_tubes={tube_type:[] for tube_type in self.CONTAINERS_SERVICED}
        for tube in empty_tubes:
            self.empty_tubes[self.get_container_type(tube)].append(tube)
        self.full = not self.empty_tubes


    def pop_next_well(self, vol=None, container_type=None):
        '''
        Gets the next available tube. If vol is specified, will return an
        appropriately sized tube. Otherwise it will return a tube. It makes no guarentees that
        tube will be the correct size. It is not recommended this method be called without
        a volume argument
        params:
            float vol: used to determine an appropriate sized tube
            str container_type: the type of container requested
        returns:
            str: loc the location of the smallest next tube that can accomodate the volume
            None: if it can't be accomodated
        '''
        if not self.full:
            if container_type:
                #here I'm assuming you wouldn't want to put more volume in a tube than it can fit
                viable_tubes = self.empty_tubes[container_type]
            elif vol:
                #neat trick
                viable_tubes = self.empty_tubes[self.get_container_type(vol=vol)]
                if not viable_tubes:
                    #but if it didn't work you need to check everything
                    for tube_type in self.CONTAINERS_SERVICED:
                        viable_tubes = self.empty_tubes[tube_type]
                        if viable_tubes:
                            #check if the volume is still ok
                            capacity = self.labware.wells_by_name()[viable_tubes[0]]._geometry._max_volume
                            if vol < capacity:
                                break
            else:
                #volume was not specified
                #return the next smallest tube.
                #this always returns because you aren't empty
                for tube_type in self.CONTAINERS_SERVICED:
                    if self.empty_tubes[tube_type]:
                        viable_tubes = self.empty_tubes[tube_type]
                        break
            if viable_tubes:
                tube_loc = viable_tubes.pop()
                self.update_full()
                return tube_loc
            else:
                return None
        else:
            #self.empty_tubes is empty!
            return None

    def update_full(self):
        '''
        updates self.full
        '''
        self.full=True
        for tube_type in self.CONTAINERS_SERVICED:
            if self.empty_tubes[tube_type]:
                self.full = False
                return
        
    def get_container_type(self, loc=None, vol=None):
        '''
        NOTE internally, this method is a little different, but the user should use as 
        outlined below
        returns type of container
        params:
            str loc: the location on this labware
        returns:
            str: the type of container at that loc
        '''
        if not vol:
            tube_capacity = self.labware.wells_by_name()[loc]._geometry._max_volume
        else:
            tube_capacity = vol
        if tube_capacity <= 2000:
            return 'Tube2000uL'
        elif tube_capacity <= 20000:
            return 'Tube20000uL'
        else:
            return 'Tube50000uL'

class WellPlate(Labware):
    '''
    subclass of labware for dealing with plates
    INHERITED METHODS:
        pop_next_well(vol=None,container_type=None) str: vol should be provided to 
          check if well is big enough. container_type is for compatibility
        get_container_type(loc) str
    INHERITED_ATTRIBUTES:
        Opentrons.Labware labware, bool full, int deck_pos, str name
    OVERRIDEN CONSTANTS:
        list<str> CONTAINERS_SERVICED
    ATTRIBUTES:
        int current_well: the well number your on (NOT loc!)
    '''

    CONTAINERS_SERVICED = ['Well96']

    def __init__(self, labware, first_well, deck_pos):
        super().__init__(labware, deck_pos)
        #allow for none initialization
        all_wells = labware.wells()
        self.current_well = 0
        while self.current_well < len(all_wells) and all_wells[self.current_well]._impl._name != first_well:
            self.current_well += 1
        #if you overflowed you'll be correted here
        self.full = self.current_well >= len(labware.wells())

    def pop_next_well(self, vol=None,container_type=None):
        '''
        returns the next well if there is one, otherwise returns None
        params:
            float vol: used to determine if your reaction can be fit in a well
            str container_type: should never be used. Here for compatibility
        returns:
            str: the well loc if it can accomadate the request
            None: if can't accomodate request
        '''
        if not self.full:
            well = self.labware.wells()[self.current_well] 
            capacity = well._geometry._max_volume
            if capacity > vol:
                #have a well that works
                self.current_well += 1
                self.full = self.current_well >= len(self.labware.wells())
                return well._impl._name
            else:
                #requested volume is too large
                return None
        else:
            #don't have any more room
            return None
    
    def get_container_type(self, loc):
        '''
        params:
            str loc: loc on the labware
        returns:
            str: the type of container
        '''
        return 'Well96'

#Robot
class OT2Controller():
    """
    The big kahuna. This class contains all the functions for controlling the robot
    ATTRIBUTES:
        str ip: IPv4 LAN address of this machine
        Dict<str, Container> containers: maps from a common name to a Container object
        Dict<str, Obj> tip_racks: maps from a common name to a opentrons tiprack labware object
        Dict<str, Obj> labware: maps from labware common names to opentrons labware objects. tip racks not included?
        Dict<str:Dict<str:Obj>>: JSON style dict. First key is the arm_pos second is the attribute
            'size' float: the size of this pipette in uL
            'last_used' str: the chem_name of the last chemical used. 'clean' is used to denote a
              clean pipette
        Opentrons...ProtocolContext protocol: the protocol object of this session
    """

    #Don't try to read this. Use an online json formatter 
    _LABWARE_TYPES = {"96_well_plate":{"opentrons_name":"corning_96_wellplate_360ul_flat","groups":["well_plate"],'definition_path':""},"24_well_plate":{"opentrons_name":"corning_24_wellplate_3.4ml_flat","groups":["well_plate"],'definition_path':""},"48_well_plate":{"opentrons_name":"corning_48_wellplate_1.6ml_flat","groups":["well_plate"],'definition_path':""},"tip_rack_20uL":{"opentrons_name":"opentrons_96_tiprack_20ul","groups":["tip_rack"],'definition_path':""},"tip_rack_300uL":{"opentrons_name":"opentrons_96_tiprack_300ul","groups":["tip_rack"],'definition_path':""},"tip_rack_1000uL":{"opentrons_name":"opentrons_96_tiprack_1000ul","groups":["tip_rack"],'definition_path':""},"tube_holder_10":{"opentrons_name":"opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical","groups":["tube_holder"],'definition_path':""},"temp_mod_24_tube":{"opentrons_name":"opentrons_24_aluminumblock_generic_2ml_screwcap","groups":["tube_holder","temp_mod"],'definition_path':""},"platereader4":{"opentrons_name":"","groups":["well_plate","platereader"],"definition_path":"LabwareDefs/plate_reader_4.json"},"platereader7":{"opentrons_name":"","groups":["well_plate","platereader"],"definition_path":"LabwareDefs/plate_reader_7.json"},"platereader":{"opentrons_name":"","groups":["well_plate","platereader"]}}
    _PIPETTE_TYPES = {"300uL_pipette":{"opentrons_name":"p300_single_gen2"},"1000uL_pipette":{"opentrons_name":"p1000_single_gen2"},"20uL_pipette":{"opentrons_name":"p20_single_gen2"}}


    def __init__(self, simulate, using_temp_ctrl, temp, labware_df, instruments, reagent_df, my_ip, controller_ip, portal):
        '''
        params:
            bool simulate: if true, the robot will run in simulation mode only
            bool using_temp_ctrl: true if you want to use the temperature control module
            float temp: the temperature to keep the control module at.
            df labware_df:
                str name: the common name of the labware
                str first_usable: the first tip/well to use
                int deck_pos: the position on the deck of this labware
                str empty_list: the available slots for empty tubes format 'A1,B2,...' No specific
                  order
            Dict<str:str> instruments: keys are ['left', 'right'] corresponding to arm slots. vals
              are the pipette names filled in
            df reagent_df: info on reagents. columns from sheet. See excel specification
            str my_ip: the ip address of the robot
            bool simulate: whether to simulate protocol or not
                
        postconditions:
            protocol has been initialzied
            containers and tip_racks have been created
            labware has been initialized
            CAUTION: the values of tip_racks and containers must be sent from the client.
              it is the client's responsibility to make sure that these are initialized prior
              to operating with them
        '''
        #convert args back to df
        labware_df = pd.DataFrame(labware_df)
        reagent_df = pd.DataFrame(reagent_df)
        self.containers = {}
        self.pipettes = {}
        self.my_ip = my_ip
        self.portal = portal
        self.controller_ip = controller_ip
        self.simulate = simulate

        #self.log = logging.getLogger('eve_logger')
        #self.log.addHandler(logging.FileHandler('.eve.log',encoding='utf8'))
        #self.log.warning('uh oh')

        #like protocol.deck, but with custom labware wrappers
        self.lab_deck = np.full(12, None, dtype='object') #note first slot not used

        if simulate:
            # define version number and define protocol object
            self.protocol = opentrons.simulate.get_protocol_api('2.9')
        else:
            self.protocol = opentrons.execute.get_protocol_api('2.9')
            self.protocol.set_rail_lights(on = True)
            self.protocol.rail_lights_on 
        self.protocol.home() # Homes the pipette tip
        #empty list was in comma sep form for easy shipping. unpack now to list
        labware_df['empty_list'] = labware_df['empty_list'].apply(lambda x: x.split(',')
                if x else [])
        self._init_params()
        self._init_directories()
        self._init_labware(labware_df, using_temp_ctrl, temp)
        self._init_instruments(instruments, labware_df)
        self._init_containers(reagent_df)

    def _init_directories(self):
        '''
        The debug/directory structure of the robot is not intended to be stored for long periods
        of time. This is becuase the files should be shipped over FTP to laptop. In the event
        of an epic fail, e.g. where network went down and has no means to FTP back to laptop
        Postconditions: the following directory structure has been contstructed
            Eve_Out: root
                Debug: populated with error information. Used on crash
                Logs: log files for eve
        '''
        #clean up last time
        if os.path.exists('Eve_Out'):
            shutil.rmtree('Eve_Out')
        #make new folders
        os.mkdir('Eve_Out')
        os.mkdir('Eve_Out/Debug')
        os.mkdir('Eve_Out/Logs')
        self.root_p = 'Eve_Out/'
        self.debug_p = os.path.join(self.root_p, 'Debug')
        self.logs_p = os.path.join(self.root_p, 'Logs')


    def _init_containers(self, reagent_df):
        '''
        params:
            df reagent_df: as passed to init
        Postconditions:
            the dictionary, self.containers, has been initialized to have name keys to container
              objects
        '''
        container_types = reagent_df['deck_pos'].apply(lambda d: self.lab_deck[d])
        container_types = reagent_df[['deck_pos','loc']].apply(lambda row: 
                self.lab_deck[row['deck_pos']].get_container_type(row['loc']),axis=1)
        container_types.name = 'container_type'

        for name, conc, loc, deck_pos, mass, container_type in reagent_df.join(container_types).itertuples():
            self.containers[name] = self._construct_container(container_type, name, deck_pos,loc, mass=mass, conc=conc)
    
    def _construct_container(self, container_type, name, deck_pos, loc, **kwargs):
        '''
        params:
            str container_type: the type of container you want to instantiate
            str name: the chemical name
            int deck_pos: labware position on deck
            str loc: the location on the labware
          **kwargs:
            float mass: the mass of the starting contents
            float conc: the concentration of the starting components
        returns:
            Container: a container object of the type you specified
        '''
        if container_type == 'Tube2000uL':
            return Tube2000uL(name, deck_pos, loc, **kwargs)
        elif container_type == 'Tube20000uL':
            return Tube20000uL(name, deck_pos, loc, **kwargs)
        elif container_type == 'Tube50000uL':
            return Tube50000uL(name, deck_pos, loc, **kwargs)
        elif container_type == 'Well96':
            #Note we don't yet have a way to specify volume since we assumed that we would
            #always be weighing in the input template. Future feature allows volume to be
            #specified in sheets making this last step more interesting
            return Well96(name, deck_pos, loc, **kwargs)
        else:
            raise Exception('Invalid container type')
       
    def _init_params(self):
        '''
        TODO: if this still just initializes speed when we're done, it should be named such
        '''
        self.protocol.max_speeds['X'] = 250
        self.protocol.max_speeds['Y'] = 250

    def _init_temp_mod(self, name, using_temp_ctrl, temp, deck_pos, empty_tubes):
        '''
        initializes the temperature module
        params:
            str name: the common name of the labware
            bool using_temp_ctrl: true if using temperature control
            float temp: the temperature you want it at
            int deck_pos: the deck_position of the temperature module
            list<tup<str, float>> empty_tubes: the empty_tubes associated with this tube holder
              the tuple holds the name of the tube and the volume associated with it
        Postconditions:
            the temperature module has been initialized
            the labware wrapper for these tubes has been initialized and added to the deck
        '''
        if using_temp_ctrl:
            temp_module = self.protocol.load_module('temperature module gen2', 3)
            temp_module.set_temperature(temp)
            opentrons_name = self._LABWARE_TYPES[name]['opentrons_name']
            labware = temp_module.load_labware(opentrons_name,label=name)
            #this will always be a tube holder
            self._add_to_deck(name, deck_pos, labware, empty_containers=empty_tubes)


    def _init_custom_labware(self, name, deck_pos, **kwargs):
        '''
        initializes custom built labware by reading from json
        initializes the labware_deck
        params:
            str name: the common name of the labware
            str deck_pos: the position on the deck for the labware
        kwargs:
            NOTE this is really here for compatibility since it's just one keyword that should
            always be passed. It's here in case we decide to use other types of labware in the
            future
            str first_well: the first available well in the labware
        '''
        with open(self._LABWARE_TYPES[name]['definition_path'], 'r') as labware_def_file:
            labware_def = json.load(labware_def_file)
        labware = self.protocol.load_labware_from_definition(labware_def, deck_pos,label=name)
        self._add_to_deck(name, deck_pos, labware, **kwargs)

    def _add_to_deck(self, name, deck_pos, labware, **kwargs):
        '''
        constructs the appropriate labware object
        params:
            str name: the common name for the labware
            int deck_pos: the deck position of the labware object
            Opentrons.labware: labware
            kwargs:
                list empty_containers<str>: the list of the empty locations on the labware
                str first_well: the first available well in the labware
        Postconditions:
            an entry has been added to the lab_deck
        '''
        if 'tube_holder' in self._LABWARE_TYPES[name]['groups']:
            self.lab_deck[deck_pos] = TubeHolder(labware, kwargs['empty_containers'], deck_pos)
        elif 'well_plate' in self._LABWARE_TYPES[name]['groups']:
            self.lab_deck[deck_pos] = WellPlate(labware, kwargs['first_well'], deck_pos)
        else:
            raise Exception("Sorry, Illegal Labware Option. Your labware is not a tube or plate")

    def _init_labware(self, labware_df, using_temp_ctrl, temp):
        '''
        initializes the labware objects in the protocol and pipettes.
        params:
            df labware_df: as recieved in __init__
        Postconditions:
            The deck has been initialized with labware
        '''
        for deck_pos, name, first_usable, empty_list in labware_df.itertuples(index=False):
            #diff types of labware need diff initializations
            if self._LABWARE_TYPES[name]['definition_path']:
                #plate readers (or other custom?)
                self._init_custom_labware(name, deck_pos, first_well=first_usable)
            elif 'temp_mod' in self._LABWARE_TYPES[name]['definition_path']:
                #temperature controlled racks
                self._init_temp_mod(name, using_temp_ctrl, 
                        temp, deck_pos, empty_tubes=empty_list)
            else:
                #everything else
                opentrons_name = self._LABWARE_TYPES[name]['opentrons_name']
                labware = self.protocol.load_labware(opentrons_name,deck_pos,label=name)
                if 'well_plate' in self._LABWARE_TYPES[name]['groups']:
                    self._add_to_deck(name, deck_pos, labware, first_well=first_usable)
                elif 'tube_holder' in self._LABWARE_TYPES[name]['groups']:
                    self._add_to_deck(name, deck_pos, labware, empty_containers=empty_list)
                #if it's none of the above, it's a tip rack. We don't need them on the deck

        
    def _init_instruments(self,instruments, labware_df):
        '''
        initializes the opentrons instruments (pipettes) and sets first tips for pipettes
        params:
            Dict<str:str> instruments: as recieved in __init__
            df labware_df: as recieved in __init__
        Postconditions:
            the pipettes have been initialized and 
            tip racks have been given first tips
        '''
        for arm_pos, pipette_name in instruments.items():
            #lookup opentrons name
            opentrons_name = self._PIPETTE_TYPES[pipette_name]['opentrons_name']
            #get the size of this pipette
            pipette_size = pipette_name[:pipette_name.find('uL')]
            #get the row inds for which the size is the same
            tip_row_inds = labware_df['name'].apply(lambda name: 
                'tip_rack' in self._LABWARE_TYPES[name]['groups'] and pipette_size == 
                name[name.rfind('_')+1:name.rfind('uL')])
            tip_rows = labware_df.loc[tip_row_inds]
            #get the opentrons tip rack objects corresponding to the deck positions that
            #have tip racks
            tip_racks = [self.protocol.loaded_labwares[deck_pos] for deck_pos in tip_rows['deck_pos']]
            #load the pipette
            pipette = self.protocol.load_instrument(opentrons_name,arm_pos,tip_racks=tip_racks)
            #get the row with the largest lexographic starting tip e.g. (B1 > A0)
            #and then get the deck position
            #this is the tip rack that has used tips
            used_rack_row = tip_rows.loc[self._lexo_argmax(tip_rows['first_usable'])]
            #get opentrons object
            used_rack = self.protocol.loaded_labwares[used_rack_row['deck_pos']]
            #set starting tip
            pipette.starting_tip = used_rack.well(used_rack_row['first_usable'])
            pipette.pick_up_tip()
            #update self.pipettes
            self.pipettes[arm_pos] = {'size':float(pipette_size),'last_used':'clean','pipette':pipette}
        return

    def _lexo_argmax(self, s):
        '''
        pandas does not have a lexographic idxmax, so I have supplied one
        Params:
            pd.Series s: a series of strings to be compared lexographically
        returns:
            Object: the pandas index associated with that string
        '''
        max_str = ''
        max_idx = None
        for i, val in s.iteritems():
            max_str = max(max_str, val)
            max_idx = i
        return i
 
    def _exec_init_containers(self, product_df):
        '''
        used to initialize empty containers, which is useful before transfer steps to new chemicals
        especially if we have preferences for where those chemicals are put
        Params:
            df product_df: as generated in client init_robot
        Postconditions:
            every container has been initialized according to the parameters specified
        '''
        for chem_name, req_labware, req_container, max_vol in product_df.itertuples():
            container = None
            #if you've already initialized this complane
            if chem_name in self.containers:
                raise Exception("you tried to initialize {},\
                        but there is already an entry for {}".format(chem_name, chem_name))
            #filter labware
            viable_labware =[]
            for viable in self.lab_deck:
                if viable:
                    labware_ok = not req_labware or (viable.name == req_labware or \
                            req_labware in self._LABWARE_TYPES[viable.name]['groups'])
                            #last bit necessary for platereader-> platereader4/platereader7
                    container_ok = not req_container or (req_container in viable.CONTAINERS_SERVICED)
                    if labware_ok and container_ok:
                        viable_labware.append(viable)
            #sort the list so that platreader slots are prefered
            viable_labware.sort(key=lambda x: self._exec_init_containers.priority[x.name])
            #iterate through the filtered labware and pick the first one that 
            loc, deck_pos, container_type  = None, None, None
            i = 0
            while not loc:
                try:
                    viable = viable_labware[i]
                except IndexError: 
                    message = 'No containers to put {} with labware type {} and container type \
                            {} with maximum volume {}.'.format(chem_name, \
                            req_labware, req_container, max_vol) 
                    raise Exception(message)
                next_container_loc = viable.pop_next_well(vol=max_vol,container_type=req_container)
                if next_container_loc:
                    #that piece of labware has space for you
                    loc = next_container_loc
                    deck_pos = viable.deck_pos
                    container_type = viable.get_container_type(loc)
                i += 1
            self.containers[chem_name] = self._construct_container(container_type, 
                    chem_name, deck_pos, loc)


    #a dictionary to assign priorities to different labwares. Right now used only to prioritize
    #platereader when no other labware has been specified
    _exec_init_containers.priority = defaultdict(lambda: 100)
    _exec_init_containers.priority['platereader4'] = 1
    _exec_init_containers.priority['platereader7'] = 2

    def execute(self, command_type, cid, arguments):
        '''
        takes the packet type and payload of an Armchair packet, and executes the command
        params:
            str command_type: the type of packet to execute
            tuple<Obj> arguments: the arguments to this command 
              (generally passed as list so no *args)
        returns:
            int: 1=ready to recieve. 0=terminated
        Postconditions:
            the command has been executed
        '''
        if command_type == 'transfer':
            self._exec_transfer(*arguments)
            self.portal.send_pack('ready', cid)
            return 1
        elif command_type == 'init_containers':
            self._exec_init_containers(pd.DataFrame(arguments[0]))
            self.portal.send_pack('ready', cid)
            return 1
        elif command_type == 'pause':
            self._exec_pause(arguments[0])
            self.portal.send_pack('ready', cid)
            return 1
        elif command_type == 'stop':
            self._exec_stop()
            self.portal.send_pack('ready', cid)
            return 1
        elif command_type == 'close':
            self._exec_close()
            return 0
        else:
            raise Exception("Unidenified command {}".format(pack_type))

    def _exec_pause(self,pause_time):
        '''
        executes a pause command by waiting for 'time' seconds
        params:
            float pause_time: time to wait in seconds
        '''
        #no need to pause for a simulation
        if not self.simulate:
            time.sleep(pause_time)

    def _exec_transfer(self, src, transfer_steps, callbacks):
        '''
        this command executes a transfer. It's usually pretty simple, unless you have
        a stop callback. If you have a stop callback it launches a new TCP connection and
        stops to wait for user input at each transfer
        params:
            str src: the chem_name of the source well
            list<tuple<str,float>> transfer_steps: each element is a dst, vol pair
            list<str> callbacks: the ordered callbacks to perform after each transfer or None
        '''
        #check to make sure that both tips are not dirty with a chemical other than the one you will pipette
        for arm in self.pipettes.keys():
            if self.pipettes[arm]['last_used'] not in ['WaterC1.0', 'clean', src]:
                self._get_clean_tips()
                break; #cause now they're clean
        callback_types = [callback for callback, _ in callbacks]
        #if you're going to be altering flow, you need to create a seperate connection with the
        #controller
        if 'stop' in callback_types:
            sock = socket.socket(socket.AF_INET)
            fail_count=0
            connected = False
            while not connected and fail_count < 3:
                try:
                    sock.connect((self.controller_ip, 50003))
                    connected = True
                except ConnectionRefusedError:
                    fail_count += 1
                    time.sleep(2**fail_count)
        for dst, vol in transfer_steps:
            self._transfer_step(src,dst,vol)
            new_tip=False #don't want to use a new tip_next_time
            if callbacks:
                for callback, args in callbacks:
                    if callback == 'pause':
                        self._exec_pause(args[0])
                    elif callback == 'stop':
                        self._exec_stop()

    def _exec_stop(self):
        '''
        executes a stop command by creating a TCP connection, telling the controller to get
        user input, and then waiting for controller response
        '''
        self.portal.send_pack('stopped')
        pack_type, _, _ = self.portal.recv_pack()
        assert (pack_type == 'continue'), "Was stopped waiting for continue, but recieved, {}".format(pack_type)

    def _transfer_step(self, src, dst, vol):
        '''
        used to execute a single tranfer from src to dst. Handles things like selecting
        appropriately sized pipettes. If you need
        more than 1 step, will facilitate that
        params:
            str src: the chemical name to pipette from
            str dst: thec chemical name to pipette into
            float vol: the volume to pipette
        '''
        #choose your pipette
        arm = self._get_preffered_pipette(vol)
        n_substeps = int(vol // self.pipettes[arm]['size']) + 1
        substep_vol = vol / n_substeps
        
        #transfer the liquid in as many steps are necessary
        for i in range(n_substeps):
            self._liquid_transfer(src, dst, substep_vol, arm)
        return

    def _get_clean_tips(self):
        '''
        checks if the both tips to see if they're dirty. Drops anything that's dirty, then picks
        up clean tips
        params:
            str ok_chems: if you're ok reusing the same tip for this chemical, no need to replace
        TODO: Wrap in try for running out of tips
        '''
        drop_list = [] #holds the pipettes that were dirty
        #drop first so no sprinkles get on rack while picking up
        ok_list = ['clean','WaterC1.0']
        for arm in self.pipettes.keys():
            if self.pipettes[arm]['last_used'] not in ok_list:
                self.pipettes[arm]['pipette'].drop_tip()
                drop_list.append(arm)
        #now that you're clean, you can pick up new tips
        for arm in drop_list:
            self.pipettes[arm]['pipette'].pick_up_tip()
            self.pipettes[arm]['last_used'] = 'clean'

    def _get_preffered_pipette(self, vol):
        '''
        returns the pipette with size, or one smaller
        params:
            float vol: the volume to be transfered in uL
        returns:
            str: in ['right', 'left'] the pipette arm you're to use
        '''
        preffered_size = 0
        if vol < 40.0:
            preffered_size = 20.0
        elif vol < 1000:
            preffered_size = 300.0
        else:
            preffered_size = 1000.0
        
        #which pipette arm has a larger pipette?
        larger_pipette=None
        if self.pipettes['right']['size'] < self.pipettes['left']['size']:
            larger_pipette = 'left'
            smaller_pipette = 'right'
        else:
            larger_pipette = 'right'
            smaller_pipette = 'left'

        FUDGE_FACTOR = 0.0001
        if self.pipettes[larger_pipette]['size'] <= preffered_size + FUDGE_FACTOR:
            #if the larger one is small enough return it
            return larger_pipette
        else:
            #if the larger one is too large return the smaller
            return smaller_pipette

    def _liquid_transfer(self, src, dst, vol, arm):
        '''
        the lowest of the low. Transfer liquid from one container to another. And mark the tip
        as dirty with src, and update the volumes of the containers it uses
        params:
            str src: the chemical name of the source container
            str dst: the chemical name of the destination container
            float vol: the volume of liquid to be transfered
            str arm: the robot arm to use for this transfer
        Postconditions:
            vol uL of src has been transfered to dst
            pipette has been adjusted to be dirty with src
            volumes of src and dst have been updated
        Preconditions:
            Both pipettes are clean or of same type
        '''
        for arm_to_check in self.pipettes.keys():
            #this is really easy to fix, but it should not be fixed here, it should be fixed in a
            #higher level function call. This is minimal step for maximum speed.
            assert (self.pipettes[arm_to_check]['last_used'] in ['clean', 'WaterC1.0', src]), "trying to transfer {}->{}, with {} arm, but {} arm was dirty with {}".format(src, dst, arm, arm_to_check, self.pipettes[arm_to_check]['last_used'])
        self.protocol._commands.append('HEAD: {} : transfering {} to {}'.format(datetime.now().strftime('%d-%b-%Y %H:%M:%S:%f'), src, dst))
        pipette = self.pipettes[arm]['pipette']
        src_cont = self.containers[src] #the src container
        dst_cont = self.containers[dst] #the dst container
        assert (src_cont.vol >= vol),'{} cannot transfer {} to {} because it only has {}uL'.format(src,vol,dst,src_cont.aspiratible_vol)
        #It is not necessary to check that the dst will not overflow because this is done when
        #containers are initialized
        #set aspiration height
        pipette.well_bottom_clearance.aspirate = self.containers[src].asp_height
        #aspirate(well_obj)
        pipette.aspirate(vol, self.lab_deck[src_cont.deck_pos].get_well(src_cont.loc))
        #update the vol of src
        src_cont.update_vol(-vol)
        #pipette is now dirty
        self.pipettes[arm]['last_used'] = src
        #touch tip
        pipette.touch_tip()
        #maybe move up if clipping
        #set dispense height 
        pipette.well_bottom_clearance.dispense = self.containers[dst].disp_height
        #dispense(well_obj)
        pipette.dispense(vol, self.lab_deck[dst_cont.deck_pos].get_well(dst_cont.loc))
        #update vol of dst
        dst_cont.update_vol(vol,src)
        #blowout
        for i in range(4):
            pipette.blow_out()
        #wiggle - touch tip (spin fast inside well)
        pipette.touch_tip(radius=0.3,speed=40)

    def dump_well_map(self):
        '''
        dumps the well_map to a file
        '''
        path=os.path.join(self.logs_p,'wellmap.tsv')
        names = self.containers.keys()
        locs = [self.containers[name].loc for name in names]
        deck_poses = [self.containers[name].deck_pos for name in names]
        vols = [self.containers[name].vol for name in names]
        def lookup_container_type(name):
            container = self.containers[name]
            labware = self.lab_deck[container.deck_pos]
            return labware.get_container_type(container.loc)
        container_types = [lookup_container_type(name) for name in names]
        well_map = pd.DataFrame({'chem_name':list(names), 'loc':locs, 'deck_pos':deck_poses, 
                'vol':vols,'container':container_types})
        well_map.sort_values(by=['deck_pos', 'loc'], inplace=True)
        well_map.to_csv(path, index=False, sep='\t')

    def dump_protocol_record(self):
        '''
        dumps the protocol record to tsv
        '''
        path=os.path.join(self.logs_p, 'protocol_record.txt')
        command_str = ''.join(x+'\n' for x in self.protocol.commands())[:-1]
        with open(path, 'w') as command_dump:
            command_dump.write(command_str)

    def dump_well_histories(self):
        '''
        gathers the history of every reaction and puts it in a single df. Writes that df to file
        '''
        path=os.path.join(self.logs_p, 'well_history.tsv')
        histories=[]
        for name, container in self.containers.items():
            df = pd.DataFrame(container.history, columns=['timestamp', 'chemical', 'vol'])
            df['container'] = name
            histories.append(df)
        all_history = pd.concat(histories, ignore_index=True)
        all_history['timestamp'] = pd.to_datetime(all_history['timestamp'], format='%d-%b-%Y %H:%M:%S:%f')
        all_history.sort_values(by=['timestamp'], inplace=True)
        all_history.reset_index(inplace=True, drop=True)
        all_history.to_csv(path, index=False, sep='\t')

    def exception_handler(self, e):
        '''
        code to handle all exceptions.
        Procedure:
            Dump a locations of all of the chemicals
        '''
        pass

    def _exec_close(self):
        '''
        close the connection in a nice way
        '''
        print('<<eve>> initializing breakdown')
        for arm_dict in self.pipettes.values():
            pipette = arm_dict['pipette']
            pipette.drop_tip()
        self.protocol.home()
        #write logs
        self.dump_protocol_record()
        self.dump_well_histories()
        self.dump_well_map()
        #ship logs
        filenames = list(os.listdir(self.logs_p))
        port = 50001 #default port for ftp 
        self.send_files(port, filenames)
        #kill link
        print('<<eve>> shutting down')
        self.portal.send_pack('close')
        self.portal.close()

    def send_files(self,port,filenames):
        '''
        used to ship files back to server
        params:
            int port: the port number to ship the files out of
            list<str> filepaths: the filepaths to ship
        '''
        #setting up a socket for FTP
        sock = socket.socket(socket.AF_INET)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.my_ip, port))
        sock.listen(5)
        #send ok to server that you are ready to accept
        print('<<eve>> initializing filetransfer')
        filepaths = [os.path.join(self.logs_p, filename) for filename in filenames]
        self.portal.send_pack('sending_files', port, filenames)
        client_sock, client_addr = sock.accept()
        for filepath in filepaths:
            with open(filepath,'rb') as local_file:
                client_sock.sendfile(local_file)
                client_sock.send(armchair.FTP_EOF)
            #wait until client has sent that they're ready to recieve again
        client_sock.close()
        sock.close()


def launch_eve_server(**kwargs):
    '''
    launches an eve server to create robot, connect to controller etc
    **kwargs:
        str my_ip: the ip address to launch the server on. required arg
        threading.Barrier barrier: if specified, will launch as a thread instead of a main
    '''
    my_ip = kwargs['my_ip']
    PORT_NUM = 50000
    #construct a socket
    sock = socket.socket(socket.AF_INET)
    sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    sock.bind((my_ip,PORT_NUM))
    print('<<eve>> listening on port {}'.format(PORT_NUM))
    sock.listen(5)
    if kwargs['barrier']:
        #running in thread mode with barrier. Barrier waits for both threads
        kwargs['barrier'].wait()
    client_sock, client_addr = sock.accept()
    print('<<eve>> connected')
    buffered_sock = BufferedSocket(client_sock, timeout=None)
    portal = Armchair(buffered_sock,'eve','Armchair_Logs')
    eve = None
    pack_type, cid, args = portal.recv_pack()
    if pack_type == 'init':
        simulate, using_temp_ctrl, temp, labware_df, instruments, reagents_df, controller_ip = args
        #I don't know why this line is needed, but without it, Opentrons crashes because it doesn't
        #like to be run from a thread
        asyncio.set_event_loop(asyncio.new_event_loop())
        eve = OT2Controller(simulate, using_temp_ctrl, temp, labware_df, instruments, reagents_df,my_ip, controller_ip, portal)
        portal.send_pack('ready', cid)
    connection_open=True
    while connection_open:
        pack_type, cid, payload = portal.recv_pack()
        connection_open = eve.execute(pack_type, cid, payload)
    sock.close()
    return

def make_unique(s):
    '''
    makes every element in s unique by adding _1, _2, ...
    params:
        pd.Series s: a series of elements with some duplicates
    returns:
        pd.Series: s with any duplicates made unique by adding a count
    e.g.
    ['sloth', 'gorilla', 'sloth'] becomes ['sloth_1', 'gorilla', 'sloth_2']
    '''
    val_counts = s.value_counts()
    duplicates = val_counts.loc[val_counts > 1]
    def _get_new_name(name):
        #mini helper func for the apply
        if name in duplicates:
            i = duplicates[name]
            duplicates[name] -= 1
            return "{}_{}".format(name, i)
        else:
            return name
    return s.apply(_get_new_name)
