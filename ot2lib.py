from abc import ABC
from abc import abstractmethod
import datetime
import json
import dill

import gspread
from df2gspread import df2gspread as d2g
from df2gspread import gspread2df as g2d
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import opentrons.execute
from opentrons import protocol_api, simulate, types
import webbrowser
from tempfile import NamedTemporaryFile

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
def pre_rxn_questions():
    '''
    asks user for control params
    returns:
        bool simulate: true if behaviour is to be simulated instead of executed
        str rxn_sheet_name: the title of the google sheet
        bool using_temp_ctrl: true if planning to use temperature ctrl module
        float temp: the temperature you want to keep the module at
    '''
    simulate = (input('Simulate or Execute: ').lower() == 'simulate')
    rxn_sheet_name = input('Enter Sheet Name as it Appears on the Spreadsheets Title: ')
    temp_ctrl_response = input('Are you using the temperature control module, yes or no?\
    (if yes, turn it on before responding): ').lower()
    using_temp_ctrl = ('y' == temp_ctrl_response or 'yes' == temp_ctrl_response)
    temp = None
    if using_temp_ctrl:
        temp = input('What temperature in Celcius do you want the module \
        set to? \n (the protocol will not proceed until the set point is reached) \n')
    return simulate, rxn_sheet_name, using_temp_ctrl, temp

def open_sheet(rxn_sheet_name, credentials):
    '''
    open the google sheet
    params:
        str rxn_sheet_name: the title of the sheet to be opened
        oauth2client.ServiceAccountCredentials credentials: credentials read from a local json
    returns:
        gspread.Spreadsheet the spreadsheet (probably of all the reactions)

    '''
    gc = gspread.authorize(credentials)
    try:
        wks = gc.open(rxn_sheet_name)
        print('Everythings Ready To Go')
    except: 
        raise Exception('Spreadsheet Not Found: Make sure the spreadsheet name is spelled correctly and that it is shared with the robot ')
    return wks

def init_credentials(rxn_sheet_name):
    '''
    this function reads a local json file to get the credentials needed to access other funcs
    '''
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    #get login credentials from local file. Your json file here
    path = 'Credentials/hendricks-lab-jupyter-sheets-5363dda1a7e0.json'
    credentials = ServiceAccountCredentials.from_json_keyfile_name(path, scope) 
    return credentials

def get_wks_key(credentials, rxn_sheet_name):
    '''
    open and search a sheet that tells you which sheet is associated with the reaction
    '''
    gc = gspread.authorize(credentials)
    name_key_wks = gc.open_by_url('https://docs.google.com/spreadsheets/d/1m2Uzk8z-qn2jJ2U1NHkeN7CJ8TQpK3R0Ai19zlAB1Ew/edit#gid=0').get_worksheet(0)
    name_key_pairs = name_key_wks.get_all_values() #list<list<str name, str key>>
    #Note the key is a unique identifier that can be used to access the sheet
    #d2g uses it to access the worksheet
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

def load_rxn_df(rxn_spreadsheet, rxn_sheet_name):
    '''
    reaches out to google sheets and loads the reaction protocol into a df and formats the df
    adds a chemical name (primary key for lots of things. e.g. robot dictionaries)
    renames some columns to code friendly as opposed to human friendly names
    params:
        gspread.Spreadsheet rxn_spreadsheet: the sheet with all the reactions
        str rxn_sheet_name: the name of the spreadsheet
    returns:
        pd.DataFrame: the information in the rxn_spreadsheet w range index. spreadsheet cols
    '''
    rxn_wks = rxn_spreadsheet.get_worksheet(0)
    data = rxn_wks.get_all_values()
    rxn_df = pd.DataFrame(data[1:], columns=data[0])
    #rename some of the clunkier columns 
    rxn_df.rename({'operation':'op', 'concentration (mM)':'conc', 'reagent (must be uniquely named)':'reagent', 'Pause before addition?':'pause', 'comments (e.g. new bottle)':'comments'}, axis=1, inplace=True)
    rxn_df.drop(columns=['comments'], inplace=True)#comments are for humans
    rxn_df.replace('', np.nan,inplace=True)
    rxn_df['chemical_name'] = rxn_df[['conc', 'reagent']].apply(get_chemical_name,axis=1)
    return rxn_df

def get_chemical_name(row):
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


def init_robot(portal, rxn_spreadsheet, rxn_df, simulate, spreadsheet_key, credentials):
    '''
    This function gets the unique reagents, interfaces with the docs to get details on those
      reagents, and ships that information to the robot so it can initialize it's labware dicts
    '''
    #pull labware etc from the sheets
    labware_df, instruments = get_labware_info(rxn_spreadsheet)

    #get unique reagents
    reagents_and_conc = rxn_df[['chemical_name', 'conc']].groupby('chemical_name').first()

    #query the docs for more info on reagents
    #DEBUG not overwriting vals for testing purposes
    #construct_reagent_sheet(reagents_and_conc, spreadsheet_key, credentials)
    input("please press enter when you've completed the reagent sheet")

    #pull the info into a df
    reagents = g2d.download(spreadsheet_key, 'reagent_info', col_names = True, row_names = True, credentials=credentials)

    #iterate through that thing and send information over armchair in a network friendly way
    portal.send_pack('init', simulate, labware_df, instruments, reagents)

    #return (god willing)
    return

def construct_reagent_sheet(reagent_df, spreadsheet_key, credentials):
    '''
    query the user with a reagent sheet asking for more details on locations of reagents, mass
    etc
    Preconditions:
        see excel_spreadsheet_preconditions.txt
    PostConditions:
        reagent_sheet has been constructed
    '''
    reagent_df[['content_type', 'product', 'mass', 'positions', 'deck_pos', 'reagent_to_dilute', 'desired_vol', 'comments']] = ''
    d2g.upload(reagent_df.reset_index(),spreadsheet_key,wks_name = 'reagent_info', row_names=False , credentials = credentials)


def get_reagent_attrs(row):
    '''
    proccess the df into an index of unique col names corresponding to volumes
    from a row in a pandas df. (can be just the two columns, ['conc', 'reagent'])
    params:
        pd.Series row: a row in the rxn_df
    returns:
        pd.series:
            Elements
            chemical_name: the name for the chemical "{}C{}".format(name, conc) or name if
              has no concentration, or nan if no name
            conc: the concentration of the chemical (only applicable to solutions)
            type: the type of substance this is (a classname in ot2_server_lib.py
    '''
    new_cols = {}
    if pd.isnull(row['reagent']):
        #this must not be a transfer. this operation has no chemical name
        new_cols['chemical_name'] = np.nan
        new_cols['conc'] = np.nan
        new_cols['content_type'] = np.nan
    elif pd.isnull(row['conc']):
        #this uses a chemical, but the chemical doesn't have a concentration (probably a mix)
        new_cols['chemical_name'] = row['conc'].replace(' ', '_')
        new_cols['conc'] = np.nan
        new_cols['content_type'] = 'mix'
    else:
        #this uses a chemical with a conc. Probably a stock solution
        new_cols['chemical_name'] = "{}C{}".format(row['reagent'], row['conc']).replace(' ', '_')
        new_cols['conc'] = row['conc']
        new_cols['content_type'] = 'dilution'
    return pd.Series(new_cols)

def get_labware_info(rxn_spreadsheet):
    '''
    Interface with sheets to get information about labware locations, first tip, etc.
    Preconditions:
        The second sheet in the worksheet must be initialized with where you've placed reagents 
        and the first thing not being used
    params:
        gspread.Spreadsheet rxn_spreadsheet: a spreadsheet object with second sheet having
          deck positions
    returns:
        df:
          str name: the common name of the labware (made unique 
        Dict<str:str>: key is 'left' or 'right' for the slots. val is the name of instrument
    '''
    #create labware df
    raw_labware_data = rxn_spreadsheet.get_worksheet(1).get_all_values()
    #the format google fetches this in is funky, so we convert it into a nice df
    labware_dict = {'name':[], 'first_tip':[],'deck_pos':[]}
    for row_i in range(0,10,3):
        for col_i in range(3):
            labware_dict['name'].append(raw_labware_data[row_i+1][col_i])
            labware_dict['first_tip'].append(raw_labware_data[row_i+2][col_i])
            labware_dict['deck_pos'].append(raw_labware_data[row_i][col_i])
    labware_df = pd.DataFrame(labware_dict)
    labware_df = labware_df.loc[labware_df['name'] != ''] #remove empty slots
    labware_df.reset_index(drop=True, inplace=True)
    labware_df['deck_pos'] = pd.to_numeric(labware_df['deck_pos'])
    #make instruments
    instruments = {}
    instruments['left'] = raw_labware_data[13][0]
    instruments['right'] = raw_labware_data[13][1]
    return labware_df, instruments

#SERVER
class Container(ABC):
    """
    
    Abstract container class to be overwritten for well, tube, etc.
    ABSTRACT ATTRIBUTES:
        str name: the common name we use to refer to this container
        float vol: the volume of the liquid in this container in uL
        Obj Labware: a pointer to the Opentrons Labware object of which this is a part
        str loc: a location on the labware object (e.g. 'A5')
    ABSTRACT METHODS:
        _update_height void: updates self.height to hieght at which to pipet (a bit below water line)
    IMPLEMENTED METHODS:
        update_vol(float aspvol) void: updates the volume upon an aspiration
    """

    def __init__(self, name, vol, labware, loc):
        self.name = name
        self.contents = Contents(vol)
        self.labware = labware
        self.loc = loc
        self.height = self._update_height()
        self.vol = vol

    @abstractmethod
    def _update_height(self):
        pass

    def update_vol(self, aspvol):
        self.contents.update_vol(aspvol)
        self._update_height()



    
class SmallTube(Container):
    #TODO update class name to be reflective of size
    """
    Spcific tube with measurements taken to provide implementations of abstract methods
    INHERITED ATTRIBUTES
        str name, float vol, Obj Labware, str loc
    INHERITED METHODS
        _update_height void, update_vol(float aspvol) void,
    """
    
    def __init__(self, name, mass, labware, loc):
        density_water_25C = 0.9970479 # g/mL
        avg_tube_mass15 = 6.6699 # grams
        reagent_mass = mass - avg_tube_mass15 # N = 1 (in grams) 
        vol = (reagent_mass / density_water_25C) * 1000 # converts mL to uL
        super().__init__(name, vol, labware, loc)
       # 15mm diameter for 15 ml tube  -5: Five mL mark is 19 mm high for the base/noncylindrical protion of tube 

    def _update_height(self):
        diameter_15 = 14.0 # mm (V1 number = 14.4504)
        vol_bottom_cylinder = 2000 # uL
        height_bottom_cylinder = 30.5  #mm
        tip_depth = 5 # mm
        self.height = ((self.contents.vol - vol_bottom_cylinder)/(math.pi*(diameter_15/2)**2))+(height_bottom_cylinder - tip_depth)
            
class BigTube(Container):
    #TODO update class name to be reflective of size
    """
    Spcific tube with measurements taken to provide implementations of abstract methods
    INHERITED ATTRIBUTES
        str name, float vol, Obj Labware, str loc
    INHERITED METHODS
        _update_height void, update_vol(float aspvol) void,
    """
    def __init__(self, name, mass, labware, loc):
        density_water_25C = 0.9970479 # g/mL
        avg_tube_mass50 = 13.3950 # grams
        reagent_mass = mass - avg_tube_mass50 # N = 1 (in grams) 
        vol = (reagent_mass / density_water_25C) * 1000 # converts mL to uL
        super().__init__(name,vol, labware, loc)
       # 15mm diameter for 15 ml tube  -5: Five mL mark is 19 mm high for the base/noncylindrical protion of tube 
        
    def _update_height(self):
        diameter_50 = 26.50 # mm (V1 number = 26.7586)
        vol_bottom_cylinder = 5000 # uL
        height_bottom_cylinder = 21 #mm
        tip_depth = 5 # mm
        self.height = ((self.contents.vol - vol_bottom_cylinder)/(math.pi*(diameter_50/2)**2)) + (height_bottom_cylinder - tip_depth) 

class tube2000uL(Container):
    """
    2000uL tube with measurements taken to provide implementations of abstract methods
    INHERITED ATTRIBUTES
         str name, float vol, Obj Labware, str loc
    INHERITED METHODS
        _update_height void, update_vol(float aspvol) void,
    """
    
    def __init__(self, name, mass, labware, loc):
        density_water_4C = 0.9998395 # g/mL
        avg_tube_mass2 =  1.4        # grams
        reagent_mass = mass - avg_tube_mass2 # N = 1 (in grams) 
        vol = (self.mass / density_water_4C) * 1000 # converts mL to uL
        super().__init__(name, vol, labware, loc)
           
    def _update_height(self):
        diameter_2 = 8.30 # mm
        vol_bottom_cylinder = 250 #uL
        height_bottom_cylinder = 10.5 #mm
        tip_depth = 4.5 # mm
        self.height = ((self.contents.vol - vol_bottom_cylinder)/(math.pi*(diameter_2/2)**2)) + (height_bottom_cylinder - tip_depth)

class well96(Container):
    """
        a well in a 96 well plate
        INHERITED ATTRIBUTES
             str name, float vol, Obj Labware, str loc
        INHERITED METHODS
            _update_height void, update_vol(float aspvol) void,
    """

    def __init__(self, name, labware, loc, vol=0):
        #vol is defaulted here because the well will probably start without anything in it
        vol = (self.mass / density_water_4C) * 1000 # converts mL to uL
        super().__init__(name, vol, labware, loc)
           
    def _update_height(self):
        #TODO develop an update height method for wells.
        pass

#TODO right now Contents is created inside the constructor of the container, but contents should
#be passed in pre baked. This requires the code for initial stoicheometry to be moved to a main
#in the initialization code of this or maybe the client
#TODO docs not updated on these
class Contents(ABC):
    """
    There is a difference between a reagent and a product, and oftentimes it has to do with
    reagents being dilutions and products being mixtures. Products need information
    about what is in them, dilutions need concentrations. These attributes relate to
    the content of the container, not the container itself. Every container has a
    substance inside it.
    This is a purely architectural class. It exists to preserve the seperation of things.
    ABSTRACT ATTRIBUTES:
        float conc: the concentration (varies in meaning based on child)
        float mass: the mass of the reagent
    METHODS:
        update_vol(float aspvol) void: updates the volume of the contents left
    """
    def __init__(self, conc, mass):
        self.conc=conc
        self.mass=mass

class Stock(Contents):
    """
    A contents with a concentration meant to represent a stock solution
    INHERITED ATTRIBUTES:
        float vol
    ATTRIBUTES:
        float conc: the concentration
    INHERITED METHODS:
        update_vol(float aspvol)
    """
    def __init__(self, conc, mass, name, molecular_weight):
        self.molecular_weight=molecular_weight
        super().__init__(conc,mass)

class Mixture(Contents):
    """
    This is probably a product or seed
    It keeps track of what is in it
    INHERITED ATTRIBUTES:
        float vol
    ATTRIBUTES:
        list<tup<str, vol>> components: reagents and volumes in this mixture
    INHERITED METHODS:
        update_vol(float aspvol)
    """
    def __init__(self, conc, mass, components):
        self.components = components
        super().__init__(conc,mass)
        return

    def add(self, name, vol,timestamp=datetime.datetime.now()):
        #TODO you'll wnat to format the timestamp however you like
        self.vol += vol
        self.components += (timestamp, name, vol)
        return

class OT2Controller():
    """
    The big kahuna. This class contains all the functions for controlling the robot
    ATTRIBUTES:
        Dict<str, str> OPENTRONS_LABWARE_NAMES: This is a constant dictionary that is used to
          translate from human readable names to opentrons labware object names.
        Dict<str, Container> containers: maps from a common name to a Container object
        Dict<str, Obj> tip_racks: maps from a common name to a opentrons tiprack labware object
        Dict<str, Obj> labware: maps from labware common names to opentrons labware objects. tip racks not included?
        Opentrons...ProtocolContext protocol: the protocol object of this session

    """

    _OPENTRONS_LABWARE_NAMES = {'96_well_plate_1':'corning_96_wellplate_360ul_flat','96_well_plate_2':'corning_96_wellplate_360ul_flat','24_well_plate_1':'corning_24_wellplate_3.4ml_flat','24_well_plate_2':'corning_24_wellplate_3.4ml_flat','48_well_plate_1':'corning_48_wellplate_1.6ml_flat','48_well_plate_2':'corning_48_wellplate_1.6ml_flat','tip_rack_20uL':'opentrons_96_tiprack_20ul','tip_rack_300uL':'opentrons_96_tiprack_300ul','tip_rack_1000uL':'opentrons_96_tiprack_1000ul','tube_holder_10':'opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical'}
    _CUSTOM_LABWARE_DEFINITION_PATHS = {'platereader4':'LabwareDefs/plate_reader_4.json','platereader7':'LabwareDefs/plate_reader_7.json'}
    _OPENTRONS_INSTRUMENT_NAMES = {'20uL_pipette':'p20_single_gen2','300uL_pipette':'p300_single_gen2','1000uL_pipette':'p1000_single_gen2'}
    _TIP_RACK_NAMES = {'tip_rack_20uL', 'tip_rack_300uL','tip_rack_1000uL'}


    def __init__(self, simulate, labware_df, instruments, reagents_df):
        '''
        params:
            bool simulate: if true, the robot will run in simulation mode only
            df labware_df:
                str name: the common name of the labware
                str first_tip: the first tip/well to use
                int deck_pos: the position on the deck of this labware
            Dict<str:str> instruments: keys are ['left', 'right'] corresponding to arm slots. vals
              are the pipette names filled in
            df reagents_df: info on reagents. columns from sheet. See excel specification
                
        postconditions:
            protocol has been initialzied
            containers and tip_racks have been created
            labware has been initialized
            CAUTION: the values of tip_racks and containers must be sent from the client.
              it is the client's responsibility to make sure that these are initialized prior
              to operating with them
        '''
        with open('cached.pkl','wb') as cache:
            dill.dump([simulate, labware_df, instruments, reagents_df], cache)
        self.containers = {}
        if simulate:
            # define version number and define protocol object
            self.protocol = opentrons.simulate.get_protocol_api('2.9')
        else:
            self.protocol = opentrons.execute.get_protocol_api('2.9')
            self.protocol.set_rail_lights(on = True)
            self.protocol.rail_lights_on 
        self.protocol.home() # Homes the pipette tip
        self._init_params()
        self._init_labware(labware_df)
        self._init_instruments(instruments, labware_df)
        breakpoint()
        #CONCLUDED!

    def _init_params(self):
        '''
        TODO: if this still just initializes speed when we're done, it should be named such
        '''
        self.protocol.max_speeds['X'] = 100
        self.protocol.max_speeds['Y'] = 100

    def _init_temp_mod(self):
        pass

    def _init_custom_labware(self, name, deck_pos):
        '''
        initializes custom built labware by reading from json
        TODO: should this take in first_tip? 
        params:
            str name: the common name of the labware
            str deck_pos: the position on the deck for the labware
        '''
        with open(self._CUSTOM_LABWARE_DEFINITION_PATHS[name], 'r') as labware_def_file:
            labware_def = json.load(labware_def_file)
        self.protocol.load_labware_from_definition(labware_def, deck_pos)

    def _init_labware(self, labware_df):
        '''
        initializes the labware objects in the protocol and pipettes.
        params:
            df labware_df: as recieved in __init__
        Postconditions:
            The deck has been initialized with labware
        TODO need to update first tip for wells 
        '''
        for name, first_tip, deck_pos in labware_df.itertuples(index=False):
            if name in self._CUSTOM_LABWARE_DEFINITION_PATHS:
                self._init_custom_labware(name, deck_pos)
            elif name == 'temp_mod':
                self._init_temp_mod()
            else:
                #yes it's slow, but there are 12 cols max
                opentrons_name = self._OPENTRONS_LABWARE_NAMES[name]
                self.protocol.load_labware(opentrons_name,deck_pos,label=name)
        
    def _init_instruments(self,instruments, labware_df):
        '''
        initializes the opentrons instruments (pipettes) and sets first tips for pipettes
        params:
            Dict<str:str> instruments: as recieved in __init__
            df labware_df: as recieved in __init__
        Postconditions:
            the pipettes have been initialized and given first tips
        '''
        for arm_pos, pipette_name in instruments.items():
            #lookup opentrons name
            opentrons_name = self._OPENTRONS_INSTRUMENT_NAMES[pipette_name]
            #get the size of this pipette
            pipette_size = pipette_name[:pipette_name.find('uL')]
            #get the row inds for which the size is the same
            tip_row_inds = labware_df['name'].apply(lambda name: 
                name in self._TIP_RACK_NAMES and pipette_size == 
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
            breakpoint()
            used_rack_row = tip_rows.loc[self._lexo_argmax(tip_rows['first_tip'])]
            #get opentrons object
            used_rack = self.protocol.loaded_labwares[used_rack_row['deck_pos']]
            #set starting tip
            pipette.starting_tip = used_rack.well(used_rack_row['first_tip'])
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





    def exec(self, command_type, args):
        '''
        takes the packet type and payload of an Armchair packet, and executes the command
        params:
            str command_type: the type of packet to execute
            tuple args: the arguments to this command
        Postconditions:
            the command has been executed
        '''
        if command_type == 'init':
            init_labware(args[0])
            init_reagents(args[1])
        else:
            raise Exception("Unidenified command {}".format(pack_type))
        return

    def make_fixed_objects(self):
        '''
        this creates the objects that are not movable (plate reader and temp mod)
        Postconditions:
            
        '''
        if temperature_module_response == 'y' or temperature_module_response == 'yes':
            labware['temp_mod'] = protocol.load_module('temperature module gen2', 3)
            labware['temp_mod'].set_temperature(set_temperature_response)
        #labware['cuvette'] = protocol.load_labware_from_definition(labware_def1, 3)
