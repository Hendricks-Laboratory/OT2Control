from abc import ABC
from abc import abstractmethod
from collections import defaultdict
import datetime
import json
import dill
import math

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

def get_wks_key(credentials, rxn_sheet_name):
    '''
    open and search a sheet that tells you which sheet is associated with the reaction
    params:
        ServiceAccountCredentials credentials: to access the sheets
        str rxn_sheet_name: the name of sheet
    returns:
        str wks_key: the key associated with the sheet. It functions similar to a url
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

def load_rxn_table(rxn_spreadsheet, rxn_sheet_name):
    '''
    reaches out to google sheets and loads the reaction protocol into a df and formats the df
    adds a chemical name (primary key for lots of things. e.g. robot dictionaries)
    renames some columns to code friendly as opposed to human friendly names
    params:
        gspread.Spreadsheet rxn_spreadsheet: the sheet with all the reactions
        str rxn_sheet_name: the name of the spreadsheet
    returns:
        pd.DataFrame: the information in the rxn_spreadsheet w range index. spreadsheet cols
        Dict<str,str>: effectively the 2nd row in excel. Gives labware preferences for products
    '''
    rxn_wks = rxn_spreadsheet.get_worksheet(0)
    data = rxn_wks.get_all_values()
    cols = make_unique(pd.Series(data[0])) 
    rxn_df = pd.DataFrame(data[2:], columns=cols)
    #rename some of the clunkier columns 
    rxn_df.rename({'operation':'op', 'dilution concentration':'dilution_conc','concentration (mM)':'conc', 'reagent (must be uniquely named)':'reagent', 'Pause before addition?':'pause', 'comments (e.g. new bottle)':'comments'}, axis=1, inplace=True)
    rxn_df.drop(columns=['comments'], inplace=True)#comments are for humans
    rxn_df.replace('', np.nan,inplace=True)
    #rename chemical names
    rxn_df['chemical_name'] = rxn_df[['conc', 'reagent']].apply(get_chemical_name,axis=1)
    rename_products(rxn_df)
    #create labware_dict
    cols = rxn_df.columns.to_list()
    product_start_i = cols.index('reagent')+1
    requested_labware = data[1][product_start_i+1:]#add one to account for the first col (labware).
    #in df this is an index, so size cols is one less
    products_to_labware = {product:labware for product, labware in zip(cols[product_start_i:], requested_labware)}
    products = products_to_labware.keys()
    #make the reagent columns floats
    rxn_df.loc[:,products] =  rxn_df[products].astype(float)
    rxn_df.loc[:,products] = rxn_df[products].fillna(0)
    return rxn_df, products_to_labware

def rename_products(rxn_df):
    '''
    renames dilutions acording to the reagent that created them
    and renames rxns to have a concentration
    Preconditions:
        dilution cols are named dilution_1/2 etc
        callback is the last column in the dataframe
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
            name = reagent_name[:reagent_name.rfind('C')+1]+row['dilution_conc']
            rename_key[col] = name
        else:
            rename_key[col] = "{}C1".format(col)

    rxn_df.rename(rename_key, axis=1, inplace=True)

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


def init_robot(portal, rxn_spreadsheet, rxn_df, simulate, spreadsheet_key, credentials, using_temp_ctrl, temp, products_to_labware):
    '''
    This function gets the unique reagents, interfaces with the docs to get details on those
      reagents, and ships that information to the robot so it can initialize it's labware dicts
    params:
        Armchair portal: the armchair object connected to robot
        gspread.Spreadsheet rxn_spreadsheet: a spreadsheet object with second sheet having
          deck positions
        df rxn_df: the input df read from sheets
        bool simulate: if the robot is to simulate or execute
        str spreadsheet_key: this is the a unique id for google sheet used for i/o with sheets
        ServiceAccount Credentials credentials: to access sheets
        Dict<str, str>: maps rxns to prefered labware
    Postconditions:
        user has been queried about reagents and response has been pulled down
    '''

    #query the docs for more info on reagents
    construct_reagent_sheet(rxn_df, spreadsheet_key, credentials)
    input("please press enter when you've completed the reagent sheet")

    #pull the info into a df
    reagent_info = g2d.download(spreadsheet_key, 'reagent_info', col_names = True, 
            row_names = True, credentials=credentials).drop(columns=['comments'])
    empty_containers = reagent_info.loc['empty'].set_index('deck_pos').drop(columns=
            ['conc', 'mass'])
    reagents = reagent_info.drop(['empty']).astype({'conc':float,'deck_pos':int,'mass':float})

    #pull labware etc from the sheets
    labware_df, instruments = get_labware_info(rxn_spreadsheet, empty_containers)

    #build_product_df with info on where to build products
    product_df = construct_product_df(rxn_df, products_to_labware)

    #send robot data to initialize itself
    portal.send_pack('init', simulate, using_temp_ctrl, temp, labware_df, instruments, reagents)

    #send robot data to initialize empty product containers. Because we know things like total
    #vol and desired labware, this makes sense for a planned experiment
    with open('product_df_cache.pkl', 'wb') as cache:
        dill.dump(product_df, cache)
    portal.send_pack('init_containers', product_df)

    return

def construct_product_df(rxn_df, products_to_labware):
    '''
    Creates a df to be used by robot to initialize containers for the products it will make
    params:
        df rxn_df: as passed to init_robot
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
    max_vols = [get_rxn_max_vol(rxn_df, product, products) for product in products]
    product_df = pd.DataFrame(products_to_labware, index=['labware']).T
    product_df['max_vol'] = max_vols
    return product_df

def get_rxn_max_vol(rxn_df, name, products):
    '''
    Preconditions:
        volume in a container can change only during a 'transfer' or 'dilution'. Easy to add more
        by changing the vol_change_rows
    params:
        df rxn_df: as returned by load_table. Should have all NaN removed from products
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
    vol_change_rows = rxn_df.loc[rxn_df['op'].apply(lambda x: x in ['transfer','dilution'])]
    aspirations = vol_change_rows['chemical_name'] == name
    max_vol = 0
    current_vol = 0
    for i, is_aspiration in aspirations.iteritems():
        if is_aspiration and rxn_df.loc[i,'op'] == 'transfer':
            #This is a row where we're transfering from this well
            current_vol -= rxn_df.loc[i, products].sum()
        else:
            current_vol += rxn_df.loc[i,name]
            max_vol = max(max_vol, current_vol)
    return max_vol

def construct_reagent_sheet(rxn_df, spreadsheet_key, credentials):
    '''
    query the user with a reagent sheet asking for more details on locations of reagents, mass
    etc
    Preconditions:
        see excel_spreadsheet_preconditions.txt
    PostConditions:
        reagent_sheet has been constructed
    '''
    rxn_names = rxn_df.loc[:, 'reagent':'chemical_name'].drop(columns=['reagent','chemical_name']).columns
    reagent_df = rxn_df[['chemical_name', 'conc']].groupby('chemical_name').first()
    reagent_df.drop(rxn_names, errors='ignore', inplace=True) #not all rxns are reagents
    reagent_df[['loc', 'deck_pos', 'mass', 'comments']] = ''
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

def get_labware_info(rxn_spreadsheet, empty_containers):
    '''
    Interface with sheets to get information about labware locations, first tip, etc.
    Preconditions:
        The second sheet in the worksheet must be initialized with where you've placed reagents 
        and the first thing not being used
    params:
        gspread.Spreadsheet rxn_spreadsheet: a spreadsheet object with second sheet having
          deck positions
        df empty_containers: this is used for tubes. it holds the containers that can be used
            int index: deck_pos
            str position: the position of the empty container on the labware
    returns:
        df:
          str name: the common name of the labware (made unique 
        Dict<str:str>: key is 'left' or 'right' for the slots. val is the name of instrument
    '''
    raw_labware_data = rxn_spreadsheet.get_worksheet(1).get_all_values()
    #the format google fetches this in is funky, so we convert it into a nice df
    labware_dict = {'name':[], 'first_usable':[],'deck_pos':[]}
    for row_i in range(0,10,3):
        for col_i in range(3):
            labware_dict['name'].append(raw_labware_data[row_i+1][col_i])
            labware_dict['first_usable'].append(raw_labware_data[row_i+2][col_i])
            labware_dict['deck_pos'].append(raw_labware_data[row_i][col_i])
    labware_df = pd.DataFrame(labware_dict)
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
    #make instruments
    instruments = {}
    instruments['left'] = raw_labware_data[13][0]
    instruments['right'] = raw_labware_data[13][1]
    return labware_df, instruments

#SERVER
#CONTAINERS
class Container(ABC):
    """
    
    Abstract container class to be overwritten for well, tube, etc.
    ABSTRACT ATTRIBUTES:
        str name: the common name we use to refer to this container
        float vol: the volume of the liquid in this container in uL
        Obj Labware: a pointer to the Opentrons Labware object of which this is a part
        str loc: a location on the labware object (e.g. 'A5')
        float conc: the concentration of the substance
    ABSTRACT METHODS:
        _update_height void: updates self.height to hieght at which to pipet (a bit below water line)
    IMPLEMENTED METHODS:
        update_vol(float aspvol) void: updates the volume upon an aspiration
    """

    def __init__(self, name, labware, loc, vol=0,  conc=1):
        self.name = name
        self.labware = labware
        self.loc = loc
        self.vol = vol
        self.height = self._update_height()
        self.conc = conc

    @abstractmethod
    def _update_height(self):
        pass

    def update_vol(self, aspvol):
        self.vol = self.vol - self.aspvol
        self._update_height()

class Tube20000uL(Container):
    """
    Spcific tube with measurements taken to provide implementations of abstract methods
    INHERITED ATTRIBUTES
        str name, float vol, Obj Labware, str loc
    INHERITED METHODS
        _update_height void, update_vol(float aspvol) void,
    """
    
    def __init__(self, name, labware, loc, mass=6.6699, conc=1):
        '''
        mass is defaulted to the avg_mass so that there is nothing in the container
        '''
        density_water_25C = 0.9970479 # g/mL
        avg_tube_mass15 = 6.6699 # grams
        self.mass = mass - avg_tube_mass15 # N = 1 (in grams) 
        vol = (self.mass / density_water_25C) * 1000 # converts mL to uL
        super().__init__(name, labware, loc, vol, conc)
       # 15mm diameter for 15 ml tube  -5: Five mL mark is 19 mm high for the base/noncylindrical protion of tube 

    def _update_height(self):
        diameter_15 = 14.0 # mm (V1 number = 14.4504)
        vol_bottom_cylinder = 2000 # uL
        height_bottom_cylinder = 30.5  #mm
        tip_depth = 5 # mm
        self.height = ((self.vol - vol_bottom_cylinder)/(math.pi*(diameter_15/2)**2))+(height_bottom_cylinder - tip_depth)
            
class Tube50000uL(Container):
    """
    Spcific tube with measurements taken to provide implementations of abstract methods
    INHERITED ATTRIBUTES
        str name, float vol, Obj Labware, str loc
    INHERITED METHODS
        _update_height void, update_vol(float aspvol) void,
    """
    def __init__(self, name, labware, loc, mass=13.3950, conc=1):
        density_water_25C = 0.9970479 # g/mL
        avg_tube_mass50 = 13.3950 # grams
        self.mass = mass - avg_tube_mass50 # N = 1 (in grams) 
        vol = (self.mass / density_water_25C) * 1000 # converts mL to uL
        super().__init__(name, labware, loc, vol, conc)
       # 15mm diameter for 15 ml tube  -5: Five mL mark is 19 mm high for the base/noncylindrical protion of tube 
        
    def _update_height(self):
        diameter_50 = 26.50 # mm (V1 number = 26.7586)
        vol_bottom_cylinder = 5000 # uL
        height_bottom_cylinder = 21 #mm
        tip_depth = 5 # mm
        self.height = ((self.vol - vol_bottom_cylinder)/(math.pi*(diameter_50/2)**2)) + (height_bottom_cylinder - tip_depth) 

class Tube2000uL(Container):
    """
    2000uL tube with measurements taken to provide implementations of abstract methods
    INHERITED ATTRIBUTES
         str name, float vol, Obj Labware, str loc
    INHERITED METHODS
        _update_height void, update_vol(float aspvol) void,
    """
    
    def __init__(self, name, labware, loc, mass=1.4, conc=1):
        density_water_4C = 0.9998395 # g/mL
        avg_tube_mass2 =  1.4        # grams
        self.mass = mass - avg_tube_mass2 # N = 1 (in grams) 
        vol = (self.mass / density_water_4C) * 1000 # converts mL to uL
        super().__init__(name, labware, loc, vol, conc)
           
    def _update_height(self):
        diameter_2 = 8.30 # mm
        vol_bottom_cylinder = 250 #uL
        height_bottom_cylinder = 10.5 #mm
        tip_depth = 4.5 # mm
        self.height = ((self.vol - vol_bottom_cylinder)/(math.pi*(diameter_2/2)**2)) + (height_bottom_cylinder - tip_depth)

class Well96(Container):
    """
        a well in a 96 well plate
        INHERITED ATTRIBUTES
             str name, float vol, Obj Labware, str loc
        INHERITED METHODS
            _update_height void, update_vol(float aspvol) void,
    """

    def __init__(self, name, labware, loc, vol=0, conc=1):
        #vol is defaulted here because the well will probably start without anything in it
        super().__init__(name, labware, loc, vol, conc)
           
    def _update_height(self):
        #TODO develop an update height method for wells.
        pass


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
    ABSTRACT METHODS:
        get_container_type(loc) str: returns the type of container at that location
        pop_next_well(vol=None) str: returns the index of the next available well
          If there are no available wells of the volume requested, return None
    '''

    def __init__(self, labware, deck_pos):
        self.labware = labware
        self.full = False
        self.deck_pos = deck_pos

    @abstractmethod
    def pop_next_well(self, vol=None):
        '''
        returns the next available well
        Or returns None if there is no next availible well with specified volume
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
    ATTRIBUTES:
        list<str> empty_tubes: contains locs of the empty tubes. Necessary because the user may
          not put tubes into every slot
    '''
    def __init__(self, labware, empty_tubes, deck_pos):
        self.empty_tubes = empty_tubes
        super().__init__(labware,deck_pos)
        self.full = not self.empty_tubes

    def pop_next_well(self, vol=None):
        '''
        Gets the next available tube. If vol is specified, will return an
        appropriately sized tube. Otherwise it will return a tube. It makes no guarentees that
        tube will be the correct size. It is not recommended this method be called without
        a volume argument
        kwargs:
            float vol: used to determine an appropriate sized tube
        returns:
            str: loc the location of the next tube
        '''
        if not self.full:
            if vol:
                for i, loc in enumerate(self.empty_tubes):
                    tube_capacity = self.labware.wells_by_name()[loc]._geometry._max_volume
                    if tube_capacity > vol:
                        tube_loc = self.empty_tubes.pop(i)
                        self.full = not self.empty_tubes
                        return tube_loc
                #you checked all empty tubes and none can accomodate!
                return None
            else:
                #volume was not specified
                #return the next tube. God know's what you'll get
                tube = self.empty_tubes.pop(0)
                self.full = not self.empty_tubes
                return tube
        else:
            #self.empty_tubes is empty!
            return None

    def get_container_type(self, loc):
        '''
        returns type of container
        params:
            str loc: the location on this labware
        returns:
            str: the type of container at that loc
        '''
        tube_capacity = self.labware.wells_by_name()[loc]._geometry._max_volume
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
        pop_next_well(vol=None) str: vol should be provided to check if well is big enough
        get_container_type(loc) str
    INHERITED_ATTRIBUTES:
        Opentrons.Labware labware, bool full, int deck_pos, str name
    ATTRIBUTES:
        int current_well: the well number your on (NOT loc!)
    '''
    def __init__(self, labware, first_well, deck_pos):
        n_rows = len(labware.columns()[0])
        col = first_well[:1]#alpha part
        row = first_well[1:]#numeric part
        self.current_well = (ord(col)-64)*n_rows #transform alpha num to num
        super().__init__(labware, deck_pos)
        self.full = self.current_well >= len(labware.wells())

    def pop_next_well(self, vol=None):
        '''
        returns the next well if there is one, otherwise returns None
        params:
            float vol: used to determine if your reaction can be fit in a well
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
        Dict<str, Container> containers: maps from a common name to a Container object
        Dict<str, Obj> tip_racks: maps from a common name to a opentrons tiprack labware object
        Dict<str, Obj> labware: maps from labware common names to opentrons labware objects. tip racks not included?
        Opentrons...ProtocolContext protocol: the protocol object of this session

    """

    _OPENTRONS_LABWARE_NAMES = {'96_well_plate':'corning_96_wellplate_360ul_flat','24_well_plate':'corning_24_wellplate_3.4ml_flat','48_well_plate':'corning_48_wellplate_1.6ml_flat','tip_rack_20uL':'opentrons_96_tiprack_20ul','tip_rack_300uL':'opentrons_96_tiprack_300ul','tip_rack_1000uL':'opentrons_96_tiprack_1000ul','tube_holder_10':'opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical','temp_mod_24_tube':'opentrons_24_aluminumblock_generic_2ml_screwcap'}
    _CUSTOM_LABWARE_DEFINITION_PATHS = {'platereader4':'LabwareDefs/plate_reader_4.json','platereader7':'LabwareDefs/plate_reader_7.json'}
    _OPENTRONS_INSTRUMENT_NAMES = {'20uL_pipette':'p20_single_gen2','300uL_pipette':'p300_single_gen2','1000uL_pipette':'p1000_single_gen2'}
    _TIP_RACK_NAMES = {'tip_rack_20uL', 'tip_rack_300uL','tip_rack_1000uL'}
    _TEMP_MOD_NAMES = {'temp_mod_24_tube'}
    _PLATE_NAMES = {'96_well_plate','48_well_plate','24_well_plate', 'platereader4', 'platereader7'}
    _TUBE_HOLDER_NAMES = {'tube_holder_10','temp_mod_24_tube'}


    def __init__(self, simulate, using_temp_ctrl, temp, labware_df, instruments, reagents_df):
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
            df reagents_df: info on reagents. columns from sheet. See excel specification
                
        postconditions:
            protocol has been initialzied
            containers and tip_racks have been created
            labware has been initialized
            CAUTION: the values of tip_racks and containers must be sent from the client.
              it is the client's responsibility to make sure that these are initialized prior
              to operating with them
        '''
        #DEBUG
        with open('cache.pkl','wb') as cache:
            dill.dump([simulate, using_temp_ctrl, temp, labware_df, instruments, reagents_df], cache)
        self.containers = {}
        
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
        self._init_labware(labware_df, using_temp_ctrl, temp)
        self._init_instruments(instruments, labware_df)
        self._init_containers(reagents_df)
        #CONCLUDED!

    def _init_containers(self, reagents_df):
        '''
        params:
            df reagents_df: as passed to init
        Postconditions:
            the dictionary, self.containers, has been initialized to have name keys to container
              objects
        '''
        container_types = reagents_df['deck_pos'].apply(lambda d: self.lab_deck[d])
        container_types = reagents_df[['deck_pos','loc']].apply(lambda row: 
                self.lab_deck[row['deck_pos']].get_container_type(row['loc']),axis=1)
        container_types.name = 'container_type'

        for name, conc, loc, deck_pos, mass, container_type in reagents_df.join(container_types).itertuples():
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
        self.protocol.max_speeds['X'] = 100
        self.protocol.max_speeds['Y'] = 100

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
            opentrons_name = self._OPENTRONS_LABWARE_NAMES[name]
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
        with open(self._CUSTOM_LABWARE_DEFINITION_PATHS[name], 'r') as labware_def_file:
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
        if name in self._TUBE_HOLDER_NAMES:
            self.lab_deck[deck_pos] = TubeHolder(labware, kwargs['empty_containers'], deck_pos)
        elif name in self._PLATE_NAMES:
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
            if name in self._CUSTOM_LABWARE_DEFINITION_PATHS:
                #plate readers (or other custom?)
                self._init_custom_labware(name, deck_pos, first_well=first_usable)
            elif name in self._TEMP_MOD_NAMES:
                #temperature controlled racks
                self._init_temp_mod(name, using_temp_ctrl, 
                        temp, deck_pos, empty_tubes=empty_list)
            else:
                #everything else
                opentrons_name = self._OPENTRONS_LABWARE_NAMES[name]
                labware = self.protocol.load_labware(opentrons_name,deck_pos,label=name)
                if name in self._PLATE_NAMES:
                    self._add_to_deck(name, deck_pos, labware, first_well=first_usable)
                elif name in self._TUBE_HOLDER_NAMES:
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
            used_rack_row = tip_rows.loc[self._lexo_argmax(tip_rows['first_usable'])]
            #get opentrons object
            used_rack = self.protocol.loaded_labwares[used_rack_row['deck_pos']]
            #set starting tip
            pipette.starting_tip = used_rack.well(used_rack_row['first_usable'])
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
 
    def _init_empty_containers(self, product_df):
        '''
        used to initialize empty containers, which is useful before transfer steps to new chemicals
        especially if we have preferences for where those chemicals are put
        Params:
            df product_df: as generated in client init_robot
        Postconditions:
            every container has been initialized according to the parameters specified
        '''
        for chem_name, labware, max_vol in product_df.itertuples():
            
            container = None
            #if you've already initialized this complane
            if chem_name in self.containers:
                raise Exception("you tried to initialize {},\
                        but there is already an entry for {}".format(chem_name, chem_name))
            #filter labware
            viable_labware = [viable for viable in self.lab_deck if viable]
            if labware:
                if labware == 'platereader':
                    #platereader is speacial because there are two platereaders
                    viable_labware = [viable for viable in viable_labware if \
                    viable.name in ['platereader4','platereader7']]
                else:
                    viable_labware = [viable for viable in viable_labware if viable.name == labware]
            #sort the list so that platreader slots are prefered
            viable_labware.sort(key=lambda x: self._init_empty_containers.priority[x.name])
            #iterate through the filtered labware and pick the first one that 
            loc, deck_pos, container_type  = None, None, None
            i = 0
            while not loc:
                try:
                    viable = viable_labware[i]
                except IndexError: 
                    raise Exception('No viable slots to put {}.'.format(chem_name))
                next_container_loc = viable.pop_next_well(vol=max_vol)
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
    _init_empty_containers.priority = defaultdict(lambda: 100)
    _init_empty_containers.priority['platereader4'] = 1
    _init_empty_containers.priority['platereader7'] = 2

    def execute(self, command_type, arguments):
        '''
        takes the packet type and payload of an Armchair packet, and executes the command
        params:
            str command_type: the type of packet to execute
            tuple<Obj> arguments: the arguments to this command 
              (generally passed as list so no *args)
        Postconditions:
            the command has been executed
        '''
        if command_type == 'transfer':
            init_labware(arguments[0])
            init_reagents(arguments[1])
        elif command_type == 'init_containers':
            self._init_empty_containers(arguments[0])
        else:
            raise Exception("Unidenified command {}".format(pack_type))

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
