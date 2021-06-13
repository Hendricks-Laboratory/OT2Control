import gspread
from df2gspread import df2gspread as d2g
from df2gspread import gspread2df as g2d
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np

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
    labware_df = get_labware_info(rxn_spreadsheet)

    #get unique reagents
    reagents_and_conc = rxn_df[['chemical_name', 'conc']].groupby('chemical_name').first()

    #query the docs for more info on reagents
    #DEBUG not overwriting vals for testing purposes
    #construct_reagent_sheet(reagents_and_conc, spreadsheet_key, credentials)
    input("please press enter when you've completed the reagent sheet")

    #pull the info into a df
    reagents = g2d.download(spreadsheet_key, 'reagent_info', col_names = True, row_names = True, credentials=credentials)
    breakpoint()

    #iterate through that thing and send information over armchair in a network friendly way

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
    '''
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
    return 
