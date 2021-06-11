import gspread
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
    return credentials

def load_rxn_df(rxn_sheet, rxn_sheet_name):
    '''
    reaches out to google sheets and loads the reaction protocol into a df and formats the df
    adds a chemical name (primary key for lots of things. e.g. robot dictionaries)
    renames some columns to code friendly as opposed to human friendly names
    params:
        gspread.Spreadsheet rxn_sheet: the sheet with all the reactions
        str rxn_sheet_name: the name of the spreadsheet
    returns:
        pd.DataFrame: the information in the rxn_sheet w range index. spreadsheet cols
    '''
    rxn_wks = rxn_sheet.get_worksheet(0)
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


def init_robot(rxn_df, simulate):
    '''
    This function gets the unique reagents, interfaces with the docs to get details on those
      reagents, and ships that information to the robot so it can initialize it's labware dicts
    '''
    #get unique reagents
    reagents = rxn_df[['chemical_name', 'conc']].groupby('chemical_name').first()
    reagents['content_type'] = reagents.apply(lambda row: 'mix' if pd.isnull(row['conc']) else 'dilution',axis=1)
    breakpoint()

    #query the docs about labware etc

    #pull labware etc from the docs

    #construct df with "{}C{}".format(reagent,conc) key and a bunch of values needed to construct
    #the object

    #iterate through that thing and send information over armchair in a network friendly way

    #return (god willing)
    return



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

#MAYBE USEFULL???
#non_vol_cols = ['op','scan_params','conc','reagent','callback']
#reagent_cols = rxn_df.drop(columns=non_vol_cols).loc[rxn_df['op'] == 'transfer']
#grouped = rxn_df.loc[rxn_df['op'] == 'transfer'].groupby('chemical_name', dropna=False)
#unique_reagents = grouped.sum()
