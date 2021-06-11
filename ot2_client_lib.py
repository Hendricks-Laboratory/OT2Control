import gspread
def pre_rxn_questions():
    '''
    asks user for control params
    returns:
        bool simulate: true if behaviour is to be simulated instead of executed
        str sheet_name: the title of the google sheet
        bool using_temp_ctrl: true if planning to use temperature ctrl module
        float temp: the temperature you want to keep the module at
    '''
    simulate = (input('Simulate or Execute: ').lower() == 'simulate')
    sheet_name = input('Enter Sheet Name as it Appears on the Spreadsheets Title: ')
    temp_ctrl_response = input('Are you using the temperature control module, yes or no?\
    (if yes, turn it on before responding): ').lower()
    using_temp_ctrl = ('y' == temp_ctrl_response or 'yes' == temp_ctrl_response)
    temp = None
    if using_temp_ctrl:
        temp = input('What temperature in Celcius do you want the module \
        set to? \n (the protocol will not proceed until the set point is reached) \n')
    return simulate, sheet_name, using_temp_ctrl, temp

def open_wks(sheet_name, credentials):
    '''
    open the given google sheet with name, 'sheet_name'
    '''
    gc = gspread.authorize(credentials)
    try:
        wks = gc.open(sheet_name)
        print('Everythings Ready To Go')
    except: 
        raise Exception('Spreadsheet Not Found: Make sure the spreadsheet name is spelled correctly and that it is shared with the robot ')
    return wks

def init_credentials(sheet_name):
    '''
    this function reads a local json file to get the credentials needed to access other funcs
    '''
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    #get login credentials from local file. Your json file here
    path = '../../OT2Control/Credentials/hendricks-lab-jupyter-sheets-5363dda1a7e0.json'
    credentials = ServiceAccountCredentials.from_json_keyfile_name(path, scope) 
    return credentials

def get_wks_key(credentials, sheet_name):
    '''
    open and search a sheet that tells you which sheet is associated with the reaction
    '''
    gc = gspread.authorize(credentials)
    name_key_wks = gc.open_by_url('https://docs.google.com/spreadsheets/d/1m2Uzk8z-\
    qn2jJ2U1NHkeN7CJ8TQpK3R0Ai19zlAB1Ew/edit#gid=0').get_worksheet(0)
    name_key_pairs = name_key_wks.get_all_values() #list<list<str name, str key>>
    #Note the key is a unique identifier that can be used to access the sheet
    #d2g uses it to access the worksheet
    try:
        i=0
        wks_key = None
        while not wks_key and i < len(name_key_pairs):
            row = name_key_pairs[i]
            if row[0] == reaction_sheet_name:
                wks_key = row[1]
            i+=1
    except Indexerror:
        raise Exception('Spreadsheet Name/Key pair was not found. Check the dict spreadsheet \
        and make sure the spreadsheet name is spelled exactly the same as the reaction \
        spreadsheet.')
    return credentials
