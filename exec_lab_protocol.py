import socket
import os
import sys
import threading
import time

import dill
import pandas as pd

import ot2lib 
from Armchair.armchair import Armchair
import eve_server

#SERVERADDR = "10.25.15.209"
SERVERADDR = '127.0.0.1' #loopback for testing only
PORT = 50000


def main():
    #launch an eve server in background for simulation purposes
    #jenky way of spinng off a thread
    eve_thread = threading.Thread(target=run_eve)
    eve_thread.start()
    time.sleep(2) #it's a little jenky for now
    #get user input
    #simulate, rxn_sheet_name, using_temp_ctrl, temp = ot2lib.pre_rxn_questions()
    #DEBUG
    simulate, rxn_sheet_name, using_temp_ctrl, temp = (True, 'test_wombo_combo', True, 22.5)
    #credentials are needed for a lot of i/o operations with google sheets
    credentials = ot2lib.init_credentials(rxn_sheet_name)
    #wks_key is also needed for google sheets i/o. It functions like a url
    wks_key = ot2lib.get_wks_key(credentials, rxn_sheet_name)
    #rxn_sheet is the entire spreadsheet
    rxn_spreadsheet = ot2lib.open_sheet(rxn_sheet_name, credentials)
    rxn_df, products_to_labware = ot2lib.load_rxn_table(rxn_spreadsheet, rxn_sheet_name)
    #establish connection
    #DEBUG sockets and armchair commented out in order to allow testing without sockets
    sock = socket.socket(socket.AF_INET)
    sock.connect((SERVERADDR, PORT))
    print("<<controller>> connected")
    portal = Armchair(sock,'controller','Armchair_Logs')
    ot2lib.init_robot(portal, rxn_spreadsheet, rxn_df, simulate, wks_key, credentials, using_temp_ctrl, temp, products_to_labware)
    ot2lib.run_protocol(rxn_df, portal)
    ot2lib.close_connection(portal, SERVERADDR)

def run_eve():
    os.system('python eve_server.py')

if __name__ == '__main__':
    main()
