import socket
import sys

import dill
import pandas as pd

import ot2_client_lib as ot2cl
from Armchair.armchair import Armchair

#SERVERADDR = "10.25.15.209"
SERVERADDR = '127.0.0.1' #loopback for testing only
PORT = 50000


def main():
    #get user input
    #simulate, rxn_sheet_name, using_temp_ctrl, temp = ot2cl.pre_rxn_questions()
    simulate, rxn_sheet_name, using_temp_ctrl, temp = (True, 'pchem_week5', False, None)
    #credentials are needed for a lot of i/o operations with google sheets
    credentials = ot2cl.init_credentials(rxn_sheet_name)
    #wks_key is also needed for google sheets i/o. It functions like a url
    wks_key = ot2cl.get_wks_key(credentials, rxn_sheet_name)
    #rxn_sheet is the entire spreadsheet
    rxn_spreadsheet = ot2cl.open_sheet(rxn_sheet_name, credentials)
    rxn_df = ot2cl.load_rxn_df(rxn_spreadsheet, rxn_sheet_name)
    #establish connection
    #DEBUG sockets and armchair commented out in order to allow testing without sockets
    print("connecting to socket port {} at {}".format(PORT, SERVERADDR))
    sock = socket.socket(socket.AF_INET)
    sock.connect((SERVERADDR, PORT))
    print("connected")
    portal = Armchair(sock)
    ot2cl.init_robot(portal, rxn_spreadsheet, rxn_df,simulate, wks_key, credentials)
    breakpoint()

    #sock.close()

if __name__ == '__main__':
    main()
