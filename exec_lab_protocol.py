import socket
import sys

import dill
import pandas as pd

import ot2_client_lib as ot2cl
from Armchair.armchair import Armchair

SERVERADDR = "10.25.15.209"
PORT = 50000


def main():
    #establish connection
    #DEBUG sockets and armchair commented out in order to allow testing without sockets
    #sock = socket.socket(socket.AF_INET)
    #sock.connect((SERVERADDR, PORT))
    #portal = Armchair(sock)
    #get user input
    #simulate, rxn_sheet_name, using_temp_ctrl, temp = ot2cl.pre_rxn_questions()
    simulate, rxn_sheet_name, using_temp_ctrl, temp = (True, 'pchem_week5', False, None)
    #credentials are needed for a lot of i/o operations with google sheets
    credentials = ot2cl.init_credentials(rxn_sheet_name)
    #wks_key is also needed for google sheets i/o. It functions like a url
    wks_key = ot2cl.get_wks_key(credentials, rxn_sheet_name)
    rxn_sheet = ot2cl.open_sheet(rxn_sheet_name, credentials)
    rxn_df = ot2cl.load_rxn_df(rxn_sheet, rxn_sheet_name)
    ot2cl.init_robot(rxn_df,simulate)
    breakpoint()

    #sock.close()

if __name__ == '__main__':
    main()
