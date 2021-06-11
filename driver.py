import socket
import sys

import dill

import ot2_client_lib as ot2cl
from Armchair.armchair import Armchair

SERVERADDR = "10.25.15.209"
PORT = 50000


def main():
    #establish connection
    sock = socket.socket(socket.AF_INET)
    sock.connect((SERVERADDR, PORT))
    portal = Armchair(sock)
    #get user input
    simulate, sheet_name, using_temp_ctrl, temp = ot2cl.pre_rxn_questions()
    #credentials are needed for a lot of i/o operations with google sheets
    credentials = ot2cl.init_credentials(sheet_name)
    #wks_key is also needed for google sheets i/o. It functions like a url
    wks_key = ot2cl.get_wks_key(credentials, sheet_name)
    wks = ot2cl.open_wks(sheet_name, credentials)

    sock.close()

if __name__ == '__main__':
    main()
