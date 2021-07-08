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

SERVERADDR = "10.25.19.212"
#SERVERADDR = "10.4.6.151"


def main():
    #instantiate a controller
    rxn_sheet_name = input('<<controller>> please input the sheet name')
    #using the cache bypasses google docs communication and uses the last rxn you loaded
    use_cache = 'y' == input('<<controller>> would you like to use the spreadsheet cache? [yn] ')
    #use_cache = True
    #rxn_sheet_name = 'test_Oh_the_horrible_things_I_can_do'
    my_ip = socket.gethostbyname(socket.gethostname())
    controller = ot2lib.ProtocolExecutor(rxn_sheet_name, my_ip, SERVERADDR, use_cache=use_cache)

    tests_passed = controller.run_simulation()

    if tests_passed:
        if input('would you like to run the protocol? [yn] ').lower() == 'y':
            controller.run_protocol(SERVERADDR)
    else:
        print('Failed Some Tests. Please fix your errors and try again')

if __name__ == '__main__':
    main()
