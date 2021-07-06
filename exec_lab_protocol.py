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

SERVERADDR = "127.0.0.1"


def main():
    #instantiate a controller
    #rxn_sheet_name = input('<<controller>> please input the sheet name')
    rxn_sheet_name = 'test_inLab01'
    controller = ot2lib.ProtocolExecutor(rxn_sheet_name, use_cache=True)

    tests_passed = controller.run_simulation()

    if tests_passed:
        if input('would you like to run the protocol? [yn] ').lower() == 'y':
            controller.run_protocol(SERVERADDR)
    else:
        print('Failed Some Tests. Please fix your errors and try again')

if __name__ == '__main__':
    main()
