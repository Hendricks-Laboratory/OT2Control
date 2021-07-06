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
    b = threading.Barrier(2,timeout=20)
    eve_thread = threading.Thread(target=eve_server.main, kwargs={'my_ip':'','barrier':b})
    eve_thread.start()

    #do create a connection
    b.wait()
    sock = socket.socket(socket.AF_INET)
    sock.connect((SERVERADDR, PORT))
    print("<<controller>> connected")
    portal = Armchair(sock,'controller','Armchair_Logs')
    
    #instantiate a controller
    #rxn_sheet_name = input('<<controller>> please input the sheet name')
    rxn_sheet_name = 'test_wombo_combo'
    controller = ot2lib.ProtocolExecutor(portal,rxn_sheet_name, use_cache=True)

    #execute the protocol
    controller.init_robot(True)
    controller.run_protocol()
    controller.close_connection(SERVERADDR)
    
    #run post execution tests
    controller.run_all_tests()

    #collect the eve thread
    eve_thread.join()

if __name__ == '__main__':
    main()
