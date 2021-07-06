'''
All this code does is run the ot2lib.launch_eve_server with appropriate args
'''
import socket

import ot2lib

if __name__ == '__main__':
    my_ip = socket.gethostname()
    ot2lib.launch_eve_server(my_ip=my_ip, barrier=None)
