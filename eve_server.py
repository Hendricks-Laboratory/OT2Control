'''
All this code does is run the ot2lib.launch_eve_server with appropriate args
'''
import socket

import ot2lib

def hack_to_get_ip():
    '''
    author @zags from stack overflow
    courtesy of stack overflow
    '''
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    my_ip = s.getsockname()[0]
    s.close()
    return my_ip

if __name__ == '__main__':
    my_ip = hack_to_get_ip()
    ot2lib.launch_eve_server(my_ip=my_ip, barrier=None)


