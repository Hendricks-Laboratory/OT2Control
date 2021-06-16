
"""
This code is for deomonstration of how the armchair protocol works only. It is not part of 
the project
"""
import socket
import sys
from time import sleep

from armchair import Armchair

SERVERADDR = "127.0.0.1"
PORT = 50000

client_sock = socket.socket(socket.AF_INET)
client_sock.connect((SERVERADDR, PORT))
portal = Armchair(client_sock)
pack_type = 'transfer'
w1 = 'w1'
w2 = 'w2'
vol = 77
sleep(10)
portal.send_pack(pack_type, w1,w2,vol)
client_sock.close()
