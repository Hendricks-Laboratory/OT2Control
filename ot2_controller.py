import socket
import sys

import dill

from armchair import Armchair

SERVERADDR = "127.0.0.1"
PORT = 50000

def init

def main():
    sock = socket.socket(socket.AF_INET)
    sock.connect((SERVERADDR, PORT))
    portal = Armchair(sock)
    portal.send_pack(pack_type, payload)
    client_sock.close()

if __name__ == '__main__':
    main()
