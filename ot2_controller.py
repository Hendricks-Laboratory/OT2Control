import socket
import sys

import dill

from Armchair.armchair import Armchair

SERVERADDR = "10.25.15.209"
PORT = 50000


def main():
    sock = socket.socket(socket.AF_INET)
    sock.connect((SERVERADDR, PORT))
    portal = Armchair(sock)
    payload=None
    pack_type = 'transfer'
    portal.send_pack(pack_type, payload)
    sock.close()

if __name__ == '__main__':
    main()
