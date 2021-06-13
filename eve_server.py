import socket

from boltons.socketutils import BufferedSocket
import dill

from ot2_server_lib import OT2Controller
from armchair import Armchair

PORT_NUM = 50000

def main(client_sock):
    buffered_sock = BufferedSocket(client_sock)
    portal = Armchair(buffered_sock)
    eve = None
    while not eve:
        pack_type, args = portal.recv_pack()
        if pack_type == 'init':
            simulate = args[0]
            labware_df = args[1]
            reagents_df = args[2]
            eve = OT2Controller(simulate, labware_df, reagents_df)
    connection_open=True
    while connection_open:
        pack_type, payload = portal.recv_pack()
        print('header type = {}, payload = {}'.format(pack_type, payload))

if __name__ == '__main__':
    #construct a socket
    sock = socket.socket(socket.AF_INET)
    sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    sock.bind(('',PORT_NUM))
    print('server listening on port {}'.format(PORT_NUM))
    while True:
        #spin until you get a connection
        sock.listen(5)
        client_sock, client_addr = sock.accept()
        main(client_sock)
