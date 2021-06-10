import socket

from boltons.socketutils import BufferedSocket
import dill

from armchair import Armchair

PORT_NUM = 50000

def main(client_sock):
    buffered_sock = BufferedSocket(client_sock)
    portal = Armchair(buffered_sock)
    connection_open=True
    while connection_open: 
        pack_type, payload = portal.recv_pack()
        if pack_type:
            #meh. do something based on pack type
            args = dill.loads(payload)
            print('header type = {}, payload = {}'.format(pack_type, args))

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
