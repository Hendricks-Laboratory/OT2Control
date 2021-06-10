"""
This code is for deomonstration of how the armchair protocol works only. It is not part of 
the project
"""
import socket

from boltons.socketutils import BufferedSocket
import dill

from armchair import Armchair

PORT_NUM = 50000

if __name__ == '__main__':
    
sock = socket.socket(socket.AF_INET)
sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
sock.bind(('',PORT_NUM))

print('server listening on port {}'.format(PORT_NUM))
while True:
    sock.listen(5)
    client_sock, client_addr = sock.accept()
    main(client_sock)
    connection_open=True
    buffered_sock = BufferedSocket(client_sock)
    portal = Armchair(buffered_sock)
    while connection_open: 
        pack_type, payload = portal.recv_pack()
        args = dill.loads(payload)
        print('header type = {}, payload = {}'.format(pack_type, args))
        break;
    print("connection recieved from {}".format(client_addr))
    data = client_sock.recv(1024)
