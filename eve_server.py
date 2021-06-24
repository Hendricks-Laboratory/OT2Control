import socket

from boltons.socketutils import BufferedSocket

from ot2lib import OT2Controller
from Armchair.armchair import Armchair

PORT_NUM = 50000
ip = ''

def main(client_sock):
    print('<<eve>> connected')
    buffered_sock = BufferedSocket(client_sock, timeout=None)
    portal = Armchair(buffered_sock,'eve','Armchair_Logs')
    eve = None
    pack_type, cid, args = portal.recv_pack()
    if pack_type == 'init':
        simulate, using_temp_ctrl, temp, labware_df, instruments, reagents_df = args
        eve = OT2Controller(simulate, using_temp_ctrl, temp, labware_df, instruments, reagents_df,ip, portal)
        portal.send_pack('ready', cid)
    connection_open=True
    while connection_open:
        pack_type, cid, payload = portal.recv_pack()
        connection_open = eve.execute(pack_type, cid, payload)

if __name__ == '__main__':
    #construct a socket
    sock = socket.socket(socket.AF_INET)
    sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    sock.bind(('',PORT_NUM))
    print('<<eve>> listening on port {}'.format(PORT_NUM))
    sock.listen(5)
    client_sock, client_addr = sock.accept()
    main(client_sock)
