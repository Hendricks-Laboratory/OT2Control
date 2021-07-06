import socket
import asyncio

from boltons.socketutils import BufferedSocket

from ot2lib import OT2Controller
from Armchair.armchair import Armchair


def main(**kwargs):
    my_ip = kwargs['my_ip']
    PORT_NUM = 50000
    #construct a socket
    sock = socket.socket(socket.AF_INET)
    sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    sock.bind((my_ip,PORT_NUM))
    print('<<eve>> listening on port {}'.format(PORT_NUM))
    sock.listen(5)
    if kwargs['barrier']:
        #running in thread mode with barrier. Barrier waits for both threads
        kwargs['barrier'].wait()
    client_sock, client_addr = sock.accept()
    print('<<eve>> connected')
    buffered_sock = BufferedSocket(client_sock, timeout=None)
    portal = Armchair(buffered_sock,'eve','Armchair_Logs')
    eve = None
    pack_type, cid, args = portal.recv_pack()
    if pack_type == 'init':
        simulate, using_temp_ctrl, temp, labware_df, instruments, reagents_df = args
        #I don't know why this line is needed, but without it, Opentrons crashes because it doesn't
        #like to be run from a thread
        asyncio.set_event_loop(asyncio.new_event_loop())
        eve = OT2Controller(simulate, using_temp_ctrl, temp, labware_df, instruments, reagents_df,my_ip, portal)
        portal.send_pack('ready', cid)
    connection_open=True
    while connection_open:
        pack_type, cid, payload = portal.recv_pack()
        connection_open = eve.execute(pack_type, cid, payload)

if __name__ == '__main__':
    my_ip = socket.gethostname()
    main(my_ip=my_ip, barrier=None)
