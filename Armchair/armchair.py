from bidict import bidict
from datetime import datetime
import os
import dill

#used for armchair file transfer initialized from armchair instructions
FTP_EOF = 'AFKJldkjvaJKDvJDFDFGHowCouldYouEverHaveThisInAFile'.encode('ascii')

class Armchair():
    def __init__(self, socket, name, log_path='', buffsize=4):
        self.sock = socket
        self.cid = 0
        self.buffsize = buffsize
        self._inflight_packs = []
        #bidirectional dictionary for conversions from byte codes to string names for commands and back
        self.pack_types = bidict({'init':b'\x00','close':b'\x01','error':b'\x02','ready':b'\x03','transfer':b'\x04','init_containers':b'\x05','sending_files':b'\x06','pause':b'\x07','stop':b'\x08','continue':b'\x09'})
        self.name = name
        self.log_path = log_path
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        with open(os.path.join(self.log_path, '{}_armchair.log'.format(self.name)), 'w') as armchair_log:
            armchair_log.write('Armchair Log: {}, {}\n'.format(self.name, datetime.now().strftime('%H:%M:%S:%f')))
            

    def get_len(self,header):
        '''
        params:
            bytes header: the header of the packet
        returns:
            int: the number of bytes in payload
        '''
        return int.from_bytes(header[:8], 'big')
    
    def get_type(self,header):
        '''
        params:
            bytes header: the header of the packet
        returns:
            str: the string name of the command type
        '''
        return self.pack_types.inv[header[8:9]]

    def get_cid(self,header):
        '''
        params:
            bytes header: the header of the packet
        returns:
            int: the cid of this command
        '''
        return int.from_bytes(header[9:17], 'big')
    
    def construct_head(self,n_bytes, pack_type):
        '''
        params:
            int n_bytes: the number of bytes in the payload
            str pack_type: the type of the packet
        '''
        self.cid+=1
        return n_bytes.to_bytes(8,'big') + self.pack_types[pack_type] + self.cid.to_bytes(8,'big')

    def recv(self):
        '''
        If you're looking at this, you probably want recv_pack. recv is a lower level command that
        will simply get you the next thing in the pipe. recv_pack is a wrapper that gives you the
        next significant thing
        '''
        header = self.sock.recv_size(17)
        header_len = self.get_len(header)
        header_type = self.get_type(header)
        header_cid = self.get_cid(header)
        if header_len > 0: #if there were arguments
            payload = self.sock.recv_size(header_len)
            payload = dill.loads(payload)
        else:
            payload = None
        with open(os.path.join(self.log_path, '{}_armchair.log'.format(self.name)), 'a+') as armchair_log:
                armchair_log.write("{}\trecieved {}, cid {}\n".format(datetime.now().strftime('%H:%M:%S:%f'),header_type,self.cid))
        return header_type, header_cid, payload

    def recv_pack(self):
        '''
        processes the next packet and returns None if nothing to read
        If the next packet was a ready packet it will be ignored and corresponding send will be
        removed
        returns: if packet in buffer
            str: type of packet
            bytes: the argmuents/payload
        returns: else
            (None,None)
        '''
        print("reciveing. curr buff is {}".format(self._inflight_packs))
        header_type = 'ready' #do while
        while header_type == 'ready':
            header_type, header_cid, payload = self.recv()
            if header_type == 'ready':
                self._inflight_packs.remove(header_cid)
        return header_type, header_cid, payload
        
    def send_pack(self, pack_type, *args):
        '''
        will first check the buffer. If buffer size is exceded will wait on a ready
        constructs a packet and sends it over network
        params:
            str pack_type: the type of packet being sent (string form)
            args*: the objects to pickle
        returns:
            int: the cid of the sent packet
        Postconditions:
            An armchair packet has been constructed and sent over the socket
            has created log entry of send
            cid has been appended to self._inflight_packs
        '''
        print(self._inflight_packs)
        if len(self._inflight_packs) > self.buffsize:
            self._block_on_ready()
        if args:
            payload = dill.dumps(args)
            n_bytes = len(payload)
            header = self.construct_head(n_bytes, pack_type)
            self.sock.send(header+payload)
        else:
            n_bytes = 0
            header = self.construct_head(n_bytes, pack_type)
            self.sock.send(header)
        with open(os.path.join(self.log_path, '{}_armchair.log'.format(self.name)), 'a+') as armchair_log:
            armchair_log.write("{}\tsending {}, cid {}\n".format(datetime.now().strftime('%H:%M:%S:%f'), pack_type,self.cid))
        if pack_type != 'ready':
            self._inflight_packs.append(self.cid)
        return self.cid

#prob not necessary
#
#    def burn_inflight(self):
#        '''
#        used to burn through the list of inflight packets.
#        Postconditions:
#            there are no more inflight packets. Everything has been acknowledged
#        '''
#        while self._inflight_packs:
#            self._block_on_ready()

    def _block_on_ready(self):
        '''
        used to block until the other side responds with a 'ready' packet
        Preconditions: self._inflight_packs contains cids of packets that have been sent 
        not yet acknowledged
        Postconditions:
            has stalled until a ready command was recieved.
            The cid in the ready command has been removed from self.inflight_packs
        '''
        pack_type, _, arguments = self.recv()
        assert (pack_type == 'ready'), "was expecting a ready packet, but instead recieved a {}".format(pack_type)
        cid = arguments[0]
        self._inflight_packs.remove(cid)

    def close(self):
        self.sock.close()
