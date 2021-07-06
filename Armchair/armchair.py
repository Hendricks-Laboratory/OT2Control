from bidict import bidict
from datetime import datetime
import os
import dill

#used for armchair file transfer initialized from armchair instructions
FTP_EOF = 'AFKJldkjvaJKDvJDFDFGHowCouldYouEverHaveThisInAFile'.encode('ascii')

class Armchair():
    def __init__(self, socket, name, log_path=''):
        self.sock = socket
        self.cid = 0
        #bidirectional dictionary for conversions from byte codes to string names for commands and back
        self.pack_types = bidict({'init':b'\x00','close':b'\x01','error':b'\x02','ready':b'\x03','transfer':b'\x04','init_containers':b'\x05','sending_files':b'\x06'})
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

    def recv_pack(self):
        '''
        processes the next packet and returns None if nothing to read
        returns: if packet in buffer
            str: type of packet
            bytes: the argmuents/payload
        returns: else
            (None,None)
        '''
        header = self.sock.recv(17)
        header_len = self.get_len(header)
        header_type = self.get_type(header)
        header_cid = self.get_cid(header)
        payload = self.sock.recv(header_len)
        if payload: #if there were arguments
            payload = dill.loads(payload)
        with open(os.path.join(self.log_path, '{}_armchair.log'.format(self.name)), 'a+') as armchair_log:
            armchair_log.write("{}\trecieved {}, cid {}\n".format(datetime.now().strftime('%H:%M:%S:%f'),header_type,self.cid))
        return header_type, header_cid, payload
        
    def send_pack(self, pack_type, *args):
        '''
        constructs a packet and sends it over network
        params:
            str pack_type: the type of packet being sent (string form)
            args*: the objects to pickle
        returns:
            int: the cid of the sent packet
        Postconditions:
            An armchair packet has been constructed and sent over the socket
            has created log entry of send
        '''
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
        return self.cid

    def close(self):
        self.sock.close()
