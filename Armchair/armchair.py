from bidict import bidict
import dill

class Armchair():
    def __init__(self, socket):
        #bidirectional dictionary for conversions from byte codes to string names for commands and back
        self.sock = socket
        self.pack_types = bidict({'init':b'\x00','close':b'\x01','error':b'\x02','status':b'\x03','transfer':b'\x04','init_containers':b'\x05'})

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
    
    def construct_head(self,n_bytes, pack_type):
        '''
        params:
            int n_bytes: the number of bytes in the payload
            str pack_type: the type of the packet
        '''
        return n_bytes.to_bytes(8,'big') + self.pack_types[pack_type]

    def recv_pack(self):
        '''
        processes the next packet and returns None if nothing to read
        returns: if packet in buffer
            str: type of packet
            bytes: the argmuents/payload
        returns: else
            (None,None)
        '''
        header = self.sock.recv(9)
        if header:
            header_len = self.get_len(header)
            header_type = self.get_type(header)
            payload = self.sock.recv(header_len)
            if payload: #if there were arguments
                payload = dill.loads(payload)
            return header_type, payload
        else:
            return None, None
        
    def send_pack(self, pack_type, *args):
        '''
        constructs a packet and sends it over network
        params:
            str pack_type: the type of packet being sent (string form)
            args*: the objects to pickle
        Postconditions:
            An armchair packet has been constructed and sent over the socket
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
        return
