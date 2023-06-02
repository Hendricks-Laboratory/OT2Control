from bidict import bidict
from datetime import datetime
import os
import dill
import functools

#used for armchair file transfer initialized from armchair instructions

class Armchair():
    '''
    This class facilitates all Armchair interactions over socket. Detailed documentation on the
    protocol can be found in acompanying armchair_specs.txt
    The class maintains a buffer of packets in a conversation, and if the buffer is full, it will
    block until the buffer is emptied. Note however that this abstraction is broken in the case
    of GHOST packets, which are sent without waiting for a ready response and without changing the
    cid  
    ATTRIBUTES:  
        socket sock: a connected socket  
        int buffsize: this is the number of allowed inflight packets (i.e. the number of packets
          to send to the reciever before waiting for a ready response)  
        int cid: the current command id. This is a constantly increasing number. It is used to 
          communicate between Armchair objects about where the other end is in terms of processing
          packages.  
        str name: the name of this armchair object. Used for loging purposes  
        str log_path: the path to the log file for this Armchair  
        int state: the state of the robot. 0 is good. 1 is bad
    CONSTANTS:  
        bidict PACK_TYPES: this is a key for translating between byte codes and string labels
          for different packet types  
        GHOST_TYPES: this is a list of all str types that classify as GHOST packets. i.e. they do
          not go to the buffer, are sent immediately without waiting for a ready. The responses to
          these packets are controlled outside of the Armchair class by the user  
    METHODS:  
        recv_pack() tuple<str,int,Obj>: recieve a packet. This method will make
          sure that the buffer is managed appropriately if necessary. Returns type, cid, and 
          payload  tuple<str,int,Obj>: like recv, but 
        recv_first(str pack_type) tuple<str,int,Obj>: recieves the first packet that has the 
          given type. Discards any other packets read in the process.  
        send_pack(pack_type, *args) int: returns the cid it sent. This is used to send a packet 
          of pack_type with args.  
        burn_pipe() void: This command waits until the pipe is clear  
        close() void: terminate the connection  
    '''

    FTP_EOF = 'AFKJldkjvaJKDvJDFDFGHowCouldYouEverHaveThisInAFile'.encode('ascii')
    PACK_TYPES = bidict({'init':b'\x00','close':b'\x01','error':b'\x02','ready':b'\x03','transfer':b'\x04','init_containers':b'\x05','sending_files':b'\x06','pause':b'\x07','stop':b'\x08','continue':b'\x09','stopped':b'\x0A','loc_req':b'\x0B','loc_resp':b'\x0C','home':b'\x0D','make':b'\x0E','mix':b'\x0F','save':b'\x10'})
    GHOST_TYPES = ['continue', 'stopped', 'loc_resp','loc_req', 'save','error'] 
    #These are necessary because we never want to wait on a
    #buffer. These packs should be send as soon as possible
    #They also do not require ready's / are not added to inflight packs. Do not modify CID.


    def __init__(self, socket, name, log_path='', buffsize=4):
        self.state = 0
        self.error_payload = None
        self.sock = socket
        self.cid = 0
        self.buffsize = buffsize
        self._inflight_packs = []
        #bidirectional dictionary for conversions from byte codes to string names for commands and back
        self.name = name
        self.log_path = log_path
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        with open(os.path.join(self.log_path, '{}_armchair.log'.format(self.name)), 'w') as armchair_log:
            armchair_log.write('Armchair Log: {}, {}\n'.format(self.name, datetime.now().strftime('%H:%M:%S:%f')))

    def state_dependent(func):
        def decorated(*args, **kwargs):
            self = args[0]
            if self.state == 0:
                return func(*args, **kwargs)
            else:
                raise ConnectionError('Armchair Error: cannot send or recieve with error state {}, please call reset_error first'.format(self.state))
        return decorated
            

    def _get_len(self,header):
        '''
        params:  
            bytes header: the header of the packet  
        returns:  
            int: the number of bytes in payload  
        '''
        return int.from_bytes(header[:8], 'big')
    
    def _get_type(self,header):
        '''
        params:  
            bytes header: the header of the packet  
        returns:  
            str: the string name of the command type  
        '''
        return self.PACK_TYPES.inv[header[8:9]]

    def _get_cid(self,header):
        '''
        params:  
            bytes header: the header of the packet  
        returns:  
            int: the cid of this command  
        '''
        return int.from_bytes(header[9:17], 'big')
    
    def _construct_head(self,n_bytes, pack_type):
        '''
        params:  
            int n_bytes: the number of bytes in the payload  
            str pack_type: the type of the packet  
        '''
        if pack_type not in self.GHOST_TYPES:
            self.cid+=1
        return n_bytes.to_bytes(8,'big') + self.PACK_TYPES[pack_type] + self.cid.to_bytes(8,'big')

    @state_dependent
    def _recv(self):
        '''
        If you're looking at this, you probably want recv_pack. recv is a lower level command that
        will simply get you the next thing in the pipe. recv_pack is a wrapper that gives you the
        next significant thing  
        returns: if packet in buffer  
            str: type of packet  
            int: the cid of the recived packet  
            Obj: the argmuents/payload  
        raises:  
            ConnectionError: if an error packet is recieved
        '''
        header = self.sock.recv_size(17)
        header_len = self._get_len(header)
        header_type = self._get_type(header)
        header_cid = self._get_cid(header)
        if header_len > 0: #if there were arguments
            payload = self.sock.recv_size(header_len)
            payload = dill.loads(payload)
        else:
            payload = None
        with open(os.path.join(self.log_path, '{}_armchair.log'.format(self.name)), 'a+') as armchair_log:
                armchair_log.write("{}\trecieved {}, cid {}\n".format(datetime.now().strftime('%H:%M:%S:%f'),header_type,self.cid))
        if header_type == 'error':
            self._handle_error_pack(payload)
        return header_type, header_cid, payload

    def _handle_error_pack(self, payload):
        '''
        handles the recipt of an error packet by setting error attributes, and then raising
        an error  
        params:  
            list<Obj> payload: the arguments of the error packet (see specs)  
        '''
        self.error_payload = payload
        self.state = 1
        raise ConnectionError('Armchair Error: error pack recieved')

    def reset_error(self):
        '''
        This makes a lot of sense to do if, e.g. you want to continue to use the Armchair
        connection for ftp before you shutdown
        Postconditions:  
            state is reset  
            error_payload is wiped  
            inflight_packs are cleared without waiting for ready
        '''
        self.state = 0
        self.error_payload = None
        self._inflight_packs = []

    def recv_pack(self):
        '''
        processes the next packet and returns None if nothing to read
        If the next packet was a ready packet it will be ignored and corresponding send will be
        removed  
        Not state dependent because recv is state dependent
        returns:  
            str: type of packet  
            int: the cid of the recived packet  
            Obj: the argmuents/payload  
        '''
        header_type = 'ready' #do while
        while header_type == 'ready':
            header_type, header_cid, payload = self._recv()
            if header_type == 'ready':
                self._inflight_packs.remove(payload[0])
        return header_type, header_cid, payload
        
    @state_dependent
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
        if len(self._inflight_packs) > self.buffsize and pack_type not in self.GHOST_TYPES:
            self._block_on_ready()
        if args:
            payload = dill.dumps(args)
            n_bytes = len(payload)
            header = self._construct_head(n_bytes, pack_type)
            self.sock.send(header+payload)
        else:
            n_bytes = 0
            header = self._construct_head(n_bytes, pack_type)
            self.sock.send(header)
        with open(os.path.join(self.log_path, '{}_armchair.log'.format(self.name)), 'a+') as armchair_log:
            armchair_log.write("{}\tsending {}, cid {}\n".format(datetime.now().strftime('%H:%M:%S:%f'), pack_type,self.cid))
        if pack_type != 'ready' and pack_type not in self.GHOST_TYPES:
            self._inflight_packs.append(self.cid)
        return self.cid

    def burn_pipe(self):
        '''
        burns through the pipe by reading all of the ready commands  
        This is not state dependent because _block_on_ready calls recv_pack which is state dep
        Postconditions:  
            Nothing left in the inflight packets buffer  
        '''
        while self._inflight_packs:
            self._block_on_ready()

    def recv_first(self,pack_type):
        '''
        This method should be used sparingly, but is applicable in error handling.  
        It reads through packets until it finds one with the given pack_type.  
        Packets of other types that were read in the process are ignored  
        params:  
            str pack_type: the type of packet to wait for  
        returns:  
            str: type of packet  
            int: the cid of the recived packet  
            Obj: the argmuents/payload  
        '''
        pack_type, cid, payload = self._recv()
        while pack_type != pack_type: #Executor will request save. Burn other prexisting packs
            pack_type, cid, payload = self._recv()
        return pack_type, cid, payload

    def recv_ftp(self):
        '''
        violates the armchair protocol a little bit because rather than sending an entire  
        file as a payload, ftp is sent in a raw stream with delimiters after a sending_files   
        is sent  
        returns:  
            list<tuple<str:bytes>>: linked filenames and files as bytestreams  
        '''
        pack_type, cid, arguments = self.recv_pack()
        assert(pack_type == 'sending_files')
        filenames = arguments[0]
        files = []
        for filename in filenames:
            file_bytes = self.sock.recv_until(self.FTP_EOF)
            files.append((filename, file_bytes))
        #ack the send files
        self.send_pack('ready',cid)
        return files

    def send_ftp(self, filepaths):
        '''
        params:  
            list<str> filenames: the names of the files to send  
            str filepath: the path where all the files live  
        '''
        self.send_pack('sending_files', [os.path.basename(filepath) for filepath in filepaths])
        for filepath in filepaths:
            with open(filepath,'rb') as local_file:
                self.sock.sock.sendfile(local_file)
                self.sock.send(self.FTP_EOF)

    def _block_on_ready(self):
        '''
        used to block until the other side responds with a 'ready' packet  
        Preconditions: self._inflight_packs contains cids of packets that have been sent 
        not yet acknowledged  
        Postconditions:  
            has stalled until a ready command was recieved.  
            The cid in the ready command has been removed from self.inflight_packs  
        '''
        pack_type, _, arguments = self._recv()
        assert (pack_type == 'ready'), "was expecting a ready packet, but instead recieved a {}".format(pack_type)
        cid = arguments[0]
        self._inflight_packs.remove(cid)

    def close(self):
        '''
        shutsdown this Armchair. Will burn through all readys before closing connection
        '''
        self.burn_pipe()
        self.sock.close()
