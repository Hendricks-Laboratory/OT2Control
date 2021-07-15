import os
#start spectrostar_nano if not already working
#spectro_star_path =r'C:\"Program Files"\"SPECTROstar Nano V5.50"\ExeDLL\SPECTROstar_Nano.exe'
#os.system(spectro_star_path)

class PlateReader():
    '''
    This class handles all platereader interactions
    '''
    SPECTRO_ROOT_PATH = "/mnt/c/Program\\ Files/SPECTROstar\\ Nano\\ V5.50/"
    PROTOCOL_PATH = r"'C:\Program Files\SPECTROstar Nano V5.50\User\Definit'"

    def __init__(self):
        self.exec_macro("dummy")
        self.exec_macro("init")
        
    def exec_macro(self, macro, *args):
        '''
        sends a macro command to the platereader and blocks waiting for response. If response
        not ok, it'll crash and burn
        params:
            str macro: should be a macro from the documentation
            *args: associated arguments of the macto
        Postconditions:
            The command has been sent to the PlateReader, if the return status was not 0 (good)
            an error will be thrown
        '''
        exec_str = "{}Cln/DDEClient.exe {}".format(self.SPECTRO_ROOT_PATH, macro)
        #add arguments
        for arg in args:
            exec_str += ' {}'.format(arg)
        exit_code = os.system(exec_str)
        print(exit_code)
        try:
            assert (exit_code == 0)
        except:
            if exit_code < 1000:
                raise Exception("PlateReader rejected command Error")
            elif exit_code == 1000:
                raise Exception("PlateReader Nonexistent Protocol Name Error")
            elif exit_code == 2000:
                raise Exception("PlateReader Communication Error")
            else:
                raise Exception("PlateReader Error. Exited with code {}".format(exit_code))

    def shake(self):
        '''
        executes a shake
        '''
        macro = "Shake"
        shake_type = 2
        shake_freq = 300
        shake_time = 60
        self.exec_macro(macro, shake_type, shake_freq, shake_time)

    def shake_protocol(self):
        '''
        executes  '60s shake'
        '''
        macro = 'run'
        protocol_name = "'60s shake'"
        data_path = r"'C:\Program Files\SPECTROstar Nano V5.50\User\Data'"
        self.exec_macro(macro, protocol_name, self.PROTOCOL_PATH, data_path)

    def edit_layout(self, protocol_name, layout):
        '''
        params:
            str protocol_name: the name of the protocol that will be edited
            list<str> wells: the wells that you want to be used for the protocol ordered.
              (first will be X1, second X2 etc.
        Postcondtions:
            The protocol has had it's layout updated to include only the wells specified
        '''
        well_entries = []
        for i, well in enumerate(layout):
            well_entries.append("{}=X{}".format(well, i))
        well_arg = "'EmptyLayout {}'".format(' '.join(well_entries))
        self.exec_macro('EditLayout', protocol_name, self.PROTOCOL_PATH, well_arg)


    def run_protocol(self, protocol_name, data_path=r"'C:\Program Files\SPECTROstar Nano V5.50\User\Data'", layout=None):
        r'''
        params:
            str protocol_name: the name of the protocol that will be edited
            str data_path: windows path raw string. no quotes on outside. e.g.
              C:\Program Files\SPECTROstar Nano V5.50\User
            list<str> layout: the wells that you want to be used for the protocol ordered.
              (first will be X1, second X2 etc. If not specified will not alter layout)
        '''
        if layout:
            self.edit_layout(protocol_name, layout)
        macro = 'run'
        protocol_name = "'{}'".format(protocol_name)
        data_path = "'{}'".format(data_path)
        self.exec_macro(macro, protocol-name, self.PROTOCOL_PATH, data_path)

    def shutdown(self):
        '''
        closes connection. Use this if you're done with this object at cleanup stage
        '''
        self.exec_macro('Terminate')
