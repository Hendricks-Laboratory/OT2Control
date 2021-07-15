import os
#start spectrostar_nano if not already working
#spectro_star_path =r'C:\"Program Files"\"SPECTROstar Nano V5.50"\ExeDLL\SPECTROstar_Nano.exe'
#os.system(spectro_star_path)

class PlateReader():
    '''
    This class handles all platereader interactions
    '''
    SPECTRO_ROOT_PATH = "/mnt/c/Program\\ Files/SPECTROstar\\ Nano\\ V5.50/"

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
        try:
            assert (exit_code == 0)
        except:
            if exit_code < 1000:
                raise Exception("PlateReader Parameter Range Error")
            elif exit_code == 1000:
                raise Exception("PlateReader Nonexistent Protocol Name Error")
            elif exit_code == 2000:
                raise Exception("PlateReader Communication Error")
            else:
                raise Exception("PlateReader Error. Exited with code {}".format(exit_code))

    def shake(self):
        macro = "Shake"
        shake_type = "0"
        shake_freq = 100
        shake_time = 1
        self.exec_macro(macro, shake_type, shake_freq, shake_time)

    def shake_protocol(self):
        macro = 'run'
        protocol_name = "'60s shake'"
        protocol_path = r"'C:\Program Files\SPECTROstar Nano V5.50\User\Definit'"
        #protocol_path = r"'C:\Program Files\SPECTROstar Nano V5.50\User\DefLVIs'"
        data_path = r"'C:\Program Files\SPECTROstar Nano V5.50\User\Data'"
        breakpoint()
        self.exec_macro(macro, protocol_name, protocol_path, data_path)

    def 
