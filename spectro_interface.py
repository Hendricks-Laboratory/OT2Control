import os
import win32ui
import dde
#start spectrostar_nano if not already working
#spectro_star_path =r'C:\"Program Files"\"SPECTROstar Nano V5.50"\ExeDLL\SPECTROstar_Nano.exe'
#os.system(spectro_star_path)

class PlateReader():
    '''
    This class handles all platereader interactions
    '''
    def __init__(self):
        self.server = dde.CreateServer()
        self.server.Create('MyDataProxy')
        self.conversation = dde.CreateConversation(self.server)
        #self.conversation.ConnectTo("SPECTROstar_Nano", "DDEServerConv1")
        #self.conversation.Exec('Dummy')
        conv2 = self.conversation.ConnectTo("SPECTROstar_Nano","SZDDESYS_TOPIC")
        breakpoint()
        self.conversation.Request('DdeServerStatus')

    def platein(self):
        '''
        pulls in the plate if it's out
        '''
        self.conversation.Exec("PlateIn")

    def plateout(self):
        '''
        sticks out the plate if it's in
        '''
        self.conversation.Exec("PlateOut")
