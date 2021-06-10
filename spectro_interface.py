import os
import win32ui
import dde
#start spectrostar_nano if not already working
#spectro_star_path =r'C:\"Program Files"\"SPECTROstar Nano V5.50"\ExeDLL\SPECTROstar_Nano.exe'
#os.system(spectro_star_path)
print('launched')
server = dde.CreateServer()
server.Create('MyDataProxy')
conversation = dde.CreateConversation(server)
conversation.ConnectTo("SPECTROstar_Nano", "DDEServerConv1")
print('connection established')
conversation.Exec("PlateIn")
print('command executed')
