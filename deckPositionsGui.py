import customtkinter
from CTkToolTip import *
import gspread
import random
from oauth2client.service_account import ServiceAccountCredentials
import sys


class Board:
    #positions goes row by row
    # board is associated with the position of the same index
    def __init__(self):
        #self.get_contents()
        
        """Initiates Board and calls get content on each square of the board"""
        self.board=[(0,[0]),(0,[0]),(-1, ),(-1,[0]),(0,[0]),(0,[0]),(-1,[0]),(0,[0]),(0,[0]),(0,[0]),(0,[0]),(0,[0])]
        self.positions=[[0,75],[200,75],[400,75],[0,200],[200,200],[400,200],[0,325],[200,325],[400,325],[0,450],[200,450],[400,450]]
        self.singlePosition=(0,[0])
    def create_full_board(self,controller):
        controller.draw_board(self)
        
        for i in range(len(self.board)):
                # calls the get content function and depending on that it will set up the board to match the values gotten from the excell sheet
                
                #self.get_contents()
                
                # if that board share is set to 1 it draws the reagents and if it is 2 it draws the pipets
                if self.board[i][0]==1:
                    controller.draw_small_reagents(self.positions[i][0],self.positions[i][1],self)
                elif self.board[i][0]==2:
                    controller.draw_small_pipets(self.positions[i][0],self.positions[i][1],self)
                elif self.board[i][0]==3:
                    controller.draw_small_chem(self.positions[i][0],self.positions[i][1],self)
    
    def create_single_cell(self,controller):
   
        iposition=None
        for i in range(len(self.positions)):
            iposition=i
            
            if self.singlePosition[0]==self.positions[i][0] and self.singlePosition[1]==self.positions[i][1]:
                break
        if iposition is not None:
            if self.board[iposition][0]==0 or self.board[iposition][0]==None:
                controller.draw_cell()
            elif self.board[iposition][0]==1:
                controller.draw_large_reagents(self)
            elif self.board[iposition][0]==2:
                controller.draw_large_pipets(self)
            elif self.board[iposition][0]==3:
                controller.draw_large_chem(self)
            
            
    def change_single_position(self,pos):
        self.singlePosition=pos
        
        
    def get_contents(self):
        """
        This is the function which will access different parts of the excel sheet based on which square it is looking at at that moment.
        It will then set that board place as equal to that so that we can then have a fillout function
        """
        # using the .json in order to ask for permission to get access to the google sheet. 
        # will need to do this again on the lab computer
        
        # Open Spreadsheet by name
        creds=self.get_credentials()
        try:
            spreadsheet = self._get_key_wks(creds)
            for item in spreadsheet:
                #what it should be
                if item[0]==sys.argv[1]:
                    worksheet=self.find_types(creds,item[1])
                    self.get_chemicals(creds,item[1])
        except IndexError:
            raise Exception('Spreadsheet Name/Key pair was not found. Check the dict spreadsheet \
            and make sure the spreadsheet name is spelled exactly the same as the reaction \
            spreadsheet.')
        
        # gets the values from the first row of the sheet
        # values_list = worksheet.row_values(1)
        # print(values_list)
        return -1

    #the next three functions were taken from controler py and are used toget credentials for the google sheet
    def get_credentials(self):
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        #get login credentials from local file. Your json file here
        path = '/mnt/c/Users/science_356_lab/Robot_Files/OT2Control/Credentials/hendricks-lab-jupyter-sheets-5363dda1a7e0.json'
        credentials = ServiceAccountCredentials.from_json_keyfile_name(path, scope) 
        return credentials
    

    def find_types(self,credentials,url):
        gc = gspread.authorize(credentials)
        spreadsheet = gc.open_by_url('https://docs.google.com/spreadsheets/d/'+url+'/edit#gid=0')
        worksheet=spreadsheet.get_worksheet(2)
        
        self.board[0]=(self.name_to_num(worksheet.cell(2, 1).value),[])
        self.board[1]=(self.name_to_num(worksheet.cell(2, 2).value),[])
        
        self.board[4]=(self.name_to_num(worksheet.cell(5, 2).value),[])
        self.board[5]=(self.name_to_num(worksheet.cell(5, 3).value),[])
        
        self.board[7]=(self.name_to_num(worksheet.cell(8, 2).value),[])
        self.board[8]=(self.name_to_num(worksheet.cell(8, 3).value),[])
        
        self.board[9]=(self.name_to_num(worksheet.cell(11, 1).value),[])
        self.board[10]=(self.name_to_num(worksheet.cell(11, 2).value),[])
        self.board[11]=(self.name_to_num(worksheet.cell(11, 3).value),[])
        return worksheet
    
    def get_chemicals(self,credentials,url):
        gc = gspread.authorize(credentials)
        spreadsheet = gc.open_by_url('https://docs.google.com/spreadsheets/d/'+url+'/edit#gid=0')
        worksheet=spreadsheet.get_worksheet(4)
        rows=worksheet.get_all_values()
        for row in rows:
           #self.board= [(0,[0]),(0,[0]),(-1,[0]),(-1,[0]),(0,[0]),(0,[0]),(-1,[0]),(0,[0]),(0,[0]),(0,[0]),(0,[0]),(0,[0])]
            if row[3].isnumeric():
                
                spot=self.get_spot(int(row[3]))
                self.board[spot][1].append((row[0],row[2]))
                
        


    def _get_key_wks(self, credentials):
        gc = gspread.authorize(credentials)
        name_key_wks = gc.open_by_url('https://docs.google.com/spreadsheets/d/1m2Uzk8z-qn2jJ2U1NHkeN7CJ8TQpK3R0Ai19zlAB1Ew/edit#gid=0').get_worksheet(0)
        name_key_pairs = name_key_wks.get_all_values()
        return name_key_pairs
    
    def name_to_num(self,name):
        if name=="tip_rack_20uL":
            return 2
        elif name=="tip_rack_300uL":
            return 2
        elif name=="tip_rack_1000uL":
            return 2
        elif name=="96_well_plate":
            return 2
        elif name=="24_well_plate":
            return 1
        elif name=="tube_holder_10":
            return 3
        elif name=="temp_mod_24_tube":
            return 1
        elif name=="":
            return 0
        
        
    def get_spot(self,loc):
        if loc==10:
            return 0
        elif loc==11:
            return 1
        elif loc==8:
            return 4
        elif loc==9:
            return 5
        elif loc==5:
            return 7
        elif loc==6:
            return 8
        elif loc==1:
            return 9
        elif loc==2:
            return 10
        elif loc==3:
            return 11
        
              

  
class CTkinterApp(customtkinter.CTk):
    # __init__ function for class tkinterApp 
    def __init__(self): 
         
        # __init__ function for class Tk
        customtkinter.CTk.__init__(self)
        self.position=()
        # creating a container
        container = customtkinter.CTkFrame(self,fg_color="#252526")  
        self.geometry("850x800")

        self.title("Deck Positions")

        container.pack(side = "top", fill = "both", expand = True) 
        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)
        self.board=Board()
        # initializing frames to an empty array
        self.frames = {}  
        self.c1=None
        self.c2=None
        # iterating through a tuple consisting
        # of the different page layouts
        i=0
        frame = Page1(container, self)
        # initializing frame of that object from
        # startpage, page1
        self.frames[Page1] = frame 

        frame.grid(row = 0, column = 0, sticky ="nsew")
        frame = StartPage(container, self)
        # initializing frame of that object from
        # startpage, page1
        self.frames[StartPage] = frame 

        frame.grid(row = 0, column = 0, sticky ="nsew")
        
  
        self.show_frame(StartPage)
  
    # to display the current frame passed as
    # parameter
    def show_frame(self, cont):
        self.clear_labels()
        frame = self.frames[cont]
        frame.tkraise()
  
        
    def close(self): 
        self.destroy
        print("Should be closing")  
        
    def draw_small_pipets(self,x,y,board):

        """
        Canvas is the canvas object
        x and y are the x and y cordinates
        """
        filled="black"
        index=board.positions.index([x,y])
        for i in range(8):
            for j in range(12):
                strin=chr(i+65)+str(j+1)
                out=[touple for touple in board.board[index][1] if touple[1]==strin]
                if len(out)!=0: 
                    filled='blue' 
                else:
                    filled='black'
                oval=self.c1.create_oval(x+10+(j*15) ,y+3+(i*12)+(i*3),x+22+(j*15) ,y+15.66+(i*12)+(i*3), outline="black", fill=filled,width=2)
                self.c1.tag_bind(oval, '<Button-1>',lambda z: [ board.change_single_position((x,y)), self.show_frame(Page1),board.create_single_cell(self)])

    def draw_small_reagents(self,x,y,board):
        filled="black"
        index=board.positions.index([x,y])
        for i in range(4):
            for j in range(6):
                strin=chr(i+65)+str(j+1)
                out=[touple for touple in board.board[index][1] if touple[1]==strin]
                if len(out)!=0: 
                    filled='blue' 
                else:
                    filled='black'
                oval=self.c1.create_oval(x+10+(j*32) ,y+6+(i*30),x+30+(j*32) ,y+26+(i*30), outline="black", fill=filled,width=2)
                self.c1.tag_bind(oval, '<Button-1>',lambda z: [ board.change_single_position((x,y)), self.show_frame(Page1),board.create_single_cell(self)])
    
    def draw_small_chem(self,x,y,board):
        #first column
        filled="black"
        index=board.positions.index([x,y])

        #first column
        out=[touple for touple in board.board[index][1] if touple[1]=="A1"]
        if len(out)!=0: 
            filled='blue' 
        else:
            filled='black'
        a=self.c1.create_oval(x+25,y+10,x+55 ,y+40, outline="black", fill=filled,width=2)
        self.c1.tag_bind(a, '<Button-1>',lambda z: [ board.change_single_position((x,y)), self.show_frame(Page1), board.create_single_cell(self)])
        out=[touple for touple in board.board[index][1] if touple[1]=="B1"]
        if len(out)!=0: 
            filled='blue' 
        else:
            filled='black'
        b=self.c1.create_oval(x+25,y+45,x+55 ,y+75, outline="black", fill=filled,width=2)
        self.c1.tag_bind(b, '<Button-1>',lambda z: [ board.change_single_position((x,y)), self.show_frame(Page1), board.create_single_cell(self)])
        out=[touple for touple in board.board[index][1] if touple[1]=="C1"]
        if len(out)!=0: 
            filled='blue' 
        else:
            filled='black'
        c=self.c1.create_oval(x+25,y+80,x+55 ,y+110, outline="black", fill=filled,width=2)
        self.c1.tag_bind(c, '<Button-1>',lambda z: [ board.change_single_position((x,y)), self.show_frame(Page1), board.create_single_cell(self)])
        #second column
        out=[touple for touple in board.board[index][1] if touple[1]=="A2"]
        if len(out)!=0: 
            filled='blue' 
        else:
            filled='black'
        d=self.c1.create_oval(x+60,y+10,x+90 ,y+40, outline="black", fill=filled,width=2)
        self.c1.tag_bind(d, '<Button-1>',lambda z: [ board.change_single_position((x,y)), self.show_frame(Page1), board.create_single_cell(self)])
        out=[touple for touple in board.board[index][1] if touple[1]=="B2"]
        if len(out)!=0: 
            filled='blue' 
        else:
            filled='black'
        e=self.c1.create_oval(x+60,y+45,x+90 ,y+75, outline="black", fill=filled,width=2)
        self.c1.tag_bind(e, '<Button-1>',lambda z: [ board.change_single_position((x,y)), self.show_frame(Page1), board.create_single_cell(self)])
        out=[touple for touple in board.board[index][1] if touple[1]=="C2"]
        if len(out)!=0: 
            filled='blue' 
        else:
            filled='black'
        f=self.c1.create_oval(x+60,y+80,x+90 ,y+110, outline="black", fill=filled,width=2)
        self.c1.tag_bind(f, '<Button-1>',lambda z: [ board.change_single_position((x,y)), self.show_frame(Page1), board.create_single_cell(self)])
        # third column
        out=[touple for touple in board.board[index][1] if touple[1]=="A3"]
        if len(out)!=0: 
            filled='blue' 
        else:
            filled='black'
        g=self.c1.create_oval(x+95,y+20,x+135 ,y+60, outline="black", fill=filled,width=2)
        self.c1.tag_bind(g, '<Button-1>',lambda z: [ board.change_single_position((x,y)), self.show_frame(Page1), board.create_single_cell(self)])
        out=[touple for touple in board.board[index][1] if touple[1]=="B3"]
        if len(out)!=0: 
            filled='blue' 
        else:
            filled='black'
        h=self.c1.create_oval(x+95,y+65,x+135 ,y+105, outline="black", fill=filled,width=2)
        self.c1.tag_bind(h, '<Button-1>',lambda z: [ board.change_single_position((x,y)), self.show_frame(Page1), board.create_single_cell(self)])
        #forth column
        out=[touple for touple in board.board[index][1] if touple[1]=="A4"]
        if len(out)!=0: 
            filled='blue' 
        else:
            filled='black'
        i=self.c1.create_oval(x+140,y+20,x+180 ,y+60, outline="black", fill=filled,width=2)
        self.c1.tag_bind(i, '<Button-1>',lambda z: [ board.change_single_position((x,y)), self.show_frame(Page1), board.create_single_cell(self)])
        out=[touple for touple in board.board[index][1] if touple[1]=="B4"]
        if len(out)!=0: 
            filled='blue' 
        else:
            filled='black'
        j=self.c1.create_oval(x+140,y+65,x+180 ,y+105, outline="black", fill=filled,width=2)
        self.c1.tag_bind(j, '<Button-1>',lambda z: [ board.change_single_position((x,y)), self.show_frame(Page1), board.create_single_cell(self)])
        
    def draw_board(self,bor):
        """
        bor.change_single_position(pos)
        """
        #column one
        rect=self.c1.create_rectangle(0, 75, 200, 200, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((0,75)), self.show_frame(Page1),bor.create_single_cell(self)])
        rect=self.c1.create_rectangle(0, 200, 200, 325, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((0,200)), self.show_frame(Page1),bor.create_single_cell(self)])
        rect=self.c1.create_text(100, 260, text="Plate Reader", fill="white", font=('Helvetica 15 bold'))
        rect=self.c1.create_rectangle(0, 325, 200, 450, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((0,325)), self.show_frame(Page1),bor.create_single_cell(self)])
        rect=self.c1.create_text(100, 385, text="Plate Reader", fill="white", font=('Helvetica 15 bold'))
        rect=self.c1.create_rectangle(0, 450, 200, 575, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((0,450)), self.show_frame(Page1),bor.create_single_cell(self)])
        #column two
        rect=self.c1.create_rectangle(200, 75, 400, 200, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((200,75)), self.show_frame(Page1),bor.create_single_cell(self)])
        rect=self.c1.create_rectangle(200, 200, 400, 325, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((200,200)), self.show_frame(Page1),bor.create_single_cell(self)])
        rect=self.c1.create_rectangle(200, 325, 400, 450, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((200,325)), self.show_frame(Page1),bor.create_single_cell(self)])
        rect=self.c1.create_rectangle(200, 450, 400, 575, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((200,450)), self.show_frame(Page1),bor.create_single_cell(self)])
        #column three
        #trash
        self.c1.create_rectangle(400, 0, 650, 200, fill="#7A797B",outline="black")
        self.c1.create_text(525, 100, text="Trash", fill="white", font=('Helvetica 20 bold'))
        #rest of column three
        rect=self.c1.create_rectangle(400, 200, 600, 325, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((400,200)), self.show_frame(Page1),bor.create_single_cell(self)])
        rect=self.c1.create_rectangle(400, 325, 600, 450, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((400,325)), self.show_frame(Page1),bor.create_single_cell(self)])
        rect=self.c1.create_rectangle(400, 450, 600, 575, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((400,450)), self.show_frame(Page1),bor.create_single_cell(self)])
        
        
    def get_canvas_items(self):
        item_list = self.c1.find_all()
        for item in item_list:
            item_type = self.c1.type(item)   # e.g. "text", "line", etc.
            
    
    def clear_labels(self):
        """Clears the labels."""
        if hasattr(self, 'lbl'):
            self.lbl.destroy()
            
    def draw_large_reagents(self,board):
        """_summary_

        Args:
            canvas (_type_): _description_
        """
        self.draw_cell()
        filled='black'
        self.clear_labels()
        lbl = customtkinter.CTkLabel(self.c2,text='',fg_color="blue",text_color='white')
        lbl.place(x=0, y=0, anchor="n")
        self.lbl = lbl
        index=board.positions.index(list(board.singlePosition))
        for i in range(4):
            for j in range(6):
                strin=chr(i+65)+str(j+1)
                out=[touple for touple in board.board[index][1] if touple[1]==strin]
                if len(out)!=0:
                    filled='blue'
                    texttag=out[0][0]
                else:
                    filled='black'
                    texttag="empty"
                ovalname=str(i)+str(j)
                ovalname=self.c2.create_oval(30+(j*100) ,25+(i*90),120+(j*100) ,105+(i*90), outline="black", fill=filled,width=2)
                self.c2.tag_bind(ovalname, "<Enter>", lambda event: on_enter(event,self.c2,lbl,filled,50+(i*100),50+(j*90)))
                self.c2.tag_bind(ovalname, "<Leave>", lambda event: on_leave(event,lbl))
                
    def draw_large_pipets(self,board):
        """
        Canvas is the canvas object
        x and y are the x and y cordinates
        """
        self.draw_cell()
        filled="black"
        self.clear_labels()
        lbl = customtkinter.CTkLabel(self.c2,text='',fg_color="blue",text_color='white')
        lbl.place(x=0, y=0, anchor="n")
        self.lbl = lbl
        index=board.positions.index(list(board.singlePosition))
        d={}
        for i in range(8):
            for j in range(12):
                strin=chr(i+65)+str(j+1)
                out=[touple for touple in board.board[index][1] if touple[1]==strin]
                if len(out)!=0:
                    filled='blue'
                    texttag=out[0][0]
                else:
                    filled='black'
                    texttag="empty"
                
                d["group"+str(i)+str(j)]=(self.c2.create_oval(30+(j*50), 15+(i*50),75+(j*50) ,60+(i*50), outline="black", fill=filled,width=2,tag=texttag),i,j)
                self.c2.tag_bind(d["group"+str(i)+str(j)], "<Enter>", lambda event: on_enter(event,self.c2,self.lbl,filled,50+(d["group"+str(i)+str(j)][1]*50),45+(d["group"+str(i)+str(j)][2]*50)))
                self.c2.tag_bind(d["group"+str(i)+str(j)], "<Leave>", lambda event: on_leave(event,self.lbl))
            
                
                
    def draw_large_chem(self,board):
        self.draw_cell()
        #lbl = customtkinter.CTkLabel(self.c2)
        filled='black'
        index=board.positions.index(list(board.singlePosition))
        self.clear_labels()
        lbl = customtkinter.CTkLabel(self.c2,text='',fg_color="blue",text_color='white')
        lbl.place(x=0, y=0, anchor="nw")
        self.lbl = lbl

        #first column
        out=[touple for touple in board.board[index][1] if touple[1]=="A1"]
        if len(out)!=0:
            filled='blue'
            texttag=out[0][0]
        else:
            filled='black'
            texttag="empty"
        A=self.c2.create_oval(25,25,145 ,145, outline="black", fill=filled,width=2,tag=texttag)
        self.c2.tag_bind(A, "<Enter>", lambda event: on_enter(event,self.c2,lbl,filled,80,75))
        self.c2.tag_bind(A, "<Leave>", lambda event: on_leave(event,lbl))
        out=[touple for touple in board.board[index][1] if touple[1]=="B1"]
        if len(out)!=0:
            filled='blue'
            texttag=out[0][0]
        else:
            filled='black'
            texttag="empty"
        b=self.c2.create_oval(25,150,145,270, outline="black", fill=filled,width=2,tag=texttag)
        self.c2.tag_bind(b, "<Enter>", lambda event: on_enter(event,self.c2,lbl,filled,80,205))
        self.c2.tag_bind(b, "<Leave>", lambda event: on_leave(event,lbl))
        out=[touple for touple in board.board[index][1] if touple[1]=="C1"]
        if len(out)!=0:
            filled='blue'
            texttag=out[0][0]
        else:
            texttag="empty"  
        c=self.c2.create_oval(25,275,145,395, outline="black", fill=filled,width=2,tag=texttag)
        self.c2.tag_bind(c, "<Enter>", lambda event: on_enter(event,self.c2,lbl,filled,80,330))
        self.c2.tag_bind(c, "<Leave>", lambda event: on_leave(event,lbl))
        
        #second column
        out=[touple for touple in board.board[index][1] if touple[1]=="A2"]
        if len(out)!=0:
            filled='blue'
            texttag=out[0][0]
        else:
            filled='black'
            texttag="empty"
        d=self.c2.create_oval(155, 25, 275, 145, outline="black", fill=filled,width=2,tag=texttag)
        self.c2.tag_bind(d, "<Enter>", lambda event: on_enter(event,self.c2,lbl,filled,210,80))
        self.c2.tag_bind(d, "<Leave>", lambda event: on_leave(event,lbl))
        out=[touple for touple in board.board[index][1] if touple[1]=="B2"]
        if len(out)!=0:
            filled='blue'
            texttag=out[0][0]
        else:
            filled='black'
            texttag="empty"
        e=self.c2.create_oval(155, 150, 275, 270, outline="black", fill=filled,width=2,tag=texttag)
        self.c2.tag_bind(e, "<Enter>", lambda event: on_enter(event,self.c2,lbl,filled,210,205))
        self.c2.tag_bind(e, "<Leave>", lambda event: on_leave(event,lbl))
        out=[touple for touple in board.board[index][1] if touple[1]=="C2"]
        if len(out)!=0:
            filled='blue'
            texttag=out[0][0]
        else:
            filled='black'
            texttag="empty"
        f=self.c2.create_oval(155, 275, 275, 395, outline="black", fill=filled,width=2,tag=texttag)
        self.c2.tag_bind(f, "<Enter>", lambda event: on_enter(event,self.c2,lbl,filled,210,330))
        self.c2.tag_bind(f, "<Leave>", lambda event: on_leave(event,lbl))
        
        # third column
        out=[touple for touple in board.board[index][1] if touple[1]=="A3"]
        if len(out)!=0:
            filled='blue'
            texttag=out[0][0]
        else:
            filled='black'
            texttag="empty"
        g=self.c2.create_oval(280, 50, 440, 210, outline="black", fill=filled,width=2,tag=texttag)
        self.c2.tag_bind(g, "<Enter>", lambda event: on_enter(event,self.c2,lbl,filled,350,125))
        self.c2.tag_bind(g, "<Leave>", lambda event: on_leave(event,lbl))
        
        out=[touple for touple in board.board[index][1] if touple[1]=="B3"]
        if len(out)!=0:
            filled='blue'
            texttag=out[0][0]
        else:
            filled='black'
            texttag="empty"
        h=self.c2.create_oval(280, 215, 440, 375, outline="black", fill=filled,width=2,tag=texttag)
        self.c2.tag_bind(h, "<Enter>", lambda event: on_enter(event,self.c2,lbl,filled,350,290))
        self.c2.tag_bind(h, "<Leave>", lambda event: on_leave(event,lbl))
        
        
        #forth column
        out=[touple for touple in board.board[index][1] if touple[1]=="A4"]
        if len(out)!=0:
            filled='blue'
            texttag=out[0][0]
        else:
            filled='black'
            texttag="empty"
        i=self.c2.create_oval(445,50,605 ,210, outline="black", fill=filled,width=2,tag=texttag)
        self.c2.tag_bind(i, "<Enter>", lambda event: on_enter(event,self.c2,lbl,filled,525,125))
        self.c2.tag_bind(i, "<Leave>", lambda event: on_leave(event,lbl))
    
        out=[touple for touple in board.board[index][1] if touple[1]=="B4"]
        if len(out)!=0:
            filled='blue'
            texttag=out[0][0]
        else:
            filled='black'
            texttag="empty"
        j=self.c2.create_oval(445,215,605 ,375, outline="black", fill=filled,width=2,tag=texttag)
        self.c2.tag_bind(j, "<Enter>", lambda event: on_enter(event,self.c2,lbl,filled,525,290))
        self.c2.tag_bind(j, "<Leave>", lambda event: on_leave(event,lbl))
        
    def draw_cell(self):

        self.c2.delete("all")
        rect=self.c2.create_rectangle(0, 0, 650, 425, fill="#7A797B",outline="black")
        
        
        
        
def on_enter(e,canvas,lbl,filled,xaxis,yaxis):

    # find the canvas item below mouse cursor
    item = canvas.find_withtag("current")
    # get the tags for the item
    if item :
        tags = canvas.gettags(item)
        if tags :
            lbl.place(x=xaxis,y=yaxis, anchor="n")
            # show it using the label
            lbl.configure(text=tags[0],fg_color=filled)
        else:
        # clear the label text if no canvas item found
            lbl.configure(text="")
    
def on_leave(e,lbl):
    # clear the label text
    #lbl.configure(text="")
    #lbl.place(x=0,y=0)
    if e.x < 0 or e.y < 0 or e.x > e.widget.winfo_width() or e.y > e.widget.winfo_height():
        lbl.place_forget()
    




    
# first window frame startpage
  
class StartPage(customtkinter.CTkFrame):
    StartPageCanvas=None
    def __init__(self, parent, controller): 
        customtkinter.CTkFrame.__init__(self, parent)
         
        # label of frame Layout 2
        label = customtkinter.CTkLabel(self, text ="Startpage", font = ("Verdana", 35))
        
        #Create a canvas object
        controller.c1 = customtkinter.CTkCanvas(self, width=650, height=575, borderwidth=0, highlightthickness=0,bg ='#d9d9d9')
        
        controller.board.get_contents()
        

        controller.c1.place(x=100,y=50)


        controller.board.create_full_board(controller)

        button= customtkinter.CTkButton(self,text="Close",width=200,command=parent.destroy)
        button.place(x=300,y=650)

# second window frame page1 
class Page1(customtkinter.CTkFrame):
    Page1Canvas=None
     
    def __init__(self, parent, controller):
        customtkinter.CTkFrame.__init__(self, parent)
        label = customtkinter.CTkLabel(self, text ="Page 1", font = ("Verdana", 35))
        controller.c2 = customtkinter.CTkCanvas(self, width=650, height=425, borderwidth=0, highlightthickness=0,bg ='#d9d9d9')


        controller.c2.place(x=100,y=50)
        controller.draw_cell()

        controller.board.create_single_cell(controller)
        # button to show frame 2 with text
        # layout2
        button1 = customtkinter.CTkButton(self, text ="StartPage", command = lambda : controller.show_frame(StartPage))
        tooltip_1 = CTkToolTip(button1, message="50")
        
        # putting the button in its place 
        # by using grid
        button1.grid(row = 1, column = 1, padx = 10, pady = 10)
        
  
# Driver Code
app = CTkinterApp()
app.mainloop()
