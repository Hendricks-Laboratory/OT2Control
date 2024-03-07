import customtkinter
from CTkToolTip import *
import gspread
import random
from oauth2client.service_account import ServiceAccountCredentials
import sys


class deck:
    
    def __init__(self):
        #self.get_contents()
        
        """
        Initiates Booard
            Includes a deck array, positions array, and single positions touple.
            when populated the deck arry takes on the form of:
            [(type of deck,[(Chemical,Postiotion),(Chemical,Position)]),(type of deck,[(Chemical,Postiotion),(Chemical,Position)])...]
            The positions array is never changed
            Single position is for when you click on one of the positions of the full deck position 
        """
        
        self.deck=[(0,[0]),(0,[0]),(0,[0]),(-1,[0]),(0,[0]),(0,[0]),(-1,[0]),(0,[0]),(0,[0]),(0,[0]),(0,[0]),(-1,[0])]
        self.positions=[[0,450],[200,450],[400,450],[0,325],[200,325],[400,325],[0,200],[200,200],[400,200],[0,75],[200,75],[400,75],]
        self.singlePosition=(0,[0])
        
    def create_full_deck(self,controller):
        """checks for each position on the deck whether or not there is something there. If there is than it calls the function that should populate that square.

        Args:
            controller (Tkinter frame and canvas): This controlls and divises up work amoung its function for which the Tkinter canvas is apart of.
        """
        controller.draw_deck()
        for i in range(len(self.deck)):
                # calls the get content function and depending on that it will set up the deck to match the values gotten from the excell sheet
                
                #self.get_contents()
                if type(self.deck[i][0])==int or self.deck[i][0]==None:
                    pass
                # if that deck share is set to 1 it draws the reagents and if it is 2 it draws the pipets
                elif self.deck[i][0]=="24_well_plate" or self.deck[i][0]=="temp_mod_24_tube":
                    controller.draw_small_24_well_plate(self.positions[i][0],self.positions[i][1])
                elif self.deck[i][0]=="96_well_plate":
                    controller.draw_small_96_well_plate(self.positions[i][0],self.positions[i][1])
                elif self.deck[i][0]=="tube_holder_10":
                    controller.draw_small_tube_holder(self.positions[i][0],self.positions[i][1])
                elif self.deck[i][0][:3]=="tip":
                    controller.draw_small_tiprack(self.positions[i][0],self.positions[i][1])

    def create_single_cell(self,controller):
        """
        analyzes the specific deck position that is selected and whether or not something is there it calls the functions to make sure the right thing is displayed

        Args:
            controller (Tkinter frame and canvas): This controlls and divises up work amoung its function for which the Tkinter canvas is apart of.
        """
   
        iposition=None
        for i in range(len(self.positions)):
            iposition=i
            
            if self.singlePosition[0]==self.positions[i][0] and self.singlePosition[1]==self.positions[i][1]:
                break
        if iposition is not None and type(self.deck[iposition][0])!=int:
            
            if self.deck[iposition][0]==0 or self.deck[iposition][0]==None:
                controller.draw_cell()
            elif self.deck[iposition][0]=="24_well_plate" or self.deck[iposition][0]=="temp_mod_24_tube":
                controller.draw_large_24_well_plate()
            elif self.deck[iposition][0]=="96_well_plate":
                controller.draw_large_96_well_plate()
            elif self.deck[iposition][0]=="tube_holder_10":
                controller.draw_large_tube_holder()
            elif self.deck[iposition][0][:3]=="tip":
                controller.draw_large_tiprack()
            
    def change_single_position(self,pos):
        """changes self.singlePosition to be equal to the position for which is clicked. Needs to be a function because of the way lambda processes commands

        Args:
            pos (touple): Touple to match the same position that is clicked
        """
        self.singlePosition=pos
        
        
    def get_contents(self):
        """
        This is the function which will access different parts of the excel sheet based on which square it is looking at at that moment.
        It will then set that deck place as equal to that so that we can then have a fillout function
        """
        # Open Spreadsheet by name
        creds=self.get_credentials()
        try:
            spreadsheet = self._get_key_wks(creds)
            for item in spreadsheet:
                #what it should be
                if item[0]==sys.argv[1]:
                    worksheet=self.find_types(creds,item[1])
                    self.get_chemicals(creds,item[1])
                # if item[0]=='MPH_test8':
                #     self.find_types(creds,item[1])
                #     self.get_chemicals(creds,item[1])
        except IndexError:
            raise Exception('Spreadsheet Name/Key pair was not found. Check the dict spreadsheet \
            and make sure the spreadsheet name is spelled exactly the same as the reaction \
            spreadsheet.')
        return -1

    #the next three functions were taken from controler py and are used toget credentials for the google sheet
    def get_credentials(self):
        """accesses the json file which contails the permission file for the class of google sheets we need to access.
        it then uses the gspread function from_json_keyfile_name to get the credentials from the json

        Returns:
            credentials: google sheet credentials.
        """
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        #get login credentials from local file. Your json file here
        path = '/mnt/c/Users/science_356_lab/Robot_Files/OT2Control/Credentials/hendricks-lab-jupyter-sheets-5363dda1a7e0.json'
        credentials = ServiceAccountCredentials.from_json_keyfile_name(path, scope) 
        return credentials
    

    def find_types(self,credentials,url):
        """_summary_

        Args:
            credentials: google sheet credentials.
            url (_type_): _description_
        """
        gc = gspread.authorize(credentials)
        spreadsheet = gc.open_by_url('https://docs.google.com/spreadsheets/d/'+url+'/edit#gid=0')
        worksheet=spreadsheet.get_worksheet(2)
        rackval=worksheet.cell(2, 1).value
        self.deck[9]=(rackval,[])
        if rackval!=None:
            if rackval[0:3]=='tip':
                self.deck[9][1].append(worksheet.cell(3,1).value)
        rackval=worksheet.cell(2, 2).value
        self.deck[10]=(rackval,[])
        if rackval!=None:
            if rackval[0:3]=='tip':
                self.deck[10][1].append(worksheet.cell(3,2).value)
            
        rackval=worksheet.cell(5, 2).value
        self.deck[7]=(rackval,[])
        if rackval!=None:
            if rackval[0:3]=='tip':
                self.deck[7][1].append(worksheet.cell(6,2).value)
        rackval=worksheet.cell(5, 3).value
        self.deck[8]=(rackval,[])
        if rackval!=None:
            if rackval[0:3]=='tip':
                self.deck[8][1].append(worksheet.cell(6,3).value)
            
        rackval=worksheet.cell(8, 2).value
        self.deck[4]=(rackval,[])
        if rackval!=None:
            if rackval[0:3]=='tip':
                self.deck[4][1].append(worksheet.cell(9,2).value)
        rackval=worksheet.cell(8, 3).value
        self.deck[5]=(rackval,[])
        if rackval!=None:
            if rackval[0:3]=='tip':
                self.deck[5][1].append(worksheet.cell(9,3).value)
            
        rackval=worksheet.cell(11, 1).value
        self.deck[0]=(rackval,[])
        if rackval!=None:
            if rackval[0:3]=='tip':
                self.deck[0][1].append(worksheet.cell(12,1).value)
        rackval=worksheet.cell(11, 2).value
        self.deck[1]=(rackval,[])
        if rackval!=None:
            if rackval[0:3]=='tip':
                self.deck[1][1].append(worksheet.cell(12,2).value)
        rackval=worksheet.cell(11, 3).value
        self.deck[2]=(rackval,[])
        if rackval!=None:
            if rackval[0:3]=='tip':
                self.deck[2][1].append(worksheet.cell(12,3).value)
        print(self.deck)


    def get_chemicals(self,credentials,url):
        """
        Accesses the google sheets and checks through reagents tab of the google sheet. It then takes all of the needed data regarding
        the differnt reagents and populates the 

        Args:
            credentials (googlesheet athenticationcredentials): gotten by calling the get_credentials function. 
            url (string): part of the url that changes.
        """
        gc = gspread.authorize(credentials)
        spreadsheet = gc.open_by_url('https://docs.google.com/spreadsheets/d/'+url+'/edit#gid=0')
        worksheet=spreadsheet.worksheet("reagent_info")
        columns=worksheet.get_all_values()
        for column in columns:
            if column[3].isnumeric():
                self.deck[int(column[3])-1][1].append((column[0],column[2]))
        
    def _get_key_wks(self, credentials):
        """

        Args:
            credentials (googlesheet athenticationcredentials): gotten by calling the get_credentials function. 

        Returns:
            list of tuples where the first element of the touple is the same of the google sheet and the second element is the part of the url specific
            to that google sheet
        """
        gc = gspread.authorize(credentials)
        name_key_wks = gc.open_by_url('https://docs.google.com/spreadsheets/d/1m2Uzk8z-qn2jJ2U1NHkeN7CJ8TQpK3R0Ai19zlAB1Ew/edit#gid=0').get_worksheet(0)
        name_key_pairs = name_key_wks.get_all_values()
        return name_key_pairs

  
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
        self.deckPositions=deck()
        self.frames = {}  
        self.c1=None
        self.c2=None
        self.deckNameLabel=None
        self.reagentLabel=None
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
        
    def draw_small_96_well_plate(self,x,y):

        """
        x and y are the x and y cordinates of where the 96 well plate is placed on the deck
        """
        filled="black"
        index=self.deckPositions.deck.positions.index([x,y])
        for i in range(8):
            for j in range(12):
                positionalString=chr(i+65)+str(j+1)
                positionalOutput=[touple for touple in self.deckPositions.deck[index][1] if touple[1]==positionalString]
                if len(positionalOutput)!=0: 
                    filled='blue' 
                else:
                    filled='black'
                oval=self.c1.create_oval(x+10+(j*15) ,y+3+(i*12)+(i*3),x+22+(j*15) ,y+15.66+(i*12)+(i*3), outline="black", fill=filled,width=2)
                
                
    def draw_small_tiprack(self,x,y):
        """
        x and y are the x and y cordinates of where the tiprack is placed on the deck

        """
        filled="black"
        index=self.deckPositions.positions.index([x,y])
        for i in range(12):
            for j in range(8):
                positionalString=chr(j+65)+str(i+1)
                if self.deckPositions.deck[index][1][0]==positionalString:
                    filled='#F5DEB3'
                oval=self.c1.create_oval(x+10+(i*15) ,y+3+(j*15),x+22+(i*15) ,y+15.66+(j*15), outline="black", fill=filled,width=2)
                self.c1.tag_bind(oval, '<Button-1>',lambda z: [ self.deckPositions.change_single_position((x,y)), self.show_frame(Page1),self.deckPositions.create_single_cell(self)])
    
    def draw_small_24_well_plate(self,x,y):
        """
        x and y are the x and y cordinates of where the 24 well plate is placed on the deck

        """
        try:
            filled="black"
            index=self.deckPositions.positions.index([x,y])
            for i in range(4):
                for j in range(6):
                    positionString=chr(i+65)+str(j+1)
                    positionalOutput=[touple for touple in self.deckPositions.deck[index][1] if touple[1]==positionString]
                    if len(positionalOutput)!=0: 
                        filled='blue' 
                        if positionalOutput[0][0]=="empty":
                            filled="black"
                    else:
                        filled='black'
                    oval=self.c1.create_oval(x+10+(j*32) ,y+6+(i*30),x+30+(j*32) ,y+26+(i*30), outline="black", fill=filled,width=2)
                    self.c1.tag_bind(oval, '<Button-1>',lambda z: [ self.deckPositions.change_single_position((x,y)), self.show_frame(Page1),self.deckPositions.create_single_cell(self)])
        except TypeError:
            pass
    def draw_small_tube_holder(self,x,y):
        """
        x and y are the x and y cordinates of where the tube holder is placed on the deck

        """
        index=self.deckPositions.positions.index([x,y])
        #first column
        self.draw_individual_small_10_wells(x,y,index,25,55,10,40,"A1")
        self.draw_individual_small_10_wells(x,y,index,25,55,45,75,"B1")
        self.draw_individual_small_10_wells(x,y,index,25,55,80,110,"C1")
        #second column
        self.draw_individual_small_10_wells(x,y,index,60,90,10,40,"A2")
        self.draw_individual_small_10_wells(x,y,index,60,90,45,75,"B2")
        self.draw_individual_small_10_wells(x,y,index,60,90,80,110,"C2")
        # third column
        self.draw_individual_small_10_wells(x,y,index,95,135,20,60,"A3")
        self.draw_individual_small_10_wells(x,y,index,95,135,65,105,"B3")
        #forth column
        self.draw_individual_small_10_wells(x,y,index,140,180,20,60,"A4")
        self.draw_individual_small_10_wells(x,y,index,140,180,65,105,"B4")
        
        
    def draw_individual_small_10_wells(self, x,y,index,x1,x2,y1,y2,spot):
        """draws the 10 well plate on the dexk

        Args:
            x (int): x coordinate
            y (int): y co0rdanate
            index (int): index of the position where the 10 wells plate is to be placed
            x1 (int): oval coordinate x1
            x2 (int): oval coordinate x2
            y1 (int): oval coordinate y1
            y2 (int): oval coordinate y2
            spot (string): location on physical deck
        """
        try:
            out=[touple for touple in self.deckPositions.deck[index][1] if touple[1]==spot]
            if len(out)!=0: 
                filled='blue' 
            else:
                filled='black'
            j=self.c1.create_oval(x+x1,y+y1,x+x2 ,y+y2, outline="black", fill=filled,width=2)
            self.c1.tag_bind(j, '<Button-1>',lambda z: [ self.deckPositions.change_single_position((x,y)), self.show_frame(Page1), self.deckPositions.create_single_cell(self)])
        except TypeError:
            pass
        
    def draw_deck(self):
        """
        draws the plain deck with nothing on it
        """
        #column one
        rect=self.c1.create_rectangle(0, 75, 200, 200, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ self.deckPositions.change_single_position((0,75)), self.show_frame(Page1),self.deckPositions.create_single_cell(self)])
        rect=self.c1.create_rectangle(0, 200, 200, 325, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ self.deckPositions.change_single_position((0,200)), self.show_frame(Page1),self.deckPositions.create_single_cell(self)])
        rect=self.c1.create_text(100, 260, text="Plate Reader", fill="white", font=('Helvetica 15 bold'))
        rect=self.c1.create_rectangle(0, 325, 200, 450, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ self.deckPositions.change_single_position((0,325)), self.show_frame(Page1),self.deckPositions.create_single_cell(self)])
        rect=self.c1.create_text(100, 385, text="Plate Reader", fill="white", font=('Helvetica 15 bold'))
        rect=self.c1.create_rectangle(0, 450, 200, 575, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ self.deckPositions.change_single_position((0,450)), self.show_frame(Page1),self.deckPositions.create_single_cell(self)])
        #column two
        rect=self.c1.create_rectangle(200, 75, 400, 200, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ self.deckPositions.change_single_position((200,75)), self.show_frame(Page1),self.deckPositions.create_single_cell(self)])
        rect=self.c1.create_rectangle(200, 200, 400, 325, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ self.deckPositions.change_single_position((200,200)), self.show_frame(Page1),self.deckPositions.create_single_cell(self)])
        rect=self.c1.create_rectangle(200, 325, 400, 450, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ self.deckPositions.change_single_position((200,325)), self.show_frame(Page1),self.deckPositions.create_single_cell(self)])
        rect=self.c1.create_rectangle(200, 450, 400, 575, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ self.deckPositions.change_single_position((200,450)), self.show_frame(Page1),self.deckPositions.create_single_cell(self)])
        #column three
        #trash
        self.c1.create_rectangle(400, 0, 650, 200, fill="#7A797B",outline="black")
        self.c1.create_text(525, 100, text="Trash", fill="white", font=('Helvetica 20 bold'))
        #rest of column three
        rect=self.c1.create_rectangle(400, 200, 600, 325, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ self.deckPositions.change_single_position((400,200)), self.show_frame(Page1),self.deckPositions.create_single_cell(self)])
        rect=self.c1.create_rectangle(400, 325, 600, 450, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ self.deckPositions.change_single_position((400,325)), self.show_frame(Page1),self.deckPositions.create_single_cell(self)])
        rect=self.c1.create_rectangle(400, 450, 600, 575, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ self.deckPositions.change_single_position((400,450)), self.show_frame(Page1),self.deckPositions.create_single_cell(self)])
        
            
    
    def clear_labels(self):
        """Clears the labels."""
        if hasattr(self, 'lbl'):
            self.lbl.destroy()
            
    def draw_large_24_well_plate(self):
        """
        draws the close up 24 well plate
        """
        self.draw_cell()
        self.clear_labels()
        lbl = customtkinter.CTkLabel(self.c2,text='',fg_color="blue",text_color='white')
        lbl.place(x=0, y=0, anchor="n")
        self.lbl = lbl
        index=self.deckPositions.positions.index(list(self.deckPositions.singlePosition))
        self.deckNameLabel.configure(text=self.deckPositions.deck[index][0])
        self.draw_individual_small_24_wells(index,lbl,0,0)
        self.draw_individual_small_24_wells(index,lbl,0,1)
        self.draw_individual_small_24_wells(index,lbl,0,2)
        self.draw_individual_small_24_wells(index,lbl,0,3)
        self.draw_individual_small_24_wells(index,lbl,0,4)
        self.draw_individual_small_24_wells(index,lbl,0,5)
        
        self.draw_individual_small_24_wells(index,lbl,1,0)
        self.draw_individual_small_24_wells(index,lbl,1,1)
        self.draw_individual_small_24_wells(index,lbl,1,2)
        self.draw_individual_small_24_wells(index,lbl,1,3)
        self.draw_individual_small_24_wells(index,lbl,1,4)
        self.draw_individual_small_24_wells(index,lbl,1,5)
        
        self.draw_individual_small_24_wells(index,lbl,2,0)
        self.draw_individual_small_24_wells(index,lbl,2,1)
        self.draw_individual_small_24_wells(index,lbl,2,2)
        self.draw_individual_small_24_wells(index,lbl,2,3)
        self.draw_individual_small_24_wells(index,lbl,2,4)
        self.draw_individual_small_24_wells(index,lbl,2,5)
        
        self.draw_individual_small_24_wells(index,lbl,3,0)
        self.draw_individual_small_24_wells(index,lbl,3,1)
        self.draw_individual_small_24_wells(index,lbl,3,2)
        self.draw_individual_small_24_wells(index,lbl,3,3)
        self.draw_individual_small_24_wells(index,lbl,3,4)
        self.draw_individual_small_24_wells(index,lbl,3,5)


    def draw_individual_small_24_wells(self,index,lbl,i,j):
        """
        draws the individual wells of the 24 well plate

        Args:
            index (int): int of the index of the deck which was clicked on
            lbl (lable): custom tkinter label object
            i (int): x offset
            j (int): y offset
        """
        try:
            positionString=chr(i+65)+str(j+1)
            positionalOutput=[touple for touple in self.deckPositions.deck[index][1] if touple[1]==positionString]
            if len(positionalOutput)!=0:
                filled='blue'
                texttag=positionalOutput[0][0]
                if positionalOutput[0][0]=="empty":
                            filled="black"
            else:
                filled='black'
                texttag="empty"
            ovalname=str(i)+str(j)
            ovalname=self.c2.create_oval(30+(j*100) ,25+(i*90),120+(j*100) ,105+(i*90), outline="black", fill=filled,width=2,tag=texttag)
            self.c2.tag_bind(ovalname, "<Enter>", lambda event: on_enter(event,self.c2,lbl,filled,75+(j*100),50+(i*90)))
            self.c2.tag_bind(ovalname, "<Leave>", lambda event: on_leave(event,lbl))
        except: 
            pass    
    def draw_large_96_well_plate(self):
        """
        draws the close up 96 well plate
        """
        self.draw_cell()
        filled="black"
        self.clear_labels()
        lbl = customtkinter.CTkLabel(self.c2,text='',fg_color="blue",text_color='white')
        lbl.place(x=0, y=0, anchor="n")
        self.lbl = lbl
        index=self.deckPositions.positions.index(list(self.deckPositions.singlePosition))
        self.deckNameLabel.configure(text=self.deckPositions.deck[index][0])
        for i in range(8):
            for j in range(12):
                positionString=chr(i+65)+str(j+1)
                positionalOutput=[touple for touple in self.deckPositions.deck[index][1] if touple[1]==positionString]
                if len(positionalOutput)!=0:
                    filled='blue'
                    if positionalOutput[0][0]=="empty":
                        filled="black"
                    texttag=positionalOutput[0][0]
                else:
                    filled='black'
                    texttag="empty"
                self.c2.create_oval(30+(j*50), 15+(i*50),75+(j*50) ,60+(i*50), outline="black", fill=filled,width=2,tag=texttag)
               
                
    
    def draw_large_tiprack(self):
        """
        draws the close up of the tiprack
        """
        self.draw_cell()
        index=self.deckPositions.positions.index(list(self.deckPositions.singlePosition))
        self.deckNameLabel.configure(text=self.deckPositions.deck[index][0])
        filled='black'
        for i in range(12):
            for j in range(8):
                strin=chr(j+65)+str(i+1)
                if self.deckPositions.deck[index][1][0]==strin:
                    filled='#F5DEB3'
                self.c2.create_oval(30+(i*50), 15+(j*50),75+(i*50) ,60+(j*50), outline="black", fill=filled,width=2)
                
                
    def draw_large_tube_holder(self):
        """
        draws the close up of the tubeholder_10
        """
        self.draw_cell()
        #lbl = customtkinter.CTkLabel(self.c2)
        index=self.deckPositions.positions.index(list(self.deckPositions.singlePosition))
        self.deckNameLabel.configure(text=self.deckPositions.deck[index][0])
        self.clear_labels()
        lbl = customtkinter.CTkLabel(self.c2,text='',fg_color="blue",text_color='white')
        lbl.place(x=0, y=0, anchor="nw")
        self.lbl = lbl
        #first column
        self.draw_individual_large_reagent(index,lbl, 25,145,25,145,"A1")
        self.draw_individual_large_reagent(index,lbl, 25,145,150,270,"B1")
        self.draw_individual_large_reagent(index,lbl, 25,145,275,395,"C1")        
        #second column
        self.draw_individual_large_reagent(index,lbl, 155,275,25,145,"A2")
        self.draw_individual_large_reagent(index,lbl, 155,275,150,270,"B2")
        self.draw_individual_large_reagent(index,lbl, 155,275,275,395,"C2")        
        # third column
        self.draw_individual_large_reagent(index,lbl, 280,440,50,210,"A3")
        self.draw_individual_large_reagent(index,lbl, 280,440,215,375,"B3")  
        #forth column
        self.draw_individual_large_reagent(index,lbl, 445,605,50,210,"A4")
        self.draw_individual_large_reagent(index,lbl, 445,605,215,375,"B4")
        
    def draw_individual_large_reagent(self,index,lbl,x1,x2,y1,y2,spot):
        """
        draws the individual reagent 

        Args:
            Args:
            index (int): index of the position where the 10 wells plate is to be placed
            lbl (Label object) Tkinter label object
            x1 (int): oval coordinate x1
            x2 (int): oval coordinate x2
            y1 (int): oval coordinate y1
            y2 (int): oval coordinate y2
            spot (string): location on physical deck
        """
        positionalOutput=[touple for touple in self.deckPositions.deck[index][1] if touple[1]==spot]
        if len(positionalOutput)!=0:
            filled='blue'
            texttag=positionalOutput[0][0]
            if texttag=="empty":
                filled="black"
        else:
            filled='black'
            texttag="empty"
        j=self.c2.create_oval(x1,y1,x2 ,y2, outline="black", fill=filled,width=2,tag=texttag)
        self.c2.tag_bind(j, "<Enter>", lambda event: on_enter(event,self.c2,lbl,filled,(x1+x2)/2,((y1+y2)/2)-5))
        self.c2.tag_bind(j, "<Leave>", lambda event: on_leave(event,lbl))
        
    def draw_cell(self):
        """
        draws the blank close up cell
        """

        self.c2.delete("all")
        rect=self.c2.create_rectangle(0, 0, 650, 425, fill="#7A797B",outline="black")
        
def on_enter(e,canvas,lbl,filled,xaxis,yaxis):
    # find the canvas item below mouse cursor
    try:
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
    except:
        pass
    
def on_leave(e,lbl):
    # clear the label text
    #lbl.configure(text="")
    #lbl.place(x=0,y=0)
    if e.x < 0 or e.y < 0 or e.x > e.widget.winfo_width() or e.y > e.widget.winfo_height():
        lbl.place_forget()
    




    
# first window frame startpage
  
class StartPage(customtkinter.CTkFrame):
    def __init__(self, parent, controller): 
        customtkinter.CTkFrame.__init__(self, parent)
         
        # label of frame Layout 2
        #label = customtkinter.CTkLabel(self, text ="Startpage", font = ("Verdana", 35))
        
        #Create a canvas object
        controller.c1 = customtkinter.CTkCanvas(self, width=650, height=575, borderwidth=0, highlightthickness=0,bg ='#d9d9d9')
        
        controller.deckPositions.get_contents()

        controller.c1.place(x=100,y=50)


        controller.deckPositions.create_full_deck(controller)

        button= customtkinter.CTkButton(self,text="Close",width=200,command=parent.destroy)
        button.place(x=300,y=650)

# second window frame page1 
class Page1(customtkinter.CTkFrame):
     
    def __init__(self, parent, controller):
        customtkinter.CTkFrame.__init__(self, parent)
        controller.deckNameLabel = customtkinter.CTkLabel(self, text ="", font = ("Verdana", 35),anchor='n')
        controller.deckNameLabel.place(x=100,y=50)
        controller.c2 = customtkinter.CTkCanvas(self, width=650, height=425, borderwidth=0, highlightthickness=0,bg ='#d9d9d9')
        controller.c2.place(x=100,y=100)
        controller.draw_cell()
        controller.reagentLabel = customtkinter.CTkLabel(self, text ="", font = ("Verdana", 35),anchor='n')
        controller.reagentLabel.place(x=400,y=50)
        

        controller.deckPositions.create_single_cell(controller)
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
