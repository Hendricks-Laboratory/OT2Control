import customtkinter
import gspread
import random
import sys
from oauth2client.service_account import ServiceAccountCredentials


class Board:
    #positions goes row by row
    # board is associated with the position of the same index
    board=[(2,[0]),(2,[0]),(-1,[0]),(0,[0]),(0,[0]),(0,[0]),(0,[0]),(0,[0]),(0,[0]),(0,[0]),(3,[0]),(1,[0])]
    positions=[[0,75],[200,75],[400,75],[0,200],[200,200],[400,200],[0,325],[200,325],[400,400],[0,450],[200,450],[400,450]]
    singlePosition=(2,[0])
    def __init__(self):
        #self.get_contents()
        print("initialized")
        """Initiates Board and calls get content on each square of the board"""
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
        print("create_single_cell")
   
        
        iposition=None
        for i in range(len(self.positions)):
            iposition=i
            
            if self.singlePosition[0]==self.positions[i][0] and self.singlePosition[1]==self.positions[i][1]:
                break
        if iposition is not None:
            if self.board[iposition][0]==0:
                controller.draw_cell()
            elif self.board[iposition][0]==1:
                controller.draw_large_reagents(self)
            elif self.board[iposition][0]==2:
                controller.draw_large_pipets(self)
            elif self.board[iposition][0]==3:
                controller.draw_large_chem(self)
            
            
    def change_single_position(self,pos):
        self.singlePosition=pos
        
        
        #self.print_board()
    def get_contents(self):
        """
        This is the function which will access different parts of the excel sheet based on which square it is looking at at that moment.
        It will then set that board place as equal to that so that we can then have a fillout function
        """
        # using the .json in order to ask for permission to get access to the google sheet. 
        # will need to do this again on the lab computer
        #gc = gspread.service_account("C:/Users/gabep/OneDrive/Documents/school/4th year whitman/GuiTeam/OT2Control/deck-position-gui-556e4624293c.json")
        # Open Spreadsheet by name
        creds=self.get_credentials()
        try:
            spreadsheet = self._get_key_wks(creds)
            for item in spreadsheet:
                #what it should be
                # if item[0]==sys.argv[1]:
                #     print(item[1])
                # what i am using for testing
                if item[0]=='MPH_test8':
                    worksheet=self.find_types(creds,item[1])
            print(self.board)
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
        path = '/home/gabepm100/Documents/hendricks-lab-jupyter-sheets-5363dda1a7e0.json'
        credentials = ServiceAccountCredentials.from_json_keyfile_name(path, scope) 
        return credentials
    
    #  board=[(2,[0]),(2,[0]),(-1,[0]),(-1,[0]),(0,[0]),(0,[0]),      (-1,[0]),(0,[0]),(0,[0]),(0,[0]),(3,[0]),(1,[0])]
    # positions=[[0,75],[200,75],[400,75],[0,200],[200,200],[400,200],[0,325],[200,325],[400,400],[0,450],[200,450],[400,450]]
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
    
    def _get_key_wks(self, credentials):
        gc = gspread.authorize(credentials)
        name_key_wks = gc.open_by_url('https://docs.google.com/spreadsheets/d/1m2Uzk8z-qn2jJ2U1NHkeN7CJ8TQpK3R0Ai19zlAB1Ew/edit#gid=0').get_worksheet(0)
        name_key_pairs = name_key_wks.get_all_values()
        return name_key_pairs
    
    def name_to_num(self,name):
        if name=="tip_rack_20uL":
            return 1
        if name=="tip_rack_300uL":
            return 2
        if name=="tip_rack_1000uL":
            return 3
        if name=="96_well_plate":
            return 4
        if name=="24_well_plate":
            return 5
        if name=="tube_holder_10":
            return 6
        if name=="temp_mod_24_tube":
            return 7
        if name=="":
            return 0
        
    

            
    
              
LARGEFONT =("Verdana", 35)
  
class CTkinterApp(customtkinter.CTk):
    # __init__ function for class tkinterApp 
    def __init__(self): 
         
        # __init__ function for class Tk
        customtkinter.CTk.__init__(self)
        self.position=()
        # creating a container
        container = customtkinter.CTkFrame(self)  
        self.geometry("850x800")
        self.configure(bg="#d9d9d9")
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
        frame = self.frames[cont]
        frame.tkraise()
  
    # def show_frame2(self,page1,board,Page1Canvas):
    #     print("show frame")
    #     frame = self.frames[page1]
    #     frame.tkraise()
    #     board.create_single_cell(,self)
        
        
    def close(self): 
        self.destroy
        print("Should be closing")  
        
    def draw_small_pipets(self,x,y,bor):

        """
        Canvas is the canvas object
        x and y are the x and y cordinates
        """
        #just a mess
        Whetherfilled=[0]*32 +[1]*66
        random.shuffle(Whetherfilled)
        for i in range(len(Whetherfilled)):
            if Whetherfilled[i]==0:
                Whetherfilled[i]="black"
            else:
                Whetherfilled[i]="blue"
        for i in range(8):
            for j in range(12):
                oval=self.c1.create_oval(x+10+(j*15) ,y+3+(i*12)+(i*3),x+22+(j*15) ,y+15.66+(i*12)+(i*3), outline="black", fill=Whetherfilled[i+j],width=2)
                self.c1.tag_bind(oval, '<Button-1>',lambda z: [ bor.change_single_position((x,y)), self.show_frame(Page1)])

    def draw_small_reagents(self,x,y,bor):
        Whetherfilled=[0]*18 +[1]*6
        random.shuffle(Whetherfilled)
        for i in range(len(Whetherfilled)):
            if Whetherfilled[i]==0:
                Whetherfilled[i]="black"
            else:
                Whetherfilled[i]="blue"
                
        for i in range(4):
            for j in range(6):
                oval=self.c1.create_oval(x+10+(j*32) ,y+6+(i*30),x+30+(j*32) ,y+26+(i*30), outline="black", fill=Whetherfilled[i+j],width=2)
                self.c1.tag_bind(oval, '<Button-1>',lambda z: [ bor.change_single_position((x,y)), self.show_frame(Page1)])
    
    def draw_small_chem(self,x,y,bor):
        #first column
        Whetherfilled=[1,0,1,0,1,0,0,0,1,0]
        for i in range(len(Whetherfilled)):
            if Whetherfilled[i]==0:
                Whetherfilled[i]="black"
            else:
                Whetherfilled[i]="blue"
        a=self.c1.create_oval(x+25,y+10,x+55 ,y+40, outline="black", fill=Whetherfilled[0],width=2)
        self.c1.tag_bind(a, '<Button-1>',lambda z: [ bor.change_single_position((x,y)), self.show_frame(Page1)])
        b=self.c1.create_oval(x+25,y+45,x+55 ,y+75, outline="black", fill=Whetherfilled[1],width=2)
        self.c1.tag_bind(b, '<Button-1>',lambda z: [ bor.change_single_position((x,y)), self.show_frame(Page1)])
        c=self.c1.create_oval(x+25,y+80,x+55 ,y+110, outline="black", fill=Whetherfilled[2],width=2)
        self.c1.tag_bind(c, '<Button-1>',lambda z: [ bor.change_single_position((x,y)), self.show_frame(Page1)])
        #second column
        d=self.c1.create_oval(x+60,y+10,x+90 ,y+40, outline="black", fill=Whetherfilled[3],width=2)
        self.c1.tag_bind(d, '<Button-1>',lambda z: [ bor.change_single_position((x,y)), self.show_frame(Page1)])
        e=self.c1.create_oval(x+60,y+45,x+90 ,y+75, outline="black", fill=Whetherfilled[4],width=2)
        self.c1.tag_bind(e, '<Button-1>',lambda z: [ bor.change_single_position((x,y)), self.show_frame(Page1)])
        f=self.c1.create_oval(x+60,y+80,x+90 ,y+110, outline="black", fill=Whetherfilled[5],width=2)
        self.c1.tag_bind(f, '<Button-1>',lambda z: [ bor.change_single_position((x,y)), self.show_frame(Page1)])
        # third column
        g=self.c1.create_oval(x+95,y+20,x+135 ,y+60, outline="black", fill=Whetherfilled[6],width=2)
        self.c1.tag_bind(g, '<Button-1>',lambda z: [ bor.change_single_position((x,y)), self.show_frame(Page1)])
        h=self.c1.create_oval(x+95,y+65,x+135 ,y+105, outline="black", fill=Whetherfilled[7],width=2)
        self.c1.tag_bind(h, '<Button-1>',lambda z: [ bor.change_single_position((x,y)), self.show_frame(Page1)])
        #forth column
        i=self.c1.create_oval(x+140,y+20,x+180 ,y+60, outline="black", fill=Whetherfilled[8],width=2)
        self.c1.tag_bind(i, '<Button-1>',lambda z: [ bor.change_single_position((x,y)), self.show_frame(Page1)])
        j=self.c1.create_oval(x+140,y+65,x+180 ,y+105, outline="black", fill=Whetherfilled[9],width=2)
        self.c1.tag_bind(j, '<Button-1>',lambda z: [ bor.change_single_position((x,y)), self.show_frame(Page1)])
        
    def draw_board(self,bor):
        """
        bor.change_single_position(pos)
        """
        #column one
        rect=self.c1.create_rectangle(0, 75, 200, 200, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((0,75)), self.show_frame(Page1)])
        rect=self.c1.create_rectangle(0, 200, 200, 325, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((0,200)), self.show_frame(Page1)])
        rect=self.c1.create_text(100, 260, text="Plate Reader", fill="white", font=('Helvetica 15 bold'))
        rect=self.c1.create_rectangle(0, 325, 200, 450, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((0,325)), self.show_frame(Page1)])
        rect=self.c1.create_text(100, 385, text="Plate Reader", fill="white", font=('Helvetica 15 bold'))
        rect=self.c1.create_rectangle(0, 450, 200, 575, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((0,450)), self.show_frame(Page1)])
        #column two
        rect=self.c1.create_rectangle(200, 75, 400, 200, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((200,75)), self.show_frame(Page1)])
        rect=self.c1.create_rectangle(200, 200, 400, 325, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((200,200)), self.show_frame(Page1)])
        rect=self.c1.create_rectangle(200, 325, 400, 450, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((200,325)), self.show_frame(Page1)])
        rect=self.c1.create_rectangle(200, 450, 400, 575, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((200,450)), self.show_frame(Page1)])
        #column three
        #trash
        self.c1.create_rectangle(400, 0, 650, 200, fill="#7A797B",outline="black")
        self.c1.create_text(525, 100, text="Trash", fill="white", font=('Helvetica 20 bold'))
        #rest of column three
        rect=self.c1.create_rectangle(400, 200, 600, 325, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((400,200)), self.show_frame(Page1)])
        rect=self.c1.create_rectangle(400, 325, 600, 450, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((400,325)), self.show_frame(Page1)])
        rect=self.c1.create_rectangle(400, 450, 600, 575, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((400,450)), self.show_frame(Page1)])
        
    # def getorigin(self,eventorigin,board):
    #     global x,y
    #     x = eventorigin.x
    #     y = eventorigin.y
    #     print(x,y)
        
    def get_canvas_items(self):
        item_list = self.c1.find_all()
        for item in item_list:
            item_type = self.c1.type(item)   # e.g. "text", "line", etc.
            #if item_type=="oval":
                #print(item_type)
                
    def show_pop_up(self,board,xy):
        print(xy)
        if (xy[0]>0 and xy[0]<200) and (xy[1]>75 and xy[1]<200):
            print("1,1")
        if (xy[0]>0 and xy[0]<200) and (xy[1]>200 and xy[1]<325):
            print("1,2")
        if (xy[0]>0 and xy[0]<200) and (xy[1]>325 and xy[1]<450):
            print("1,3")
        if (xy[0]>0 and xy[0]<200) and (xy[1]>450 and xy[1]<575):
            print("1,4")
        if (xy[0]>200 and xy[0]<400) and (xy[1]>75 and xy[1]<200):
            print("2,1")
        if (xy[0]>200 and xy[0]<400) and (xy[1]>200 and xy[1]<325):
            print("2,2")
        if (xy[0]>200 and xy[0]<400) and (xy[1]>325 and xy[1]<450):
            print("2,3")
        if (xy[0]>200 and xy[0]<400) and (xy[1]>450 and xy[1]<575):
            print("2,4")
        if (xy[0]>400 and xy[0]<600) and (xy[1]>200 and xy[1]<325):
            print("3,2")
        if (xy[0]>400 and xy[0]<600) and (xy[1]>325 and xy[1]<450):
            print("3,3")
        if (xy[0]>400 and xy[0]<600) and (xy[1]>450 and xy[1]<575):
            print("3,4")
            
    def draw_large_reagents(self,board):
        """_summary_

        Args:
            canvas (_type_): _description_
        """
        self.draw_cell()
        Whetherfilled=[0]*18 +[1]*6
        random.shuffle(Whetherfilled)
        for i in range(len(Whetherfilled)):
            if Whetherfilled[i]==0:
                Whetherfilled[i]="black"
            else:
                Whetherfilled[i]="blue"
                
        for i in range(4):
            for j in range(6):
                oval=self.c2.create_oval(30+(j*100) ,25+(i*90),120+(j*100) ,105+(i*90), outline="black", fill=Whetherfilled[i+j],width=2)
                
    def draw_large_pipets(self):
        """
        Canvas is the canvas object
        x and y are the x and y cordinates
        """
        print("large_pipets")
        self.draw_cell()
        #just a mess
        Whetherfilled=[0]*32 +[1]*66
        random.shuffle(Whetherfilled)
        for i in range(len(Whetherfilled)):
            if Whetherfilled[i]==0:
                Whetherfilled[i]="black"
            else:
                Whetherfilled[i]="blue"
        for i in range(8):
            for j in range(12):
                oval=self.c2.create_oval(30+(j*50), 15+(i*50),75+(j*50) ,60+(i*50), outline="black", fill=Whetherfilled[i+j],width=2)

    def draw_large_chem(self,board):
        self.draw_cell()
        Whetherfilled=[1,0,1,0,1,0,0,0,1,0]
        for i in range(len(Whetherfilled)):
            if Whetherfilled[i]==0:
                Whetherfilled[i]="black"
            else:
                Whetherfilled[i]="blue"
        #first column
        a=self.c2.create_oval(25,25,145 ,145, outline="black", fill=Whetherfilled[0],width=2)
        b=self.c2.create_oval(25,150,145,270, outline="black", fill=Whetherfilled[1],width=2)
        c=self.c2.create_oval(25,275,145,395, outline="black", fill=Whetherfilled[2],width=2)
        #second column
        d=self.c2.create_oval(155, 25, 275, 145, outline="black", fill=Whetherfilled[3],width=2)
        e=self.c2.create_oval(155, 150, 275, 270, outline="black", fill=Whetherfilled[4],width=2)
        f=self.c2.create_oval(155, 275, 275, 395, outline="black", fill=Whetherfilled[5],width=2)
        # third column
        g=self.c2.create_oval(280, 50, 440, 210, outline="black", fill=Whetherfilled[6],width=2)
        h=self.c2.create_oval(280, 215, 440, 375, outline="black", fill=Whetherfilled[7],width=2)
        #forth column
        i=self.c2.create_oval(445,50,605 ,210, outline="black", fill=Whetherfilled[8],width=2)
        j=self.c2.create_oval(445,215,605 ,375, outline="black", fill=Whetherfilled[9],width=2)
        
    def draw_cell(self):
        print("draw cell")
        self.c2.delete("all")
        rect=self.c2.create_rectangle(0, 0, 650, 425, fill="#7A797B",outline="black")
    
# first window frame startpage
  
class StartPage(customtkinter.CTkFrame):
    StartPageCanvas=None
    def __init__(self, parent, controller): 
        customtkinter.CTkFrame.__init__(self, parent)
         
        # label of frame Layout 2
        label = customtkinter.CTkLabel(self, text ="Startpage", font = LARGEFONT)
        
        #Create a canvas object
        controller.c1 = customtkinter.CTkCanvas(self, width=650, height=575, borderwidth=0, highlightthickness=0,bg ='#d9d9d9')
        
        controller.board.get_contents()
        

        controller.c1.place(x=100,y=50)
        print('in start page')

        controller.board.create_full_board(controller)

        button= customtkinter.CTkButton(self,text="Close",width=200,command=parent.destroy)
        button.place(x=300,y=650)

# second window frame page1 
class Page1(customtkinter.CTkFrame):
    Page1Canvas=None
     
    def __init__(self, parent, controller):
        customtkinter.CTkFrame.__init__(self, parent)
        label = customtkinter.CTkLabel(self, text ="Page 1", font = LARGEFONT)
        controller.c2 = customtkinter.CTkCanvas(self, width=650, height=425, borderwidth=0, highlightthickness=0,bg ='#d9d9d9')
        print("in page 1")

        controller.c2.place(x=100,y=50)
        controller.draw_cell()

        controller.board.create_single_cell(controller)
        # button to show frame 2 with text
        # layout2
        button1 = customtkinter.CTkButton(self, text ="StartPage", command = lambda : controller.show_frame(StartPage))
        
        # putting the button in its place 
        # by using grid
        button1.grid(row = 1, column = 1, padx = 10, pady = 10)
        
  
# Driver Code
app = CTkinterApp()
app.mainloop()
