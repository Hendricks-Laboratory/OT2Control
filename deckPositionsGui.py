import customtkinter
import gspread
import random


class Board:
    #positions goes row by row
    # board is associated with the position of the same index
    
    def __init__(self):
        self.board=[(0,[0]),(0,[0]),(-1,[0]),(-1,[0]),(0,[0]),(0,[0]),(-1,[0]),(0,[0]),(0,[0]),(0,[0]),(0,[0]),(0,[0])]
        self.positions=[[0,75],[200,75],[400,75],[0,200],[200,200],[400,200],[0,325],[200,325],[400,400],[0,450],[200,450],[400,450]]
        self.singlePosition=(0,[0])
        #self.get_contents()
        print("initialized")
        """Initiates Board and calls get content on each square of the board"""
        controller.drawBoard(canvas,self.board,controller)
        
        for i in range(len(self.board)):
                # calls the get content function and depending on that it will set up the board to match the values gotten from the excell sheet
                
                #self.get_contents()
                
                # if that board share is set to 1 it draws the reagents and if it is 2 it draws the pipets
                if self.board[i][0]==1:
                    controller.drawReagents(canvas,self.positions[i][0],self.positions[i][1],self.board,controller)
                elif self.board[i][0]==2:
                    controller.drawPipets(canvas,self.positions[i][0],self.positions[i][1],self.board,controller)
                elif self.board[i][0]==3:
                    controller.draw_small_chem(self.positions[i][0],self.positions[i][1],self)
    
    def create_single_cell(self,controller):
        print("create_single_cell")
        print(self.singlePosition)
        
        iposition=None
        for i in range(len(self.positions)):
            iposition=i
            
            if self.singlePosition[0]==self.positions[i][0] and self.singlePosition[1]==self.positions[i][1]:
                break
        if iposition is not None:
            if self.board[iposition][0]==0 or self.board[iposition][0]==None:
                controller.draw_cell()
            elif self.board[iposition][0]==1:
                controller.draw_large_reagents()
            elif self.board[iposition][0]==2:
                controller.draw_large_pipets()
            elif self.board[iposition][0]==3:
                controller.draw_large_chem()
            
            
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
        #spreadsheet = gc.open_by_url("https://docs.google.com/spreadsheets/d/1m2Uzk8z-qn2jJ2U1NHkeN7CJ8TQpK3R0Ai19zlAB1Ew/edit#gid=0")
        spreadsheet = gspread.open_by_url("https://docs.google.com/spreadsheets/d/1m2Uzk8z-qn2jJ2U1NHkeN7CJ8TQpK3R0Ai19zlAB1Ew/edit#gid=0")
        
        #opens the sheet by name. sheet one is the name of the page inside the spreadsheet
        worksheet = spreadsheet.sheet1
        try:
            spreadsheet = self._get_key_wks(creds)
            for item in spreadsheet:
                #what it should be
                # if item[0]==sys.argv[1]:
                #     print(item[1])
                # what i am using for testing
                if item[0]=='MPH_test8':
                    worksheet=self.find_types(creds,item[1])
        except IndexError:
            raise Exception('Spreadsheet Name/Key pair was not found. Check the dict spreadsheet \
            and make sure the spreadsheet name is spelled exactly the same as the reaction \
            spreadsheet.')
        
        # gets the values from the first row of the sheet
        values_list = worksheet.row_values(1)
        print(values_list)
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
        #change the number swhen we design
        if name=="tip_rack_20uL":
            return 1
        elif name=="tip_rack_300uL":
            return 1
        elif name=="tip_rack_1000uL":
            return 6
        elif name=="96_well_plate":
            return 1
        elif name=="24_well_plate":
            return 2
        elif name=="tube_holder_10":
            return 3
        elif name=="temp_mod_24_tube":
            return 7
        elif name==None:
            return 0
        
    

    def print_board(self):
        for i in range(len(self.board)):
            print(self.board[i])
            
    
              
LARGEFONT =("Verdana", 35)
  
class tkinterApp(customtkinter.CTk):
     
    # __init__ function for class tkinterApp 
    def __init__(self, *args, **kwargs): 
         
        # __init__ function for class Tk
        customtkinter.CTk.__init__(self, *args, **kwargs)
         
        # creating a container
        container = customtkinter.CTkFrame(self)  
        self.geometry("850x800")
        self.configure(bg="#d9d9d9")
        self.title("Deck Positions")

        container.pack(side = "top", fill = "both", expand = True) 
  
        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)
  
        # initializing frames to an empty array
        self.frames = {}  
  
        # iterating through a tuple consisting
        # of the different page layouts
        for F in (StartPage, Page1):
  
            frame = F(container, self)
  
            # initializing frame of that object from
            # startpage, page1, page2 respectively with 
            # for loop
            self.frames[F] = frame 
  
            frame.grid(row = 0, column = 0, sticky ="nsew")
  
        self.show_frame(StartPage)
  
    # to display the current frame passed as
    # parameter
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
        
        
    def Close(self): 
        self.destroy
        print("Should be closing")  
        
    def drawPipets(self,canvas,x,y,bor,controller):

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
                self.c1.tag_bind(oval, '<Button-1>',lambda z: [ bor.change_single_position((x,y)), self.show_frame(Page1),bor.create_single_cell(self)])

    def drawReagents(self,canvas,x,y,bor,controller):
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
                self.c1.tag_bind(oval, '<Button-1>',lambda z: [ bor.change_single_position((x,y)), self.show_frame(Page1),bor.create_single_cell(self)])
    
    def drawChem(self,canvas,x,y,bor,controller):
        #first column
        Whetherfilled=[1,0,1,0,1,0,0,0,1,0]
        for i in range(len(Whetherfilled)):
            if Whetherfilled[i]==0:
                Whetherfilled[i]="black"
            else:
                Whetherfilled[i]="blue"
        a=self.c1.create_oval(x+25,y+10,x+55 ,y+40, outline="black", fill=Whetherfilled[0],width=2)
        self.c1.tag_bind(a, '<Button-1>',lambda z: [ bor.change_single_position((x,y)), self.show_frame(Page1),bor.create_single_cell(self)])
        b=self.c1.create_oval(x+25,y+45,x+55 ,y+75, outline="black", fill=Whetherfilled[1],width=2)
        self.c1.tag_bind(b, '<Button-1>',lambda z: [ bor.change_single_position((x,y)), self.show_frame(Page1),bor.create_single_cell(self)])
        c=self.c1.create_oval(x+25,y+80,x+55 ,y+110, outline="black", fill=Whetherfilled[2],width=2)
        self.c1.tag_bind(c, '<Button-1>',lambda z: [ bor.change_single_position((x,y)), self.show_frame(Page1),bor.create_single_cell(self)])
        #second column
        d=self.c1.create_oval(x+60,y+10,x+90 ,y+40, outline="black", fill=Whetherfilled[3],width=2)
        self.c1.tag_bind(d, '<Button-1>',lambda z: [ bor.change_single_position((x,y)), self.show_frame(Page1),bor.create_single_cell(self)])
        e=self.c1.create_oval(x+60,y+45,x+90 ,y+75, outline="black", fill=Whetherfilled[4],width=2)
        self.c1.tag_bind(e, '<Button-1>',lambda z: [ bor.change_single_position((x,y)), self.show_frame(Page1),bor.create_single_cell(self)])
        f=self.c1.create_oval(x+60,y+80,x+90 ,y+110, outline="black", fill=Whetherfilled[5],width=2)
        self.c1.tag_bind(f, '<Button-1>',lambda z: [ bor.change_single_position((x,y)), self.show_frame(Page1),bor.create_single_cell(self)])
        # third column
        g=self.c1.create_oval(x+95,y+20,x+135 ,y+60, outline="black", fill=Whetherfilled[6],width=2)
        self.c1.tag_bind(g, '<Button-1>',lambda z: [ bor.change_single_position((x,y)), self.show_frame(Page1),bor.create_single_cell(self)])
        h=self.c1.create_oval(x+95,y+65,x+135 ,y+105, outline="black", fill=Whetherfilled[7],width=2)
        self.c1.tag_bind(h, '<Button-1>',lambda z: [ bor.change_single_position((x,y)), self.show_frame(Page1),bor.create_single_cell(self)])
        #forth column
        i=self.c1.create_oval(x+140,y+20,x+180 ,y+60, outline="black", fill=Whetherfilled[8],width=2)
        self.c1.tag_bind(i, '<Button-1>',lambda z: [ bor.change_single_position((x,y)), self.show_frame(Page1),bor.create_single_cell(self)])
        j=self.c1.create_oval(x+140,y+65,x+180 ,y+105, outline="black", fill=Whetherfilled[9],width=2)
        self.c1.tag_bind(j, '<Button-1>',lambda z: [ bor.change_single_position((x,y)), self.show_frame(Page1),bor.create_single_cell(self)])
        
    def drawBoard(self,canvas,bor,controller):
        """
        
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
        rect=canvas.create_rectangle(400, 0, 650, 200, fill="#7A797B",outline="black")
        rect=canvas.create_text(525, 100, text="Trash", fill="white", font=('Helvetica 20 bold'))
        #rest of column three
        rect=self.c1.create_rectangle(400, 200, 600, 325, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((400,200)), self.show_frame(Page1),bor.create_single_cell(self)])
        rect=self.c1.create_rectangle(400, 325, 600, 450, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((400,325)), self.show_frame(Page1),bor.create_single_cell(self)])
        rect=self.c1.create_rectangle(400, 450, 600, 575, fill="#7A797B",outline="black")
        self.c1.tag_bind(rect, '<Button-1>',lambda x: [ bor.change_single_position((400,450)), self.show_frame(Page1),bor.create_single_cell(self)])
        
    # def getorigin(self,eventorigin,board):
    #     global x,y
    #     x = eventorigin.x
    #     y = eventorigin.y
    #     print(x,y)
        
    def get_canvas_items(self,canvas):
        item_list = canvas.find_all()
        for item in item_list:
            item_type = canvas.type(item)   # e.g. "text", "line", etc.
            #if item_type=="oval":
                #print(item_type)
                
    def showPopUp(self,board,xy):
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
            
    def draw_large_reagents(self):
        print("hello")
    
# first window frame startpage
  
class StartPage(customtkinter.CTkFrame):
    def __init__(self, parent, controller): 
        customtkinter.CTkFrame.__init__(self, parent)
         
        # label of frame Layout 2
        label = customtkinter.CTkLabel(self, text ="Startpage", font = LARGEFONT)
         
        
        
        #Create a canvas object
        c = customtkinter.CTkCanvas(self, width=650, height=575, borderwidth=0, highlightthickness=0,bg ='#d9d9d9')
        

        c.place(x=100,y=50)
        board=Board(c,controller)

        button= customtkinter.CTkButton(self,text="Close",width=200,command=parent.destroy)
        button.place(x=300,y=650)
  
  
  
# second window frame page1 
class Page1(customtkinter.CTkFrame):
     
    def __init__(self, parent, controller):
         
        customtkinter.CTkFrame.__init__(self, parent)
        label = customtkinter.CTkLabel(self, text ="Page 1", font = LARGEFONT)
        c = customtkinter.CTkCanvas(self, width=650, height=425, borderwidth=0, highlightthickness=0,bg ='#d9d9d9')
        c.place(x=100,y=50)
        rect=c.create_rectangle(0, 0, 650, 425, fill="#7A797B",outline="black")
        self.drawChem(c)
        # button to show frame 2 with text
        # layout2
        button1 = customtkinter.CTkButton(self, text ="StartPage", command = lambda : controller.show_frame(StartPage))
     
        # putting the button in its place 
        # by using grid
        button1.grid(row = 1, column = 1, padx = 10, pady = 10)
        
    def drawReagents(self,canvas):
        Whetherfilled=[0]*18 +[1]*6
        random.shuffle(Whetherfilled)
        for i in range(len(Whetherfilled)):
            if Whetherfilled[i]==0:
                Whetherfilled[i]="black"
            else:
                Whetherfilled[i]="blue"
                
        for i in range(4):
            for j in range(6):
                oval=canvas.create_oval(30+(j*100) ,25+(i*90),120+(j*100) ,105+(i*90), outline="black", fill=Whetherfilled[i+j],width=2)
                
    def drawPipets(self,canvas):

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
                oval=self.c2.create_oval(30+(j*50), 15+(i*50),75+(j*50) ,60+(i*50), outline="black", fill=Whetherfilled[i+j],width=2)

    def draw_large_chem(self):
        self.draw_cell()
        Whetherfilled=[1,0,1,0,1,0,0,0,1,0]
        for i in range(len(Whetherfilled)):
            if Whetherfilled[i]==0:
                Whetherfilled[i]="black"
            else:
                Whetherfilled[i]="blue"
        a=canvas.create_oval(25,25,145 ,145, outline="black", fill=Whetherfilled[0],width=2)
        b=canvas.create_oval(25,150,145,270, outline="black", fill=Whetherfilled[1],width=2)
        c=canvas.create_oval(25,275,145,395, outline="black", fill=Whetherfilled[2],width=2)
        #second column
        d=canvas.create_oval(155, 25, 275, 145, outline="black", fill=Whetherfilled[3],width=2)
        e=canvas.create_oval(155, 150, 275, 270, outline="black", fill=Whetherfilled[4],width=2)
        f=canvas.create_oval(155, 275, 275, 395, outline="black", fill=Whetherfilled[5],width=2)
        # third column
        g=canvas.create_oval(280, 140, 440, 300, outline="black", fill=Whetherfilled[6],width=2)
        h=canvas.create_oval(280, 305, 440, 465, outline="black", fill=Whetherfilled[7],width=2)
        #forth column
        i=canvas.create_oval(405,90,410 ,210, outline="black", fill=Whetherfilled[8],width=2)
        j=canvas.create_oval(405,215,410 ,335, outline="black", fill=Whetherfilled[9],width=2)


                
  
       
  
# Driver Code
app = tkinterApp()
app.mainloop()
