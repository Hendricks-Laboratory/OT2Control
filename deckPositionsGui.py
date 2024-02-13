import customtkinter
import gspread
import random

class Board:
    #positions goes row by row
    # board is associated with the position of the same index
    board=[(2,[0]),(2,[0]),(-1,[0]),(0,[0]),(0,[0]),(0,[0]),(0,[0]),(0,[0]),(0,[0]),(0,[0]),(3,[0]),(1,[0])]
    positions=[[0,75],[200,75],[400,75],[0,200],[200,200],[400,200],[0,325],[200,325],[400,400],[0,450],[200,450],[400,450]]
    def __init__(self,canvas,controller):
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
                    controller.drawChem(canvas,self.positions[i][0],self.positions[i][1],self.board,controller)
        
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
            cell = worksheet.find("Dash")
            url = worksheet.row_values(cell.row)[1]
            spreadsheet = gspread.open_by_url("https://docs.google.com/spreadsheets/d/"+url)
        except IndexError:
            raise Exception('Spreadsheet Name/Key pair was not found. Check the dict spreadsheet \
            and make sure the spreadsheet name is spelled exactly the same as the reaction \
            spreadsheet.')
        
        
        
        # gets the values from the first row of the sheet
        values_list = worksheet.row_values(1)
        print(values_list)
        return -1
    

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
        Whetherfilled=[0]*32 +[1]*66
        random.shuffle(Whetherfilled)
        
        for i in range(len(Whetherfilled)):
            if Whetherfilled[i]==0:
                Whetherfilled[i]="black"
            else:
                Whetherfilled[i]="blue"
                
        for i in range(8):
            for j in range(12):
                oval=canvas.create_oval(x+10+(j*15) ,y+3+(i*12)+(i*3),x+22+(j*15) ,y+15.66+(i*12)+(i*3), outline="black", fill=Whetherfilled[i+j],width=2)
                canvas.tag_bind(oval, '<Button-1>',lambda x: [self.show_frame(Page1),self.showPopUp((x,y))])

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
                oval=canvas.create_oval(x+10+(j*32) ,y+6+(i*30),x+30+(j*32) ,y+26+(i*30), outline="black", fill=Whetherfilled[i+j],width=2)
                canvas.tag_bind(oval, '<Button-1>',lambda x: [self.show_frame(Page1),self.showPopUp((x,y))])
    



        
    def drawChem(self,canvas,x,y,bor,controller):
        #first column
        Whetherfilled=[1,0,1,0,1,0,0,0,1,0]
        for i in range(len(Whetherfilled)):
            if Whetherfilled[i]==0:
                Whetherfilled[i]="black"
            else:
                Whetherfilled[i]="blue"
        a=canvas.create_oval(x+25,y+10,x+55 ,y+40, outline="black", fill=Whetherfilled[0],width=2)
        canvas.tag_bind(a, '<Button-1>',lambda x: [self.show_frame(Page1),self.showPopUp((x,y))])
        b=canvas.create_oval(x+25,y+45,x+55 ,y+75, outline="black", fill=Whetherfilled[1],width=2)
        canvas.tag_bind(b, '<Button-1>',lambda x: [self.show_frame(Page1),self.showPopUp((x,y))])
        c=canvas.create_oval(x+25,y+80,x+55 ,y+110, outline="black", fill=Whetherfilled[2],width=2)
        canvas.tag_bind(c, '<Button-1>',lambda x: [self.show_frame(Page1),self.showPopUp((x,y))])
        #second column
        d=canvas.create_oval(x+60,y+10,x+90 ,y+40, outline="black", fill=Whetherfilled[3],width=2)
        canvas.tag_bind(d, '<Button-1>',lambda x: [self.show_frame(Page1),self.showPopUp((x,y))])
        e=canvas.create_oval(x+60,y+45,x+90 ,y+75, outline="black", fill=Whetherfilled[4],width=2)
        canvas.tag_bind(e, '<Button-1>',lambda x: [self.show_frame(Page1),self.showPopUp((x,y))])
        f=canvas.create_oval(x+60,y+80,x+90 ,y+110, outline="black", fill=Whetherfilled[5],width=2)
        canvas.tag_bind(f, '<Button-1>',lambda x: [self.show_frame(Page1),self.showPopUp((x,y))])
        # third column
        g=canvas.create_oval(x+95,y+20,x+135 ,y+60, outline="black", fill=Whetherfilled[6],width=2)
        canvas.tag_bind(g, '<Button-1>',lambda event: self.getorigin(event,board=bor))
        h=canvas.create_oval(x+95,y+65,x+135 ,y+105, outline="black", fill=Whetherfilled[7],width=2)
        canvas.tag_bind(h, '<Button-1>',lambda x: [self.show_frame(Page1),self.showPopUp((x,y))])
        #forth column
        i=canvas.create_oval(x+140,y+20,x+180 ,y+60, outline="black", fill=Whetherfilled[8],width=2)
        canvas.tag_bind(i, '<Button-1>',lambda x: [self.show_frame(Page1),self.showPopUp((x,y))])
        j=canvas.create_oval(x+140,y+65,x+180 ,y+105, outline="black", fill=Whetherfilled[9],width=2)
        canvas.tag_bind(j, '<Button-1>',lambda x: [self.show_frame(Page1),self.showPopUp((x,y))])
        
    def drawBoard(self,canvas,bor,controller):
        """
        
        """
        #column one
        rect=canvas.create_rectangle(0, 75, 200, 200, fill="#7A797B",outline="black")
        canvas.tag_bind(rect, '<Button-1>',lambda x: [self.show_frame(Page1),self.showPopUp((0,75))])
        rect=canvas.create_rectangle(0, 200, 200, 325, fill="#7A797B",outline="black")
        canvas.tag_bind(rect, '<Button-1>',lambda x: [self.show_frame(Page1),self.showPopUp((0,200))])
        rect=canvas.create_rectangle(0, 325, 200, 450, fill="#7A797B",outline="black")
        canvas.tag_bind(rect, '<Button-1>',lambda x: [self.show_frame(Page1),self.showPopUp((0,325))])
        rect=canvas.create_rectangle(0, 450, 200, 575, fill="#7A797B",outline="black")
        canvas.tag_bind(rect, '<Button-1>',lambda x: [self.show_frame(Page1),self.showPopUp((0,450))])
        #column two
        rect=canvas.create_rectangle(200, 75, 400, 200, fill="#7A797B",outline="black")
        canvas.tag_bind(rect, '<Button-1>',lambda x: [self.show_frame(Page1),self.showPopUp((200,75))])
        rect=canvas.create_rectangle(200, 200, 400, 325, fill="#7A797B",outline="black")
        canvas.tag_bind(rect, '<Button-1>',lambda x: [self.show_frame(Page1),self.showPopUp((200,200))])
        rect=canvas.create_rectangle(200, 325, 400, 450, fill="#7A797B",outline="black")
        canvas.tag_bind(rect, '<Button-1>',lambda x: [self.show_frame(Page1),self.showPopUp((200,325))])
        rect=canvas.create_rectangle(200, 450, 400, 575, fill="#7A797B",outline="black")
        canvas.tag_bind(rect, '<Button-1>',lambda x: [self.show_frame(Page1),self.showPopUp((200,450))])
        #column three
        #trash
        rect=canvas.create_rectangle(400, 0, 650, 200, fill="#7A797B",outline="black")
        rect=canvas.create_text(525, 100, text="Trash", fill="white", font=('Helvetica 20 bold'))
        #rest of column three
        rect=canvas.create_rectangle(400, 200, 600, 325, fill="#7A797B",outline="black")
        canvas.tag_bind(rect, '<Button-1>',lambda x: [self.show_frame(Page1),self.showPopUp((400,200))])
        rect=canvas.create_rectangle(400, 325, 600, 450, fill="#7A797B",outline="black")
        canvas.tag_bind(rect, '<Button-1>',lambda x: [self.show_frame(Page1),self.showPopUp((400,325))])
        rect=canvas.create_rectangle(400, 450, 600, 575, fill="#7A797B",outline="black")
        canvas.tag_bind(rect, '<Button-1>',lambda x: [self.show_frame(Page1),self.showPopUp((400,450))])
        


        
    def getorigin(self,eventorigin,board):
        global x,y
        x = eventorigin.x
        y = eventorigin.y
        self.expand(x,y,board)
        print(x,y)
        
    def get_canvas_items(self,canvas):
        item_list = canvas.find_all()
        for item in item_list:
            item_type = canvas.type(item)   # e.g. "text", "line", etc.
            #if item_type=="oval":
                #print(item_type)
                
    def showPopUp(self,xy):
        
        print(xy)
        
    def expand(self,x,y,board):
        if (x>0 and x<200)and (y>75 and y<200):
            print("1,1")
        if (x>0 and x<200)and (y>200 and y<325):
            print("1,2")
        if (x>0 and x<200)and (y>325 and y<450):
            print("1,3")
        if (x>0 and x<200)and (y>450 and y<575):
            print("1,4")
        if (x>200 and x<400)and (y>75 and y<200):
            print("2,1")
        if (x>200 and x<400)and (y>200 and y<325):
            print("2,2")
        if (x>200 and x<400)and (y>325 and y<450):
            print("2,3")
        if (x>200 and x<400)and (y>450 and y<575):
            print("2,4")
        if (x>400 and x<600)and (y>200 and y<325):
            print("3,2")
        if (x>400 and x<600)and (y>325 and y<450):
            print("3,3")
        if (x>400 and x<600)and (y>450 and y<575):
            print("3,4")
    
  
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
        c = customtkinter.CTkCanvas(self, width=650, height=575, borderwidth=0, highlightthickness=0,bg ='#d9d9d9')
    
        # button to show frame 2 with text
        # layout2
        button1 = customtkinter.CTkButton(self, text ="StartPage", command = lambda : controller.show_frame(StartPage))
     
        # putting the button in its place 
        # by using grid
        button1.grid(row = 1, column = 1, padx = 10, pady = 10)
  
       
  
# Driver Code
app = tkinterApp()
app.mainloop()
