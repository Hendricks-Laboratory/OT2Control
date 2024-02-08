import customtkinter
import gspread


def drawChem(canvas,x,y):
    #first column
    canvas.create_oval(x+25,y+10,x+55 ,y+40, outline="black", fill="white",width=2)
    canvas.create_oval(x+25,y+45,x+55 ,y+75, outline="black", fill="white",width=2)
    canvas.create_oval(x+25,y+80,x+55 ,y+110, outline="black", fill="white",width=2)
    #second column
    canvas.create_oval(x+60,y+10,x+90 ,y+40, outline="black", fill="white",width=2)
    canvas.create_oval(x+60,y+45,x+90 ,y+75, outline="black", fill="white",width=2)
    canvas.create_oval(x+60,y+80,x+90 ,y+110, outline="black", fill="white",width=2)
    # third column
    canvas.create_oval(x+95,y+20,x+135 ,y+60, outline="black", fill="white",width=2)
    canvas.create_oval(x+95,y+65,x+135 ,y+105, outline="black", fill="white",width=2)
    #forth column
    canvas.create_oval(x+140,y+20,x+180 ,y+60, outline="black", fill="white",width=2)
    canvas.create_oval(x+140,y+65,x+180 ,y+105, outline="black", fill="white",width=2)
    
    
    
def drawPipets(canvas,x,y):
    """
    Canvas is the canvas object
    x and y are the x and y cordinates
    """
    for i in range(8):
        for j in range(12):
            canvas.create_oval(x+10+(j*15) ,y+3+(i*12)+(i*3),x+22+(j*15) ,y+15.66+(i*12)+(i*3), outline="black", fill="red",width=2)


def drawReagents(canvas,x,y):
    for i in range(4):
        for j in range(6):
            canvas.create_oval(x+10+(j*32) ,y+6+(i*30),x+30+(j*32) ,y+26+(i*30), outline="black", fill="white",width=2)
    


def drawBoard(canvas):
    """
    
    """
    canvas.create_rectangle(0, 75, 200, 200, fill="#7A797B",outline="black")
    
    canvas.create_rectangle(200, 75, 400, 200, fill="#7A797B",outline="black")

    canvas.create_rectangle(0, 200, 200, 325, fill="#7A797B",outline="black")

    canvas.create_rectangle(200, 200, 400, 325, fill="#7A797B",outline="black")
    #trash
    canvas.create_rectangle(400, 0, 650, 200, fill="#7A797B",outline="black")

    canvas.create_rectangle(200, 325, 400, 450, fill="#7A797B",outline="black")

    canvas.create_rectangle(400, 200, 600, 325, fill="#7A797B",outline="black")

    canvas.create_rectangle(0, 325, 200, 450, fill="#7A797B",outline="black")

    canvas.create_rectangle(400, 325, 600, 450, fill="#7A797B",outline="black")

    canvas.create_rectangle(0, 450, 200, 575, fill="#7A797B",outline="black")

    canvas.create_rectangle(400, 450, 600, 575, fill="#7A797B",outline="black")

    canvas.create_rectangle(200, 450, 400, 575, fill="#7A797B",outline="black")

    canvas.create_text(525, 100, text="Trash", fill="white", font=('Helvetica 20 bold'))


class Board:
    #positions goes row by row
    # board is associated with the position of the same index
    board=[(2,[0]),(2,[0]),(-1,[0]),(0,[0]),(0,[0]),(0,[0]),(0,[0]),(0,[0]),(0,[0]),(0,[0]),(3,[0]),(1,[0])]
    positions=[[0,75],[200,75],[400,75],[0,200],[200,200],[400,200],[0,325],[200,325],[400,400],[0,450],[200,450],[400,450]]
    def __init__(self,canvas):
        """Initiates Board and calls get content on each square of the board"""
        drawBoard(canvas)
        
        for i in range(len(self.board)):

                # calls the get content function and depending on that it will set up the board to match the values gotten from the excell sheet
                #self.get_contents()
                # if that board share is set to 1 it draws the reagents and if it is 2 it draws the pipets
                if self.board[i][0]==1:
                    drawReagents(canvas,self.positions[i][0],self.positions[i][1])
                elif self.board[i][0]==2:
                    drawPipets(canvas,self.positions[i][0],self.positions[i][1])
                elif self.board[i][0]==3:
                    drawChem(canvas,self.positions[i][0],self.positions[i][1])
        
        #self.print_board()
    def get_contents(self):
        """
        This is the function which will access different parts of the excel sheet based on which square it is looking at at that moment.
        It will then set that board place as equal to that so that we can then have a fillout function
        """
        # using the .json in order to ask for permission to get access to the google sheet. 
        # will need to do this again on the lab computer
        gc = gspread.service_account("C:/Users/gabep/OneDrive/Documents/school/4th year whitman/GuiTeam/OT2Control/deck-position-gui-556e4624293c.json")
        # Open Spreadsheet by name
        spreadsheet = gc.open_by_url("https://docs.google.com/spreadsheets/d/124PqDU-h_bPa-WcAnTXrShyv81nVTYWc4jws1f0IWmM/edit#gid=0")

        #opens the sheet by name. sheet one is the name of the page inside the spreadsheet
        worksheet = spreadsheet.sheet1
        
        # gets the values from the first row of the sheet
        values_list = worksheet.row_values(1)
        print(values_list)
        return -1
    

    def print_board(self):
        for i in range(len(self.board)):
            print(self.board[i])


def Close(): 
    print("Should be closing")


def run():
    customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
    customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green
    win = customtkinter.CTk()  # create CTk window like you do with the Tk window
    win.geometry("850x800")

    #Create a canvas object
    c = customtkinter.CTkCanvas(win, width=650, height=575, borderwidth=0, highlightthickness=0,bg ='#d9d9d9')
    c.place(x=100,y=50)
    board=Board(c)
    button= customtkinter.CTkButton(win,text="Close",width=200,command=Close)
    button.place(x=300,y=750)
    win.title("Deck Positions")
    win.mainloop()
run()
