import customtkinter
import gspread


def drawPipets(canvas,x,y):
    """
    Canvas is the canvas object
    x and y are the x and y cordinates
    """
    for i in range(12):
        for j in range(8):
            canvas.create_oval(x+20.66+(j*20) ,y+6.66+(i*12)+(i*3),x+32.66+(j*20) ,y+18.66+(i*12)+(i*3), outline="black", fill="red",width=2)


def drawReagents(canvas,x,y):
    canvas.create_oval(x+56.66,y+26.66,x+96.66,y+66.66, outline="black", fill="white",width=2)

    canvas.create_oval(x+106.66,y+26.66,x+146.66,y+66.66, outline="black", fill="white",width=2)

    canvas.create_oval(x+56.66,y+76.66,x+96.66,y+116.66, outline="black", fill="white",width=2)
    
    canvas.create_oval(x+106.66,y+76.66,x+146.66,y+116.66, outline="black", fill="white",width=2)
    
    canvas.create_oval(x+56.66,y+126.66,x+96.66,y+166.66, outline="black", fill="white",width=2)
    
    canvas.create_oval(x+106.66,y+126.66,x+146.66,y+166.66, outline="black", fill="white",width=2)


def drawBoard(canvas):
    """
    
    """
    canvas.create_rectangle(0, 50, 200, 250, fill="#7A797B",outline="black")
    
    canvas.create_rectangle(200, 50, 400, 250, fill="#7A797B",outline="black")

    canvas.create_rectangle(0, 250, 200, 450, fill="#7A797B",outline="black")

    canvas.create_rectangle(200, 250, 400, 450, fill="#7A797B",outline="black")

    canvas.create_rectangle(400, 0, 650, 250, fill="#7A797B",outline="black")

    canvas.create_rectangle(200, 450, 400, 650, fill="#7A797B",outline="black")

    canvas.create_rectangle(400, 250, 600, 450, fill="#7A797B",outline="black")

    canvas.create_rectangle(0, 450, 200, 650, fill="#7A797B",outline="black")

    canvas.create_rectangle(400, 450, 600, 650, fill="#7A797B",outline="black")

    canvas.create_rectangle(0, 650, 200, 850, fill="#7A797B",outline="black")

    canvas.create_rectangle(400, 650, 600, 850, fill="#7A797B",outline="black")

    canvas.create_rectangle(200, 650, 400, 850, fill="#7A797B",outline="black")

    canvas.create_text(525, 125, text="Trash", fill="white", font=('Helvetica 20 bold'))


class Board:
    board=[(2,[0]),(1,[0]),(-1,[0]),(0,[0]),(0,[0]),(0,[0]),(0,[0]),(1,[0]),(0,[0]),(0,[0]),(1,[0]),(0,[0])]
    positions=[[0,50],[200,50],[0,250],[200,250],[200,450],[0,450],[400,250],[400,450],[600,50],[600,250],[600,450]]
    def __init__(self,canvas):
        """Initiates Board and calls get content on each square of the board"""
        drawBoard(canvas)
        square=0
        for i in range(len(self.board)):
            if square<8:
                # calls the get content function and depending on that it will set up the board to match the values gotten from the excell sheet
                #self.get_contents()
                # if that board share is set to 1 it draws the reagents and if it is 2 it draws the pipets
                if self.board[square][0]==1:
                    drawReagents(canvas,self.positions[square][0],self.positions[square][1])
                elif self.board[square][0]==2:
                    drawPipets(canvas,self.positions[square][0],self.positions[square][1])
            square+=1

            """
            list1 = list(self.board[i])
            self.board[i]=self.get_contents(i)
            tuple1 = tuple(list1)
            self.board[i]=self.get_contents(i)
            """
        
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
    win.geometry("850x1000")

    #Create a canvas object
    c = customtkinter.CTkCanvas(win, width=650, height=850, borderwidth=0, highlightthickness=0,bg ='#242424')
    c.place(x=100,y=50)
    board=Board(c)
    button= customtkinter.CTkButton(win,text="Close",width=200,command=Close)
    button.place(x=300,y=950)
    win.title("Deck Positions")
    win.mainloop()
run()
