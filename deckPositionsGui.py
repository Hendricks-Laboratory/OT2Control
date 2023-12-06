import customtkinter

customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

win = customtkinter.CTk()  # create CTk window like you do with the Tk window
win.geometry("800x800")


#Create a canvas object
c = customtkinter.CTkCanvas(win, width=600, height=600, borderwidth=0, highlightthickness=0, bg="white")
c.place(x=100,y=50)


radius = 50 #set the arc radius
canvas_middle = [int(c['width'])/2, int(c['height'])/2] #find the middle of the canvas



def Create_Circle(canvas,x,y,radius,color):
    canvas.create_arc(canvas_middle[0] - radius, canvas_middle[1] - radius, canvas_middle[0] + radius, canvas_middle[1] + radius, extent=359.9999, fill="Grey")

def Create_Rectangle(canvas,x,y,width,height,fill,outline):
    a = canvas.create_rectangle(x, y, x+width, x+height, fill=fill,outline=outline)

def drawBoard(canvas):
    canvas.create_rectangle(0, 0, 600, 600, fill="grey",outline="black")

    canvas.create_rectangle(0, 0, 200, 200, fill="#7A797B",outline="black")
    
    canvas.create_rectangle(200, 0, 400, 200, fill="#7A797B",outline="black")

    canvas.create_rectangle(0, 200, 200, 400, fill="#7A797B",outline="black")

    canvas.create_rectangle(200, 200, 400, 400, fill="#7A797B",outline="black")

    canvas.create_rectangle(400, 0, 600, 200, fill="#7A797B",outline="black")

    canvas.create_rectangle(200, 400, 400, 600, fill="#7A797B",outline="black")

    canvas.create_rectangle(400, 200, 600, 400, fill="#7A797B",outline="black")

    canvas.create_rectangle(0, 400, 200, 600, fill="#7A797B",outline="black")

    canvas.create_rectangle(400, 400, 600, 660, fill="#7A797B",outline="black")

    canvas.create_text(500, 100, text="Trash", fill="white", font=('Helvetica 20 bold'))


class Board:
    board=[[[],[],[]],[[],[],[]],[[],[],[]]]
    def __init__(self,canvas):
        """Initiates Board and calls get content on each square of the board"""
        drawBoard(canvas)
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                self.board[i][j]=self.get_contents(i,j)
        self.print_board()


    def get_contents(self,row,column):
        """
        This is the function which will access different parts of the excel sheet based on which square it is looking at at that moment.
        It will then set that board place as equal to that so that we can then have a fillout function
        """
        return -1
    

    def print_board(self):
        for i in range(len(self.board)):
            print(self.board[i])




board=Board(c)
button= customtkinter.CTkButton(win,text="Hello",width=200)
button.place(x=300,y=700)
win.title("Circles and Arcs")
win.mainloop()
