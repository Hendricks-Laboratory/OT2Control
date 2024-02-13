import tkinter
import customtkinter
from customtkinter import IntVar, CHECKBUTTON
import subprocess
import threading
import os


#Create an instance of Tkinter frame
win= tkinter.Tk()
#Set the geometry of Tkinter frame
win.geometry("750x350")
win.configure(background= '#404040')

def display_text():
   global entry
   string= entry.get()
   label.configure(text=string)

def run():
   os.chdir('/home/gabepm100/Hendrix-Lab-Ubuntu-Gui')
   execute_python_file('deckPositionsGui.py','')

def input1(output,sim,auto):
    global entry
    string= entry.get()
    os.chdir('/home/gabepm100/Documents/OT2Control')
    #test one
    command="controller.py"
    output=execute_python_file(command,string)
    T.insert(customtkinter.CTkEnd,output) #FIX#
    #real one 
    #command="python controller.py -n "+string

def execute_python_file(file_Name,argument):
   try:
      completed_process = subprocess.run(['python3',file_Name, argument], capture_output=True)
      if completed_process.returncode == 0:
         print("Execution successful.")
         print("Output:")
         return completed_process.stdout
      else:
         print(f"Error: Failed to execute.")
         print("Error output:")
         print(completed_process.stderr)
   except FileNotFoundError:
      print(f"Error: The file does not exist.")
    
#Initialize a Label to display the User Input
label = customtkinter.CTkLabel(master=win, text="", font=("Courier", 22, "bold"))
#label=Label(win, text="", font=("Courier 22 bold"))
label.pack()

# Name Label
l = customtkinter.CTkLabel(master= win, text = "What is the name?")
l.configure(font =("Courier", 14), text_color="white")
l.pack()

#Create an Entry widget to accept User Input
entry= customtkinter.CTkEntry(master=win, width= 400)
entry.focus_set()
entry.pack()

#Sim checkbox

sim = IntVar()
c2 = customtkinter.CTkCheckBox(master= win, text='Sim?',variable=sim, onvalue=1, offvalue=0, fg_color= "404040", text_color= "white", border_color = "#A7A6A6")
c2.configure(border_width= 2)
c2.pack(padx=20, pady=10)

#Sim checkbox
auto = IntVar()
c2 = customtkinter.CTkCheckBox(master= win, text='Auto?',variable=auto, onvalue=1, offvalue=0, text_color= "white", border_color = "#A7A6A6")
c2.configure(border_width= 2)
c2.pack()
output="hello"
#Create a Button to validate Entry Widget
customtkinter.CTkButton(win, text= "Execute?",width= 20, command= lambda : [display_text(),input1(output,sim,auto)]).pack(pady=20)

customtkinter.CTkButton(win, text= "Check Deck Positions?",command=run, width=30).pack()

# print("should be")
# Create text widget and specify size.
T = customtkinter.CTkTextbox(win, height = 5, width = 52)

# Create label
l = customtkinter.CTkLabel(win, text = "Output")
l.configure(font =("Courier", 14))
l.pack()

# Create text widget and specify size.
T = customtkinter.CTkTextbox(win, height = 50, width = 400)
T.pack()



win.mainloop()
