from tkinter import *
from tkinter import ttk
import tkinter as tk
import subprocess
import threading
import os
import customtkinter


#Create an instance of Tkinter frame
win= Tk()

#Set the geometry of Tkinter frame
win.geometry("750x350")

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
    T.insert(tk.END,output)
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
label=Label(win, text="", font=("Courier 22 bold"))
label.pack()

# Name Label
l = Label(win, text = "What is the name?")
l.config(font =("Courier", 14))
l.pack()

#Create an Entry widget to accept User Input
entry= Entry(win, width= 40)
entry.focus_set()
entry.pack()

#Sim checkbox

sim = tk.IntVar()
c2 = tk.Checkbutton(win, text='Sim?',variable=sim, onvalue=1, offvalue=0)
c2.pack()

#Sim checkbox
auto = tk.IntVar()
c2 = tk.Checkbutton(win, text='Auto?',variable=auto, onvalue=1, offvalue=0)
c2.pack()
output="hello"
#Create a Button to validate Entry Widget
ttk.Button(win, text= "Execute?",width= 20, command= lambda : [display_text(),input1(output,sim,auto)]).pack(pady=20)

ttk.Button(win, text= "Check Deck Positions?",command=run, width=30).pack()

# print("should be")
# Create text widget and specify size.
T = Text(win, height = 5, width = 52)

# Create label
l = Label(win, text = "Output")
l.config(font =("Courier", 14))
l.pack()

# Create text widget and specify size.
T = Text(win, height = 5, width = 52)
T.pack()



win.mainloop()
