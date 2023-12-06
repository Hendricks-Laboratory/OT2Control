from tkinter import *
from tkinter import ttk
import tkinter as tk
import subprocess
from subprocess import CREATE_NEW_CONSOLE
import threading

#Create an instance of Tkinter frame
win= Tk()

#Set the geometry of Tkinter frame
win.geometry("750x350")

def display_text():
   global entry
   string= entry.get()
   label.configure(text=string)

def input1(output,sim,auto):
    global entry
    string= entry.get()
    command="python controller.py -n "+string
    command1 = "wsl ~ -e cd Dcouments"
    command2 = "ls"


    sub=subprocess.run(['wsl',"~","ls"],capture_output=True,)

    print(command)
    print(sub.stdout)
    output=sub.stdout
    

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

# Create text widget and specify size.
T = Text(win, height = 5, width = 52)

# Create label
l = Label(win, text = "Output")
l.config(font =("Courier", 14))
l.pack()

# Create text widget and specify size.
T = Text(win, height = 5, width = 52)
T.pack()

T.insert(tk.END, "hello")

win.mainloop()