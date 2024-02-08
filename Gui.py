from tkinter import *
from tkinter import ttk
import tkinter as tk
import subprocess
import threading
import os


#Create an instance of Tkinter frame
win= Tk()

#Set the geometry of Tkinter frame
win.geometry("800x400")

def display_text():
   global entry
   string= entry.get()
   label.configure(text=string)

def run():
   os.chdir('/home/gabepm100/OT2Control')
   execute_python_file('deckPositionsGui.py',entry.get())

def input1(output,sim,auto):
   global entry
   
   ent=" -n " +entry.get()
   if len(ent)==4:
      T.delete("1.0","end")
      T.insert(tk.END," Need Name Input",'warning')
      return -1
      
   os.chdir('/home/gabepm100/Documents/OT2Control')
   if sim.get()==1:
      ent=ent + " --no-sim"
   if auto.get():
      ent = ent+ " -m auto"
   #test one
   
   command="controller.py"
   
   output=execute_python_file(command,ent)
   T.delete("1.0","end")
   T.insert(tk.END,output)


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
c2 = tk.Checkbutton(win, text='Sim?',variable=sim, onvalue=0, offvalue=1)
c2.pack()

#Sim checkbox
auto = tk.IntVar()
c2 = tk.Checkbutton(win, text='Auto?',variable=auto, onvalue=1, offvalue=0)
c2.pack()
output="hello"

#Create a Button to validate Entry Widget
ttk.Button(win, text= "Execute?",width= 20, command= lambda : [display_text(),input1(output,sim,auto)]).pack(pady=20)

ttk.Button(win, text= "Check Deck Positions?",command=run, width=30).pack()

# Create label
l = Label(win, text = "Output")
l.config(font =("Courier", 14))
l.pack()

v=Scrollbar(win,orient='vertical')
v.pack(side=RIGHT, fill='y')

# Create text widget and specify size.
T = Text(win, height = 5, width = 70, yscrollcommand=v.set)
T.tag_config('warning',foreground="red")
T.pack(side=LEFT,expand=True,fill=BOTH)



win.mainloop()
