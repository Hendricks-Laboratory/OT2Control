from tkinter import *
from tkinter import ttk
import tkinter as tk
import subprocess
import os

#Create an instance of Tkinter frame
win= Tk()
win.title("OT2Control")

#Set the geometry of Tkinter frame
win.geometry("800x400")

def display_text():
   global entry
   string= entry.get()
   label.configure(text=string)

def run():
   os.chdir
   os.chdir("/home/fatimakowdan/OT2Control")
   execute_python_file('deckPositionsGui.py',entry.get())


def input1(output,sim,auto):
   global entry
   
   ent=" -n " +entry.get()
   if len(ent)==4:
      T.delete("1.0",tk.END)
      T.insert(tk.END, "Need Name Input", 'warning')
      return -1
      
   os.chdir("/home/fatimakowdan/OT2Control")
   if sim.get()==1:
      ent=ent + " --no-sim"
   if auto.get():
      ent = ent+ " -m auto"
   #test one
   
   command="controller.py"
   
   output=execute_python_file(command,ent)
   #output=output.stdout
   T.delete("1.0","end")
   T.insert(tk.END,output)

def execute_python_file(file_Name, argument):
   try:
      completed_process = subprocess.run(['python3', file_Name, argument], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      if completed_process.returncode == 0:
         print("Execution successful.")
         return completed_process.stdout
      else:
         print(f"Error: Failed to execute.")
         return completed_process.stderr
   except FileNotFoundError:
      print(f"Error: The file does not exist.")
      
def execute_command(command):
   # executes the given command and returns the process
   process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
   return process

def read_stdout(process):
   # reads stdout of the given process line by line and update the output
   while True:
      output = process.stdout.readline().decode('utf-8')
      if not output:
         break
      update_output(output)

def read_stderr(process):
   # reads stderr of the provided process line by line and the output
   while True:
      error = process.stderr.readline().decode('utf-8')
      if not error:
         break
      update_output(error)

def update_output(text):
   # updates the output text
   T.insert(tk.END, text)
   T.see(tk.END)


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

#auto checkbox
auto = tk.IntVar()
c2 = tk.Checkbutton(win, text='Auto?',variable=auto, onvalue=1, offvalue=0)
c2.pack()
output="hello" 

#Create a Button to Execute given file name
ttk.Button(win, text= "Execute?",width= 20, command= lambda : [display_text(),input1(output,sim,auto)]).pack(pady=20)  

# Bind the <Return> event to the execute_button's command
win.bind('<Return>', lambda event: [display_text(), input1(output, sim, auto)])

#show deck positions
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

#win.bind('<Return>',input2)

win.mainloop()