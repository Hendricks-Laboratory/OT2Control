import tkinter
import customtkinter
from customtkinter import IntVar, CHECKBUTTON
import subprocess
import os

#Create an instance of Tkinter frame
win= tkinter.Tk()
win.title("OT2Control")
#Set the geometry of Tkinter frame

win.geometry("750x350")
win.configure(background= '#303030')
win.title("OT2Control")

def display_text():
   global entry
   string= entry.get()
   label.configure(text=string)

def run():
   os.chdir
   os.chdir("/home/halversm/OT2Control")
   execute_python_file('deckPositionsGui.py',entry.get())


def input1(output,sim,auto):
   global entry
   
   ent=" -n " +entry.get()
   if len(ent)==4:
      T.delete("1.0",customtkinter.END)
      T.insert(customtkinter.END, "Need Name Input", 'warning')
      return -1
      
   os.chdir("/home/halversm/OT2Control")
   if sim.get()==1:
      ent=ent + " --no-sim"
   if auto.get():
      ent = ent+ " -m auto"
   #test one
   
   command="controller.py"
   
   output=execute_python_file(command,ent)
   #output=output.stdout
   T.delete("1.0",customtkinter.END)
   T.insert(customtkinter.END,output) #FIX#

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
   T.insert(customtkinter.END, text)
   T.see(customtkinter.END)


#Initialize a Label to display the User Input
label = customtkinter.CTkLabel(master=win, text="", font=("Inter", 22, "bold"))
#label=Label(win, text="", font=("Courier 22 bold"))
label.pack()

# Name Label
l = customtkinter.CTkLabel(master= win, text = "What is the name?")
l.configure(font =("Inter", 16), text_color="white")
l.pack()

#Create an Entry widget to accept User Input
entry= customtkinter.CTkEntry(master=win, width= 400)
entry.configure(fg_color= "#585858", text_color= "white")
entry.focus_set()
entry.pack()

#Sim checkbox

sim = IntVar()
c2 = customtkinter.CTkCheckBox(master= win, text='Sim?',variable=sim, onvalue=1, offvalue=0, fg_color= "303030", text_color= "white", border_color = "#A7A6A6")
c2.configure(border_width= 2, font= ("Inter", 12))
c2.pack(padx=20, pady= (15, 10))

#Sim checkbox
auto = IntVar()
c2 = customtkinter.CTkCheckBox(master= win, text='Auto?',variable=auto, onvalue=1, offvalue=0, text_color= "white", border_color = "#A7A6A6")
c2.configure(border_width= 2, font= ("Inter", 12))
c2.pack()
output="hello"
#Create a Button to validate Entry Widget
customtkinter.CTkButton(win, text= "Execute",width= 20, font= ("Inter", 12) ,command= lambda : [display_text(),input1(output,sim,auto)]).pack(pady=(20, 13))




# Bind the <Return> event to the execute_button's command
win.bind('<Return>', lambda event: [display_text(), input1(output, sim, auto)])

#show deck positions
customtkinter.CTkButton(win, text= "Check Deck Positions", font= ("Inter", 12), command=run, width=30).pack(pady= (0, 17))

# print("should be")
# Create text widget and specify size.
T = customtkinter.CTkTextbox(win, height = 5, width = 52)

# Create label
l = customtkinter.CTkLabel(win, text = "Output", text_color= "white")
l.configure(font =("Inter", 14))
l.pack()

v=customtkinter.CTkScrollbar(win,orientation='vertical') 
v.pack(side="right", fill='y')  

# Create text widget and specify size.
T = customtkinter.CTkTextbox(win, height = 50, width = 400)
T.configure(fg_color= "#585858", text_color= "white")
#T.tag_config('warning',foreground="red")
T.focus_set()
T.pack(side='left',expand=True,fill='both')



win.mainloop()