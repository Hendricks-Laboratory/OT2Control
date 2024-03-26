import customtkinter
from customtkinter import IntVar, CHECKBUTTON
import subprocess
from subprocess import check_output
import io
import os
import pickle
import threading
import json 
import pty

def run():
   os.chdir("/mnt/c/Users/science_356_lab/Robot_Files/OT2Control")
   execute_python_file('deckPositionsGui.py',mynumber.get(),T)


def input1(sim,auto,combobox):
   global mynumber
   update_pickle(mynumber.get(),combobox)
   ent=mynumber.get()
   if len(ent)==4:
      T.delete("1.0",customtkinter.END)
      T.insert(customtkinter.END, "Need Name Input", 'warning')
      return -1
        
   os.chdir("/mnt/c/Users/science_356_lab/Robot_Files/OT2Control")
   if sim.get()==1:
      ent=ent + " --no-sim"
   if auto.get():
      ent = ent+ " -m auto"
    #test one
    
   command="controller.py"
    
   execute_python_file(command,ent,T)
   #output=output.stdout
   # T.delete("1.0",customtkinter.END)
   # T.insert(customtkinter.END,output) #FIX#

def execute_python_file(file_name, argument,Textbox):
   try:
      process = start_subprocess(["python3", file_name,'-n',argument], Textbox)
   except FileNotFoundError:
      print("Error: The file does not exist.")
      
def read_output(process, textbox):
    for line in iter(process.stdout.readline, b''):
        textbox.insert(customtkinter.END, line)
    process.stdout.close()

def send_input(process, entry):
    process.stdin.write((entry.get() + '\n').encode('utf-8'))
    process.stdin.flush()
    entry.delete(0, customtkinter.END)

def start_subprocess(command, textbox):
    try:
        master_fd, slave_fd = pty.openpty() #PseudoTerminal
        process = subprocess.Popen(command, stdout=slave_fd, stderr=subprocess.STDOUT, stdin=slave_fd, universal_newlines=True)
        os.close(slave_fd)
        threading.Thread(target=read_output, args=(master_fd, textbox), daemon=True).start()
    except Exception as e:
        print(f"Error starting subprocess: {e}")
        update_output("Error starting subprocess: " + str(e), textbox)
        return None
    return process

def write_stdin(process):
   process.stdin.write()
   
def read_output(master_fd, textbox):
    try:
        while True:
            output = os.read(master_fd, 1024)
            if not output:
                break
            textbox.insert(customtkinter.END, output)
    except Exception as e:
        print(f"Error reading output from subprocess: {e}")
        update_output("Error reading output from subprocess: " + str(e), textbox)


def read_stderr(process):
   # reads stderr of the provided process line by line and the output
   while True:
      error = process.stderr.readline().decode('utf-8')
      if not error:
         break
      update_output(error)

def update_output(text,Textbox):
   print("in update")
   Textbox.configure(state="normal") # Make the state normal

   Textbox.insert(customtkinter.END, text)
   Textbox.configure(state="disabled") # Make the state disabled again
   print("out of update")
def update_pickle(val,combobox):
   global comboboxlist
   vals=list(comboboxlist)
   try:
      if isinstance(val,str) and val!='':
         filename='pickle.pk'
         if os.path.isfile(filename):

            vals.append(val)
            with open(filename, 'wb') as g:
               vals=list(dict.fromkeys(vals))
               if len(vals)>10:
                  vals=vals[1:]
               pickle.dump(vals,g)
               g.close()
      else:
         print("Sheetname is not string")
      combobox.configure(values=vals)
   except:
      print("updating pickle didnt work. Please try doing something different")
      

def read_pickle():
   try:
      with open('pickle.pk', 'rb') as fi:
         loadedval=pickle.load(fi)
         loadedval=[x for x in list(loadedval) if x]
         fi.close()
         return list(dict.fromkeys(loadedval))
   except:
      print("couldnt read pickle")
      return []

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme('dark-blue')
#Create an instance of Tkinter frame
win= customtkinter.CTk()
win.title("OT2Control")
#Set the geometry of Tkinter frame

win.geometry("750x450")
win.configure(fg_color= '#252526')
win.title("OT2Control")

# Name Label
l = customtkinter.CTkLabel(master= win, text = "What is the name?")
l.configure(font =("Inter", 16), text_color="white")
l = customtkinter.CTkLabel(master= win, text = "What is the name?")
l.configure(font =("Inter", 16), text_color="white")
l.pack()

#Create an Entry widget to accept User Input
mynumber = customtkinter.StringVar()
combobox = customtkinter.CTkComboBox(win, width = 400 , variable = mynumber,fg_color='#3e3e42')
v=read_pickle()
combobox.configure(values = v)
comboboxlist=v
combobox.focus_set()
combobox.pack()

#Sim checkbox

sim = IntVar()
c2 = customtkinter.CTkCheckBox(master= win, text='Sim?',variable=sim, onvalue=1, offvalue=0, fg_color= "#303030", text_color= "white", border_color = "#A7A6A6")
c2.configure(border_width= 2, font= ("Inter", 12))
c2.pack(padx=20, pady= (15, 10))


#Sim checkbox
auto = IntVar()
c2 = customtkinter.CTkCheckBox(master= win, text='Auto?',variable=auto, onvalue=1, offvalue=0, text_color= "white", border_color = "#A7A6A6")
c2.configure(border_width= 2, font= ("Inter", 12))
c2.pack()
output="hello"
#Create a Button to validate Entry Widget
customtkinter.CTkButton(win, text= "Execute",width= 20,fg_color='#007acc', font= ("Inter", 12) ,command= lambda : [input1(sim,auto,combobox)]).pack(pady=(20, 13))
# Bind the <Return> event to the execute_button's command
win.bind('<Return>', lambda event: [input1(sim, auto,combobox)])

#show deck positions
customtkinter.CTkButton(win, text= "Check Deck Positions",fg_color='#007acc', font= ("Inter", 12), command=run, width=30).pack(pady= (0, 17))

# Create label
l = customtkinter.CTkLabel(win, text = "Output", text_color= "white")
l.configure(font =("Inter", 14))
l = customtkinter.CTkLabel(win, text = "Output", text_color= "white")
l.configure(font =("Inter", 14))
l.pack()

v=customtkinter.CTkScrollbar(win,orientation='vertical') 
v.pack(side="right", fill='y')  


# Create text widget and specify size.
T = customtkinter.CTkTextbox(win, height = 50, width = 400)
T.configure(fg_color= "#3e3e42", text_color= "white")
T.pack(side='left',expand=True,fill='both')





win.mainloop()