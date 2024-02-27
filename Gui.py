import customtkinter
from customtkinter import IntVar, CHECKBUTTON
import subprocess
import os
import pickle

def run():
   os.chdir
   os.chdir("/home/gabepm100/OT2Control")
   execute_python_file('deckPositionsGui.py',mynumber.get())


def input1(sim,auto,combobox):
    global mynumber
    update_pickle(mynumber.get(),combobox)
    ent=" -n " +mynumber.get()
    if len(ent)==4:
        T.delete("1.0",customtkinter.END)
        T.insert(customtkinter.END, "Need Name Input", 'warning')
        return -1
        
    os.chdir("/home/gabepm100/OT2Control")
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
   
def update_pickle(val,combobox):
   global comboboxlist
   print("update_pickle")
   vals=list(comboboxlist)
   try:
      if isinstance(val,str) and val!='':
         filename='pickle.pk'
         if os.path.isfile(filename):
            print('file exists')
            vals.append(val)
            with open(filename, 'wb') as g:
               print("combo")
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
   print("read_pickle")
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
print(v)
combobox.configure(values = v)
comboboxlist=v
combobox.pack()

#Sim checkbox

sim = IntVar()
c2 = customtkinter.CTkCheckBox(master= win, text='Sim?',variable=sim, onvalue=1, offvalue=0, fg_color= "303030", text_color= "white", border_color = "#A7A6A6")
c2.configure(border_width= 2, font= ("Inter", 12))
c2.pack(padx=20, pady= (15, 10))

sim = IntVar()
c2 = customtkinter.CTkCheckBox(master= win, text='Sim?',variable=sim, onvalue=1, offvalue=0, fg_color= "303030", text_color= "white", border_color = "#A7A6A6")
c2.configure(border_width= 2, font= ("Inter", 12))
c2.pack(padx=20, pady= (15, 10))

#Sim checkbox
auto = IntVar()
c2 = customtkinter.CTkCheckBox(master= win, text='Auto?',variable=auto, onvalue=1, offvalue=0, text_color= "white", border_color = "#A7A6A6")
c2.configure(border_width= 2, font= ("Inter", 12))
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
# Create text widget and specify size.
T = customtkinter.CTkTextbox(win, height = 5, width = 52)
# Create label
l = customtkinter.CTkLabel(win, text = "Output", text_color= "white")
l.configure(font =("Inter", 14))
l = customtkinter.CTkLabel(win, text = "Output", text_color= "white")
l.configure(font =("Inter", 14))
l.pack()

v=customtkinter.CTkScrollbar(win,orientation='vertical') 
v.pack(side="right", fill='y')  
v=customtkinter.CTkScrollbar(win,orientation='vertical') 
v.pack(side="right", fill='y')  

# Create text widget and specify size.
T = customtkinter.CTkTextbox(win, height = 50, width = 400)
T.configure(fg_color= "#3e3e42", text_color= "white")
T.focus_set()
T.pack(side='left',expand=True,fill='both')




win.mainloop()