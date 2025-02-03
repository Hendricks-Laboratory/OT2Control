import customtkinter
from CTkMessagebox import CTkMessagebox
from customtkinter import IntVar, CHECKBUTTON
import subprocess
from subprocess import check_output
import os
import pickle
import threading
import pty
from controller import run_as_thread

from threadManager import ThreadManager, QueueManager



def run():
   os.chdir(os.curdir)
   execute_python_file('deckPositionsGui.py',mynumber.get(),T)

def run_controller(sim, auto, sheet_name, combobox):

   """
   This function will pass the arguements entered by the use in GUI to controller, and run controller as a thread in paralell with 
   the update thread. 
   parameters:
      sim:
         checkbox user entry for "--no-sim" flag
      auto:
         checkbox user entry for "-m" flag 
      sheet_name:
         user text entry for sheet name
   """

   
   update_pickle(sheet_name.get(),combobox)


   # mimic cli args for controller
   cli_args = []

   if sheet_name.get():
      cli_args.append(f"-n{sheet_name.get().strip()}")
   if auto.get():
      cli_args.append(f"-mauto")
   if sim.get():
      cli_args.append(f"--no-sim")

   # shared resource between threads
   # create status obj

   thread_manager = ThreadManager()

   thread_manager.start_thread(target=update_status, args=(T,))
   thread_manager.start_thread(target=listen_input)
   thread_manager.start_thread(target=run_as_thread, args=(cli_args, ))
   

def update_status(T):
   """
   status_q: singleto
   """
   status_q = QueueManager.get_status_queue()
   while True:
      msg = status_q.get()
      T.insert(customtkinter.END, msg + "\n")


def listen_input():
   """

   """
   input_q = QueueManager.get_input_queue()
   while True:
      pass


def handle_yn(msg):
   msgbox = CTkMessagebox(message = msg,
                 icon="check",
                 option_1="yes",
                 option_2="no")
   response = msgbox.get() 
   if response == "yes":
      return True
   else:
      return False
   

def handle_step():
   """
   This un-sets the event indicator, and switches the threads from read to write mode
   """
   event.set()


def execute_python_file(file_name, argument,Textbox):
   try:
      process = start_subprocess(["python3", file_name,'-n',argument], Textbox)
   except FileNotFoundError:
      print("Error: The file does not exist.")
      
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

# def write_stdin(process):
#    process.stdin.write()
   
# def read_output(master_fd, textbox):
#     try:
#         while True:
#             output = os.read(master_fd, 1024)
#             if not output:
#                 break
#             textbox.insert(customtkinter.END, output)
#     except Exception as e:
#         print(f"Error reading output from subprocess: {e}")
#         update_output("Error reading output from subprocess: " + str(e), textbox)


# def read_stderr(process):
#    # reads stderr of the provided process line by line and the output
#    while True:
#       error = process.stderr.readline().decode('utf-8')
#       if not error:
#          break
#       update_output(error)

# def update_output(text,Textbox):
#    print("in update")
#    Textbox.configure(state="normal") # Make the state normal

#    Textbox.insert(customtkinter.END, text)
#    Textbox.configure(state="disabled") # Make the state disabled again
#    print("out of update")

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
# l = customtkinter.CTkLabel(master= win, text = "What is the name?")
# l.configure(font =("Inter", 16), text_color="white")
l = customtkinter.CTkLabel(master= win, text = "Sheetname")
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
c2 = customtkinter.CTkCheckBox(master= win, text='Sim',variable=sim, onvalue=1, offvalue=0, fg_color= "#303030", text_color= "white", border_color = "#A7A6A6")
c2.configure(border_width= 2, font= ("Inter", 12))
c2.pack(padx=20, pady= (15, 10))


#Sim checkbox
auto = IntVar()
c2 = customtkinter.CTkCheckBox(master= win, text='Auto',variable=auto, onvalue=1, offvalue=0, text_color= "white", border_color = "#A7A6A6")
c2.configure(border_width= 2, font= ("Inter", 12))
c2.pack()
output="hello"
#Create a Button to validate Entry Widget
customtkinter.CTkButton(win, text= "Execute",width= 20,fg_color='#007acc', font= ("Inter", 12) ,command= lambda : [run_controller(sim,auto,mynumber,combobox)]).pack(pady=(20, 13))
# Bind the <Return> event to the execute_button's command
win.bind('<Return>', lambda event: [run_controller(sim, auto, mynumber,combobox)])


#show deck positions
customtkinter.CTkButton(win, text= "Check Deck Positions",fg_color='#007acc', font= ("Inter", 12), command=run, width=30).pack(pady= (0, 17))


I = customtkinter.CTkButton(win, text= "Step",fg_color='#007acc', font= ("Inter", 12), command=handle_step, width=30)
I.pack(pady= (0, 17))


# Create label
# l = customtkinter.CTkLabel(win, text = "Output", text_color= "white")
# l.configure(font =("Inter", 14))
l = customtkinter.CTkLabel(win, text = "Output", text_color= "white")
l.configure(font =("Inter", 14))
l.pack()


v=customtkinter.CTkScrollbar(win,orientation='vertical') 
v.pack(side="right", fill='y')  


# Create text widget and specify size.
T = customtkinter.CTkTextbox(win, height = 50, width = 400)
T.configure(fg_color= "#3e3e42", text_color= "white")
T.pack(side='left',expand=True,fill='both')


# handle_yn("hello there")

# customtkinter.CTkButton(win, text= "Step",width= 20,fg_color='#007acc', font= ("Inter", 12) ,command= lambda : [step_controller()]).pack(pady=(20, 13))
win.mainloop()