import tkinter as tk
from tkinter import *
import subprocess
from subprocess import CREATE_NEW_CONSOLE
import os
import pexpect

class tkinterApp(tk.Tk):
	def __init__(self, *args, **kwargs): 
		
		tk.Tk.__init__(self, *args, **kwargs)
		process= subprocess.Popen(['Ubuntu'], creationflags=CREATE_NEW_CONSOLE, stdin=subprocess.PIPE,stdout=subprocess.PIPE)

		# creating a container
		container = tk.Frame(self,height=500,width=700)
		container.pack(side = "top", fill = "both", expand = True)
		
		self.frames = {} 
		#iterate through all pages and create frame for each
		for F in (StartPage, Page1, Page2, Page3):

			frame = F(self,container, process)
			self.frames[F] = frame 

		self.show_frame(StartPage)
		
	def show_frame(self, cont):
		frame = self.frames[cont]
		frame.tkraise()


class StartPage(tk.Frame):
	def __init__(self, controller,parent,process): 
		tk.Frame.__init__(self, parent)
        # Label for start page
		Label(self, text ="Do you want auto or simulation?", font="Helvetica",width=60).pack()
		Label.place(self,x=50,y=50)
		# page 1 button
		Button(self, text ="Page 1", fg="#1a1a1a", highlightbackground="#1a1a1a", command = lambda : [controller.show_frame(Page1),self.input1()]).pack()
		Button.place(self,x=25,y=75)
		# page 2 button
		Button(self, text ="Page 2", command = lambda : [controller.show_frame(Page2),self.input2(process)]).pack()
		Button.place(self,x=50,y=75)
	#function that is called when clicking the first button
	def input1(self):
		os.system("ls")
	#function that is called when clicking the Second button
	def input2(self,process):
		process.stdin.write("ls".encode())
		process.stdin.flush()
		print(process.stdout.read())
				 

class Page1(tk.Frame):
	def __init__(self, controller,parent,process):
		
		tk.Frame.__init__(self, parent)
		#create label  bg="#1a1a1a"
		Label(self, text ="Page 1", font="Helvetica",width=60).pack()
		Label.place(self,x=50,y=50)
		#create page 1 button
		Button(self, text ="StartPage", command = lambda :[ controller.show_frame(StartPage),self.input1()]).pack()
		Button.place(self,x=25,y=75)
		#create page 2 button
		Button(self, text ="Page 2", command = lambda : [controller.show_frame(Page2),self.input2(process)]).pack()
		Button.place(self,x=50,y=75)
	# function that is called when clicking the first button
	def input1(self):
		subprocess.call("Helloworld")
	# function that is called when clicking the Second button
	def input2(self,process):
		process.communicate(input="ls", stdin=PIPE)

class Page2(tk.Frame): 
	def __init__(self, controller, parent,process):
		tk.Frame.__init__(self, parent)
		#label
		Label(self, text ="Page 2", font="Helvetica",width=60).pack()
		Label.place(self,x=50,y=50)
		#create page 1 button
		Button(self, text ="Page 1", command = lambda : [controller.show_frame(Page1),self.input1(process)]).pack()
		Button.place(self,x=25,y=75)
		#create start page button
		Button(self, text ="Startpage", command = lambda :[ controller.show_frame(StartPage),self.input2(process)]).pack()
		Button.place(self,x=50,y=75)
	# function that is called when clicking the first button
	def input1(self,process):
		subprocess.call("Helloworld")
	# function that is called when clicking the Second button
	def input2(self,process):
		process.stdin.write("ls".encode())
		process.stdin.read()

class Page3(tk.Frame): 
	def __init__(self, controller,parent,process):
		tk.Frame.__init__(self, parent)
		#label
		Label(self, text ="Page 2", font="Helvetica",width=60).pack()
		Label.place(self,x=50,y=50)
		#create page 1 button
		Button(self, text ="Page 1", command = lambda : [controller.show_frame(Page1),self.input1()]).pack()
		Button.place(self,x=25,y=75)
		#create start page button
		Button(self, text ="Startpage", command = lambda :[ controller.show_frame(StartPage),self.input2(process)]).pack()
		Button.place(self,x=50,y=75)
	# function that is called when clicking the first button
	def input1(self):
		subprocess.call("Helloworld")
	# function that is called when clicking the Second button
	def input2(self,process):
		subprocess.call("Helloworld")


		
app = tkinterApp()
app.mainloop()