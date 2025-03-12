import tkinter as tk
from tkinter import messagebox, ttk, simpledialog
import os
import pickle
from queue import Queue
from threadManager import QueueManager, ThreadManager
from controller import run_as_thread
from emailNotifier import EmailNotifier
from deckPositionsGui import run_deckpos

class PickleManager:
    """
    This class manages adding and retrieving data from the Pickle file
    """
    def __init__(self, filename="pickle.pk"):
        self.filename = filename
        self.default_data = []
        self.data = self._load_or_create()

    def _load_or_create(self):
        """find the existing pickle file or creat a new one"""
        if not os.path.isfile(self.filename):  
            return self._save(self.default_data)
        try:
            with open(self.filename, 'rb') as file:
                return pickle.load(file)
        except (EOFError, pickle.UnpicklingError):
            return self._save(self.default_data)
        except Exception:
            return self.default_data

    def _save(self, data):
        """save to pickle"""
        with open(self.filename, 'wb') as file:
            pickle.dump(data, file)
        return data
    
    def add_entry(self, entry):
        """add a piece of data to the list for pickle and for gui"""
        if isinstance(entry, str) and entry.strip():
            if entry not in self.data:
                self.data.append(entry)
                if len(self.data) > 10:
                    self.data = self.data[1:]
                self._save(self.data)
    
    def get_data(self):
        """get data from pickle"""
        return self.data

class GUIApp(tk.Tk):
    """
    Main GUI object
    This class represents the main GUI window and passes selected arguements to controller
    """
    def __init__(self):
        super().__init__()
        self.title("OT2Control")
        self.geometry("750x450")
        self.configure(bg='#252526')

        self.input_queue = QueueManager().get_input_queue()
        self.status_queue = QueueManager().get_status_queue()
        self.pickle = PickleManager()
        self.thread_manager = ThreadManager()
        
        self.sheet_name = tk.StringVar()
        self.sim = tk.IntVar()
        self.auto = tk.IntVar()
        
        self.recipient_email = None

        self.create_interface()
        self.listen_input()
        self.protocol("WM_DELETE_WINDOW", self.thread_manager.stop_all_threads()) # make sure all threads close if window closed
        self.thread_manager.start_thread(target=self.update_run_status) # begin update thread
        self.thread_manager.start_thread(target=self.listen_input)


    def create_interface(self):
        """initializes all of the UI elements in the main window"""
        tk.Label(self, text="Sheetname", font=("Inter", 16), fg="white", bg="#252526").pack()
        
        self.sheet_input = ttk.Combobox(self, textvariable=self.sheet_name, values=self.pickle.get_data(), width=50)
        self.sheet_input.pack()
        
        ttk.Checkbutton(self, text="Sim", variable=self.sim).pack(pady=5)
        ttk.Checkbutton(self, text="Auto", variable=self.auto).pack(pady=5)

        tk.Button(self, text="Execute", command=self.run_controller).pack(pady=10)
        tk.Button(self, text="Check Deck Positions", command=self.run_deckpos).pack(pady=10)
        
        tk.Button(self, text="Notify Me", command=self.ask_email).pack(pady=10)

        self.output_text = tk.Text(self, height=10, width=50, bg="#3e3e42", fg="white")
        self.output_text.pack(expand=True, fill="both", pady=10)
        
    def ask_email(self):
        """prompt the user for their email"""
        email = simpledialog.askstring("Notification Signup", "Enter your email for reaction updates:")
        
        if email:
            self.chemist_email = email
            messagebox.showinfo("Success", f"You will be notified at {email} when the reaction is complete.")
        else:
            messagebox.showwarning("Input Required", "Please enter a valid email address.")


    def show_popup(self, type, msg):
        """
        show a popup, type indicated by string
        yesno = a popup with a yes and not option
        continue = a popup which gives the options for ok or cancel
        input = a popup which prompts the user for text input
        """
        if type == "yesno":
            return messagebox.askyesno("User Input", msg)
        elif type == "continue":
            return messagebox.askokcancel("User Input", msg)
        elif type == "input":
            #workaround from Stackoverflow
            newWin = tk.Tk()
            newWin.withdraw()
            returnval = simpledialog.askstring("User Input", msg, parent=newWin)
            newWin.destroy()
            return returnval

    def run_controller(self):
        """
        initializes the controller, status, and email threads
        """
        cli_args = []
        if self.sheet_name.get():
            cli_args.append(f"-n{self.sheet_name.get().strip()}")
        if self.auto.get():
            cli_args.append("-mauto")
        else:
            cli_args.append("-mprotocol")
        if self.sim.get():
            cli_args.append("-s")
        else:
            cli_args.append("--no-sim")
        self.pickle.add_entry(self.sheet_name.get())
        
        if self.recipient_email:
            try:
                self.notifier = EmailNotifier(self.recipient_email)
                messagebox.showinfo("Email Notification", f"Notifications setup for {self.recipient_email}")
                self.thread_manager.start_thread(target=self.notifier.watch_event)
            except Exception as e:
                messagebox.showerror(e)

        self.thread_manager.start_thread(target=run_as_thread, args=(cli_args, )) # begin controller thread
        self.thread_manager.start_thread(target=self.update_run_status) # begin update thread

    def run_deckpos(self):
        """run deckpositions"""
        if self.sheet_name.get() == "":
            self.status_queue.put("no sheetname provided")
        else:
            QueueManager().set_sheetname(str(self.sheet_name.get()))
            self.thread_manager.start_thread(target=run_deckpos)

    def update_run_status(self):
        while True:
            msg = self.status_queue.get()
            self.output_text.insert(tk.END, str(msg) + "\n")
            self.output_text.see(tk.END)
    
    def listen_input(self):
        """manage making popups given requests from input queue, sending return vals to response queue"""
        response_queue = QueueManager.get_response_queue()
        if not self.input_queue.empty():
            type, msg = self.input_queue.get()
            response = self.show_popup(type, msg)
            response_queue.put(response)
        self.after(100, self.listen_input)

        
    def on_close(self):
        """make sure the window is destroyed"""
        self.destroy()

if __name__ == "__main__":
    app = GUIApp()
    app.mainloop()
