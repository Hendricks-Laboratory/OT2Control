import customtkinter
from customtkinter import IntVar, CHECKBUTTON
# from subprocess import check_output
# import subprocess
# import threading
# import pty
import os
import pickle
from controller import run_as_thread
# from deckPositionsGui import CTkinterApp
from threadManager import ThreadManager, QueueManager

class PickleManager():
    """This class manages the saving and retrieving of data from pickle"""
    def __init__(self, filename="pickle.pk"):
        self.filename = filename
        self.default_data = []
        self.data = self._load_or_create()
        self.status_queue = QueueManager.get_status_queue()
        print(self.data)

    def _load_or_create(self):
        """loads pickle data or creates a new file if it doesn't exist."""
        if not os.path.isfile(self.filename):  
            self.status_queue.put(f"File {self.filename} does not exist. Creating a new one...")
            return self._save(self.default_data)
        try:
            with open(self.filename, 'rb') as file:
                return pickle.load(file)
            
        except (EOFError, pickle.UnpicklingError):
            self.status_queue.put(f"Error: {self.filename} is empty or corrupted. Resetting file.")
            return self._save(self.default_data)
        except Exception as e:
            self.status_queue.put(f"Unexpected error reading {self.filename}: {e}")
            return self.default_data

    def _save(self, data):
        """save some data in pickle"""
        with open(self.filename, 'wb') as file:
            pickle.dump(data, file)
        return data
    
    def add_entry(self, entry): 
        """Adds a new entry to the pickle file (if unique) and updates the file."""
        if isinstance(entry, str) and entry.strip():
            if entry not in self.data:
                self.data.append(entry)
                if len(self.data) > 10:  # Limit stored values
                    self.data = self.data[1:]
                self._save(self.data)
        else:
            self.status_queue.put("Invalid entry: Not a valid string.")
            
    def get_data(self):
        return self.data

class GUIApp(customtkinter.CTk):
    """
    This class represents the main gui app
    """
    def __init__(self):
        """initializes the GUI window, and variables associated with the inputs"""
        super().__init__()
        self.title("OT2Control") # create main window
        self.geometry("750x450")
        self.configure(fg_color= '#252526')

        self.input_queue = QueueManager.get_input_queue() # intialize the shared queues and the threadmanager
        self.status_queue = QueueManager.get_status_queue()
        self.response_queue = QueueManager.get_response_queue()
        self.pickle = PickleManager()
        self.thread_manager = ThreadManager()
        # self.deck_positions = CTkinterApp()

        self.sheet_list = [] # variables for widgets in window
        self.sim = customtkinter.IntVar()
        self.auto = customtkinter.IntVar()
        self.sheet_name = customtkinter.StringVar()
        self.create_interface()

        self.protocol("WM_DELETE_WINDOW", self.thread_manager.stop_all_threads()) # make sure all threads close if window closed
        self.thread_manager.start_thread(target=self.update_run_status) # begin update thread
        self.thread_manager.start_thread(target=self.listen_input)

    def create_interface(self):
        """Create elements within the window"""
        self.sheet_label = customtkinter.CTkLabel(self, text="Sheetname", font=("Inter", 16), text_color="white")
        self.sheet_label.pack()

        self.sheet_input = customtkinter.CTkComboBox(self, width=400, variable=self.sheet_name, values = self.pickle.get_data(),fg_color='#3e3e42')
        self.sheet_input.pack()

        self.sim_checkbox = customtkinter.CTkCheckBox(self, text="Sim", variable=self.sim, fg_color="#303030", text_color="white")
        self.sim_checkbox.pack(pady=(15, 10))

        self.auto_checkbox = customtkinter.CTkCheckBox(self, text="Auto", variable=self.auto, text_color="white")
        self.auto_checkbox.pack()

        self.execute_btn = customtkinter.CTkButton(self, text="Execute", command=self.run_controller)
        self.execute_btn.pack(pady=(20, 13))
        
        self.output_text = customtkinter.CTkTextbox(self, height=200, width=400, fg_color="#3e3e42", text_color="white")
        self.output_text.pack(expand=True, fill="both")

        self.deck_pos = customtkinter.CTkButton(self, text= "Check Deck Positions",fg_color='#007acc', font= ("Inter", 12), command=self.run_deckpos(), width=30)
        self.deck_pos.pack(pady= (0, 17))

    def show_popup(self, type, msg):
        """
        Create a popup of three predefined types
        type = yesno:
            creates a popup window including msg, which has options for yes or no
        type = continue:
            creates a popup window which prompts to continue, includes msg
        type = input:
            creates a popup that takes text input, returns text input to initial call for input
        
        The return values of this function will differ depending on the type and user input, keep this in mind when using 
        return values from this method.

        """
        popup = customtkinter.CTkToplevel(self)
        popup.title("User Input")
        popup.geometry("300x150")
        popup.transient(self)  # Keep popup on top
        popup.grab_set()  # Make it modal

        label = customtkinter.CTkLabel(popup, text=msg, font=("Inter", 14))
        label.pack(pady=10)

        result = {"value": None}  # Store the result in a mutable dictionary

        def close_popup(value=None):
            result["value"] = value
            popup.grab_release()
            popup.destroy()
            print("popup closed")

        if type == "yesno":
            yes_button = customtkinter.CTkButton(popup, text="Yes", command=lambda: close_popup(True))
            no_button = customtkinter.CTkButton(popup, text="No", command=lambda: close_popup(False))
            yes_button.pack(side="left", expand=True, padx=10, pady=10)
            no_button.pack(side="right", expand=True, padx=10, pady=10)

        elif type == "continue":
            continue_button = customtkinter.CTkButton(popup, text="Continue", command=lambda: close_popup(True))
            continue_button.pack(pady=20)

        elif type == "input":
            entry = customtkinter.CTkEntry(popup, width=200)
            entry.pack(pady=10)
            submit_button = customtkinter.CTkButton(popup, text="Submit", command=lambda: close_popup(value=entry.get()))
            submit_button.pack(pady=10)

        def check_popup():
            # this is generated by chatgpt, for some reason wait_window does not work, makes the windows unresponsive
            if not popup.winfo_exists():
                self.response_queue.put(result["value"])
                # print("response sent")
            else:
                self.after(100, check_popup)

        check_popup()

    def run_controller(self):
        """
        initializes arguements and calls controller as a thread
        """
        cli_args = []

        if self.sheet_name.get():
            cli_args.append(f"-n{self.sheet_name.get().strip()}")
        if self.auto.get():
            cli_args.append("-mauto")
        if self.sim.get():
            cli_args.append("--no-sim")

        self.pickle.add_entry(self.sheet_name.get())

        self.thread_manager.start_thread(target=run_as_thread, args=(cli_args,))

    def update_run_status(self):
        """
        listen for status messages from the status queue
        """
        while True:
            msg = self.status_queue.get()
            self.output_text.insert(customtkinter.END, msg + "\n")
    
    def listen_input(self):
        """
        listen for messages from the input queue
        """
        while True:
            if not self.input_queue.empty():
                type, msg = self.input_queue.get()
                self.show_popup(type, msg)

    #not gonna deal with this rn
    def run_deckpos(self):
        # self.thread_manager.start_thread(self.deck_pos.mainloop())
        # def execute_python_file(file_name, argument,Textbox):
        #     try:
        #         process = start_subprocess(["python3", file_name,'-n',argument], Textbox)
        #     except FileNotFoundError:
        #         print("Error: The file does not exist.")
                
        # def start_subprocess(command, textbox):
        #     try:
        #         master_fd, slave_fd = pty.openpty() #PseudoTerminal
        #         process = subprocess.Popen(command, stdout=slave_fd, stderr=subprocess.STDOUT, stdin=slave_fd, universal_newlines=True)
        #         os.close(slave_fd)
        #         # threading.Thread(target=read_output, args=(master_fd, textbox), daemon=True).start()
        #     except Exception as e:
        #         print(f"Error starting subprocess: {e}")
        #         # update_output("Error starting subprocess: " + str(e), textbox)
        #         return None
        #     return process
        
        # execute_python_file('deckPositionsGui.py',self.sheet_name.get(),self.output_text)
        pass
    
    
window = GUIApp()
window.mainloop()

