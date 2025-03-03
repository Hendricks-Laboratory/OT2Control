import tkinter as tk
from tkinter import messagebox, ttk, simpledialog
import os
import pickle
from queue import Queue
from threadManager import QueueManager, ThreadManager
from controller import run_as_thread
class PickleManager:
    def __init__(self, filename="pickle.pk"):
        self.filename = filename
        self.default_data = []
        self.data = self._load_or_create()

    def _load_or_create(self):
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
        with open(self.filename, 'wb') as file:
            pickle.dump(data, file)
        return data
    
    def add_entry(self, entry):
        if isinstance(entry, str) and entry.strip():
            if entry not in self.data:
                self.data.append(entry)
                if len(self.data) > 10:
                    self.data = self.data[1:]
                self._save(self.data)
    
    def get_data(self):
        return self.data

class GUIApp(tk.Tk):
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

        self.create_interface()

        self.protocol("WM_DELETE_WINDOW", self.thread_manager.stop_all_threads()) # make sure all threads close if window closed
        

    def create_interface(self):
        tk.Label(self, text="Sheetname", font=("Inter", 16), fg="white", bg="#252526").pack()
        
        self.sheet_input = ttk.Combobox(self, textvariable=self.sheet_name, values=self.pickle.get_data(), width=50)
        self.sheet_input.pack()
        
        ttk.Checkbutton(self, text="Sim", variable=self.sim).pack(pady=5)
        ttk.Checkbutton(self, text="Auto", variable=self.auto).pack(pady=5)

        tk.Button(self, text="Execute", command=self.run_controller).pack(pady=10)
        
        self.output_text = tk.Text(self, height=10, width=50, bg="#3e3e42", fg="white")
        self.output_text.pack(expand=True, fill="both", pady=10)

    def show_popup(self, type, msg):
        if type == "yesno":
            return messagebox.askyesno("User Input", msg)
        elif type == "continue":
            return messagebox.askokcancel("User Input", msg)
        elif type == "input":
            return simpledialog.askstring("User Input", msg)

    def run_controller(self):
        cli_args = []
        if self.sheet_name.get():
            cli_args.append(f"-n{self.sheet_name.get().strip()}")
        if self.auto.get():
            cli_args.append("-mauto")
        if self.sim.get():
            cli_args.append("--no-sim")
        self.pickle.add_entry(self.sheet_name.get())

        self.thread_manager.start_thread(target=run_as_thread)

        self.thread_manager.start_thread(target=self.update_run_status) # begin update thread
        self.thread_manager.start_thread(target=self.listen_input)

    def update_run_status(self):
        while True:
            msg = self.status_queue.get()
            self.output_text.insert(tk.END, msg + "\n")
            self.output_text.see(tk.END)
    
    def listen_input(self):
        while True:
            if not self.input_queue.empty():
                type, msg = self.input_queue.get()
                self.show_popup(type, msg)

    def on_close(self):
        self.destroy()

if __name__ == "__main__":
    app = GUIApp()
    app.mainloop()
