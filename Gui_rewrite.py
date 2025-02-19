import tkinter as tk
from tkinter import ttk, messagebox
import os
import pickle
from controller import run_as_thread
from threadManager import ThreadManager, QueueManager

class PickleManager:
    """Manages the saving and retrieving of data from pickle."""
    def __init__(self, filename="pickle.pk"):
        self.filename = filename
        self.default_data = []
        self.data = self._load_or_create()
        self.status_queue = QueueManager.get_status_queue()

    def _load_or_create(self):
        """Loads pickle data or creates a new file if it doesn't exist."""
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
        """Save data in pickle."""
        with open(self.filename, 'wb') as file:
            pickle.dump(data, file)
        return data
    
    def add_entry(self, entry):
        """Adds a new entry if unique and updates the file."""
        if isinstance(entry, str) and entry.strip() and entry not in self.data:
            self.data.append(entry)
            if len(self.data) > 10:  # Limit stored values
                self.data = self.data[1:]
            self._save(self.data)

    def get_data(self):
        return self.data


class GUIApp(tk.Tk):
    """Main GUI application."""
    def __init__(self):
        super().__init__()
        self.title("OT2Control")
        self.geometry("750x450")
        self.configure(bg="#252526")

        self.input_queue = QueueManager.get_input_queue()
        self.status_queue = QueueManager.get_status_queue()
        self.response_queue = QueueManager.get_response_queue()
        self.pickle = PickleManager()
        self.thread_manager = ThreadManager()

        self.sheet_list = []
        self.sim = tk.IntVar()
        self.auto = tk.IntVar()
        self.sheet_name = tk.StringVar()

        self.create_interface()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.thread_manager.start_thread(target=self.update_run_status)
        self.thread_manager.start_thread(target=self.listen_input)

    def create_interface(self):
        """Create UI elements."""
        label = tk.Label(self, text="Sheetname", font=("Arial", 16), fg="white", bg="#252526")
        label.pack()

        self.sheet_input = ttk.Combobox(self, width=50, textvariable=self.sheet_name, values=self.pickle.get_data())
        self.sheet_input.pack()

        self.sim_checkbox = tk.Checkbutton(self, text="Sim", variable=self.sim, fg="white", bg="#252526", selectcolor="#252526")
        self.sim_checkbox.pack(pady=(15, 10))

        self.auto_checkbox = tk.Checkbutton(self, text="Auto", variable=self.auto, fg="white", bg="#252526", selectcolor="#252526")
        self.auto_checkbox.pack()

        self.execute_btn = tk.Button(self, text="Execute", command=self.run_controller, bg="#007acc", fg="white")
        self.execute_btn.pack(pady=(20, 13))

        self.output_text = tk.Text(self, height=10, width=60, bg="#3e3e42", fg="white", wrap=tk.WORD)
        self.output_text.pack(expand=True, fill="both")

        self.deck_pos = tk.Button(self, text="Check Deck Positions", bg="#007acc", fg="white", command=self.run_deckpos)
        self.deck_pos.pack(pady=(0, 17))

    def show_popup(self, popup_type, msg):
        """
        Create a popup of three predefined types:
        - "yesno": Yes/No buttons
        - "continue": Continue button
        - "input": User input field
        """
        popup = tk.Toplevel(self)
        popup.title("User Input")
        popup.geometry("300x150")
        popup.transient(self)  # Keep popup on top
        popup.grab_set()  # Make it modal

        label = tk.Label(popup, text=msg, font=("Arial", 14))
        label.pack(pady=10)

        result = {"value": None}

        def close_popup(value=None):
            result["value"] = value
            popup.grab_release()
            popup.destroy()

        if popup_type == "yesno":
            btn_frame = tk.Frame(popup)
            btn_frame.pack()
            tk.Button(btn_frame, text="Yes", command=lambda: close_popup(True)).pack(side="left", padx=10)
            tk.Button(btn_frame, text="No", command=lambda: close_popup(False)).pack(side="right", padx=10)

        elif popup_type == "continue":
            tk.Button(popup, text="Continue", command=lambda: close_popup(True)).pack(pady=20)

        elif popup_type == "input":
            entry = tk.Entry(popup, width=30)
            entry.pack(pady=10)
            tk.Button(popup, text="Submit", command=lambda: close_popup(entry.get())).pack(pady=10)

        popup.wait_window()
        return result["value"]

    def run_controller(self):
        """Initialize arguments and call controller as a thread."""
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
        """Listen for status messages from the status queue."""
        while True:
            msg = self.status_queue.get()
            self.output_text.insert(tk.END, msg + "\n")
            self.output_text.see(tk.END)  # Auto-scroll

    def listen_input(self):
        """Listen for messages from the input queue."""
        while True:
            if not self.input_queue.empty():
                popup_type, msg = self.input_queue.get()
                self.show_popup(popup_type, msg)

    def run_deckpos(self):
        """Placeholder for running deck positions."""
        pass

    def on_close(self):
        """Handle window close event."""
        if self.show_popup("yesno", "Are you sure you want to quit?"):
            self.thread_manager.stop_all_threads()
            self.destroy()


if __name__ == "__main__":
    window = GUIApp()
    window.mainloop()
