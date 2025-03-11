from queue import Queue
import threading

import threading
from queue import Queue

class QueueManager:
    """
    Singleton object for managing the queues used for inter-thread communication.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.status_queue = Queue()
                cls._instance.input_queue = Queue()
                cls._instance.response_queue = Queue()
                cls._instance._sheetname = ""
                cls._instance._sheetname_lock = threading.Lock()  # Lock for string access
                cls._instance.completion_event = threading.Event()
        return cls._instance

    @staticmethod
    def get_status_queue():
        return QueueManager()._instance.status_queue

    @staticmethod
    def get_input_queue():
        return QueueManager()._instance.input_queue

    @staticmethod
    def get_response_queue():
        return QueueManager()._instance.response_queue
    
    @staticmethod
    def get_completion_event():
        return QueueManager()._instance.completion_event

    @staticmethod
    def get_sheetname():
        with QueueManager()._instance._sheetname_lock:
            return QueueManager()._instance._sheetname

    @staticmethod
    def set_sheetname(name):
        with QueueManager()._instance._sheetname_lock:
            QueueManager()._instance._sheetname = name
    

class ThreadManager:
    """
    Object for starting, managing and cleanly ending all threads
    """
    def __init__(self):
        self.threads = []

    def start_thread(self, target, args=(), daemon=True):
        """Starts a new thread and keeps track of it"""
        thread = threading.Thread(target=target, args=args, daemon=daemon)
        self.threads.append(thread)
        thread.start()
        print(thread.getName)
        return thread

    def stop_all_threads(self):
        """Stops all running threads"""
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1)  # Give some time to stop
