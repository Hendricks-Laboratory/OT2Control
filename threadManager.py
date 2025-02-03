from queue import Queue
import threading

class QueueManager:
    """
    Singleton object for managing the two queues used for inter-thread communication
    status_queue: queue for sending status text from controller to the GUI
    input_queue: queue for sending input (yes, no, continue) to controller from GUI
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
