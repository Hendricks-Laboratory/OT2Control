import cv2
import time 
from threadManager import QueueManager, ThreadManager

class CameraManager:
    def _init_(self):
        self.camera_index = 1 # USB ccamera index
        self.thread_manger = ThreadManger()
        self.camera_queue = QueueManger.get_camera_queue()
        self.status_queue = QueueManger.get_camera_queue()
        self.log_callback = None
        self.timelapse_running = False
        self.timelapse_thread = None

    # These two functions allows the GUI to set a logging function so messages can be displayed in the application insead of the terminal
    def set_log_callback(self, callback):
        self.log_callback = callback
     
    
    # displays messages in gui textbox    
    def log_message(self, message): #Logs message to GUI if available, otherwise prints to terminal.
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)  # Fallback to terminal output
            self.status_queue.put(message) # Send status messages to the queue so the GUI can display them


    # This function task is just to capture a single photo using the USB -connected camera and saves it as jpg.
    def capture_photo(self):
        try:
            camera = cv2.VideoCapture(self.camera_index)  # since we are using USB
            if not camera.isOpened():
                self.log_message("Error: Could not access the robot camera.")
                return
            ret, frame = camera.read()
            if ret:
                filename = os.path.expanduser("~/Desktop/reaction_photo.jpg")
                cv2.imwrite(filename, frame)
                self.log_message(f"Photo saved as {filename}")
            else:
                self.log_message("Failed to capture photo")  
            
            camera.release()
            
        except Exception as e:
            self.log_message(f"Error capturing photo: {e}")


#This function handels timelapse with a specified interval (in seconds)
    def start_timelapse(self,interval):
        if self.timelapse_running:
            self.log_message("Timelapse is already running.")
            return
        try:
            interval = int(interval)  
            if interval <= 0:
                self.log_message("Error: Interval must be greater than 0 seconds")
                return

            self.log_message(f"Starting timelapse with {interval}-second intervals...")
            self.timelapse_running = True
            self.timelapse_thread = self.start_thread(self.timelapse_photos, daemon = true)
            
        #self.threading.Thread(target=timelapse_photos, args=(interval,), daemon=True).start()
        
        except ValueError:
            self.log_message("Error: Please enter a valid number for the interval")



# Captures photos continuously at the specified interval until stopped.
    def timelapse_photos(self,interval):
        #global timelapse_running 
        try:
            camera = cv2.VideoCapture(self.camera_index) 
            if not camera.isOpened():
                self.log_message("Error: Could not access the robot camera.")
                return

            while self.timelapse_running:
                ret, frame = camera.read()
                if ret:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")# formats the current date and time into a string using the given format Year,month,day,hour,minute and seconds.
                    filename = f"reaction_photo_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    self.log_message(f"Timelapse photo saved as {filename}")
                else:
                    self.log_message("Failed to capture photo")
                
                time.sleep(interval)
    
            camera.release()
        except Exception as e:
            self.log_message(f"Error in timelapse: {e}")
        
    # This function will Stops the running timelapse. by turning the global variable to fasle and prints a message
    def stop_timelapse():
    #global timelapse_running
        if not self.timelapse_running:
            self.log_message("Timelapse stopped.")
            return
       
        self.timelapse_running = False
        self.log_message("Timelapse stopped")
        self.timelapse_thread.stop_thread()
        

    # This function will handel camera tasks from the queue.
    def process_camera_tasks():
        while True:
            task = self.camera_queue.get()
            if task == "capture_photo":
                self.capture_photo()
            elif task.startswith("start_timelapse"):
                _, interval = task.split(":")
                self.start_timelapse(interval)
            elif task == "stop_timelapse":
                self.stop_timelapse()
            elif task == "reaction_done":
                self.log_message("Reaction completed! Capturing final photo...")
                self.capture_photo()

#Starts a background thread to continuously process camera tasks from the queue.
#threading.Thread(target=process_camera_tasks, daemon=True).start()
