import cv2
import time 
import os
from threadManager import QueueManager, ThreadManager

class CameraManager:
    def __init__(self):
        self.camera_index = 1
        self.thread_manager = ThreadManager()
        self.completion_event = QueueManager.get_completion_event()
        
    def capture_photo(self):
        try:
            camera = cv2.VideoCapture(self.camera_index)
            if not camera.isOpened():
                print("Error: Could not access the camera.")
                return
            ret, frame = camera.read()
            if ret:
                filename = os.path.expanduser("~/Desktop/reaction_photo.jpg")
                cv2.imwrite(filename, frame)
                print(f"Photo saved as {filename}")
            else:
                print("failed to capture photo")
            camera.release()
        except Exception as e:
            print(f"Error capturing photo: {e}")
            
    def start_timelapse(self, interval):
        try:
            interval = int(interval)
            if interval <= 0:
                print("Error: Interval must be greater than 0 seconds.")
                return
            print(f"Starting timelapse with {interval}-second interval.")
            self.thread_manager.start_thread(target = self.timelapse_photos, args=(interval,))
        except ValueError:
            print("Error: Please enter valid number for the interval.")
            
    def timelapse_photos(self, interval):
        try:
            camera = cv2.VideoCapture(self.camera_index)
            if not camera.isOpened():
                print("Error: Could not access the robot camera.")
                return
            while not self.completion_event.is_set():  # Wait until reaction is done
                ret, frame = camera.read()
                if ret:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = os.path.expanduser(f"~/Desktop/reaction_photo_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"Timelapse photo saved as {filename}")
                else:
                    print("Failed to capture photo.")
                time.sleep(interval)
            camera.release()
            print("Timelapse stopped: Reaction completed.")
        except Exception as e:
            print(f"Error in timelapse: {e}")