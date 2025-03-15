import time 
from camera_manager import CameraManager
from threadManager import QueueManager

camera_manager = CameraManager()


time.sleep(2)

print("capturing single photo")
camera_manager.capture_photo()

time.sleep(3)

print("starting timelapse")
camera_manager.start_timelapse(3)


time.sleep(10)

print("stopping time lapse using completion event")
QueueManager.get_completion_event().set()

time.sleep(2)

print("test complete")
             