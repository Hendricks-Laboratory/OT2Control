import time 
from camera import CameraManager
from threadManager import ThreadManager, QueueManager

thread_manager = ThreadManager()
camera_manager = CameraManager()

thread_manager.start_camera_manager(camera_manager)

time.sleep(2)
print("test 1: capture 1 pic")
QueueManager.get_camera_queue().put("capture_photo")

time.sleep(2)

print("test 2: starting timelapse")
QueueManager.get_camera_queue().put("start_timelapse")

print("test 3: stopping timelpase")
QueueManager.get_camera_queue().put("stop_timelapse")

time.sleep(2)

print("test complete")
             