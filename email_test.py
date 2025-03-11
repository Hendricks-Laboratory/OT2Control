import time
from emailNotifier import EmailNotifier
from threadManager import QueueManager, ThreadManager

recipient = "mocke@whitman.edu"

# QueueManager.get_completion_event().t()

notifier = EmailNotifier(recipient)

tm = ThreadManager()
tm.start_thread(target=notifier.watch_event)

print("testing: waiting 5 seconds before starting event.")
time.sleep(5)

print("testing: setting completion event.")
QueueManager.get_completion_event().set()

time.sleep(5) #process time
print("testing: completed. check email.")
