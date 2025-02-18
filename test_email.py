import time
from emailUpdated import EmailNotifier
from threadManager import QueueManager

test_email = "estradam@whitman.edu"

notifier = EmailNotifier(test_email)

print("testing: waiting 5 seconds before starting event.")
time.sleep(5)

print("testing: setting completion event.")
QueueManager.get_completion_event().set()

time.sleep(5) #process time
print("testing: completed. check email.")

#from emailNotif import EmailNotifier

#test_email = "estradam@whitman.edu"

#notifier = EmailNotifier(test_email)
#notifier.send_email()

#completion_event = QueueManager.get_completion_event()
    
#if completion_event:
 #   print("Setting completion_event!")
#    completion_event.set()
#else:
    #print("Error: completion_event is None!")