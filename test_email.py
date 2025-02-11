
from emailNotif import EmailNotifier
from threadManager import QueueManager

test_email = "estradam@whitman.edu"

notifier = EmailNotifier(test_email)

completion_event = QueueManager.get_completion_event()
completion_event.set()  # Simulate reaction completion


#from emailNotif import EmailNotifier

#test_email = "estradam@whitman.edu"

#notifier = EmailNotifier(test_email)
#notifier.send_email()

#completion_event = QueueManager.get_completion_event()
    
if completion_event:
    print("Setting completion_event!")
    completion_event.set()
else:
    print("Error: completion_event is None!")