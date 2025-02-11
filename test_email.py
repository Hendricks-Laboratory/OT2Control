

#from threadManager import QueueManager

#completion_event = QueueManager.get_completion_event()
#completion_event.set()  # Simulate reaction completion

from emailNotif import EmailNotifier

test_email = "estradam@whitman.edu"

notifier = EmailNotifier(test_email)
notifier.send_email()