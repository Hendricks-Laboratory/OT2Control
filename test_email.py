from emailNotif import EmailNotifier

test_email = "ericanmock@gmail.com"

notifier = EmailNotifier(test_email)
notifier.send_email()