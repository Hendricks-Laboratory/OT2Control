import os
import httplib2
import oauth2client
from oauth2client import client, tools, file
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from apiclient import errors, discovery
from email.mime.base import MIMEBase

from threadManager import QueueManager, ThreadManager

class EmailNotifier:
    """
    Sends an email to the user when the robot reaction is complete.
    Completion is triggered by `completion_event` being set in `controller.py`.
    """

    SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
    CLIENT_SECRET_FILE = "hendricks-gmail-api-send-email.json"
    APP_NAME = "Gmail API Send Email"

    @staticmethod
    def get_credentials():
        home_dir = os.path.expanduser('~')
        credential_dir = os.path.join(home_dir, '.credentials')
        if not os.path.exists(credential_dir):
            os.makedirs(credential_dir)
        credential_path = os.path.join(credential_dir, "hendricks-gmail-api-send-email.json")

        store = oauth2client.file.Storage(credential_path)
        credentials = store.get()
        if not credentials or credentials.invalid:
            flow = client.flow_from_clientsecrets(EmailNotifier.CLIENT_SECRET_FILE, EmailNotifier.SCOPES)
            flow.user_agent = EmailNotifier.APP_NAME
            credentials = tools.run_flow(flow, store)
            print('Storing credentials to ' + credential_path)
        return credentials

    def send_message(self, sender, to, subject, msgHtml, msgPlain):
        credentials = self.get_credentials()
        http = credentials.authorize(httplib2.Http())
        service = discovery.build('gmail', 'v1', http=http)
        message1 = self.create_message_html(sender, to, subject, msgHtml, msgPlain)
        result = self.send_message_internal(service, "me", message1)
        return result
    
    @staticmethod
    def send_message_internal(service, user_id, message):
        try:
            message = (service.users().messages().send(userId=user_id, body=message).execute())
            print('Message Id: %s' % message['id'])
            return message
        except errors.HttpError as error:
            print('An error occurred: %s' % error)
            return "Error"
    
    @staticmethod
    def create_message_html(sender, to, subject, msgHtml, msgPlain):
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = to
        msg.attach(MIMEText(msgPlain, 'plain'))
        msg.attach(MIMEText(msgHtml, 'html'))
        return {'raw': base64.urlsafe_b64encode(msg.as_bytes()).decode()}  # Fix encoding issue

    def __init__(self, recipient_email):
        self.recipient_email = recipient_email
        self.completion_event = QueueManager.get_completion_event()

    def watch_event(self):
        """Waits for the completion event and sends an email notification."""
        print("EmailNotifier: Waiting for completion event...")

        while True:
            self.completion_event.wait()  # Blocks until `set()` is called
            print("EmailNotifier: Event detected. Sending email.")

            try:
                self.send_message(
                    sender="your-email@example.com",
                    to=self.recipient_email,
                    subject="Task Completed",
                    msgHtml="<p>The task has been completed.</p>",
                    msgPlain="The task has been completed."
                )
            except Exception as e:
                print("Error in watch_event():", e)
            
            self.completion_event.clear()