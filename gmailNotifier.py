import os
import base64
from email.mime.text import MIMEText
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from threadManager import QueueManager, ThreadManager 

class EmailNotifier:
    """
    Sends an email to the user when the robot reaction is complete.
    Completion is triggered by `completion_event` being set in `controller.py`.
    """

    SCOPES = [
        "https://www.googleapis.com/auth/gmail.send",
        #"https://www.googleapis.com/auth/gmail.readonly",
        #"https://www.googleapis.com/auth/userinfo.email"  
        ]

    def __init__(self, recipient):
        self.recipient = recipient
        self.completion_event = QueueManager.get_completion_event()
        self.credentials = self.get_credentials()
        if not self.credentials:
            print("Failed to authenticate Gmail API.")
            return
        self.service = build("gmail", "v1", credentials=self.credentials)

    def get_credentials(self):
        """Authenticate user with OAuth 2.0 and return credentials
        """
        print("Inside get_credentials function")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        credentials_path = "/mnt/c/Users/science_356_lab/Robot_Files/OT2Control/Credentials/credentials.json"
        #os.path.join(script_dir, "Credentials/credentials.json")
        print(f"Using credentials file at: {credentials_path}")
        try:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, self.SCOPES)
            creds = flow.run_local_server(port=0)
            print("Authentication successful!")
            return creds
        except Exception as e:
            print(e)
            return None
        
    def create_email(self, sender, to, subject, body):
        """Create a MIME email message
        """
        message = MIMEText(body)
        message["to"] = to
        message["from"] = sender
        message["subject"] = subject
        return {"raw": base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")}

    def send_email(self):
        """Sends an email using the Gmail API."""
        try:
            sender = "your-email@gmail.com"
            #recipient = input("Enter your email address: ")
            subject = "Reaction Completed Notification"
            body = "Hello! Your reaction is complete. You can now view your results."
            message = self.create_email(sender, self.recipient, subject, body)
            sent_message = self.service.users().messages().send(userId="me", body=message).execute()
            print(f"Email sent successfully!{sent_message['id']}")
        except Exception as error:
            print(f"An error occurred while sending email: {error}")

    def watch_event(self):
        """Waits for the completion event and sends an email notification."""
        print("Waiting for completion event...")
        while True:
            self.completion_event.wait()  # Blocks until `set()` is called
            print("Event detected. Sending email.")
            self.send_email()
            self.completion_event.clear()