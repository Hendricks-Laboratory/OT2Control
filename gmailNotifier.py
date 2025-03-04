import base64
import json
import os
from email.mime.text import MIMEText
from google.oauth2 import service_account
from googleapiclient.discovery import build
from threadManager import QueueManager, ThreadManager

class EmailNotifier:
    """
    Sends an email to the user when the robot reaction is complete.
    Completion is triggered by `completion_event` being set in `controller.py`.
    """

    SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

    # Path to credentials file (Updated with correct repository structure)
    REPO_DIR = os.path.dirname(os.path.abspath(__file__))  # Gets script directory
    SERVICE_ACCOUNT_FILE = os.path.join(REPO_DIR, "Credentials", "hendricks-lab-jupyter-sheets-5363dda1a7e0.json")

    def __init__(self, recipient_email):
        self.recipient_email = recipient_email
        self.completion_event = QueueManager.get_completion_event()

        # Authenticate Gmail API and extract service account email
        self.service, self.sender_email = self.authenticate_gmail()

        # Start listening for the completion event

    def authenticate_gmail(self):
        """Authenticate using service account credentials and retrieve sender email."""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.SERVICE_ACCOUNT_FILE, scopes=self.SCOPES
            )

            # Extract service account email dynamically
            with open(self.SERVICE_ACCOUNT_FILE, "r") as file:
                credentials_data = json.load(file)
                sender_email = credentials_data.get("client_email")  # Gets service account email

            # If using a Google Workspace account, delegate credentials
            delegated_credentials = credentials.with_subject(sender_email)
            service = build("gmail", "v1", credentials=delegated_credentials)

            print(f"Gmail API authenticated successfully! Using sender: {sender_email}")
            return service, sender_email

        except Exception as e:
            print("Error authenticating Gmail API:", e)
            return None, None

    def watch_event(self):
        """Waits for the completion event and sends an email notification."""
        print("EmailNotifier: Waiting for completion event...")

        while True:
            print("checking for event")
            if self.completion_event.is_set():
                print("Event detected before wait.")
            self.completion_event.wait()  # Blocks until `set()` is called
            print("EmailNotifier: Event detected after wait. Sending email.")

            try:
                self.send_email()
                # print("sending......")
            except Exception as e:
                print("Error in watch_event():", e)
            self.completion_event.clear()

    def send_email(self):
        """Sends an email notification using Gmail API."""
        if not self.recipient_email:
            print("⚠️ No recipient email provided. Skipping notification.")
            return

        if not self.sender_email:
            print("Error: Sender email not available from credentials.")
            return

        print(f"📩 send_email() called for {self.recipient_email}")

        msg = MIMEText("Your reaction is complete! You can now check your results.")
        msg["Subject"] = "Reaction Complete Notification"
        msg["From"] = self.sender_email
        msg["To"] = self.recipient_email

        encoded_message = base64.urlsafe_b64encode(msg.as_bytes()).decode()

        try:
            send_message = (
                self.service.users()
                .messages()
                .send(userId="me", body={"raw": encoded_message})
                .execute()
            )
            print(f"Email successfully sent to {self.recipient_email}!")
            print("Message ID:", send_message["id"])

        except Exception as e:
            print("Error sending email:", e)