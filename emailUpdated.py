import os
import base64
import logging
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from threadManager import QueueManager, ThreadManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Scopes required for sending emails
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

class EmailNotifier:
    """
    Sends an email when the robot reaction is complete.
    The completion event triggers this action automatically.
    """

    def __init__(self, recipient_email):
        self.recipient_email = recipient_email
        self.completion_event = QueueManager.get_completion_event()
        self.thread_manager = ThreadManager()
        self.running = True  # Flag to control the thread loop

        # Authenticate Gmail API
        self.service = self.authenticate_gmail()

        # Start monitoring event in a separate thread
        self.thread_manager.start_thread(self.watch_event, daemon=True)

    def authenticate_gmail(self):
        """Authenticates and returns the Gmail API service instance."""
        creds = None
        token_file = "token.json"
        credentials_file = "credentials.json"  # Ensure this file is downloaded from Google Cloud Console

        # Load existing token if available
        if os.path.exists(token_file):
            creds = Credentials.from_authorized_user_file(token_file, SCOPES)

        # If no valid credentials, go through OAuth flow
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
                creds = flow.run_local_server(port=0)
            # Save new credentials for future use
            with open(token_file, "w") as token:
                token.write(creds.to_json())

        return build("gmail", "v1", credentials=creds)

    def watch_event(self):
        """Waits for the completion event and sends an email notification."""
        logging.info("EmailNotifier: Waiting for completion event...")

        while self.running:
            self.completion_event.wait()  # Blocks until `set()` is called
            if not self.running:  # Stop loop if shutdown
                break

            logging.info("EmailNotifier: Event detected! Sending email.")

            try:
                self.thread_manager.start_thread(self.send_email, daemon=True)
            except Exception as e:
                logging.error("Error in watch_event(): %s", e)

            self.completion_event.clear()  # Reset event for future triggers

    def send_email(self):
        """Sends an email notification using Gmail API."""
        if not self.recipient_email:
            logging.warning("No recipient email provided. Skipping notification.")
            return

        logging.info(f"send_email() triggered for {self.recipient_email}")

        # Create email message
        message = MIMEText("Your reaction is complete! You can now check your results.")
        message["to"] = self.recipient_email
        message["subject"] = "Reaction Complete Notification"

        # Encode message for Gmail API
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        try:
            logging.info(f"Sending email to {self.recipient_email} via Gmail API...")
            send_message = {"raw": raw_message}
            self.service.users().messages().send(userId="me", body=send_message).execute()
            logging.info(f"Email successfully sent to {self.recipient_email}!")
        except Exception as e:
            logging.error("Failed to send email via Gmail API: %s", e)

    def stop(self):
        """Stops the event watcher and cleans up threads."""
        logging.info("Shutting down EmailNotifier...")
        self.running = False
        self.completion_event.set()  # Unblock the wait loop
        self.thread_manager.stop_all_threads()
        logging.info("EmailNotifier shutdown complete.")
