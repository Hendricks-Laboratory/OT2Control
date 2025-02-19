import smtplib
import ssl
import logging
from email.mime.text import MIMEText
from threadManager import QueueManager, ThreadManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class EmailNotifier:
    """
    Sends an email to the user when the robot reaction is complete.
    Uses SMTP with an App Password for authentication.
    """

    smtp_server = "smtp.gmail.com"
    smtp_port = 465  # SSL port

    sender_email = "mocke@whitman.edu"  # Replace with your Gmail address
    sender_password = "rxcsnflfmzyjgdpi"  # Replace with your generated App Password

    def __init__(self, recipient_email):
        self.recipient_email = recipient_email
        self.completion_event = QueueManager.get_completion_event()
        self.thread_manager = ThreadManager()
        self.running = True  # Flag to stop the watcher

        # Start monitoring event
        self.thread_manager.start_thread(self.watch_event, daemon=True)

    def watch_event(self):
        """Waits for the completion event and sends an email notification."""
        logging.info("EmailNotifier: Waiting for completion event...")

        while self.running:
            self.completion_event.wait()  # Blocks until `set()` is called
            if not self.running:  # Break loop if stopped
                break

            logging.info("EmailNotifier: Event detected! Sending email.")

            try:
                self.thread_manager.start_thread(self.send_email, daemon=True)
            except Exception as e:
                logging.error("Error in watch_event(): %s", e)

            self.completion_event.clear()  # Reset event for future triggers

    def send_email(self):
        """Sends an email notification using Google's SMTP server."""
        if not self.recipient_email:
            logging.warning("No recipient email provided. Skipping notification.")
            return

        logging.info(f"send_email() triggered for {self.recipient_email}")

        # Create email message
        msg = MIMEText("Your reaction is complete! You can now check your results.")
        msg["Subject"] = "Reaction Complete Notification"
        msg["From"] = self.sender_email
        msg["To"] = self.recipient_email

        try:
            logging.info(f"Connecting to SMTP server {self.smtp_server} on port {self.smtp_port}...")
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context) as server:
                logging.info("Connected to SMTP server!")
                logging.info("Logging in to SMTP server...")
                server.login(self.sender_email, self.sender_password)
                logging.info("Login successful!")
                logging.info("Sending email...")
                server.sendmail(self.sender_email, self.recipient_email, msg.as_string())
                logging.info(f"Email successfully sent to {self.recipient_email}!")

        except smtplib.SMTPAuthenticationError as e:
            logging.error("SMTP Authentication Error: Check email & app password. Details: %s", e)
        except smtplib.SMTPConnectError as e:
            logging.error("SMTP Connection Error: Could not connect to email server. Details: %s", e)
        except smtplib.SMTPException as e:
            logging.error("General SMTP Error: %s", e)
        except Exception as e:
            logging.error("General Error: %s", e)

    def stop(self):
        """Stops the watcher thread and cleans up."""
        logging.info("Shutting down EmailNotifier...")
        self.running = False
        self.completion_event.set()  # Unblock the wait loop
        self.thread_manager.stop_all_threads()
        logging.info("EmailNotifier shutdown complete.")
