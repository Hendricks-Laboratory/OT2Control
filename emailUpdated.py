import smtplib
import ssl
from email.mime.text import MIMEText
from threadManager import QueueManager, ThreadManager


class EmailNotifier:
    """
    Sends an email to the user when the robot reaction is complete.
    Completion is triggered by `completion_event` being set on shutdown in `controller.py`.
    """
    
    smtp_server = "smtp.gmail.com"
    smtp_port = 465  # SSL port
    
    
  

    def __init__(self, recipient_email):
        self.recipient_email = recipient_email
        self.completion_event = QueueManager.get_completion_event()
        self.thread_manager = ThreadManager()

        # Start monitoring event
        self.thread_manager.start_thread(self.watch_event, daemon=True)

    def watch_event(self):
        """Waits for the completion event and sends an email notification."""
        print("EmailNotifier: Waiting for completion event...")
        while True:
            self.completion_event.wait()  # Blocks until `set()` is called
            print("EmailNotifier: Event detected! Sending email.")

            try:
                print("Calling send_email()")
                self.send_email()
            except Exception as e:
                print("Error in watch_event():", e)

            self.completion_event.clear()

    def send_email(self):
        """Sends an email notification."""
        if not self.recipient_email:
            print("No recipient email provided. Skipping notification.")
            return
        
        print(f"send_email() was called for {self.recipient_email}")

        msg = MIMEText("Your reaction is complete! You can now check your results.")
        msg["Subject"] = "Reaction Complete Notification"
        msg["From"] = self.sender_email
        msg["To"] = self.recipient_email

        try:
            print(f"Attempting to send email to {self.recipient_email}...")
            print(f"Connecting to SMTP server {self.smtp_server} on port {self.smtp_port}...")
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context) as server:
                print("Connected to SMTP server!")
                print("Logging in to SMTP server...")
                server.login(self.sender_email, self.sender_password)
                print("Login successful!")
                print("Sending email...")
                server.sendmail(self.sender_email, self.recipient_email, msg.as_string())
                print(f"Email successfully sent to {self.recipient_email}!")

        except smtplib.SMTPAuthenticationError as e:
            print("SMTP Authentication Error: Check email & app password.")
            print("Error details:", e)
        except smtplib.SMTPConnectError as e:
            print("SMTP Connection Error: Could not connect to email server.")
            print("Error details:", e)
        except smtplib.SMTPException as e:
            print("General SMTP Error:", e)
        except Exception as e:
            print("General Error:", e)
