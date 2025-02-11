import threading
import smtplib
import ssl
from email.mime.text import MIMEText
from threadManager import QueueManager

class EmailNotifier:
    """
    Sends an email to the user when the robot reaction is complete.
    Completion is triggered by `completion_event` being set on shutdown in `controller.py`.
    """
    
    smtp_server = "smtp.gmail.com"
    smtp_port = 465  # SSL port
    
    sender_email = "mocke@whitman.edu"
    sender_password = "rxca nflf mzyj gdpi"

    def __init__(self, recipient_email):
        self.recipient_email = recipient_email
        self.completion_event = QueueManager.get_completion_event()

        # Swait for event to be set 
        self.thread = threading.Thread(target=self.watch_event, daemon=True)
        self.thread.start()

    def watch_event(self):
        """Waits for the completion event and sends an email notification."""
        print("EmailNotifier: Waiting for completion event...")
        #while loop
        while True:
            self.completion_event.wait()  # Blocks until `set()` is called
            print("EmailNotifier: Event detected! Sending email.")
            
            try:
                print("calling send_email()")
                email_thread = threading.Thread(target=self.send_email)
                email_thread.start()
                email_thread.join()
            except Exception as a:
                print("error in watch_event()")
                
            self.completion_event.clear()
            
        #threading.Thread(target=self.send_email, daemon=True).start()

    def send_email(self):
        """Sends an email notification."""
        if not self.recipient_email:
            print("No recipient email mprovided. Skipping notification.")
            return
        
        print("send_email() was called for {self.recipient_email}")

        msg = MIMEText("Your reaction is complete! You can now check your results.")
        msg["Subject"] = "Reaction Complete Notification"
        msg["From"] = self.sender_email
        msg["To"] = self.recipient_email

        try:
            print(f"🚀 Attempting to send email to {self.recipient_email}...")
            # 🔹 Debugging prints to locate where it's getting stuck
            print(f"🔍 Connecting to SMTP server {self.smtp_server} on port {self.smtp_port}...")
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context) as server:
                print("🔍 Connected to SMTP server!")  # ✅ If this prints, connection worked
                print("🔍 Logging in to SMTP server...")
                server.login(self.sender_email, self.sender_password)
                print("✅ Login successful!")  # ✅ If this prints, login was successful
                print("📨 Sending email...")
                server.sendmail(self.sender_email, self.recipient_email, msg.as_string())
                print(f"📧 Email successfully sent to {self.recipient_email}!")  # ✅ Success

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
        #response = requests.post(
        #    f"api link",
        #    auth = ("api", self.API_KEY)
        #    data = {"from": "Chem Robot <noreply@domain.com>",
        #            "to": self.recipient_email,
        #            "subject": "Reaction Complete Notification",
        #            "text": "Your reaction is coplete! You can now check your results."})
        
        #if response.status_code == 200:
        #    print(f"Email successfully sent to {self.recipient_email}!")
        #else:
        #    print("Failed to send email:", response.text)
            