import threading
import requests
from threadManager import QueueManager

class EmailNotifier:
    """
    Sends an email to the user when the robot reaction is complete.
    Completion is triggered by `completion_event` being set on shutdown in `controller.py`.
    """
    #TODO:
    API_KEY = "api key" #store safely
    DOMAIN = "domain"   
    
    #smtp_server = "smtp.gmail.com"
    #smtp_port = 465  # SSL port ? check.

    def __init__(self, recipient_email):
        self.recipient_email = recipient_email
        self.completion_event = QueueManager.get_completion_event()

        # Swait for event to be set 
        self.thread = threading.Thread(target=self.watch_event, daemon=True)
        self.thread.start()

    def watch_event(self):
        """Waits for the completion event and sends an email notification."""
        print("EmailNotifier: Waiting for completion event...")
        self.completion_event.wait()  # Blocks until `set()` is called
        print("EmailNotifier: Event detected! Sending email.")
        self.send_email()

    def send_email(self):
        """Sends an email notification."""
        if not self.recipient_email:
            print("No recipient email mprovided. Skipping notification.")
            return
        
        response = requests.post(
            f"api link",
            auth = ("api", self.API_KEY)
            data = {"from": "Chem Robot <noreply@domain.com>",
                    "to": self.recipient_email,
                    "subject": "Reaction Complete Notification",
                    "text": "Your reaction is coplete! You can now check your results."}
        )
        if response.status_code == 200:
            print(f"Email successfully sent to {self.recipient_email}!")
        else:
            print("Failed to send email:", response.text)
            
        #msg = MIMEText("Your reaction is complete! You can now check your results.")
        #msg["Subject"] = "Reaction Complete Notification"
        #msg["From"] = self.sender_email
        #msg["To"] = self.recipient_email

        #try:
        #    context = ssl.create_default_context()
        #    with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context) as server:
        #        server.login(self.sender_email, self.sender_password)
        #        server.sendmail(self.sender_email, self.recipient_email, msg.as_string())
        #    print("Email successfully sent to", self.recipient_email)
        #except Exception as e:
        #    print("Failed to send email:", e)