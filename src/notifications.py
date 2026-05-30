import os
from twilio.rest import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def send_whatsapp_alert(defect_type: str, timestamp: str):
    """
    Sends a WhatsApp alert via Twilio for a detected defect.
    """
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    from_number = os.getenv('TWILIO_FROM_NUMBER')
    to_number = os.getenv('TWILIO_TO_NUMBER')


    # If credentials are not properly set, we skip silently or print a warning
    if not all([account_sid, auth_token, from_number, to_number]) or "your_" in account_sid:
        print("[Alert] Twilio credentials not set up. Skipping WhatsApp notification.")
        return

    try:
        client = Client(account_sid, auth_token)
        
        message_body = (
            f"⚠️ *LoomVision AI Alert*\n\n"
            f"Defect Detected: *{defect_type}*\n"
            f"Time: {timestamp}\n\n"
            f"Please check the loom immediately."
        )

        message = client.messages.create(
            from_=from_number,
            body=message_body,
            to=to_number
        )
        print(f"[Alert] WhatsApp notification sent! Message SID: {message.sid}")
    except Exception as e:
        print(f"[Alert] Failed to send WhatsApp message: {e}")

