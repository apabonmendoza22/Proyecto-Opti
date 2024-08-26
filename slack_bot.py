import os
import logging
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from dotenv import load_dotenv

load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize the Slack app
slack_app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)

# Event handler for messages
@slack_app.event("message")
def handle_message(body, say):
    event = body["event"]
    if "text" in event:
        user_input = event["text"]
        logger.debug(f"Received message: {user_input}")
        # Instead of calling chat directly, we'll pass this to a function in main.py
        from main import process_slack_message
        response = process_slack_message(user_input)
        say(response)

# Create a SlackRequestHandler for use with Flask
handler = SlackRequestHandler(slack_app)