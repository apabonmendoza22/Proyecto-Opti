import os
import logging
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

slack_app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)

@slack_app.event("message")
def handle_message(body, say):
    event = body["event"]
    if "text" in event:
        user_input = event["text"]
        logger.debug(f"Received message: {user_input}")
        from main import process_message
        response = process_message(user_input)
        say(response)

handler = SlackRequestHandler(slack_app)