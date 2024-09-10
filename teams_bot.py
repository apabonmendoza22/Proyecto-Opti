from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings
from botbuilder.schema import Activity
from flask import Response
import json
import os
import logging
import asyncio

logger = logging.getLogger(__name__)

APP_ID = os.getenv("MICROSOFT_APP_ID")
APP_PASSWORD = os.getenv("MICROSOFT_APP_PASSWORD")

settings = BotFrameworkAdapterSettings(APP_ID, APP_PASSWORD)
adapter = BotFrameworkAdapter(settings)

def get_process_message_function():
    from main import process_message
    return process_message

async def handle_teams_message(request):
    if "application/json" in request.headers["Content-Type"]:
        body = request.get_json()
        activity = Activity().deserialize(body)
        auth_header = request.headers.get("Authorization", "")

        async def turn_call(turn_context):
            message = turn_context.activity.text
            logger.info(f"Received message: {message}")
            if message:
                process_message = get_process_message_function()
                response = process_message(message)
                logger.info(f"Raw response: {response}")
                
                # Extraer el texto de la respuesta
                if isinstance(response, dict):
                    response_text = response.get('result', str(response))
                else:
                    response_text = str(response)
                
                logger.info(f"Sending response: {response_text}")
                await turn_context.send_activity(response_text)
            else:
                logger.warning("Received empty message")

        try:
            await adapter.process_activity(activity, auth_header, turn_call)
            return Response(status=200)
        except Exception as e:
            logger.error(f"Error processing activity: {str(e)}", exc_info=True)
            return Response(status=500)
    else:
        return Response(status=415)