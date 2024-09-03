import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv('EMAIL_API_URL')

def send_email(email_data: dict) -> dict:
    """
    Envía un correo electrónico utilizando la API externa.
    
    :param email_data: Diccionario con la información del correo (proveedor, destinatario, asunto, mensaje)
    :return: Diccionario con el resultado de la operación
    """
    try:
        response = requests.post(f"{API_URL}/send-email", json=email_data)
        response.raise_for_status()
        return {"success": True, "message": "Correo enviado exitosamente"}
    except requests.RequestException as e:
        return {"success": False, "error": str(e)}
