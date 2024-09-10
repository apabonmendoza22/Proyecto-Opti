from langchain.tools import Tool
import requests
import os
from dotenv import load_dotenv
load_dotenv()

API_BASE_URL = os.getenv("API_URL")

def consultar_ticket(ticket_id):
    response = requests.get(f"{API_BASE_URL}/consulta_ticket", params={"ticket_id": ticket_id})
    if response.status_code == 200:
        return response.json()["response"]["resultado"]
    else:
        return f"Error al consultar el ticket: {response.status_code}"

def crear_ticket(data):
    # Valores predeterminados
    default_data = {
        "externalSystem": "SELFSERVICE",
        "ownerGroup": "I-IBM-CO-VIRTUAL-ASSISTANT",
        "classificationId": "PRO205002012001",
        "impact": 3,
        "urgency": 3
    }
    
    # Combinar los datos proporcionados con los valores predeterminados
    full_data = {**default_data, **data}
    
    # Asegurarse de que todos los campos obligatorios estén presentes
    required_fields = ["owner", "reportedBy", "affectedPerson", "description", "longDescription"]
    for field in required_fields:
        if field not in full_data:
            return {"error": f"Falta el campo obligatorio '{field}'"}
    
    response = requests.post(f"{API_BASE_URL}/crear_ticket", json=full_data)
    if response.status_code == 200:
        ticket_data = response.json()
        return {"ticketId": ticket_data.get("ticketId")}
    else:
        return {"error": f"Error al crear el ticket: {response.status_code}"}

def consultar_incidente(ticket_id):
    response = requests.get(f"{API_BASE_URL}/consultar_incidente", params={"ticket_id": ticket_id})
    if response.status_code == 200:
        return response.json()["response"]["resultado"]
    else:
        return f"Error al consultar el incidente: {response.status_code}"

def crear_incidente(data):
    # Valores predeterminados
    default_data = {
        "externalSystem": "SELFSERVICE",
        "ownerGroup": "I-IBM-SMI-EUS",
        "classificationId": "PRO108009005",
        "impact": 3,
        "urgency": 3
    }
    
    # Combinar los datos proporcionados con los valores predeterminados
    full_data = {**default_data, **data}
    
    response = requests.post(f"{API_BASE_URL}/crear_incidente", json=full_data)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error al crear el incidente: {response.status_code}"
    
def obtener_datos_usuario(prompt):
    """
    Esta función simula la interacción con el usuario para obtener datos.
    En una implementación real, esto podría ser una llamada a una API de frontend.
    """
    return {
        "prompt": prompt,
        "requires_input": True,
        "required_fields": ["reportedBy", "affectedPerson", "description", "longDescription"]
    }

def get_api_tools():
    return [
        Tool(
            name="Consultar_Ticket",
            func=consultar_ticket,
            description="Útil para consultar el estado de un ticket. Requiere el ID del ticket."
        ),
        Tool(
            name="Crear_Ticket",
            func=crear_ticket,
            description="Útil para crear un nuevo ticket. Requiere los datos del ticket en formato JSON. Los campos obligatorios son: reportedBy, affectedPerson, description, longDescription."
        ),
        Tool(
            name="Consultar_Incidente",
            func=consultar_incidente,
            description="Útil para consultar el estado de un incidente. Requiere el ID del incidente."
        ),
        Tool(
            name="Crear_Incidente",
            func=crear_incidente,
            description="Útil para crear un nuevo incidente. Requiere los datos del incidente en formato JSON. Los campos obligatorios son: reportedBy, affectedPerson, description, longDescription."
        ),
        Tool(
            name="Obtener_Datos_Usuario",
            func=obtener_datos_usuario,
            description="Útil para obtener datos del usuario necesarios para crear un ticket o incidente. Devuelve un prompt para el usuario."
        )
    ]