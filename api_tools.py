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
    response = requests.post(f"{API_BASE_URL}/crear_ticket", json=data)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error al crear el ticket: {response.status_code}"

def consultar_incidente(ticket_id):
    response = requests.get(f"{API_BASE_URL}/consultar_incidente", params={"ticket_id": ticket_id})
    if response.status_code == 200:
        return response.json()["response"]["resultado"]
    else:
        return f"Error al consultar el incidente: {response.status_code}"

def crear_incidente(data):
    response = requests.post(f"{API_BASE_URL}/crear_incidente", json=data)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error al crear el incidente: {response.status_code}"

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
            description="Útil para crear un nuevo ticket. Requiere los datos del ticket en formato JSON."
        ),
        Tool(
            name="Consultar_Incidente",
            func=consultar_incidente,
            description="Útil para consultar el estado de un incidente. Requiere el ID del incidente."
        ),
        Tool(
            name="Crear_Incidente",
            func=crear_incidente,
            description="Útil para crear un nuevo incidente. Requiere los datos del incidente en formato JSON."
        )
    ]