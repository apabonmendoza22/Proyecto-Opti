from langchain.tools import Tool
import requests
import os
from dotenv import load_dotenv
from collections import OrderedDict
import requests
import json
load_dotenv()

API_BASE_URL = os.getenv("API_URL")

def obtener_classification_id(descripcion, tipo):
    url = "https://clasification.1jgnu1o1v8pl.us-south.codeengine.appdomain.cloud/clasificar"
    payload = {"texto": descripcion, "tipo": tipo}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Esto lanzará una excepción para códigos de estado no exitosos
        result = response.json()
        clasificacion = result.get('clasificacion')
        grupo = result.get('grupo')
        print(f"Clasificación obtenida: {clasificacion}, Grupo: {grupo}")  # Log para depuración
        return clasificacion, grupo
    except requests.RequestException as e:
        print(f"Error al obtener clasificación: {str(e)}")  # Log para depuración
        return None, None

def consultar_ticket(ticket_id):
    response = requests.get(f"{API_BASE_URL}/consulta_ticket", params={"ticket_id": ticket_id})
    if response.status_code == 200:
        return response.json()["response"]["resultado"]
    else:
        return f"Error al consultar el ticket: {response.status_code}"

def crear_ticket(data):
    # Estructura base del ticket con orden específico
    ticket_template = OrderedDict([
        ("owner", ""),
        ("impact", 3),
        ("urgency", 3),
        ("ownerGroup", "I-IBM-CO-VIRTUAL-ASSISTANT"),
        ("reportedBy", "LHOLGUIN"),
        ("description", ""),  # Se llenará con los datos proporcionados
        ("affectedPerson", "LHOLGUIN"),
        ("externalSystem", "SELFSERVICE"),
        ("longDescription", ""),  # Se llenará con los datos proporcionados
        ("classificationId", "PRO205002012001")
    ])

    # Actualizar con los datos proporcionados
    if 'description' in data:
        ticket_template['description'] = data['description']
    if 'longDescription' in data:
        ticket_template['longDescription'] = data['longDescription']

    # Obtener clasificación y grupo si no se proporcionaron
    if 'classificationId' not in data or 'ownerGroup' not in data:
        classification_id, grupo = obtener_classification_id(ticket_template['description'], "SR")
        if classification_id and grupo:
            ticket_template['classificationId'] = classification_id
            ticket_template['ownerGroup'] = grupo

    print(f"URL del endpoint: {API_BASE_URL}/crear_ticket")
    print(f"Datos del ticket a crear:\n{json.dumps(ticket_template, indent=2)}")

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    try:
        response = requests.post(f"{API_BASE_URL}/crear_ticket", data=json.dumps(ticket_template), headers=headers)
        print(f"Código de estado de la respuesta: {response.status_code}")
        print(f"Encabezados de la respuesta: {response.headers}")
        print(f"Contenido de la respuesta: {response.text}")

        response.raise_for_status()

        if response.status_code == 200:
            ticket_data = response.json()
            return {"ticketId": ticket_data.get("ticketId"), "classificationId": ticket_template["classificationId"], "grupo": ticket_template["ownerGroup"]}
        else:
            return {"error": f"Error al crear el ticket: {response.status_code}", "detalles": response.text}
    except requests.exceptions.RequestException as e:
        print(f"Error en la solicitud HTTP: {str(e)}")
        return {"error": "Error en la comunicación con el servidor", "detalles": str(e)}
    except Exception as e:
        print(f"Error inesperado: {str(e)}")
        return {"error": "Error inesperado", "detalles": str(e)}

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
        "impact": 3,
        "urgency": 3
    }
    
    # Combinar los datos proporcionados con los valores predeterminados
    full_data = {**default_data, **data}
    
    # Obtener el classificationId y grupo
    classification_id, grupo = obtener_classification_id(full_data["description"], "Incident")
    if classification_id:
        full_data["classificationId"] = classification_id
        full_data["ownerGroup"] = grupo
    else:
        full_data["classificationId"] = "PRO108009005"  # Valor por defecto
        full_data["ownerGroup"] = "I-IBM-SMI-EUS"  # Grupo por defecto
    
    print(f"Datos del incidente a crear: {full_data}")  # Log para depuración
    response = requests.post(f"{API_BASE_URL}/crear_incidente", json=full_data)
    print(f"Respuesta del servidor: {response.text}")  # Log para depuración
    
    if response.status_code == 200:
        ticket_data = response.json()
        return {"ticketId": ticket_data.get("ticketId"), "classificationId": classification_id, "grupo": grupo}
    else:
        return {"error": f"Error al crear el incidente: {response.status_code}", "detalles": response.text}
    
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