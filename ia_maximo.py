import streamlit as st
from langchain_ibm import WatsonxLLM
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
import requests
import json
from dotenv import load_dotenv
import os 

load_dotenv()

api_key = os.getenv("API_KEY")
url= os.getenv("URL")
project_id= os.getenv("PROJECT_ID")
api_url= os.getenv("API_URL")

# Parámetros para WatsonxLLM
parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 100,
    "min_new_tokens": 1,
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 1,
}

# Inicializar WatsonxLLM
llm = WatsonxLLM(
    apikey=api_key,
    model_id="meta-llama/llama-3-1-70b-instruct",
    url=url,
    project_id=project_id,
    params=parameters
)

# URL base de tu API Flask
API_BASE_URL = api_url

# Funciones para interactuar con la API Flask

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

# Crear herramientas para el agente
tools = [
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

# Crear un agente que pueda usar las herramientas
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Interfaz de Streamlit
st.title('OPTI ChatBot con integración de API')

if 'messages' not in st.session_state:
    st.session_state.messages = []

prompt = st.chat_input("Hazle una pregunta a Opti")

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    with st.spinner('Procesando tu solicitud...'):
        response = agent.run(prompt)
    
    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role': 'assistant', 'content': response})

# Mostrar historial de mensajes
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])