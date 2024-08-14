# main.py
import streamlit as st
from langchain_ibm import WatsonxLLM
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
from api_tools import get_api_tools
from general_search import general_search

load_dotenv()

api_key = os.getenv("API_KEY")
url = os.getenv("URL")
project_id = os.getenv("PROJECT_ID")

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

# Obtener herramientas de API
api_tools = get_api_tools()

# Crear un agente que pueda usar las herramientas
agent = initialize_agent(
    api_tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Prompt para determinar la intención del usuario
intent_template = """
Determina la intención del usuario basándote en su pregunta. Las posibles intenciones son:
1. Consulta de ticket
2. Creación de ticket
3. Consulta de incidente
4. Creación de incidente
5. Búsqueda general

Pregunta del usuario: {question}

Responde solo con el número de la intención correspondiente.
"""

intent_prompt = PromptTemplate(
    input_variables=["question"],
    template=intent_template,
)

intent_chain = LLMChain(llm=llm, prompt=intent_prompt)

# Interfaz de Streamlit
st.title('OPTI ChatBot con integración de API')

if 'messages' not in st.session_state:
    st.session_state.messages = []

prompt = st.chat_input("Hazle una pregunta a Opti")

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    with st.spinner('Procesando tu solicitud...'):
        # Determinar la intención del usuario
        intent = intent_chain.run(prompt)
        
        if intent in ["1", "2", "3", "4"]:
            response = agent.run(prompt)
        else:
            response = general_search(llm, prompt)
    
    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role': 'assistant', 'content': response})

# Mostrar historial de mensajes
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])