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
from RAG import load_documents, create_combined_chain, query_documents

load_dotenv()

api_key = os.getenv("API_KEY")
url = os.getenv("URL")
project_id = os.getenv("PROJECT_ID")

# Parámetros para WatsonxLLM
parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 200,
    "min_new_tokens": 1,
    "temperature": 0.2,
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

# Cargar documentos y crear la cadena combinada
documents_path = os.path.join(os.path.dirname(__file__), "documents")
vectorstore = load_documents(documents_path)
combined_chain = create_combined_chain(vectorstore, llm)

# Prompt para determinar la intención del usuario
intent_template = """
Determina la intención del usuario basándote en su pregunta. Las posibles intenciones son:
1. Consulta de ticket
2. Creación de ticket
3. Consulta de incidente
4. Creación de incidente
5. Búsqueda general o consulta de documentación interna

Pregunta del usuario: {question}

Responde solo con el número de la intención correspondiente.
"""

intent_prompt = PromptTemplate(
    input_variables=["question"],
    template=intent_template,
)

intent_chain = LLMChain(llm=llm, prompt=intent_prompt)

# Interfaz de Streamlit
st.title('OPTI ChatBot con integración de API y RAG')

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
            # Intentar usar RAG primero
            rag_response, is_relevant, source_documents = query_documents(combined_chain, prompt, vectorstore)
            
            if is_relevant:
                response = rag_response
                # Mostrar los documentos fuente en la barra lateral
                st.sidebar.info("Fuentes utilizadas:")
                for i, doc in enumerate(source_documents):
                    st.sidebar.text(f"Documento {i+1}:")
                    st.sidebar.text(f"  - Fuente: {doc.metadata['source']}")
                    st.sidebar.text(f"  - Página: {doc.metadata.get('page', 'N/A')}")
                    st.sidebar.text(f"  - Contenido: {doc.page_content[:100]}...")
            else:
                # Si no es relevante, usar búsqueda general
                response = general_search(llm, prompt)
                st.sidebar.info("Esta pregunta se respondió utilizando conocimientos generales, no se utilizaron documentos específicos.")
    
    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role': 'assistant', 'content': response})

# Mostrar historial de mensajes
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])