from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import os
from api_tools import get_api_tools
from RAG import process_pdf, rag_query, llm
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import tempfile
import logging
from general_search import general_search



app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
load_dotenv()

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
6. Consulta relacionada con el PDF cargado

Pregunta del usuario: {question}

Responde solo con el número de la intención correspondiente.
"""

intent_prompt = PromptTemplate(
    input_variables=["question"],
    template=intent_template,
)

intent_chain = LLMChain(llm=llm, prompt=intent_prompt)

# Variable global para almacenar el rag_chain
rag_chain = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    global rag_chain
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith('.pdf'):
        # Crear un archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name

        try:
            rag_chain = process_pdf(temp_path)
            return jsonify({"message": "PDF processed successfully"}), 200
        finally:
            # Asegurarse de que el archivo temporal se elimine
            os.unlink(temp_path)
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Determinar la intención
    intent = intent_chain.run(prompt)
    
    # Decidir qué tipo de respuesta dar
    if intent == "6" and rag_chain:
        # Usar RAG si la intención es relacionada con el PDF y hay un RAG chain disponible
        response = rag_query(rag_chain, prompt)
    elif intent in ["1", "2", "3", "4"]:
        # Usar el agente para consultas relacionadas con tickets o incidentes
        agent_response = agent.run(prompt)
        response = {"result": agent_response, "source_documents": []}
    else:
        # Usar búsqueda general para otras consultas
        general_response = general_search(llm, prompt)
        response = {"result": general_response, "source_documents": []}

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(port='5001', host='0.0.0.0', debug=True)