
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
from typing import Dict, Any
from api_tools import crear_ticket, crear_incidente
import re
from typing import Dict, Any

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
load_dotenv()

api_tools = get_api_tools()

agent = initialize_agent(
    api_tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name

        try:
            rag_chain = process_pdf(temp_path)
            return jsonify({"message": "PDF processed successfully"}), 200
        finally:
            os.unlink(temp_path)
    return jsonify({"error": "Invalid file type"}), 400


def extract_ticket_info(prompt):
    interpretation_template = """
    Basándote en la siguiente entrada del usuario, genera una descripción corta (máximo 100 caracteres) 
    y una descripción larga (máximo 500 caracteres) para un ticket de soporte técnico.
    Por favor, proporciona SOLO el texto de las descripciones, sin ningún código o formato adicional.

    Entrada del usuario: {prompt}

    Descripción corta:
    Descripción larga:
    """
    
    interpretation_prompt = PromptTemplate(
        input_variables=["prompt"],
        template=interpretation_template,
    )
    
    interpretation_chain = LLMChain(llm=llm, prompt=interpretation_prompt)
    
    result = interpretation_chain.run(prompt)
    logger.debug(f"Raw model output: {result}")

    # Primero, intentamos extraer cualquier texto entre comillas
    description_match = re.search(r'"([^"]*)"', result)
    if description_match:
        description = description_match.group(1)
        long_description = "Descripción larga no proporcionada"
    else:
        # Si no hay texto entre comillas, buscamos las etiquetas específicas
        short_desc_match = re.search(r'Descripción corta:\s*(.*?)(?=Descripción larga:|$)', result, re.IGNORECASE | re.DOTALL)
        long_desc_match = re.search(r'Descripción larga:\s*(.*)', result, re.IGNORECASE | re.DOTALL)
        
        if short_desc_match and long_desc_match:
            description = short_desc_match.group(1).strip()
            long_description = long_desc_match.group(1).strip()
        else:
            # Si todo lo demás falla, tomamos todo el texto como descripción
            description = result.strip()
            long_description = "Descripción larga no proporcionada"

    # Limpiamos las descripciones de cualquier código o formato no deseado
    description = re.sub(r'["`\']|let\s+\w+\s*=\s*|console\.log\(.*?\);?', '', description)
    long_description = re.sub(r'["`\']|let\s+\w+\s*=\s*|console\.log\(.*?\);?', '', long_description)
    
    # Nos aseguramos de que las descripciones no estén vacías
    description = description if description.strip() else "Descripción no proporcionada"
    long_description = long_description if long_description.strip() else "Descripción larga no proporcionada"

    logger.info(f"Extracted description: {description}")
    logger.info(f"Extracted long description: {long_description}")

    return description, long_description

@app.route('/chat', methods=['POST'])
def chat() -> Dict[str, Any]:
    data = request.json
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        intent = intent_chain.run(prompt)
        logger.info(f"Detected intent: {intent}")

        if intent == "6" and rag_chain:
            logger.info("Using RAG for PDF-related query")
            response = rag_query(rag_chain, prompt)

        elif intent == "2":  # Creación de caso
            logger.info("Creating a new ticket")
            try:
                description, long_description = extract_ticket_info(prompt)
            except Exception as e:
                logger.error(f"Error extracting ticket info: {str(e)}")
                return jsonify({"error": "Failed to extract ticket information"}), 500

            ticket_data = {
                "owner": "CMEDINAM@IBM.COM",
                "impact": 3,
                "urgency": 3,
                "ownerGroup": "I-IBM-CO-VIRTUAL-ASSISTANT",
                "reportedBy": "LHOLGUIN",
                "description": description,
                "affectedPerson": "LHOLGUIN",
                "externalSystem": "SELFSERVICE",
                "longDescription": long_description,
                "classificationId": "PRO205002012001"
            }
            ticket_response = crear_ticket(ticket_data)
            logger.info(f"Ticket creation response: {ticket_response}")
            if isinstance(ticket_response, dict) and "ticketId" in ticket_response:
                response = {
            "result": f"Ticket creado con éxito. ID: {ticket_response['ticketId']}",
            "description": description,
            "long_description": long_description,
            "source_documents": []
        }
            else:
                error_message = ticket_response if isinstance(ticket_response, str) else str(ticket_response)
                response = {"result": f"Error al crear el ticket: {error_message}", "source_documents": []}

        elif intent == "4":  # Creación de incidente
            logger.info("Creating a new incidents")
            try:
                description, long_description = extract_ticket_info(prompt)
            except Exception as e:
                logger.error(f"Error extracting ticket info: {str(e)}")
                return jsonify({"error": "Failed to extract ticket information"}), 500

            ticket_data = {
                "impact": 3,
                "urgency": 3,
                "ownerGroup": "I-IBM-SMI-EUS",
                "reportedBy": "LHOLGUIN",
                "description": description,
                "affectedPerson": "LHOLGUIN",
                "externalSystem": "SELFSERVICE",
                "longDescription": long_description,
                "classificationId": "PRO108009005"
            }
            ticket_response = crear_incidente(ticket_data)
            logger.info(f"Ticket creation response: {ticket_response}")
            if isinstance(ticket_response, dict) and "ticketId" in ticket_response:
                response = {
            "result": f"Ticket creado con éxito. ID: {ticket_response['ticketId']}",
            "description": description,
            "long_description": long_description,
            "source_documents": []
        }
            else:
                error_message = ticket_response if isinstance(ticket_response, str) else str(ticket_response)
                response = {"result": f"Error al crear el ticket: {error_message}", "source_documents": []}


        elif intent in ["1", "3"]:
            logger.info("Using agent for ticket/incident related query")
            agent_response = agent.run(prompt)
            response = {"result": agent_response, "source_documents": []}
        else:
            logger.info("Using general search")
            general_response = general_search(llm, prompt)
            response = {"result": general_response, "source_documents": []}

        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": "An internal error occurred"}), 500

if __name__ == '__main__':
    app.run(port='5001', host='0.0.0.0', debug=True)


