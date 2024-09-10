
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
from slack_bot import handler as slack_handler
from pdf_processor import initialize_pdf_processor, process_query
from teams_bot import handle_teams_message  # Cambiado de 'messages as teams_messages'
import asyncio
import requests
import re
import json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
load_dotenv()

def process_message(user_input: str) -> str:
    try:
        chat_response = requests.post("https://back-opti.1jgnu1o1v8pl.us-south.codeengine.appdomain.cloud/chat", json={"prompt": user_input}).json()
        response_data = chat_response.get("response", {})
        
        # Si response_data es un diccionario, intentamos obtener el 'result'
        if isinstance(response_data, dict):
            result = response_data.get('result', str(response_data))
        else:
            result = str(response_data)
        
        logger.debug(f"Processed message. Result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        return "Lo siento, ocurrió un error al procesar tu mensaje."
    



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
7. Enviar correo electrónico

importante: si el usuario en su prompt no menciona la palabra ticket, la intención por defecto es 5 o 6. 
si el usuario menciona que quiere enviar un correo electrónico la intención es 7.


Pregunta del usuario: {question}

Responde solo con el número de la intención correspondiente.
"""

intent_prompt = PromptTemplate(
    input_variables=["question"],
    template=intent_template,
)

intent_chain = LLMChain(llm=llm, prompt=intent_prompt)

# Inicializar el procesador de PDF
pdf_directory = os.path.join(os.path.dirname(__file__), 'documents')
rag_chain = initialize_pdf_processor(pdf_directory, llm)

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


def handle_opti_identity(prompt):
    identity_keywords = ['quién eres', 'qué eres', 'eres un bot', 'eres una ia', 'eres un asistente', 'qué es opti']
    if any(keyword in prompt.lower() for keyword in identity_keywords):
        return {
            "result": "Soy OPTI, un asistente virtual de Optimize IT diseñado para ayudarte con consultas, creación de tickets, y proporcionar información sobre servicios de tu empresa. Estoy aquí para asistirte en lo que necesites.",
            "source_documents": []
        }
    return None

def extract_email_info(prompt):
    email_template = """
    Basándote en la siguiente entrada del usuario, extrae la información necesaria para enviar un correo electrónico.
    Debes proporcionar los siguientes campos en formato JSON:
    - asunto: El asunto del correo (máximo 100 caracteres)
    - mensaje: El contenido del mensaje (máximo 500 caracteres)
    - proveedor: El proveedor de correo (por defecto, usa "gmail")
    - destinatario: La dirección de correo electrónico del destinatario

    Entrada del usuario: {prompt}

    Proporciona SOLO la información en formato JSON, sin código adicional.
    """
    
    email_prompt = PromptTemplate(
        input_variables=["prompt"],
        template=email_template,
    )
    
    email_chain = LLMChain(llm=llm, prompt=email_prompt)
    
    result = email_chain.run(prompt)
    logger.debug(f"Raw email info output: {result}")

    try:
        # Intenta parsear directamente como JSON
        email_info = json.loads(result)
    except json.JSONDecodeError:
        # Si falla, intenta extraer el JSON del string
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            try:
                email_info = json.loads(json_match.group())
            except json.JSONDecodeError:
                logger.error("Failed to parse email info from extracted JSON")
                return None
        else:
            logger.error("No JSON-like structure found in the output")
            return None

    # Validar y establecer valores por defecto si es necesario
    email_info['asunto'] = email_info.get('asunto', 'Sin asunto')[:100]
    email_info['mensaje'] = email_info.get('mensaje', '')[:500]
    email_info['proveedor'] = email_info.get('proveedor', 'gmail')
    
    if 'destinatario' not in email_info or not email_info['destinatario']:
        logger.error("No recipient email address found")
        return None

    return email_info

def send_email(email_data):
    endpoint = "https://correo.1jgnu1o1v8pl.us-south.codeengine.appdomain.cloud/send-email"
    try:
        response = requests.post(endpoint, json=email_data)
        response.raise_for_status()
        return {"success": True, "message": "Correo enviado exitosamente"}
    except requests.RequestException as e:
        return {"success": False, "error": str(e)}
    

@app.route('/chat', methods=['POST'])
def chat() -> Dict[str, Any]:
    data = request.json
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:

        identity_response = handle_opti_identity(prompt)
        if identity_response:
            return jsonify({
                "response": {

                    "result": identity_response["result"],
                    "source_documents": []
                }
            })
        
        intent = intent_chain.run(prompt)
        logger.info(f"Detected intent: {intent}")


        if intent == "7":  # Envío de correo electrónico
            logger.info("Sending email")
            email_info = extract_email_info(prompt)
            if email_info:
                email_response = send_email(email_info)
                if email_response["success"]:
                    response = {"result": email_response["message"], "source_documents": []}
                else:
                    response = {"result": f"Error al enviar el correo: {email_response['error']}", "source_documents": []}
            else:
                response = {"result": "No se pudo extraer la información del correo correctamente", "source_documents": []}

        elif intent in ["5", "6"]:  # Búsqueda general o consulta relacionada con PDF
            logger.info("Using combined RAG and general search")
            response = process_query(rag_chain, lambda q: general_search(llm, q), prompt)

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
            logger.info("Unrecognized intent, falling back to combined search")
            response = process_query(rag_chain, lambda q: general_search(llm, q), prompt)
        return jsonify({"response": response})
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": "An internal error occurred"}), 500
    
    return jsonify({"response": response})
    


@app.route('/slack/events', methods=['POST'])
def slack_events():
    from slack_bot import handler
    
    if request.json and "challenge" in request.json:
        return jsonify({"challenge": request.json["challenge"]})
    
    return handler.handle(request)

@app.route("/api/messages", methods=["POST"])
def teams_webhook():
    if "application/json" in request.headers["Content-Type"]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(handle_teams_message(request))
        loop.close()
        return response
    else:
        return jsonify({"error": "Unsupported Media Type"}), 415

@app.route("/test", methods=["GET"])
def test():
    return "Bot is running!"


if __name__ == '__main__':
    app.run(port=5001, host='0.0.0.0', debug=True)


