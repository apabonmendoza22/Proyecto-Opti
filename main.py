from flask import Flask, request, jsonify, render_template
from langchain_ibm import WatsonxLLM
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
from api_tools import get_api_tools
from general_search import general_search
from RAG import process_pdf, rag_query

app = Flask(__name__)

load_dotenv()

api_key = os.getenv("API_KEY")
url = os.getenv("URL")
project_id = os.getenv("PROJECT_ID")

# Parámetros para WatsonxLLM
parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 200,
    "min_new_tokens": 1,
    "temperature": 0.1,
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
        rag_chain = process_pdf(file, llm)
        return jsonify({"message": "PDF processed successfully"}), 200
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data.get('prompt')
    use_rag = data.get('use_rag', False)

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    if use_rag and rag_chain:
        response = rag_query(rag_chain, prompt)
    else:
        intent = intent_chain.run(prompt)
        if intent in ["1", "2", "3", "4"]:
            response = agent.run(prompt)
        else:
            response = general_search(llm, prompt)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(port='5000', host='0.0.0.0', debug=True)