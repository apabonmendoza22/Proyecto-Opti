from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

# Configuración del modelo
api_key = os.getenv("API_KEY")
project_id = os.getenv("PROJECT_ID")
model_id = "meta-llama/llama-3-1-70b-instruct"

params = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MAX_NEW_TOKENS: 512,  # Aumentado para permitir respuestas más largas
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.TEMPERATURE: 0.5,  # Reducido para respuestas más deterministas
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 0.95  # Ajustado para un mejor balance entre creatividad y precisión
}

# Inicializar el modelo
model = Model(
    model_id=model_id,
    params=params,
    credentials={
        "apikey": api_key,
        "url": "https://us-south.ml.cloud.ibm.com"
    },
    project_id=project_id
)

llm = WatsonxLLM(model=model)

def process_pdf(pdf_path):
    # Crear un cargador de PDF
    loader = PyPDFLoader(pdf_path)
    
    # Cargar y dividir el documento
    documents = loader.load_and_split()
    
    # Dividir el texto en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Reducido para chunks más pequeños
        chunk_overlap=200,  # Aumentado para mejor contexto entre chunks
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    
    # Crear embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Crear una base de datos de vectores
    db = Chroma.from_documents(texts, embeddings)
    
    # Crear un retriever
    retriever = db.as_retriever(search_kwargs={"k": 8})  # Aumentado para recuperar más contexto
    
    # Definir un prompt personalizado para la cadena RAG
    rag_prompt_template = """Utiliza la siguiente información para responder a la pregunta del usuario.
    Busca cuidadosamente en el texto proporcionado cualquier información específica solicitada, como nombres, fechas o cantidades.
    Si encuentras la información solicitada, proporciona una respuesta precisa citando la parte relevante del texto.
    Si la información no está presente, di claramente que no puedes encontrar esa información específica en el texto proporcionado.
    
    
    Contexto: {context}
    
    Pregunta: {question}
    
    Respuesta:"""
    
    PROMPT = PromptTemplate(
        template=rag_prompt_template, input_variables=["context", "question"]
    )
    
    # Crear una cadena de RAG
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

def rag_query(chain, prompt):
    response = chain({"query": prompt})
    
    # Limitar las fuentes a un máximo de 3 y eliminar duplicados
    sources = list(set([doc.page_content[:150] + "..." for doc in response['source_documents']]))[:3]
    
    return {
        "result": response['result'],
        "source_documents": sources
    }