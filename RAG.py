# RAG.py
import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_documents(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            logger.info(f"Cargando documento: {file_path}")
            loader = PyPDFLoader(file_path)
            doc = loader.load()
            documents.extend(doc)
            logger.info(f"Documento {filename} cargado. Número de páginas: {len(doc)}")
    
    logger.info(f"Total de documentos cargados: {len(documents)}")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    logger.info(f"Total de chunks creados: {len(texts)}")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(texts, embeddings)
    
    return vectorstore

def create_combined_chain(vectorstore, llm):
    template = """
    Utiliza la siguiente información de contexto para responder a la pregunta del usuario.
    Si la información exacta no está en el contexto, busca información relacionada que pueda ser útil.
    Si no encuentras información relevante en el contexto, indica claramente que no tienes esa información específica.

    Contexto: {context}
    
    Pregunta: {question}
    
    Respuesta detallada:"""
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

def is_relevant_to_documents(query, vectorstore):
    # Obtener los documentos más relevantes
    docs = vectorstore.similarity_search(query, k=1)
    
    # Si no hay documentos relevantes, consideramos que la pregunta no es relevante
    if not docs:
        return False
    
    # Calcular la similitud con el documento más relevante
    similarity = vectorstore.similarity_search_with_score(query, k=1)[0][1]
    
    # Definir un umbral de similitud más estricto
    threshold = 0.85
    
    # Palabras clave que indican que la pregunta probablemente no es relevante para los documentos
    irrelevant_keywords = ["diferencia", "comparar", "versus", "vs", "mejor", "peor", "ventajas", "desventajas"]
    
    # Verificar si alguna palabra clave irrelevante está en la consulta
    if any(keyword in query.lower() for keyword in irrelevant_keywords):
        return False
    
    return similarity > threshold

def query_documents(chain, query, vectorstore):
    logger.info(f"Procesando consulta: {query}")
    
    if is_relevant_to_documents(query, vectorstore):
        result = chain({"query": query})
        logger.info("Documentos recuperados:")
        for i, doc in enumerate(result['source_documents']):
            logger.info(f"Documento {i+1}:")
            logger.info(f"  - Fuente: {doc.metadata['source']}")
            logger.info(f"  - Página: {doc.metadata.get('page', 'N/A')}")
            logger.info(f"  - Contenido: {doc.page_content[:100]}...")
        return result['result'], True, result['source_documents']
    else:
        logger.info("La consulta no es relevante para los documentos cargados.")
        return None, False, []

# Este código se ejecutará cuando importes el módulo
logger.info("Iniciando carga de documentos...")
vectorstore = load_documents("documents")
logger.info("Documentos cargados exitosamente.")