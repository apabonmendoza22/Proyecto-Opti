import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def load_pdfs_from_directory(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load_and_split())
    return documents

def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(texts, embeddings)
    return db

def create_rag_chain(db, llm):
    retriever = db.as_retriever(search_kwargs={"k": 8})
    
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
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

def initialize_pdf_processor(pdf_directory, llm):
    documents = load_pdfs_from_directory(pdf_directory)
    db = create_vector_store(documents)
    rag_chain = create_rag_chain(db, llm)
    return rag_chain

def process_query(rag_chain, general_search_function, query):
    rag_response = rag_chain({"query": query})
    
    if not rag_response['result'].strip() or rag_response['result'].strip().lower().startswith("no puedo encontrar"):
        # Si RAG no encuentra información o la respuesta está vacía, usar búsqueda general
        general_response = general_search_function(query)
        return {
            "result": general_response,
            "source": "búsqueda general"
        }
    else:
        # Limitar las fuentes a un máximo de 3 y eliminar duplicados
        sources = list(set([doc.page_content[:150] + "..." for doc in rag_response['source_documents']]))[:3]
        return {
            "result": rag_response['result'],
            "source_documents": sources,
            "source": "RAG"
        }