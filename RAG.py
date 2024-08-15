from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from io import BytesIO
import tempfile
import os

def process_pdf(pdf_file, llm):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        # Write the contents of the uploaded file to the temporary file
        temp_file.write(pdf_file.getvalue())
        temp_file_path = temp_file.name

    try:
        # Use the temporary file path with PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        
        index = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        ).from_loaders([loader])
        
        return create_rag_chain(index, llm)
    finally:
        # Remove the temporary file
        os.unlink(temp_file_path)

def create_rag_chain(index, llm):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.vectorstore.as_retriever(),
        input_key='question'
    )

def rag_query(chain, prompt):
    response = chain.invoke({"question": prompt, "language": "es"})
    return response['result']