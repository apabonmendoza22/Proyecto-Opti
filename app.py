import streamlit as st
from langchain_ibm import WatsonxLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Parameters for WatsonxLLM
parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 100,
    "min_new_tokens": 1,
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 1,
}

# Initialize WatsonxLLM
llm = WatsonxLLM(
    apikey="NuKCNUSUewnw01qcgOd8ya975Qu28vJnNRY3hIiv424p",
    model_id="ibm/granite-20b-multilingual",
    url="https://us-south.ml.cloud.ibm.com",
    project_id='86968240-f607-4103-859c-401633a699ad',
    params=parameters
)

# Load a PDF
@st.cache_resource
def load_pdf():
    pdf_name = "OTROSÍ TELETRABAJO_ADRIAN PABON.pdf"
    loaders = [PyPDFLoader(pdf_name)]

    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)).from_loaders(loaders)
    return index

index = load_pdf()

# Create a QA chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
    input_key='question'
)

# Organize the page title
st.title('OPTI ChatBot')

# Initialize session state for messages if not already initialized
if 'messages' not in st.session_state:
    st.session_state.messages = []

prompt = st.chat_input("Hazle una pregunta a Opti")

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append(
        {'role': 'user', 'content': prompt}
    )
    response = chain.invoke({"question": prompt, "language": "es"})
    st.session_state.messages.append(
        {'role': 'assistant', 'content': response['result']}
    )

# Display all messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])