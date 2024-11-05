import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_groq import ChatGroq
import os

import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')


# Test to see if the environment variable is loaded
print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))



# Initialize GROQ chat model
def init_groq_model():
    groq_api_key = os.getenv('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')
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    return ChatGroq(
        groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile", temperature=0.2
    )

llm_groq = init_groq_model()

# Extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return text

# Split text into manageable chunks for embeddings
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=3000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)

# Create a FAISS vectorstore from text chunks
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# Set up a conversational retrieval chain with memory
def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm_groq, retriever=vectorstore.as_retriever(), memory=memory
    )

# Handle user input and display chat history
def handle_userinput(user_question):
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            template = user_template if i % 2 == 0 else bot_template
            st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.warning("Please upload and process your resume first.")

def main():
    st.set_page_config(page_title="Chat with your Job Assistant", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Page header
    st.header("Chat with your Job Assistant :books:")
    user_question = st.text_input("Ask a question about your resume or job:")
    if user_question:
        handle_userinput(user_question)

    # Sidebar for document upload and processing
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your resume PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                    except Exception as e:
                        st.error(f"Error processing documents: {e}")
            else:
                st.warning("Please upload resume PDFs before processing.")

if __name__ == '__main__':
    main()
