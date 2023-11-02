import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import FAISS 
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

os.environ['OPENAI_API_KEY'] = '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'


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        read_pdf = PdfReader(pdf)
        for page in read_pdf.pages:
            text += page.extract_text()
    return text


def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstores(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversational_chains(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    return conversational_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace('{{MSG}}', message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace('{{MSG}}', message.content), unsafe_allow_html=True)




def main():
    st.set_page_config(page_title="MineBOT",page_icon=":computer:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("MineBOT :computer:")
    user_question = st.text_input("Ask your question here: ")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Mining Documents:")
        pdf_docs = st.file_uploader("Upload necesasary documents:" , accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # get pdf contents
                raw_data = get_pdf_text(pdf_docs) # here the text in pdfs converted into string
                # st.write(raw_data) this is for testing that's all don't uncomment!

                # get pdf chunks
                text_chunks = get_chunks(raw_data)
                # st.write((len(text_chunks))) to test how many chunks are there in given docs.

                #get vectors
                vectorstore = get_vectorstores(text_chunks)

                # conversational chains create
                st.session_state.conversation = get_conversational_chains(vectorstore)
                #session_state is to make sure the conversation is persistant and won't reload.

    #st.session_state.conversation #to make our conversation persistent through out.


if __name__ == '__main__':
    main()