from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import itertools



def main():
    load_dotenv()
    st.set_page_config(page_title= "Commercial Laundry")
    st.image("documents\ge-appliances-a-haier-logo.png", width = 200)
    st.header("Commercial Laundry Self Service")
    st.write("")

    file_paths = {
        'FL Dryer': 'documents/FL Dryer.pdf',
        'FL Washer': 'documents/FL Washer.pdf',
        'TLS1 Dryer': 'documents/TLS1 Dryer.pdf',
        'TLS1 Washer': 'documents/TLS1 Washer.pdf',
        'TLS2 Dryer': 'documents/TLS2 Dryer.pdf',
        'TLS2 Washer': 'documents/TLS2 Washer.pdf'
    }

    # Selecting the Device
    selected_device = st.selectbox('Select a device from dropdown', list(file_paths.keys()), key="first")
    st.write('Device selected:', selected_device)

    # setting device document path
    selected_file_path = file_paths[selected_device]

    # reading the text
    pdf_reader = PdfReader(selected_file_path)

    text = ""
    for page in pdf_reader.pages:
        text+=page.extract_text()

       # split into chunks

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function= len
    )
    chunks = text_splitter.split_text(text)

    # create embeddings 
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    user_question = st.text_input("Ask your question: ", key="second")

    if user_question:
        docs = knowledge_base.similarity_search(user_question)
            
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents = docs, question = user_question)
            print(cb)
        st. write(response)   


    
if __name__ == "__main__":
    main()
