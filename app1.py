import os
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceEndpoint
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st 
from PyPDF2 import PdfReader
from huggingface_hub import login

def main():
    load_dotenv()
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    st.set_page_config(page_title="Ask your PDF")
    st.title("Ask your PDF chat")

    # Upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # Extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Create embeddings and knowledge base
        embeddings = HuggingFaceEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        vectordb_file_path = "faiss_index"
        knowledge_base.save_local(vectordb_file_path)

        user_question = st.text_input("Ask a question about your PDF:")

        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            input_data = {'input_documents': docs, 'question': user_question}
            llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain(input_data)

            st.write(response)

if __name__ == "__main__":
    main()
