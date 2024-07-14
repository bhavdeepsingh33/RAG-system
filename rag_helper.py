import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader, UnstructuredURLLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv
import pickle

def set_api_key():
    #load openAI api key
    load_dotenv()
    groq_api_key = os.environ["GROQ_API_KEY"]
    return groq_api_key

def initialize_llm(model_name, groq_api_key):
    # Initialize LLM with required parameters
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name, temperature=0.2)
    return llm

## LOAD DATA
def load_url_data(urls):
    # https://www.moneycontrol.com/news/business/markets/wall-street-rises-as-tesla-soars-on-ai-optimism-11351111.html
    # https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html
    loaders = UnstructuredURLLoader(
        urls=urls
    )
    data = loaders.load()
    return data

## SPLIT DATA TO CREATE CHUNKS
def split_data(data):
    text_spitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap =200)
    # As data is of type documents we can directly use split_documents over split_text in order to get the chunks.
    chunks = text_spitter.split_documents(data)
    return chunks

def embedding_to_vectordb(chunks):
    ## CREATE EMBEDDINGS FOR THESE CHUNKS AND SAVE THEM TO FAISS INDEX  
    # Create the embeddings of the chunks using openAIEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectors = FAISS.from_documents(chunks, embeddings)
    return vectors

def store_vectordb(v_index, file_path):
    # Storing vector index created in local
    with open(file_path, "wb") as f:
        pickle.dump(v_index, f)

def load_vectordb(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorIndex = pickle.load(f)
    return vectorIndex

def generate_answer(query, vectorIndex, llm):
    template = """
    Answer the question based on given context only. Please provide the most accurate response based on given question
    <context>
    {context}
    <context>
    Question:{input}
    """
    prompt = ChatPromptTemplate.from_template(template) 
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorIndex.as_retriever(search_kwargs={"k":4})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input":query})
    return response['answer']