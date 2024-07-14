import streamlit as st
import rag_helper as rh

st.title("RAG System")
url_1 = st.text_input("URL 1")
question = st.text_input("Ask a question")
answer_button = st.button("Generate Answer")

if answer_button:
    if url_1:
        # st.write("Why hello there")
        groq_api_key = rh.set_api_key()
        
        data = rh.load_url_data([url_1])
        chunks = rh.split_data(data)
        vector_index = rh.embedding_to_vectordb(chunks)

        file_path = "vector_index.pkl"
        rh.store_vectordb(vector_index, file_path)
        llm = rh.initialize_llm(model_name="llama3-8b-8192", groq_api_key=groq_api_key)
        file_path = "vector_index.pkl"
        v_index = rh.load_vectordb(file_path)
        answer = rh.generate_answer(question, v_index, llm)
        # st.header("Answer")
        st.write(answer)
    else:
        st.write("Enter a URL for context!!")
    