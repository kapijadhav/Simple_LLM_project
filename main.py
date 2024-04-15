import streamlit as st
from langchain_helper import create_vector_db, get_qa_chain

st.title("LLM QA ")
btn = st.button("create Knowledgebase") #under admin use
if btn:
    pass

question = st.text_input("Question: ")
if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer: ")
    st.write(response["result"])