import streamlit as st
import openai 
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_project']='Enhanced Q&A ChatBot with OPENAI'


prompt = ChatPromptTemplate.from_messages(
    {
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question:{question}")
    }
)

def generate_response(question, api_key, model, temperature, max_tokens):
    openai.api_key = api_key

    llm = ChatOpenAI(
        model=model,
        openai_api_key=api_key,  # Pass the API key explicitly
        temperature=temperature,
        max_tokens=max_tokens,
        )
    
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({'question': question})
    return answer


st.title("Enhanced Q&A ChatBot with OPENAI")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")

llm = st.sidebar.selectbox("select an Open AI Model", ["gpt-4o", "gpt-4-turbo", "gpt-4"])

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Toekns", min_value=50, max_value=300, value=150)

st.write("Go ahead and ask any question")

user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.write(response)

else:
    st.write("please provide the query")