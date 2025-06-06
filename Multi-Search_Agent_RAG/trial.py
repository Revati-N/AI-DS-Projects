from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM # Ollama LLM is technically a third-party LLM

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

## Langchain Tracking
os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

## Streamlit Framework
st.title("Langchain Initial Chatbot using Ollama")
input_text = st.text_input("Search the topic you want to know about")

## Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's queries"),
    ("user", "Question: {question}"),
])

# Ollama LLM
llm = OllamaLLM(
    temperature=0,
    model="llama2",
    verbose=True
)
output_parser = StrOutputParser()
chain = prompt_template | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))