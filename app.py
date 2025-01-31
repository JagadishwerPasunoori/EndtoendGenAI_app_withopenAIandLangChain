import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

# Langsmith tracking:
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ["LANGCHAIN_PROJECT"] = "Q&A ChatBOT with OpenAI"

# Define prompt template:
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant, please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

# Generate response:
def generate_response(question, api_key, model_name, temperature, max_tokens):
    # Initialize OpenAI model with proper parameters
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=api_key  # Use the API key passed by the user
    )

    # Properly initialize StrOutputParser
    output_parser = StrOutputParser()

    # Create the chain correctly
    chain = prompt | llm | output_parser

    # Invoke chain
    answer = chain.invoke({"question": question})
    
    return answer

# Streamlit app:
st.title("End-to-End GenAI Q&A ChatBot with OpenAI and Langchain")

# Sidebar for fields:
st.sidebar.title("Settings")

# API Key input field
api_key = st.sidebar.text_input("Enter OpenAI API key:", type="password")

# Dropdown to select various OpenAI models:
model_name = st.sidebar.selectbox("Select an OpenAI Model", ["gpt-4o", "gpt-4-turbo", "gpt-4"])

# Sliders for response parameters:
temperature = st.sidebar.slider("Temperature", value=0.7, min_value=0.0, max_value=1.0)
max_tokens = st.sidebar.slider("Max Tokens", value=150, min_value=50, max_value=300)

# Define the main interface:
st.write("Ask any question:")
user_input = st.text_input("You:")

if user_input:
    if not api_key:
        st.warning("Please enter your OpenAI API key.")
    else:
        response = generate_response(user_input, api_key, model_name, temperature, max_tokens)
        st.write("🤖 AI:", response)
else:
    st.write("Please ask a question.")

