# InvestiChat - A chat GPT investment Assistant with NSE-listed companies' integrated reports
# and financial statements since 2010. The tools to be used are Langchain and Streamlit
# Setting up the required libraries that will be used in this project
import os  # used to set up the OpenAI API key
from langchain.llms import OpenAI  # The main OpenAI language model is imported
from langchain.embeddings import OpenAIEmbeddings  # Importing embedding modules of LLM
import streamlit as st  # This is used to develop the user interface
from langchain.document_loaders import PyPDFLoader  # This will allow us to load PDF documents
from langchain.vectorstores import chroma  # This is the vector store for handling document data
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo)  # These toolkits will allow us to manage the integration of vector stores with the LLM

# Setting up the OpenAI API key and initializing the model. The OpenAI Key grants access to the OpenAL LLM Service
os.environ['OPENAI_API_KEY'] = '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'
llm = OpenAI(temperature=0.1, verbose=True)  # This creates an instance of the LLM service with a temperature of 0.1,
# to control how creative the responses from the model will be. To allow us receive additional information about the
# models processing, the parameter verbose=True is applied
embeddings = OpenAIEmbeddings()
