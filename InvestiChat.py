# InvestiChat - A chat GPT investment Assistant with NSE-listed companies' integrated reports
# and financial statements since 2010. The tools to be used are Langchain and Streamlit
# Setting up the required libraries
import os  # os is used to set up the OpenAI API key
from langchain.llms import OpenAI  # The main OpenAI language model (LLM) service is imported from LangChain
from langchain.embeddings import OpenAIEmbeddings  # These are the LLM embeddings
import streamlit as st  # Streamlit is used to for creating the user interface
from langchain.document_loaders import PyPDFLoader  # PyPDFLoader will be used for loading PDF documents
from langchain.vectorstores import chroma  # To handle the document data, Chroma will be used as the vector store
from langchain.agents.agent_toolkits import create_vectorstore_agent
from langchain.agents.agent_toolkits import VectorStoreToolkit  # To manage the integration of vector stores with the LLM
from langchain.agents.agent_toolkits import VectorStoreInfo
# The OpenAI API Key setup so that we access the LLM service at OpenAI

