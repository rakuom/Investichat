# InvestiChat - A chat GPT investment Assistant with NSE-listed companies' integrated reports
# and financial statements since 2010. We will start with Safaricom. The tools to be used are Langchain and Streamlit
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
    VectorStoreInfo)  # These toolkits will allow us to manage the integration of vector sto

# Setting up the OpenAI API key and initializing the model. The OpenAI Key grants access to the OpenAL LLM Service
os.environ['OPENAI_API_KEY'] = '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'
llm = OpenAI(temperature=0.1, verbose=True)  # This creates an instance of the LLM service with a temperature of 0.1,
# to control how creative the responses from the model will be. To allow us receive additional information about the
# models processing, the parameter verbose=True is applied
embeddings = OpenAIEmbeddings()

#Loading and pre-processing the PDF documents
loader = PyPDFLoader('') #the PyPDYLoader is used to load and split the pages of the PDF documents.
pages = loader.load_and_split()
store = Chroma.from_documents(pages, embeddings, collection_name='annualreport') # The Chroma vector store is used
# to preprocess the document data allowing similarity searches.

#Creating VectorStoreInfo and VectorStoreToolkit
vectorstore_info = VectorStoreInfo(
    name = 'annual_report',
    description = 'a company annual report as a pdf'
    vectorstore = store
)# A vectorstoreinfo object is created that provides metadata about the pdf documents that are loaded. This includes
# the name, description and the previously processed Chroma vector store.
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info) # A toolkit is created that contains vector store info

#Creating the VectorStore Agent
agent_executor = create_vectorstore_agent(
    llm = llm,
    toolkit = toolkit,
    verbose = True
) #The VectorStoreAgent allows us to link the LLM with the previously created VectorStoreToolkit, giving the LLM access
# to the document data. The verbose=True parameter provides additional information during the agent's execution.

#Building the UI
#Streamlit UI setup
st.title('InvestiChat Assistant')
prompt = st.text_input('Ask me anything about BAT')



