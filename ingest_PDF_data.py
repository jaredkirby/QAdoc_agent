from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PDFPlumberLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_transformers import DoctranQATransformer
from langchain.schema import Document

import json
import pickle

from dotenv import load_dotenv

load_dotenv()

# Load PDF data
print("Loading data...")
loader = PDFPlumberLoader("KPM Offsite 2021 PDF .pdf")
raw_documents = loader.load()

# Doctran QA
# print("Running Doctran QA...")
# qa_transformer = DoctranQATransformer(openai_api_model="gpt-3.5-turbo")
# transformed_document = qa_transformer.transform_documents(documents)
# print(json.dumps(transformed_document[0].metadata, indent=2))

# Split text
print("Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
documents = text_splitter.split_documents(raw_documents)

# Create vectorstore
print("Creating vectorstore...")
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)
