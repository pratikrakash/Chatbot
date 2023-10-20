import os
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import DataFrameLoader
import json
import pandas as pd


os.environ['OPENAI_API_KEY'] = "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"

# read the json file
df_question = pd.read_json('./qa.json')

# model_name = "BAAI/bge-large-en"
# model_kwargs = {'device': 'cpu'}
# encode_kwargs = {'normalize_embeddings': True}

# embeddings = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs
# )

loader = DataFrameLoader(df_question, page_content_column="question")
loaded_documents = loader.load()

# Generate embeddings for your documents
# document_embeddings = hf.encode(question_list)

persist_directory = "faq_db"

# db = Chroma(persist_directory="./faq_db")
# # Store embeddings in Chroma DB
# document_collection = db.create_collection("documents")
# document_collection.add(
#     ids=list(range(len(question_list))),  # Assuming ids are the indices of the documents
#     embeddings=document_embeddings.tolist()
# )
embeddings = OpenAIEmbeddings()

vectordb = Chroma.from_documents(
    documents=loaded_documents,
    embedding=embeddings,
    persist_directory=persist_directory,
    # collection_metadata={"hnsw:space": "cosine"}
)
vectordb.persist()
