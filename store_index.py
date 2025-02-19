from src.helper import load_data, text_splitter, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

extracted_data = load_data("data/")

text_chunks = text_splitter(extracted_data)

embedding = download_hugging_face_embeddings()

index_name = "medical-chatbot"

docsearch = Pinecone.from_texts([t.page_content for t in text_chunks], embedding,index_name= index_name)