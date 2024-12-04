import logging
import os
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize fastapi
app = FastAPI(title="Chatbot", description="RAG BOT", version="0.0.1")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_credentials=True,
    allow_methods=["*"],
)

def process_pdf(path: str) -> List[Document]:
   try:
       loader = PyPDFLoader(path)
       documents = loader.load()

       # Split the document in chunks for lorger documents
       text_splitter = RecursiveCharacterTextSplitter(
           chunksize=1000,
           overlap=200,
       )

       split_documents = text_splitter.split_documents(documents)

       return split_documents

   except Exception as e:
      raise HTTPException(status_code=400, detail=f"Unable to process the PDF: {e}")

# API endpoints
@app.post("/uplaod-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # File validation check
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="NO file is provided")
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Given file is not a PDF")

    try:
        tmp_file_path = f"tmp_{file.filename}"

        # sotre the reciver file in tmp path
        with open(tmp_file_path, "wb") as tmp_file:
            content = await file.read()
            tmp_file.write(content)

        # Process the pdf
        documents = process_pdf(tmp_file_path)

        # Clean the tmp path
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error while uploading the pdf: {e}")