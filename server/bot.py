import logging
import os.path
import uuid
from typing import List

import chromadb.config
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Basic ChromaDB settings
chroma_settings = chromadb.config.Settings(
    is_persistent=True,
    persist_directory="chroma-db",
    anonymized_telemetry=False,
)

# Initialize FastAPI app
app = FastAPI(title="PDF Chatbot", description="Context-aware PDF Chat", version="1.0.0")


# Chat request model
class ChatRequest(BaseModel):
    document_id: str
    query: str

# Extract text from given document and return list of documents
def extract_text(file_path: str) -> List[Document]:
    try:
        logger.info(f"Loading pdf: {file_path}")
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        logger.info(f"Loaded pdf of {len(pages)} pages")

        logger.info(f"Splitting pdf {file_path}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        docs = text_splitter.split_documents(pages)

        return docs

    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF extraction error: {e}")

# Process the input file
def process_pdf(file_path: str) -> str:
    logger.info("Creating document id")
    document_id = str(uuid.uuid4())

    try:
        docs = extract_text(file_path)

        logger.info("Creating chromadb collection")
        Chroma.from_documents(
            docs,
            OllamaEmbeddings(model="llama3.2"),
            client_settings=chroma_settings,
            collection_name=document_id,
        )

        logger.info(f"Pdf loaded successfully. Document ID: {document_id}")
        return document_id

    except Exception as e:
        logger.error(f"Error while processing document: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error while processing document: {e}"
        )

# API to process the document
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    logger.info("Upload pdf started")

    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Please provide and input file.")

    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Only pdf is supported at this moment."
        )

    tmp_file_path = f"./tmp_{file.filename}"

    try:
        with open(tmp_file_path, "+wb") as buffer:
            buffer.write(await file.read())

        document_id = process_pdf(tmp_file_path)

        logger.info("Pdf uploaded successfully")
        return {"document_id": document_id}

    except Exception as e:
        logger.error(f"Pdf upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)