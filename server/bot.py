import logging
import requests
import os.path
import uuid
from typing import List, Dict
from urllib.parse import urlparse

import chromadb.config
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import WebBaseLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
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

# In-memory conversation history storage
conversation_history: Dict[str, List[Dict[str, str]]] = {}


# Chat request model
class ChatRequest(BaseModel):
    document_id: str
    query: str


# Website request model
class WebsiteRequest(BaseModel):
    url: str


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
            OllamaEmbeddings(model="nomic-embed-text"),
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


# Function to check wether the url is valid
def is_valid_url(url: str) -> bool:
    try:
        logger.info(f"Validating the url: {url}")
        result = urlparse(url)
        return all([result.scheme, result.netloc])
        logger.info(f"Valid url: {url}")
    except Exception as e:
        logger.info(f"Invalid url: {e}")
        return False


# Function to parse and process the webpage
def process_website(url: str) -> str:
    logger.info(f"Processing the url: {url}")
    document_id = str(uuid.uuid4())

    try:
        # Content extarction from website
        loader = WebBaseLoader(url)
        docs = loader.load()

        # Text Splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?"]
        )
        split_docs = text_splitter.split_documents(docs)

        # Store in chromadb
        logger.info("Creating chromadb collection")
        Chroma.from_documents(
            split_docs,
            OllamaEmbeddings(model="nomic-embed-text"),
            client_settings=chroma_settings,
            collection_name=document_id,
        )
        logger.info("Created chromadb collection")

        logger.info(f"Website sucessfully processed: {document_id}")
        return document_id

    except Exception as e:
        logger.error(f"Error porcessign the website: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing website: {e}")


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
        return {
            "status": "success",
            "Document ID": document_id
        }

    except Exception as e:
        logger.error(f"Pdf upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


@app.post("/website")
async def process_webiste(request: WebsiteRequest):
    logger.info(f"Recived url for parsing: {request.url}")

    # Validate the url
    if not is_valid_url(request.url):
        raise HTTPException(status_code=400, detail="Invalid url provided")

    try:
        # verify the webiste accessibility
        response = requests.head(request.url, timeout=10)
        response.raise_for_status()

        # Proces the website
        document_id = process_website(request.url)

        return {
            "status": "success",
            "Document ID": document_id
        }

    except requests.RequestException as r:
        logger.error(f"Could not access the webiste: {r}")
        raise HTTPException(status_code=400, detail=f"Could not access the website: {r}")

    except Exception as e:
        logger.error(f"Error processing the website: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing the website: {e}")


# Chat endpoint
@app.post("/chat")
async def chat_with_document(chat_request: ChatRequest):
    try:
        # Retrieve document collection
        vectorstore = Chroma(
            collection_name=chat_request.document_id,
            embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
            client_settings=chroma_settings
        )

        # Retrieve conversation history for this document
        history = conversation_history.get(chat_request.document_id, [])

        # Retrieve relevant documents
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.invoke(chat_request.query)

        # Prepare context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Create chat history context
        chat_history_str = "\n".join([
            f"{'Human' if msg['role'] == 'human' else 'Assistant'}: {msg['content']}"
            for msg in history[-5:]  # Limit to last 5 messages to avoid context overflow
        ])

        # Prepare prompt template
        prompt = ChatPromptTemplate.from_template("""
        System: You are a knowledgeable AI assistant specialized in analyzing and explaining content from both
        documents and websites.
        Your responses should be:
        - Accurate and based on the provided context
        - Well-structured and easy to understand
        - Include relevant quotes or references when appropriate
        - Clear about any uncertainties or limitations in the information

        Context Information:
        {context}

        Previous Conversation:
        {chat_history}

        Current Query: {query}

        Instructions:
        1. First analyze the context thoroughly
        2. Consider the conversation history for better continuity
        3. Provide a structured response with clear sections if needed
        4. Use bullet points for multiple items
        5. Include specific references to sources when possible
        6. If information is not in the context, clearly state that

        Response Format:
        - Start with a direct answer to the query
        - Follow with supporting details and examples
        - End with any necessary clarifications or caveats

        Assistant: Let me help you with that question...
        """)

        # Initialize Ollama chat model
        chat_model = ChatOllama(model="llama3.2")

        # Generate response
        response = chat_model.invoke(
            prompt.format(
                context=context,
                chat_history=chat_history_str,
                query=chat_request.query
            )
        )

        response_content = str(response.content)

        # Update conversation history
        if chat_request.document_id not in conversation_history:
            conversation_history[chat_request.document_id] = []

        conversation_history[chat_request.document_id].extend([
            {"role": "human", "content": chat_request.query},
            {"role": "assistant", "content": response_content}
        ])

        return {
            "response": response.content,
            "context_used": context
        }

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)


'''
-----------upload------------------
curl -X POST http://localhost:8080/upload \
-H "Content-type: multipart/form-data" \
-F "file=@./sample.pdf"
----------chat---------------------
curl -X POST http://localhost:8080/chat \
-H "Content-Type: application/json" \
-d '{
    "document_id": "e46b8630-3216-4d79-a1fc-4e255a7c98f9",
    "query": "What are the main points in this document?"
}'
-------------website-----------------
curl -X POST http://localhost:8080/website \
-H "Content-Type: application/json" \
-d '{
    "url": "https://en.wikipedia.org/wiki/Artificial_intelligence"
}'
'''
