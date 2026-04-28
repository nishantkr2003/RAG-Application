import streamlit as st
import os
import requests
import time
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ---------------------------
# LOAD CONFIGURATION
# ---------------------------
# ---------------------------
# LOAD CONFIGURATION
# ---------------------------
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

# Local .env support
load_dotenv(dotenv_path=ENV_PATH)

# Render + Local Environment Variables ONLY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "/tmp/chroma_db")

# Convert relative path locally
if not CHROMA_DB_DIR.startswith("/tmp") and not os.path.isabs(CHROMA_DB_DIR):
    CHROMA_DB_DIR = os.path.join(BASE_DIR, CHROMA_DB_DIR)

# Validation
if not GROQ_API_KEY:
    st.error("GROQ API Key not found. Set it in Render Environment Variables.")
    st.stop()


# STREAMLIT CONFIG
st.set_page_config(
    page_title="Advanced PDF RA Chatbot",
    layout="wide"
)

st.title("📄 Advanced PDF RAG Chatbot with Groq")
st.write("Upload a PDF, build a knowledge base, and ask accurate questions from your document.")

if not GROQ_API_KEY:
    st.error(f"GROQ API Key not found. Expected .env at: {ENV_PATH}")
    st.stop()


# DIRECTORIES
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)


# PDF TEXT EXTRACTION
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page_text + "\n"

    return text



# CHUNK TEXT

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_text(text)

    return [chunk.strip() for chunk in chunks if chunk.strip()]



# EMBEDDINGS MODEL

@st.cache_resource    #load model once only
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )



# CLEAR CHROMA DATABASE

def clear_chroma():
    try:
        # Release session vectorstore
        if "vectorstore" in st.session_state:
            del st.session_state["vectorstore"]

        time.sleep(1)

        # Delete files manually
        if os.path.exists(CHROMA_DB_DIR):
            for root, dirs, files in os.walk(CHROMA_DB_DIR, topdown=False):
                for file in files:
                    try:
                        os.remove(os.path.join(root, file))
                    except:
                        pass

                for dir_name in dirs:
                    try:
                        os.rmdir(os.path.join(root, dir_name))
                    except:
                        pass

        os.makedirs(CHROMA_DB_DIR, exist_ok=True)

        return True

    except Exception as e:
        st.sidebar.error(f"Clear failed: {e}")
        return False



# CREATE VECTOR STORE

def create_vector_store(chunks):
    embeddings = get_embeddings()

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name="pdf_collection"
    )

    st.session_state["vectorstore"] = vectorstore

    return vectorstore

# LOAD VECTOR STORE

def load_vector_store():
    if "vectorstore" in st.session_state:
        return st.session_state["vectorstore"]

    embeddings = get_embeddings()

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name="pdf_collection"
    )

    st.session_state["vectorstore"] = vectorstore

    return vectorstore



# GROQ API RESPONSE

def get_llm_response(context, question):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are an intelligent PDF document assistant.

Instructions:
1. Use ONLY the provided context.
2. Search thoroughly across all context.
3. Combine details from multiple sections when needed.
4. If partial answer exists, provide best possible answer.
5. Only say 'Answer not found in provided PDF.' if absolutely no relevant information exists.

Context:
{context}

Question:
{question}
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise PDF analysis assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 1200
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]

        return f"Error: {response.status_code} - {response.text}"

    except Exception as e:
        return f"Request failed: {str(e)}"



# SIDEBAR CONTROLS

st.sidebar.header("⚙️ Controls")

if st.sidebar.button("Clear Database"):
    if clear_chroma():
        st.sidebar.success("Database Cleared Successfully ✅")



# PDF UPLOAD SECTION

uploaded_file = st.file_uploader(
    "Upload PDF File",
    type="pdf"
)

if uploaded_file:
    pdf_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded Successfully: {uploaded_file.name}")

    # Extract Text
    with st.spinner("Extracting text from PDF..."):
        raw_text = extract_text_from_pdf(pdf_path)

    if raw_text.strip():

        st.subheader("PDF Preview")
        st.text_area(
            "Extracted Text Sample",
            raw_text[:5000],
            height=300
        )

        # Chunk
        with st.spinner("Chunking PDF into searchable sections..."):
            chunks = chunk_text(raw_text)

        st.write(f"Total Chunks Created: {len(chunks)}")

        # Sample Chunks
        with st.expander("View Sample Chunks"):
            for i, chunk in enumerate(chunks[:3]):
                st.write(f"### Chunk {i+1}")
                st.write(chunk[:1500])
                st.write("---")

        # Vector DB
        with st.spinner("🧠 Creating embeddings and vector database..."):
            create_vector_store(chunks)

        st.success("Knowledge Base Created Successfully")

    else:
        st.error("No readable text found in PDF.")



# QUESTION ANSWERING SECTION

st.subheader("Ask Questions From Your PDF")

question = st.text_input("Enter your question:")

if question:

    if not os.path.exists(CHROMA_DB_DIR):
        st.error("Please upload and process a PDF first.")

    else:
        with st.spinner("Retrieving the most relevant information..."):

            vectorstore = load_vector_store()

            # Better Retriever
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 8,
                    "fetch_k": 20
                }
            )

            docs = retriever.get_relevant_documents(question)

            # Fallback
            if not docs:
                docs = vectorstore.similarity_search(question, k=8)

            # Context
            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            # LLM Response
            response = get_llm_response(context, question)

        # OUTPUT
        st.subheader("Answer")
        st.write(response)

        st.write(f"Retrieved Chunks Count: {len(docs)}")

        # Source Chunks
        with st.expander("Source Chunks Used"):
            for i, doc in enumerate(docs):
                st.write(f"### Source Chunk {i+1}")
                st.write(doc.page_content)
                st.write("---")