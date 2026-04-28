# 🤖 PDF RAG Chatbot with Groq

A simple PDF-based RAG (Retrieval-Augmented Generation) chatbot built with Streamlit, ChromaDB, Sentence Transformers, and Groq API.

---

## ✨ Features

- 📄 Upload PDF files
- 🔍 Extract and process PDF text
- 🧠 Create embeddings using Sentence Transformers
- 🗄️ Store vectors in ChromaDB
- 💬 Ask questions from uploaded PDF
- ⚡ Groq-powered intelligent responses
- 📌 View source chunks used for answers

---

## 📁 Project Structure

RAG Implementation/
│── app.py
│── .env
│── .env.example
│── requirements.txt
│── uploads/

---

## 🚀 Installation

### 1. Clone or Download Project

```bash
git clone <your-repo-url>
cd RAG-Implementation
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows CMD:**
```bash
venv\Scripts\activate
```

**PowerShell:**
```bash
venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ Environment Setup

Create `.env` file using `.env.example`:

```env
GROQ_API_KEY=your_groq_api_key_here
MODEL_NAME=llama-3.1-8b-instant
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

---

## 🤖 Groq Model

**Recommended:**

```env
MODEL_NAME=llama-3.1-8b-instant
```

---

## 📖 Usage

1. Upload PDF
2. Wait for text extraction + embedding
3. Ask questions
4. Get answers from PDF context

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| UI Framework | Streamlit |
| Vector Database | ChromaDB |
| Embeddings | Sentence Transformers |
| LLM API | Groq API |
| Orchestration | LangChain |
| Language | Python |

---

## 📝 Notes

- Make sure your Groq API key is valid
- Large PDFs may take more processing time
- Best for text-based PDFs (not scanned images)

---
