# 📰 AI News Analyzer

### **Multi-Source Opinion Synthesis with AI Contradiction Checking**

AI News Analyzer is an **AI-powered research and analysis platform** designed to help users explore, compare, and verify information from **multiple news articles and PDF documents**.
It combines **Retrieval-Augmented Generation (RAG)**, **hybrid search**, **contradiction detection**, and **hallucination verification** to provide **trustworthy, source-grounded insights**.

This project was developed as a **Campus Major Project** to address the growing challenge of understanding and validating information across multiple online news sources.

---

## 🚀 Problem Statement

With the rapid increase of digital news and opinion-based reporting, users often struggle to determine:

* which information is reliable,
* what viewpoints are common across sources,
* where sources contradict each other,
* and whether AI-generated summaries are actually grounded in evidence.

Traditional summarization tools generally focus on **single-article summarization** and fail to provide **cross-source comparison, contradiction detection, and answer verification**.

**AI News Analyzer** solves this problem by acting as an intelligent multi-source research assistant.

---

## 🎯 Objectives

* To build an AI-powered platform for **multi-source news analysis**
* To enable users to **compare viewpoints across articles and PDFs**
* To generate **context-aware answers** using retrieved source evidence
* To detect **contradictions** among different sources
* To verify whether generated answers are **supported by source material**
* To assign **confidence scores** for better trust and transparency
* To export research sessions into **Word and PDF reports**

---

## ✨ Key Features

### 🔗 Multi-Source Input

* Analyze **multiple article URLs**
* Upload and process **PDF documents**
* Discover news articles by entering a **topic**

### 🧠 AI-Powered Research

* Ask questions about uploaded/processed sources
* Get **source-grounded answers**
* Supports multiple LLM providers:

  * **Groq**
  * **Gemini**
  * **OpenRouter**

### 🔍 Hybrid Retrieval

* **Semantic search** using embeddings
* **BM25 keyword-based retrieval**
* **Cross-encoder reranking** for improved relevance

### ⚡ Contradiction Detection

* Identifies **real factual contradictions** between different sources
* Highlights conflicting claims across articles

### 🧪 Hallucination Verification

* Checks whether AI-generated responses are **actually supported** by source chunks
* Improves trustworthiness of generated outputs

### 🎯 Confidence Scoring

* Displays answer confidence levels:

  * High
  * Medium
  * Low

### 📤 Export Support

* Export research sessions as:

  * **Word (.docx)**
  * **PDF (.pdf)**

---

## 🏗️ System Architecture

The project follows a **Retrieval-Augmented Generation (RAG)** architecture with multiple reliability-enhancing modules.

### Workflow:

1. User provides:

   * article URLs
   * topic for article discovery
   * or PDF documents
2. Content is extracted and cleaned
3. Text is split into manageable chunks
4. Chunks are embedded and indexed in a vector database
5. User asks a question
6. Relevant chunks are retrieved using:

   * semantic similarity
   * BM25 retrieval
   * reranking
7. LLM generates a contextual response
8. System performs:

   * contradiction detection
   * hallucination verification
   * confidence scoring
9. Results are shown in an interactive research interface

---

## 🛠️ Tech Stack

### Frontend

* **Streamlit**

### Backend

* **Python**

### AI / NLP

* **LangChain**
* **Sentence Transformers**
* **HuggingFace Embeddings**
* **Cross-Encoder Re-ranking**
* **BM25 Retrieval**

### Vector Database

* **ChromaDB**

### LLM Providers

* **Groq**
* **Google Gemini**
* **OpenRouter**

### Document Processing

* **pdfplumber**
* **python-docx**
* **reportlab**

---

## 📂 Project Structure

```bash
AI-News-Analyzer/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
├── .gitignore             # Ignored files/folders
├── README.md              # Project documentation
└── notebooks/             # Research / experimentation notebooks (optional)
```

---

## ⚙️ Installation & Local Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/AI-News-Analyzer.git
cd AI-News-Analyzer
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
```

### 3️⃣ Activate virtual environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### 4️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root and add your API keys:

```env
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

> **Note:** `NEWSAPI_KEY` is optional and not required for normal usage.

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The app will open in your browser automatically.

---

## ☁️ Streamlit Cloud Deployment

This project can be deployed easily on **Streamlit Community Cloud**.

### Deployment Steps:

1. Push the project to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Create a new app
4. Select:

   * Repository
   * Branch: `main`
   * Main file: `app.py`
5. Add the following **Secrets** in Streamlit Cloud:

```toml
GROQ_API_KEY="your_groq_api_key_here"
GEMINI_API_KEY="your_gemini_api_key_here"
OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

---

## 📊 Major Project Relevance

This project is highly relevant in the context of:

* **Artificial Intelligence**
* **Natural Language Processing**
* **Information Retrieval**
* **Fact Verification**
* **Misinformation Detection**
* **AI-assisted Research Systems**

It demonstrates practical implementation of advanced AI concepts such as:

* **Retrieval-Augmented Generation (RAG)**
* **Hybrid Search**
* **Contradiction Detection**
* **Hallucination Checking**
* **Confidence Estimation**

---

## 📚 Academic Contribution

This project contributes toward building **trustworthy AI-assisted news analysis systems** by integrating:

* multi-source evidence synthesis,
* contradiction identification,
* answer verification,
* and source-grounded research assistance.

It can be useful for:

* Students
* Researchers
* Journalists
* Analysts
* General readers

---

## 🔮 Future Enhancements

Possible future improvements include:

* Source credibility scoring
* Bias and sentiment analysis
* Timeline-based event tracking
* Claim extraction and comparison
* Named entity analysis
* Visual analytics dashboard
* Support for multilingual news analysis

---


## 📜 License

This project is intended for **academic and educational purposes**.

---
