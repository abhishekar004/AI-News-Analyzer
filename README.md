# 📰 AI News Analyzer

### **AI-Powered Multi-Source News Research & Contradiction Detection System**

AI News Analyzer is an **AI-powered research and verification platform** designed to help users **analyze, compare, and validate information** from **multiple news articles and PDF documents**.

The system combines **Retrieval-Augmented Generation (RAG)**, **hybrid retrieval**, **contradiction detection**, **hallucination verification**, and **confidence scoring** to provide **trustworthy, source-grounded insights**.

> 🎓 Developed as a **Campus Major Project** to address the growing challenge of identifying reliable information across multiple online news sources.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Key Features](#-key-features)
- [How It Works](#-how-it-works)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Environment Variables](#-environment-variables)
- [Run the Application](#-run-the-application)
- [Deployment](#-deployment)
- [Use Cases](#-use-cases)
- [Academic Relevance](#-academic-relevance)
- [Future Enhancements](#-future-enhancements)
- [Author](#-author)
- [License](#-license)

---

## 🔍 Overview

In today’s digital world, users consume information from **multiple news platforms**, often with **different perspectives, claims, and biases**. This makes it difficult to determine:

- which source is reliable,
- which claims are supported by evidence,
- where articles agree or contradict,
- and whether AI-generated summaries are actually trustworthy.

**AI News Analyzer** solves this by acting as an intelligent **multi-source research assistant** that helps users compare, verify, and understand news content more effectively.

---

## 🚀 Problem Statement

Modern news consumption is fragmented across many websites, blogs, and reports. Traditional summarization tools usually focus on **single-document summarization** and often fail to support:

- **cross-source comparison**
- **contradiction identification**
- **source-grounded question answering**
- **answer verification**
- **trust estimation**

As a result, users may receive incomplete, biased, or unsupported summaries.

### ✅ Proposed Solution

AI News Analyzer provides a **multi-source AI analysis system** that can:

- process multiple article URLs and PDF files,
- retrieve the most relevant source evidence,
- answer user questions using grounded context,
- identify contradictions between sources,
- verify whether responses are hallucinated,
- and assign confidence scores for transparency.

---

## 🎯 Objectives

The primary objectives of this project are:

- To build an **AI-powered multi-source news analysis platform**
- To enable users to **compare viewpoints across multiple sources**
- To provide **source-grounded question answering**
- To detect **contradictions** between news reports
- To verify whether generated answers are **factually supported**
- To improve trust using **confidence estimation**
- To support **research export** in document formats

---

## ✨ Key Features

## 🔗 Multi-Source Input
- Analyze **multiple article URLs**
- Upload and process **PDF documents**
- Discover articles by entering a **topic**

## 🧠 AI-Powered Research Assistant
- Ask questions about uploaded or discovered content
- Get **context-aware answers** backed by retrieved evidence
- Supports multiple LLM providers:
  - **Groq**
  - **Google Gemini**
  - **OpenRouter**

## 🔍 Hybrid Retrieval System
Combines multiple retrieval strategies for better answer quality:
- **Semantic Search** using embeddings
- **BM25 Keyword Search**
- **Cross-Encoder Re-ranking**

## ⚡ Contradiction Detection
- Identifies **conflicting factual claims**
- Highlights contradictions between sources
- Helps users understand **where reports disagree**

## 🧪 Hallucination Verification
- Checks whether generated answers are **supported by retrieved evidence**
- Reduces unsupported or misleading AI outputs

## 🎯 Confidence Scoring
Provides confidence labels such as:
- **High**
- **Medium**
- **Low**

This improves user trust and interpretability.

## 📤 Export Support
Export analysis sessions as:
- **Word (.docx)**
- **PDF (.pdf)**

---

## ⚙️ How It Works

The system follows a **Retrieval-Augmented Generation (RAG)** workflow:

1. User provides:
   - article URLs
   - topic-based news search
   - or PDF documents
2. The system extracts and cleans the text
3. Content is split into manageable chunks
4. Chunks are embedded and indexed in a vector database
5. User asks a question
6. Relevant chunks are retrieved using:
   - semantic retrieval
   - BM25 keyword matching
   - reranking
7. The LLM generates a contextual answer
8. The system performs:
   - contradiction detection
   - hallucination verification
   - confidence scoring
9. Results are shown in an interactive interface

---

## 🏗️ System Architecture

The project follows a modular AI pipeline:

```text
User Input (URLs / Topic / PDFs)
            │
            ▼
   Content Extraction & Cleaning
            │
            ▼
      Text Chunking Pipeline
            │
            ▼
 Embedding + Hybrid Indexing (ChromaDB + BM25)
            │
            ▼
      Query Understanding Layer
            │
            ▼
  Retrieval + Re-ranking Pipeline
            │
            ▼
      LLM Response Generation
            │
            ▼
 ┌───────────────────────────────────────┐
 │ Contradiction Detection               │
 │ Hallucination Verification            │
 │ Confidence Scoring                    │
 └───────────────────────────────────────┘
            │
            ▼
   Interactive Research Output + Export
```

---

## 🛠️ Tech Stack

### Frontend
- **Streamlit**

### Backend
- **Python**

### AI / NLP
- **LangChain**
- **Sentence Transformers**
- **HuggingFace Embeddings**
- **Cross-Encoder Re-ranking**
- **BM25 Retrieval**

### Vector Database
- **ChromaDB**

### LLM Providers
- **Groq**
- **Google Gemini**
- **OpenRouter**

### Document Processing
- **pdfplumber**
- **python-docx**
- **reportlab**

### Optional / Utility Libraries
- **BeautifulSoup**
- **Requests**
- **FAISS / Chroma-compatible retrieval utilities**
- **dotenv**

---

## 📂 Project Structure

```bash
AI-News-Analyzer/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
├── .gitignore             # Ignored files/folders
├── README.md              # Project documentation
├── assets/                # Images / diagrams / screenshots (optional)
└── notebooks/             # Research / experimentation notebooks (optional)
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/AI-News-Analyzer.git
cd AI-News-Analyzer
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
```

### 3️⃣ Activate the Virtual Environment

#### Windows
```bash
venv\Scripts\activate
```

#### Linux / Mac
```bash
source venv/bin/activate
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root and add the following API keys:

```env
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
NEWSAPI_KEY=your_optional_newsapi_key_here
```

> **Note:** `NEWSAPI_KEY` is optional depending on your implementation.

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

After running the command, the app will open in your browser.

---

## ☁️ Deployment

This project can be deployed on **Streamlit Community Cloud**.

### Deployment Steps

1. Push your project to GitHub
2. Open **Streamlit Community Cloud**
3. Create a new app
4. Select:
   - Repository
   - Branch: `main`
   - Main file: `app.py`
5. Add the following **Secrets**:

```toml
GROQ_API_KEY="your_groq_api_key_here"
GEMINI_API_KEY="your_gemini_api_key_here"
OPENROUTER_API_KEY="your_openrouter_api_key_here"
NEWSAPI_KEY="your_optional_newsapi_key_here"
```

---

## 💡 Use Cases

AI News Analyzer can be useful for:

- **Students** researching current affairs
- **Researchers** comparing source claims
- **Journalists** validating conflicting reports
- **Analysts** tracking issue narratives
- **General readers** trying to understand complex news events

---

## 📚 Academic Relevance

This project is highly relevant in the domains of:

- **Artificial Intelligence**
- **Natural Language Processing**
- **Information Retrieval**
- **Fact Verification**
- **Misinformation Detection**
- **AI-Assisted Research Systems**

### Concepts Demonstrated
- Retrieval-Augmented Generation (**RAG**)
- **Hybrid Search**
- **Contradiction Detection**
- **Hallucination Checking**
- **Confidence Estimation**
- **Source-Grounded Answering**

---

## 🔮 Future Enhancements

Possible future improvements include:

- Source credibility scoring
- Bias detection and sentiment analysis
- Timeline-based event tracking
- Claim extraction and comparison
- Named Entity Recognition (NER)
- Interactive visual analytics dashboard
- Multilingual news analysis support
- Real-time breaking news tracking

---

## 👨‍💻 Author

**Abhi Shekar Pulla**  
B.Tech CSE – RGUKT Srikakulam  
Campus Major Project  

- **LinkedIn:** https://www.linkedin.com/in/abhishekar004 
- **GitHub:** https://github.com/abhishekar004  

---

## 📜 License

This project is intended for **academic, educational, and research purposes**.

If you plan to extend or publish this project publicly, you may add an open-source license such as **MIT License**.

---
