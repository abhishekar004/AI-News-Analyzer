import warnings, os
warnings.filterwarnings('ignore')
os.environ.setdefault('USER_AGENT', 'AINewsAnalyzer/1.0')

# AI News Analyzer — Multi-source Opinion Synthesis with AI Contradiction-checking

import gc, re, stat, shutil, json, hashlib, time
from typing import List, Optional, Tuple
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import streamlit as st

# ── LangChain imports (version-safe) ─────────────────────────────────────────
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_community.document_loaders import WebBaseLoader
except ImportError:
    from langchain.document_loaders import WebBaseLoader

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        from langchain.embeddings import HuggingFaceEmbeddings

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from dotenv import load_dotenv
load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════════
# CENTRALISED CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    "chunk_size":        800,
    "chunk_overlap":     80,
    "hybrid_k":          15,
    "rerank_top_n":      9,
    "max_url_workers":   5,
    "urls_per_source":   5,
    "llm_max_tokens":    1024,
    "llm_temperature":   0.3,
    "llm_max_retries":   3,
    "llm_retry_delay":   2,
    "max_url_fields":    10,
    "history_context":   6,
}

PERSIST_DIR    = "chroma_store_news"
INDEX_REGISTRY = "chroma_store_news/indexed_sources.json"

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL    = "gemini-2.5-flash"
GROQ_API_URL    = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL      = "llama-3.1-8b-instant"


# ═══════════════════════════════════════════════════════════════════════════════
# RETRY HELPER
# ═══════════════════════════════════════════════════════════════════════════════
def _post_with_retry(url: str, headers: dict, payload: dict,
                     timeout: int = 60) -> requests.Response:
    delay = CONFIG["llm_retry_delay"]
    for attempt in range(CONFIG["llm_max_retries"]):
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code not in (429, 503) or attempt == CONFIG["llm_max_retries"] - 1:
            return resp
        time.sleep(delay)
        delay *= 2
    return resp


# ═══════════════════════════════════════════════════════════════════════════════
# LLM CLASSES
# ═══════════════════════════════════════════════════════════════════════════════
class GeminiLLM:
    def __init__(self, max_tokens=None, temperature=None, model=GEMINI_MODEL):
        self.model       = model
        self.max_tokens  = max_tokens  if max_tokens  is not None else CONFIG["llm_max_tokens"]
        self.temperature = temperature if temperature is not None else CONFIG["llm_temperature"]

    def invoke(self, prompt, stop=None):
        load_dotenv(override=True)
        token = os.environ.get("GEMINI_API_KEY", "").strip()
        if not token:
            return "Error: Set GEMINI_API_KEY in .env"
        url = f"{GEMINI_API_BASE}/{self.model}:generateContent?key={token}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": self.max_tokens,
                "temperature":     self.temperature,
            },
        }
        try:
            resp = _post_with_retry(url, {}, payload)
            resp.raise_for_status()
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 401:
                return "Invalid API key — check GEMINI_API_KEY in your .env file."
            return "Rate limit — please wait." if resp.status_code == 429 else f"API error: {e}"
        except Exception as e:
            return f"Request failed: {e}"
        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0].strip()
        return text


class GroqLLM:
    def __init__(self, max_tokens=None, temperature=None, model=GROQ_MODEL):
        self.model       = model
        self.max_tokens  = max_tokens  if max_tokens  is not None else CONFIG["llm_max_tokens"]
        self.temperature = temperature if temperature is not None else CONFIG["llm_temperature"]

    def invoke(self, prompt, stop=None):
        load_dotenv(override=True)
        token = os.environ.get("GROQ_API_KEY", "").strip()
        if not token:
            return "Error: Set GROQ_API_KEY in .env"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {
            "model":       self.model,
            "messages":    [{"role": "user", "content": prompt}],
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
        }
        try:
            resp = _post_with_retry(GROQ_API_URL, headers, payload)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 401:
                return "Invalid API key — check GROQ_API_KEY in your .env file."
            return f"API error: {e}"
        except Exception as e:
            return f"Request failed: {e}"
        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0].strip()
        return text


class OpenRouterLLM:
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model="meta-llama/llama-3.1-8b-instruct:free",
                 max_tokens=None, temperature=None):
        self.model       = model
        self.max_tokens  = max_tokens  if max_tokens  is not None else CONFIG["llm_max_tokens"]
        self.temperature = temperature if temperature is not None else CONFIG["llm_temperature"]

    def invoke(self, prompt, stop=None):
        load_dotenv(override=True)
        token = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not token:
            return "Error: OPENROUTER_API_KEY not found. Add it to your .env file."
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://ainewsanalyzer.app",
            "X-Title":       "AI News Analyzer",
        }
        payload = {
            "model":                 self.model,
            "messages":              [{"role": "user", "content": prompt}],
            "max_completion_tokens": self.max_tokens,
            "temperature":           self.temperature,
        }
        try:
            resp = _post_with_retry(self.OPENROUTER_URL, headers, payload)
            resp.raise_for_status()
            data    = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return f"Empty response from OpenRouter: {data}"
            text = choices[0].get("message", {}).get("content", "")
            if not text:
                return f"No content in response: {data}"
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 401:
                return "Invalid API key — double-check OPENROUTER_API_KEY in your .env file."
            if resp.status_code == 429:
                return "Rate limit — please wait and retry."
            if resp.status_code == 402:
                return "Credits exhausted — top up at openrouter.ai"
            return f"API error: {e}"
        except Exception as e:
            return f"Request failed: {e}"
        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0].strip()
        return text


# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL URL LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def load_url(url: str) -> Tuple[List[Document], Optional[str]]:
    try:
        docs = WebBaseLoader([url]).load()
        for doc in docs:
            if not doc.metadata.get("source"):
                doc.metadata["source"] = url
        return docs, None
    except Exception as e:
        return [], f"Could not load {url}: {e}"


def load_urls_parallel(urls: list, max_workers: int = None) -> List[Document]:
    max_workers = max_workers or CONFIG["max_url_workers"]
    all_docs, errors = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_url, url): url for url in urls}
        for future in as_completed(futures):
            docs, err = future.result()
            all_docs.extend(docs)
            if err:
                errors.append(err)
    for err in errors:
        st.warning(err)
    return all_docs


# ═══════════════════════════════════════════════════════════════════════════════
# HYBRID BM25 + SEMANTIC SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_reranker():
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def hybrid_search(vectorstore, query: str, all_docs: List[Document],
                  k: int = None) -> List[Document]:
    k = k or CONFIG["hybrid_k"]
    semantic_docs = vectorstore.similarity_search(query, k=k)
    semantic_set  = {d.page_content[:80] for d in semantic_docs}

    bm25_docs = []
    try:
        from rank_bm25 import BM25Okapi
        corpus    = [d.page_content.lower().split() for d in all_docs]
        bm25      = BM25Okapi(corpus)
        scores    = bm25.get_scores(query.lower().split())
        top_idx   = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        bm25_docs = [all_docs[i] for i in top_idx if scores[i] > 0]
    except ImportError:
        pass

    merged, seen = list(semantic_docs), set(semantic_set)
    for doc in bm25_docs:
        key = doc.page_content[:80]
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    return merged[:k]


def rerank_docs(reranker, query: str, docs: list, top_n: int = None) -> list:
    top_n = top_n or CONFIG["rerank_top_n"]
    if reranker is None or not docs:
        return docs[:top_n]
    try:
        pairs  = [(query, d.page_content) for d in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked[:top_n]]
    except Exception:
        return docs[:top_n]


# ═══════════════════════════════════════════════════════════════════════════════
# INCREMENTAL INDEXING
# ═══════════════════════════════════════════════════════════════════════════════
def load_indexed_registry() -> dict:
    try:
        if os.path.exists(INDEX_REGISTRY):
            return json.loads(open(INDEX_REGISTRY).read())
    except Exception:
        pass
    return {}


def save_indexed_registry(registry: dict):
    os.makedirs(os.path.dirname(INDEX_REGISTRY), exist_ok=True)
    open(INDEX_REGISTRY, "w").write(json.dumps(registry))


def content_hash(docs: List[Document]) -> str:
    combined = "".join(d.page_content for d in docs)
    return hashlib.md5(combined.encode()).hexdigest()[:12]


def _dedup_chunks(chunks: List[Document]) -> List[Document]:
    seen, unique = set(), []
    for chunk in chunks:
        norm = re.sub(r'\s+', ' ', chunk.page_content.lower().strip())
        if norm not in seen:
            seen.add(norm)
            unique.append(chunk)
    return unique


def incremental_index(new_docs: List[Document], embeddings,
                      existing_vs=None) -> Tuple[Chroma, List[Document], int]:
    registry = load_indexed_registry()
    splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"],
    )

    by_source: dict = {}
    for doc in new_docs:
        src = doc.metadata.get("source", "unknown")
        by_source.setdefault(src, []).append(doc)

    chunks_to_add = []
    new_registry  = dict(registry)

    for src, docs in by_source.items():
        chash = content_hash(docs)
        if registry.get(src) == chash:
            continue
        chunks = splitter.split_documents(docs)
        chunks = _dedup_chunks(chunks)
        chunks_to_add.extend(chunks)
        new_registry[src] = chash

    num_new = len(chunks_to_add)

    if existing_vs is not None and chunks_to_add:
        existing_vs.add_documents(chunks_to_add)
        vs = existing_vs
    elif chunks_to_add:
        vs = Chroma.from_documents(chunks_to_add, embeddings,
                                   persist_directory=None)
    elif existing_vs is not None:
        vs = existing_vs
    else:
        # In-memory: nothing to reload from disk — caller must re-process sources
        all_chunks = splitter.split_documents(new_docs)
        all_chunks = _dedup_chunks(all_chunks)
        vs = Chroma.from_documents(all_chunks, embeddings,
                                   persist_directory=None)
        num_new = len(all_chunks)

    save_indexed_registry(new_registry)
    return vs, chunks_to_add, num_new


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY REWRITING
# ═══════════════════════════════════════════════════════════════════════════════
def rewrite_query(llm, original_query: str, chat_history: list) -> str:
    if not chat_history:
        return original_query
    history_text = "\n".join(
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content'][:200]}"
        for m in chat_history[-CONFIG["history_context"]:]
    )
    prompt = f"""Given this conversation history:
{history_text}

Rewrite the following question into a clear, standalone search query (no pronouns, no vague references).
Output ONLY the rewritten query, nothing else.

Question: {original_query}
Rewritten query:"""
    rewritten = llm.invoke(prompt).strip().strip('"').strip("'")
    return rewritten if rewritten else original_query


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════
COMPARE_RE = re.compile(
    r"\b(compar|vs\.?|versus|contrast|differ|similarit|common|both|all article|across article|each article)\b",
    re.IGNORECASE
)


def _normalise(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip())


def _build_context_blocks(docs: list, article_map: dict) -> str:
    seen_prefix: set = set()
    seen_norm:   set = set()
    unique_docs: list = []
    for doc in docs:
        prefix = doc.page_content[:80]
        norm   = _normalise(doc.page_content)
        if prefix in seen_prefix or norm in seen_norm:
            continue
        seen_prefix.add(prefix)
        seen_norm.add(norm)
        unique_docs.append(doc)

    by_source: dict = {}
    for doc in unique_docs:
        src = doc.metadata.get("source", "Unknown")
        by_source.setdefault(src, []).append(doc.page_content.strip())

    blocks = []
    for src, passages in by_source.items():
        label = article_map.get(src, src)
        joined = "\n\n".join(f"• {p}" for p in passages)
        blocks.append(f"[{label}]\n{joined}")

    legend = "\n".join(f"  • {v} = {k}" for k, v in article_map.items())
    return (f"Sources:\n{legend}\n\n---EXCERPTS---\n\n"
            + "\n\n---\n\n".join(blocks) + "\n---END---")


def build_qa_prompt(question: str, docs: list, article_map: dict,
                    chat_history: list, is_comparison: bool) -> str:
    ctx = _build_context_blocks(docs, article_map)
    history_text = ""
    if chat_history:
        history_text = "Conversation so far:\n" + "\n".join(
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content'][:300]}"
            for m in chat_history[-CONFIG["history_context"]:]
        ) + "\n\n"
    num_sources = len(article_map)
    source_list = ", ".join(article_map.values())

    if is_comparison:
        style = ("FORMAT YOUR ANSWER AS:\n"
                 "1. A markdown table (columns = articles, rows = aspects)\n"
                 "2. A paragraph on common themes\n"
                 "3. A paragraph on key differences")
    elif num_sources == 1:
        only_label = list(article_map.values())[0]
        style = (f"There is ONE source: [{only_label}]. "
                 f"Do NOT split it into sub-articles or create labels like Article 2, Article 3, etc. "
                 f"Write a single cohesive answer referencing [{only_label}] only.")
    else:
        style = (f"There are {num_sources} sources: {source_list}. "
                 f"Cite them inline exactly as shown (e.g. [{list(article_map.values())[0]}]). "
                 f"Do NOT invent new article labels beyond those listed.")

    return f"""{history_text}You are a research assistant. Use ALL provided excerpts.
{style}

{ctx}

Question: {question}
Answer:"""


# ═══════════════════════════════════════════════════════════════════════════════
# CONTRADICTION DETECTION — compact, direct output
# ═══════════════════════════════════════════════════════════════════════════════
def detect_contradictions(llm, docs: list, article_map: dict) -> str:
    ctx = _build_context_blocks(docs, article_map)
    prompt = f"""You are a strict fact-checker. Find ONLY real contradictions — cases where two DIFFERENT sources state conflicting facts about the exact same thing.

STRICT RULES:
1. BOTH sides must state a concrete value/claim. "Source A doesn't mention X" is NOT a contradiction — skip it entirely.
2. The same source contradicting itself is NOT a contradiction — skip it.
3. Rounding (e.g. $795.6B vs $796B) is NOT a contradiction — skip it.
4. Different dates for different events are NOT a contradiction — skip it.
5. Output at most 5 contradictions. If fewer real ones exist, output only those.
6. If none exist, reply exactly: ✅ No real contradictions detected.

Format (one line each, nothing else):
⚡ **[Topic]**: [Source A label] says "X" — [Source B label] says "Y"

{ctx}

Contradictions:"""
    return llm.invoke(prompt)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE SCORING
# ═══════════════════════════════════════════════════════════════════════════════
def compute_confidence(query: str, top_docs: list, reranker) -> float:
    if reranker and top_docs:
        try:
            pairs  = [(query, d.page_content) for d in top_docs[:5]]
            scores = reranker.predict(pairs)
            avg    = sum(scores[:3]) / len(scores[:3])
            return min(max((avg + 5) / 10, 0.0), 1.0)
        except Exception:
            pass
    sources = len(set(d.metadata.get("source", "") for d in top_docs))
    return min(0.4 + sources * 0.15 + len(top_docs) * 0.02, 1.0)


def confidence_badge(score: float) -> str:
    if score >= 0.7:
        style = "background:var(--color-background-success);color:var(--color-text-success)"
        label = f"High ({score:.0%})"
    elif score >= 0.4:
        style = "background:var(--color-background-warning);color:var(--color-text-warning)"
        label = f"Medium ({score:.0%})"
    else:
        style = "background:var(--color-background-danger);color:var(--color-text-danger)"
        label = f"Low ({score:.0%})"
    return (f'<span style="{style};border-radius:5px;'
            f'padding:2px 8px;font-size:0.75em;font-weight:600;">🎯 Confidence: {label}</span>')


# ═══════════════════════════════════════════════════════════════════════════════
# HALLUCINATION DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
def check_hallucinations(llm, answer: str, docs: list) -> str:
    context = "\n\n".join(
        f"[Chunk {i+1}]: {d.page_content[:400]}" for i, d in enumerate(docs[:6])
    )
    prompt = f"""You are a fact-checker. Below is an AI-generated answer, followed by the source chunks it was based on.

Check each factual claim in the answer against the source chunks.
For claims that are NOT supported by any chunk, mark them as [UNSUPPORTED].
For claims that ARE supported, mark them as [✓].

Output a brief verification report listing each key claim and its status.
If all claims are supported, say "All claims verified."

---ANSWER---
{answer[:1200]}
---END ANSWER---

---SOURCE CHUNKS---
{context}
---END CHUNKS---

Verification report:"""
    return llm.invoke(prompt)


# ═══════════════════════════════════════════════════════════════════════════════
# FUZZY SOURCE MATCHING — FIX #3: ensure all sources (URLs + PDFs) are listed
# ═══════════════════════════════════════════════════════════════════════════════
def _fuzzy_match_source(src: str, known_sources: set) -> Optional[str]:
    if src in known_sources:
        return src
    src_base = os.path.basename(src).strip()
    for k in known_sources:
        if os.path.basename(k).strip() == src_base:
            return k
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE EXPANDER — FIX #3: shows all actual sources used (URLs + PDFs)
# ═══════════════════════════════════════════════════════════════════════════════
def render_source_expander(sources: list, article_map: dict):
    if not sources:
        return
    with st.expander("📎 Sources used", expanded=False):
        for i, src in enumerate(sources, 1):
            label = article_map.get(src, os.path.basename(src) if not src.startswith("http") else src)
            is_url = src.startswith("http")
            if is_url:
                st.markdown(
                    f'**[{i}]** {label} — <a href="{src}" target="_blank" '
                    f'style="color:#38bdf8;">{src}</a>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"**[{i}]** 📄 {label}")


# ═══════════════════════════════════════════════════════════════════════════════
# INLINE CITATION BADGES
# ═══════════════════════════════════════════════════════════════════════════════
def format_answer_with_citations(answer: str, article_map: dict) -> str:
    for src, label in article_map.items():
        n = re.search(r"(\d+)", label)
        if not n:
            continue
        num   = n.group(1)
        badge = (f'<sup title="{src}">'
                 f'<span style="background:#1e88e5;color:white;border-radius:4px;'
                 f'padding:1px 6px;font-size:0.7em;font-weight:bold;">[{num}]</span></sup>')
        answer = answer.replace(f"[{label}]", badge)
        answer = answer.replace(f"[Article {num}]", badge)
    return answer


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN QA FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
def answer_question(
    llm, vectorstore, question: str,
    chat_history: list, article_map: dict,
    reranker, all_docs: list, k: int = None,
) -> Tuple[str, List[str], float, list]:
    k = k or CONFIG["hybrid_k"]

    search_query = rewrite_query(llm, question, chat_history)
    raw_docs     = hybrid_search(vectorstore, search_query, all_docs, k=k)
    top_docs     = rerank_docs(reranker, search_query, raw_docs,
                               top_n=CONFIG["rerank_top_n"])

    if not top_docs:
        return "No relevant content found. Please process URLs first.", [], 0.0, []

    is_comparison = bool(COMPARE_RE.search(question))
    prompt        = build_qa_prompt(question, top_docs, article_map,
                                    chat_history, is_comparison)
    answer        = llm.invoke(prompt)

    confidence = compute_confidence(search_query, top_docs, reranker)

    # FIX #3: fuzzy match so both URLs and PDFs are always captured in sources
    known_sources = set(article_map.keys())
    sources = sorted(set(
        matched
        for doc in top_docs
        for matched in [_fuzzy_match_source(doc.metadata.get("source", ""), known_sources)]
        if matched
    ))

    return answer, sources, confidence, top_docs


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
def _md_to_plain(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*',   r'\1', text)
    text = re.sub(r'__(.+?)__',   r'\1', text)
    text = re.sub(r'_(.+?)_',     r'\1', text)
    text = re.sub(r'#+\s*',       '',    text)
    text = re.sub(r'`{1,3}[^`]*`{1,3}', '', text)
    text = re.sub(r'^\s*[-*]\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    return text.strip()


def export_to_word(chat_history: list, article_map: dict) -> bytes:
    from docx import Document as DocxDoc
    import io

    doc = DocxDoc()
    doc.add_heading("AI News Analyzer — Research Report", 0)
    doc.add_paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}")
    doc.add_paragraph("Sources: " + ", ".join(article_map.keys()))
    doc.add_heading("Research Session", level=1)

    for msg in chat_history:
        role  = "You" if msg["role"] == "user" else "Assistant"
        clean = _md_to_plain(msg["content"])
        p     = doc.add_paragraph()
        run   = p.add_run(f"{role}: ")
        run.bold = True
        p.add_run(clean)
        if msg.get("sources"):
            doc.add_paragraph("Sources: " + " | ".join(msg["sources"]),
                              style="Caption")
        doc.add_paragraph()

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def export_to_pdf(chat_history: list, article_map: dict) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import cm
    import io

    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    story.append(Paragraph("AI News Analyzer — Research Report", styles["Title"]))
    story.append(Paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}",
                            styles["Normal"]))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("Sources: " + ", ".join(article_map.keys()),
                            styles["Normal"]))
    story.append(Spacer(1, 0.6*cm))

    q_style = ParagraphStyle("Q", parent=styles["Normal"],
                              fontName="Helvetica-Bold", spaceAfter=4)
    a_style = ParagraphStyle("A", parent=styles["Normal"],
                              spaceAfter=8, leading=14)

    for msg in chat_history:
        clean = _md_to_plain(msg["content"])
        clean = (clean.replace("&", "&amp;")
                      .replace("<", "&lt;")
                      .replace(">", "&gt;"))
        if msg["role"] == "user":
            story.append(Paragraph(f"Q: {clean}", q_style))
        else:
            story.append(Paragraph(f"A: {clean}", a_style))
            if msg.get("sources"):
                src_text = "Sources: " + " | ".join(msg["sources"])
                story.append(Paragraph(
                    src_text,
                    ParagraphStyle("S", parent=styles["Normal"],
                                   fontSize=8, textColor=(0.4, 0.4, 0.4))
                ))
        story.append(Spacer(1, 0.3*cm))

    doc.build(story)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# ARTICLE PREVIEW CARDS
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_article_meta(url: str) -> dict:
    meta = {"title": urlparse(url).netloc, "description": "", "favicon": ""}
    try:
        resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        html = resp.text
        m = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)', html)
        if not m:
            m = re.search(r'<title>([^<]+)</title>', html)
        if m:
            meta["title"] = m.group(1).strip()[:80]
        m = re.search(
            r'<meta[^>]+(?:property=["\']og:description["\']|name=["\']description["\'])'
            r'[^>]+content=["\']([^"\']+)', html)
        if m:
            meta["description"] = m.group(1).strip()[:160]
        meta["favicon"] = (f"https://www.google.com/s2/favicons?"
                           f"domain={urlparse(url).netloc}&sz=32")
    except Exception:
        pass
    return meta


def render_article_cards(article_metas: list):
    if not article_metas:
        return
    cols = st.columns(max(len(article_metas), 1))
    for col, (url, meta) in zip(cols, article_metas):
        with col:
            is_url = url.startswith("http")
            link   = (f'<a href="{url}" target="_blank" '
                      f'style="color:#38bdf8;font-size:0.75em;">Open ↗</a>') if is_url else ""
            fav    = (f'<img src="{meta["favicon"]}" width="16" height="16" '
                      f'style="border-radius:3px;vertical-align:middle;">'
                      ) if meta.get("favicon") else "📄"
            st.markdown(f"""
<div style="border:1px solid #334155;border-radius:10px;padding:14px;background:#1e293b;min-height:110px;">
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">
    {fav}<span style="color:#94a3b8;font-size:0.73em;">{urlparse(url).netloc if is_url else 'PDF'}</span>
  </div>
  <p style="color:#f1f5f9;font-size:0.86em;font-weight:600;margin:0 0 5px;line-height:1.4;">{meta['title']}</p>
  <p style="color:#94a3b8;font-size:0.77em;margin:0 0 6px;line-height:1.4;">{meta['description'][:110]}{'…' if len(meta.get('description',''))>110 else ''}</p>
  {link}
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SUGGESTED QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def generate_suggested_questions(llm, article_metas: list) -> list:
    titles = "\n".join(f"- {m['title']}" for _, m in article_metas)
    prompt = f"""Given these article titles:
{titles}

Generate exactly 4 insightful research questions spanning these articles.
Output ONLY a JSON array of 4 strings. No explanation. No markdown.
Example: ["Q1?","Q2?","Q3?","Q4?"]"""
    raw = llm.invoke(prompt).strip()
    raw = re.sub(r"```[a-z]*", "", raw).strip("`").strip()
    try:
        qs = json.loads(raw)
        if isinstance(qs, list):
            return [str(q) for q in qs[:4]]
    except Exception:
        pass
    lines = [l.strip().strip("-•").strip() for l in raw.split("\n") if "?" in l]
    return lines[:4] if lines else [
        "Summarise all articles", "What are the common themes?",
        "Compare key points across articles", "What are the main differences?",
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# ARTICLE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════
def discover_articles(topic: str) -> list:
    import xml.etree.ElementTree as ET
    query   = requests.utils.quote(topic)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    max_results = CONFIG["max_url_fields"]

    def _parse_rss(xml_text):
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []
        seen, urls = set(), []
        for item in root.findall(".//item"):
            link = item.findtext("link") or ""
            if not link.startswith("http"):
                guid = item.findtext("guid") or ""
                if guid.startswith("http"):
                    link = guid
            if not link.startswith("http"):
                continue
            m = re.search(r'articles/(https?[^&"]+)', link)
            if m:
                link = requests.utils.unquote(m.group(1))
            domain = urlparse(link).netloc
            if domain and domain not in seen and len(link) < 500:
                seen.add(domain)
                urls.append(link)
            if len(urls) >= max_results:
                break
        return urls

    tried = []
    for feed_url in [
        f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en",
        f"https://www.bing.com/news/search?q={query}&format=RSS",
    ]:
        tried.append(feed_url)
        try:
            resp = requests.get(feed_url, headers=headers, timeout=10)
            if resp.status_code == 200:
                urls = _parse_rss(resp.text)
                if urls:
                    return urls
        except Exception:
            pass

    api_key = os.environ.get("NEWSAPI_KEY", "")
    if api_key:
        tried.append("NewsAPI")
        try:
            resp = requests.get(
                f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt"
                f"&pageSize={max_results}&apiKey={api_key}",
                headers=headers, timeout=10)
            if resp.status_code == 200:
                seen, urls = set(), []
                for a in resp.json().get("articles", []):
                    u = a.get("url", "")
                    d = urlparse(u).netloc
                    if u.startswith("http") and d not in seen:
                        seen.add(d)
                        urls.append(u)
                    if len(urls) >= max_results:
                        break
                if urls:
                    return urls
        except Exception:
            pass

    st.info(f"No articles found. Tried: {', '.join(tried)}. "
            "Try a different topic or add NEWSAPI_KEY to .env")
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# VECTORSTORE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def has_processed_sources() -> bool:
    return bool(st.session_state.get("processed_urls") or
                st.session_state.get("processed_pdfs"))


def _force_remove(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def safe_delete_store():
    if st.session_state.get("vectorstore") is not None:
        try:
            st.session_state.vectorstore._client.close()
        except Exception:
            pass
        del st.session_state["vectorstore"]
        gc.collect()
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR, onerror=_force_remove)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — notes/pin fields removed
# ═══════════════════════════════════════════════════════════════════════════════
def init_state():
    for k, v in {
        "processed_urls":         [],
        "processed_pdfs":         [],
        "article_metas":          [],
        "article_map":            {},
        "chat_history":           [],
        "suggested_qs":           [],
        "vectorstore":            None,
        "url_count":              3,
        "all_docs":               [],
        "show_contradiction":     False,
        "contradiction_report":   None,   # cached — cleared when sources change
        "show_hallucination_idx": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG — must be first Streamlit call
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="AI News Analyzer", page_icon="📰", layout="wide")

init_state()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.block-container{padding-top:1.2rem!important;}
div[data-testid="stSidebar"]{background:#0f172a;}
</style>""", unsafe_allow_html=True)

# Fix 1: embeddings loaded once and cached across reruns
embeddings = load_embeddings()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📰 AI News Analyzer")
    st.caption("Multi-source research with contradiction detection")
    st.divider()

    LLM_OPTIONS = {
        "Groq (free – Llama 3.1)": "groq",
        "Gemini (Google)":         "gemini",
        "OpenRouter":              "openrouter",
    }
    provider = LLM_OPTIONS[st.selectbox("🤖 LLM", list(LLM_OPTIONS.keys()), index=0)]
    if provider == "groq":
        llm = GroqLLM()
    elif provider == "gemini":
        llm = GeminiLLM()
    else:
        OR_MODELS = {
            "meta-llama/llama-3.1-8b-instruct:free": "Llama 3.1 8B (free)",
            "mistralai/mistral-7b-instruct:free":     "Mistral 7B (free)",
            "google/gemma-2-9b-it:free":              "Gemma 2 9B (free)",
            "qwen/qwen-2-7b-instruct:free":           "Qwen 2 7B (free)",
        }
        or_display = {v: k for k, v in OR_MODELS.items()}
        or_choice  = st.selectbox("Model", list(or_display.keys()), index=0)
        llm = OpenRouterLLM(model=or_display[or_choice])
        st.caption("Set OPENROUTER_API_KEY in .env — free at openrouter.ai/keys")

    st.divider()
    st.markdown("#### 🔍 Auto-discover by topic")
    topic = st.text_input("Topic", placeholder="e.g. AI regulation 2025")

    if st.button("Find articles"):
        with st.spinner("Searching Google News & Bing RSS…"):
            found = discover_articles(topic)
        if found:
            placed_count = 0
            for url in found:
                placed = False
                for i in range(st.session_state.url_count):
                    existing = st.session_state.get(f"url_{i}", "").strip()
                    if not existing:
                        st.session_state[f"url_{i}"] = url
                        placed = True
                        placed_count += 1
                        break
                if not placed:
                    if st.session_state.url_count < CONFIG["max_url_fields"]:
                        st.session_state[f"url_{st.session_state.url_count}"] = url
                        st.session_state.url_count += 1
                        placed_count += 1
                    else:
                        st.warning(f"Max {CONFIG['max_url_fields']} URL fields reached.")
            st.success(f"✅ Placed {placed_count} article(s) into URL fields ↓")
            st.rerun()

    st.divider()
    st.markdown("#### 🔗 Article URLs")
    if st.button("＋ Add URL field"):
        st.session_state.url_count = min(
            st.session_state.url_count + 1, CONFIG["max_url_fields"]
        )

    urls = [st.text_input(f"URL {i+1}", key=f"url_{i}")
            for i in range(st.session_state.url_count)]

    st.divider()
    st.markdown("#### 📄 Upload PDFs")
    pdf_files = st.file_uploader("PDFs (optional)", type=["pdf"],
                                  accept_multiple_files=True,
                                  label_visibility="collapsed")

    st.divider()
    incremental_mode = st.checkbox(
        "⚡ Incremental indexing (skip unchanged sources)", value=True,
        help="Only re-embed new/changed sources. Faster for large collections."
    )
    use_reranker = st.checkbox(
        "🎯 Use reranker (better results, slower)", value=False,
        help="Adds ~300-500ms per query. Disable on slow/free-tier machines."
    )
    reranker = load_reranker() if use_reranker else None

    st.divider()
    process_clicked = st.button("⚡ Process All Sources",
                                use_container_width=True, type="primary")

    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.session_state.chat_history:
        st.divider()
        st.markdown("#### 📤 Export Session")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Word", use_container_width=True):
                try:
                    data = export_to_word(st.session_state.chat_history,
                                          st.session_state.article_map)
                    st.download_button(
                        "⬇ Download .docx", data, "report.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                except ImportError:
                    st.warning("Install python-docx: `pip install python-docx`")
        with col2:
            if st.button("📄 PDF", use_container_width=True):
                try:
                    data = export_to_pdf(st.session_state.chat_history,
                                         st.session_state.article_map)
                    st.download_button("⬇ Download .pdf", data,
                                       "report.pdf", mime="application/pdf")
                except ImportError:
                    st.warning("Install reportlab: `pip install reportlab`")


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
if process_clicked:
    urls_to_load = [u.strip() for u in urls if u.strip()]
    if not urls_to_load and not pdf_files:
        st.sidebar.warning("Provide at least one URL or PDF.")
        st.stop()

    with st.spinner("Loading sources in parallel…"):
        all_raw_docs = []

        if urls_to_load:
            progress = st.progress(0, text="Fetching articles in parallel…")
            web_docs = load_urls_parallel(urls_to_load)
            all_raw_docs.extend(web_docs)
            progress.progress(100, text=f"Loaded {len(urls_to_load)} URL(s)")

        if pdf_files:
            try:
                import pdfplumber
                for pdf in pdf_files:
                    pages = []
                    with pdfplumber.open(pdf) as f:
                        for pg in f.pages:
                            t = pg.extract_text()
                            if t:
                                pages.append(t)
                    all_raw_docs.append(Document(
                        page_content="\n\n".join(pages),
                        metadata={"source": pdf.name, "type": "pdf"}
                    ))
            except ImportError:
                st.warning("Install pdfplumber: `pip install pdfplumber`")

        if not all_raw_docs:
            st.error("No content extracted.")
            st.stop()

    with st.spinner("Indexing…"):
        if incremental_mode:
            existing_vs = st.session_state.get('vectorstore', None)
            vs, new_chunks, num_new = incremental_index(
                all_raw_docs, embeddings, existing_vs
            )
            index_msg = f"⚡ {num_new} new chunks indexed (incremental)"
        else:
            safe_delete_store()
            splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=CONFIG["chunk_size"],
                chunk_overlap=CONFIG["chunk_overlap"],
            )
            all_chunks = splitter.split_documents(all_raw_docs)
            all_chunks = _dedup_chunks(all_chunks)
            vs = Chroma.from_documents(all_chunks, embeddings,
                                       persist_directory=None)
            index_msg = f"✅ {len(all_chunks)} chunks indexed"

        st.session_state.vectorstore    = vs
        st.session_state.all_docs       = all_raw_docs
        st.session_state.processed_urls = urls_to_load
        st.session_state.processed_pdfs = [f.name for f in (pdf_files or [])]

        seen       = []
        source_order = list(urls_to_load)
        for pdf in (pdf_files or []):
            source_order.append(pdf.name)

        for doc in all_raw_docs:
            src = doc.metadata.get("source", "Unknown")
            if src not in seen:
                if src in source_order:
                    seen.append(src)
                else:
                    for s in source_order:
                        if os.path.basename(s) == os.path.basename(src) and s not in seen:
                            seen.append(s)
                            break

        for s in source_order:
            if s not in seen:
                seen.append(s)

        article_map: dict = {}
        if len(seen) == 1:
            src = seen[0]
            if src.startswith("http"):
                base_label = urlparse(src).netloc.replace("www.", "")
            else:
                base_label = os.path.basename(src)
            article_map[src] = base_label
        else:
            for i, src in enumerate(seen):
                if src.startswith("http"):
                    base_label = urlparse(src).netloc.replace("www.", "")
                else:
                    base_label = os.path.basename(src)
                article_map[src] = f"Article {i+1} ({base_label})"

        st.session_state.article_map = article_map

        metas = [(u, fetch_article_meta(u)) for u in urls_to_load]
        for pdf in (pdf_files or []):
            metas.append((pdf.name, {"title": pdf.name,
                                     "description": "Uploaded PDF", "favicon": ""}))
        st.session_state.article_metas       = metas
        st.session_state.suggested_qs        = generate_suggested_questions(llm, metas)
        st.session_state.chat_history        = []
        st.session_state.contradiction_report = None   # invalidate cache on new sources
        st.session_state.show_contradiction  = False

    st.success(f"{index_msg} — {len(seen)} source(s) ready.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# 📰 AI News Analyzer")
st.caption("Hybrid search · Contradiction detection · Hallucination checking · Export")

if st.session_state.article_metas:
    st.markdown("### 📋 Loaded Sources")
    render_article_cards(st.session_state.article_metas)

    col_a, col_b = st.columns([1, 5])
    with col_a:
        if st.button("🔍 Detect contradictions", use_container_width=True):
            st.session_state.show_contradiction = not st.session_state.show_contradiction
            # If toggling ON and no cached result yet, trigger a fresh scan
            if st.session_state.show_contradiction:
                st.session_state.contradiction_report = None

    if (st.session_state.get('show_contradiction', False)
            and st.session_state.get('vectorstore') is not None):

        # Only call the LLM if we don't have a cached result
        if st.session_state.get('contradiction_report') is None:
            with st.spinner("Scanning for contradictions…"):
                sample_docs = st.session_state.vectorstore.similarity_search(
                    "key claims facts figures numbers dates", k=14
                )
                st.session_state.contradiction_report = detect_contradictions(
                    llm, sample_docs, st.session_state.article_map
                )

        with st.expander("🔍 Contradiction Report", expanded=True):
            report_text = st.session_state.contradiction_report or ""
            # Render each ⚡ line as its own markdown block for clean formatting
            lines = [l.strip() for l in report_text.strip().splitlines() if l.strip()]
            if lines:
                st.markdown("\n\n".join(lines))
            if st.button("🔄 Re-scan", help="Force a fresh contradiction scan"):
                st.session_state.contradiction_report = None
                st.rerun()

    st.divider()

if st.session_state.suggested_qs:
    st.markdown("#### 💡 Try asking…")
    cols = st.columns(len(st.session_state.suggested_qs))
    for col, q in zip(cols, st.session_state.suggested_qs):
        if col.button(q, use_container_width=True):
            st.session_state["pending_question"] = q

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# CHAT — full width (no notes column)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("### 💬 Chat")

# Render existing chat history
for idx, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if msg["role"] == "assistant":
            # FIX #4: verify button always shown inline with confidence + sources
            row = st.columns([2, 2, 2, 2])
            if msg.get("confidence") is not None:
                row[0].markdown(confidence_badge(msg["confidence"]),
                                unsafe_allow_html=True)
            if msg.get("sources"):
                with row[1]:
                    render_source_expander(
                        msg["sources"],
                        st.session_state.article_map,
                    )
            # FIX #4: verify button always in row[2], no conditional hiding
            if row[2].button("🧪 Verify", key=f"verify_{idx}"):
                st.session_state.show_hallucination_idx = idx

            # Show hallucination report inline if this message is selected
            hal_idx = st.session_state.get('show_hallucination_idx')
            if hal_idx == idx and msg.get("top_docs"):
                with st.spinner("Running fact-check…"):
                    report = check_hallucinations(llm, msg["content"], msg["top_docs"])
                st.markdown(
                    f'<div style="background:#0f2027;border:1px solid #1e88e5;'
                    f'border-radius:8px;padding:12px 16px;margin-top:8px;font-size:0.87em;">'
                    f'<strong>🧪 Fact-check Report</strong><br><br>'
                    f'{report.replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True
                )
                if st.button("✕ Close", key=f"close_hal_{idx}"):
                    st.session_state.show_hallucination_idx = None
                    st.rerun()

# Handle pending question from suggested Qs
pending_q = st.session_state.get("pending_question")

if pending_q and not has_processed_sources():
    st.warning("⚠️ Please process sources first using the sidebar.")
else:
    pending = st.session_state.pop("pending_question", None)
    user_input = st.chat_input("Ask anything about the articles…") or pending

    if user_input:
        if not has_processed_sources():
            st.warning("⚠️ Please process sources first using the sidebar.")
            st.stop()
        if not st.session_state.get('vectorstore'):
            st.warning("⚠️ Vector store not found. Re-process sources.")
            st.stop()

        with st.chat_message("user"):
            st.markdown(user_input)

        history_before_append = list(st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        vs       = st.session_state.vectorstore
        all_docs = st.session_state.all_docs
        k        = max(
            CONFIG["hybrid_k"],
            (len(st.session_state.get('processed_urls', []))
             + len(st.session_state.get('processed_pdfs', [])))
            * CONFIG["urls_per_source"]
        )

        with st.chat_message("assistant"):
            with st.spinner("Researching…"):
                answer, sources, confidence, top_docs = answer_question(
                    llm=llm,
                    vectorstore=vs,
                    question=user_input,
                    chat_history=history_before_append,
                    article_map=st.session_state.article_map,
                    reranker=reranker,
                    all_docs=all_docs,
                    k=k,
                )
            formatted = format_answer_with_citations(answer, st.session_state.article_map)
            st.markdown(formatted, unsafe_allow_html=True)

            # FIX #4: show confidence + sources + verify immediately after answer
            row = st.columns([2, 2, 2, 2])
            row[0].markdown(confidence_badge(confidence), unsafe_allow_html=True)
            if sources:
                with row[1]:
                    render_source_expander(sources, st.session_state.article_map)
            # Verify for new answer — will work after rerun from history
            row[2].button("🧪 Verify", key="verify_new", disabled=True,
                          help="Available after page refresh")

        st.session_state.chat_history.append({
            "role":       "assistant",
            "content":    formatted,
            "sources":    sources,
            "confidence": confidence,
            "top_docs":   top_docs,
        })