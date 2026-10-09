import os
import time
import hashlib
import numpy as np
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai

# ================= SETTINGS =================
PDF_FILE = "kushal_profile.pdf"
OWNER = "Kushal"
LLM_MODEL = "gemini-3.5-flash"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
CANDIDATES = 8
FINAL_K = 4
MEMORY_TURNS = 3
CACHE_FILE = "embeddings_cache.npz"

st.set_page_config(page_title=f"Ask about {OWNER}", page_icon="🤖")
st.title(f"🤖 Ask me about {OWNER}")
st.caption("An AI assistant answering questions from Kushal's profile document.")


@st.cache_resource
def load_pipeline():
    pages = PyPDFLoader(PDF_FILE).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)
    texts = [c.page_content for c in chunks]

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    sig_text = f"{PDF_FILE}-{os.path.getsize(PDF_FILE)}-{CHUNK_SIZE}-{CHUNK_OVERLAP}-{len(chunks)}"
    signature = hashlib.md5(sig_text.encode()).hexdigest()

    chunk_embeddings = None
    if os.path.exists(CACHE_FILE):
        saved = np.load(CACHE_FILE, allow_pickle=False)
        if str(saved["sig"]) == signature:
            chunk_embeddings = saved["emb"]

    if chunk_embeddings is None:
        chunk_embeddings = embed_model.encode(texts, normalize_embeddings=True)
        np.savez(CACHE_FILE, emb=chunk_embeddings, sig=np.array(signature))

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return chunks, texts, embed_model, chunk_embeddings, reranker


chunks, texts, embed_model, chunk_embeddings, reranker = load_pipeline()

# ================= GEMINI CLIENT =================
# Reads the key from Streamlit secrets (set this up before deploying)
api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
client = genai.Client(api_key=api_key)

# ================= RETRIEVAL =================


def retrieve(question):
    q = embed_model.encode(question, normalize_embeddings=True)
    scores = chunk_embeddings @ q
    cand_ids = np.argsort(-scores)[:min(CANDIDATES, len(texts))]
    pairs = [(question, texts[i]) for i in cand_ids]
    rerank_scores = reranker.predict(pairs)
    best = np.argsort(-rerank_scores)[:FINAL_K]
    return [int(cand_ids[i]) for i in best]

# ================= GENERATION =================


def ask_llm(prompt):
    for attempt in range(4):
        try:
            return client.models.generate_content(model=LLM_MODEL, contents=prompt).text
        except Exception as e:
            if attempt == 3:
                return None
            time.sleep(2 * (2 ** attempt))


def rewrite_question(question, history):
    if not history:
        return question
    convo = "\n".join(f"User: {q}\nAssistant: {a}" for q,
                      a in history[-MEMORY_TURNS:])
    prompt = (
        "Rewrite the user's last question so it is fully self-contained, using the "
        "conversation for context. Return ONLY the rewritten question.\n\n"
        f"Conversation:\n{convo}\n\nLast question: {question}"
    )
    return (ask_llm(prompt) or question).strip()


def answer(question, history):
    standalone = rewrite_question(question, history)
    ids = retrieve(standalone)
    context = "\n\n".join(texts[i] for i in ids)

    prompt = (
        f"You are an AI assistant on {OWNER}'s portfolio website. Visitors such as "
        f"recruiters and teachers ask you about {OWNER}.\n"
        "Rules:\n"
        f"- Answer in the third person (say '{OWNER} ...' or 'he ...').\n"
        "- Use ONLY the context below. You may combine several parts, summarize, "
        "and explain in your own words.\n"
        "- Do NOT invent facts, dates, grades, or experience that are not in the context.\n"
        f"- If the context does not have the answer, say that {OWNER}'s profile does not "
        "mention it.\n"
        "- Be friendly, clear and concise. Use short bullet points for lists.\n\n"
        f"Context:\n{context}\n\nQuestion: {standalone}"
    )
    return ask_llm(prompt)


# ================= CHAT UI =================
if "messages" not in st.session_state:
    st.session_state.messages = []

for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.write(text)

user_q = st.chat_input(f"Ask about {OWNER}...")
if user_q:
    st.session_state.messages.append(("user", user_q))
    with st.chat_message("user"):
        st.write(user_q)

    history_pairs = [
        (st.session_state.messages[i][1], st.session_state.messages[i + 1][1])
        for i in range(0, len(st.session_state.messages) - 1, 2)
        if i + 1 < len(st.session_state.messages)
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = ask_llm  # placeholder, replaced below
            reply = answer(user_q, history_pairs)
        if reply is None:
            reply = "Sorry, the AI is busy right now. Please try again shortly."
        st.write(reply)
    st.session_state.messages.append(("assistant", reply))
