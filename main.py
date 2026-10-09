import os
import time
import hashlib
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai

# ================= SETTINGS =================
PDF_FILE = "kushal_profile.pdf"
OWNER = "Kushal_profile.pdf"
LLM_MODEL = "gemini-3.5-flash"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
CANDIDATES = 8
FINAL_K = 4
MEMORY_TURNS = 3
CACHE_FILE = "embeddings_cache.npz"

# =================  LOAD + SPLIT =================
if not os.path.exists(PDF_FILE):
    raise SystemExit(
        f"Cannot find '{PDF_FILE}'. Put it in the same folder as main.py.")

pages = PyPDFLoader(PDF_FILE).load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(pages)
texts = [c.page_content for c in chunks]
print(f"Loaded {PDF_FILE} -> {len(chunks)} chunks")

# =================  EMBEDDINGS =================
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
sig_text = f"{PDF_FILE}-{os.path.getsize(PDF_FILE)}-{CHUNK_SIZE}-{CHUNK_OVERLAP}-{len(chunks)}"
signature = hashlib.md5(sig_text.encode()).hexdigest()

chunk_embeddings = None
if os.path.exists(CACHE_FILE):
    saved = np.load(CACHE_FILE, allow_pickle=False)
    if str(saved["sig"]) == signature:
        chunk_embeddings = saved["emb"]
        print("Loaded saved embeddings (fast start).")

if chunk_embeddings is None:
    print("Creating embeddings...")
    chunk_embeddings = embed_model.encode(texts, normalize_embeddings=True)
    np.savez(CACHE_FILE, emb=chunk_embeddings, sig=np.array(signature))

# =================  SMART RETRIEVAL (search + re-rank) =================
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def retrieve(question):
    q = embed_model.encode(question, normalize_embeddings=True)
    # similarity with every chunk
    scores = chunk_embeddings @ q
    # best candidates by meaning
    cand_ids = np.argsort(-scores)[:min(CANDIDATES, len(texts))]
    pairs = [(question, texts[i]) for i in cand_ids]
    # judge each one carefully
    rerank_scores = reranker.predict(pairs)
    # keep the best few
    best = np.argsort(-rerank_scores)[:FINAL_K]
    return [int(cand_ids[i]) for i in best]


# ================= STEP 4: GENERATION =================
client = genai.Client()


def ask_llm(prompt):
    """Call Gemini. If the server is busy (503), wait and retry."""
    for attempt in range(4):
        try:
            return client.models.generate_content(model=LLM_MODEL, contents=prompt).text
        except Exception as e:
            if attempt == 3:
                print("\n[AI error]", str(e)[:200])
                return None
            wait = 2 * (2 ** attempt)
            print(f"(Server busy, retrying in {wait}s...)")
            time.sleep(wait)


def rewrite_question(question, history):
    """Turns 'what was his role in it?' into a full standalone question."""
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
        "mention it, and share any related information you did find.\n"
        "- Be friendly, clear and concise. Use short bullet points for lists.\n\n"
        f"Context:\n{context}\n\nQuestion: {standalone}"
    )
    reply = ask_llm(prompt)
    pages_used = sorted({chunks[i].metadata.get("page", 0) + 1 for i in ids})
    return reply, pages_used


# ================= CHAT LOOP =================
print(f"\nAsk me anything about {OWNER}.")
print("Commands: 'clear' = forget conversation, 'exit' = quit")
history = []
while True:
    q = input("\nYou: ").strip()
    if not q:
        continue
    if q.lower() == "exit":
        break
    if q.lower() == "clear":
        history = []
        print("Conversation cleared.")
        continue

    reply, pages_used = answer(q, history)
    if reply is None:
        print("Bot: Sorry, the AI is busy right now. Please try again in a minute.")
        continue
    print("\nBot:", reply)
    print("(Source pages:", pages_used, ")")
    history.append((q, reply))
