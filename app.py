import os

# Must be set BEFORE importing sentence_transformers (hides Windows symlink warning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

NO_INFO_MSG = "I do not have the relevant information in the provided document context."

# Tried in order. "-latest" aliases always point to a currently available model.
DEFAULT_MODELS = [
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-lite-latest",
]

# ------------------------------------------------------------------
# 1. PAGE SETUP
# ------------------------------------------------------------------
st.set_page_config(page_title="Campus Study Buddy", page_icon="📚", layout="centered")
st.title("📚 Campus Study Buddy")
st.caption("A RAG chatbot: it answers ONLY from the knowledge base below (open-book mode).")

# ------------------------------------------------------------------
# 2. KNOWLEDGE BASE
# ------------------------------------------------------------------
BASE_DOCUMENTS = [
    "Operating Systems: A deadlock occurs when processes wait forever for resources held by each other.",
    "Operating Systems: Virtual memory lets a program use more memory than physically available by swapping pages to disk.",
    "Operating Systems: Round Robin scheduling gives each process a fixed time quantum in a cyclic order.",
    "Computer Networks: TCP is connection-oriented and reliable, while UDP is connectionless and faster but unreliable.",
    "Computer Networks: DNS translates human-readable domain names into IP addresses.",
    "Computer Networks: The OSI model has 7 layers, from Physical up to Application.",
    "Data Structures: A stack follows LIFO order, while a queue follows FIFO order.",
    "Data Structures: Binary search runs in O(log n) time but needs a sorted array.",
    "Campus Policy: Library books can be issued for 14 days and renewed once online.",
    "Campus Policy: Students need at least 75 percent attendance to be eligible for semester exams.",
    "Campus Policy: Hostel gates close at 10 PM, and late entry requires a warden's written permission.",
]


@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~90 MB)...")
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def embed_documents(docs: tuple):
    """Embed documents once; cached until the document list changes."""
    model = load_embedder()
    return model.encode(list(docs), normalize_embeddings=True)


# ------------------------------------------------------------------
# 3. SIDEBAR
# ------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")

api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    value=os.environ.get("GEMINI_API_KEY", ""),
    help="Get a free key from https://aistudio.google.com/apikey",
)
top_k = st.sidebar.slider("Chunks to retrieve (top-k)", 1, 5, 3)
threshold = st.sidebar.slider(
    "Minimum similarity", 0.0, 0.8, 0.25, 0.05,
    help="If the best match scores below this, the bot says it doesn't know instead of guessing.",
)
custom_model = st.sidebar.text_input("Custom model name (optional)", placeholder="e.g. gemini-2.5-flash")

st.sidebar.markdown("---")
st.sidebar.subheader("➕ Add your own notes")
extra_notes = st.sidebar.text_area(
    "One fact per line",
    placeholder="Physics: Newton's second law states F = ma.",
    height=120,
)

if st.sidebar.button("🧹 Clear chat"):
    st.session_state.messages = []
    st.rerun()

extra_docs = [line.strip() for line in extra_notes.splitlines() if line.strip()]
documents = BASE_DOCUMENTS + extra_docs
doc_embeddings = embed_documents(tuple(documents))

with st.sidebar.expander(f"📖 Knowledge base ({len(documents)} chunks)"):
    for d in documents:
        st.write("• " + d)


# ------------------------------------------------------------------
# 4. RAG LOGIC
# ------------------------------------------------------------------
def retrieve(query: str, k: int):
    """Return the top-k (chunk, score) pairs by cosine similarity."""
    model = load_embedder()
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    scores = doc_embeddings @ q_emb  # normalized vectors -> dot product = cosine
    top_idx = np.argsort(-scores)[:k]
    return [(documents[i], float(scores[i])) for i in top_idx]


def generate_answer(query: str, chunks: list, key: str):
    """Call Gemini with the retrieved context. Returns (answer, model_used, errors)."""
    context = "\n".join(f"- {c}" for c, _ in chunks)
    system_instruction = (
        "You are a friendly campus study assistant. Answer using ONLY the provided "
        "context. Keep the answer short and clear. If the answer is not in the context, "
        f"reply exactly: '{NO_INFO_MSG}'"
    )
    prompt = f"Context:\n{context}\n\nQuestion: {query}"

    client = genai.Client(api_key=key)
    models = ([custom_model.strip()] if custom_model.strip() else []) + DEFAULT_MODELS
    errors = []

    for model_id in models:
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                ),
            )
            if response.text:
                return response.text, model_id, errors
            errors.append(f"{model_id}: empty response")
        except Exception as exc:  # try the next model
            errors.append(f"{model_id}: {exc}")

    return None, None, errors


# ------------------------------------------------------------------
# 5. CHAT UI
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    if sources:
        with st.expander("👁️ Retrieved context (open-book verification)"):
            for i, (chunk, score) in enumerate(sources, 1):
                st.info(f"**Chunk {i}** (similarity {score:.2f}): {chunk}")


# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources"))

question = st.chat_input("Ask about OS, Networks, Data Structures, or Campus Policy...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not api_key:
            answer, sources = "⚠️ Please enter your Gemini API key in the sidebar first.", []
            st.warning(answer)
        else:
            with st.spinner("Searching notes and thinking..."):
                results = retrieve(question, top_k)
                best_score = results[0][1]

                if best_score < threshold:
                    # Nothing relevant: skip the API call entirely
                    answer, sources = NO_INFO_MSG, results
                    st.markdown(answer)
                else:
                    text, used_model, errs = generate_answer(question, results, api_key)
                    if text is None:
                        answer = "❌ All models failed:\n\n" + "\n".join(f"- {e}" for e in errs)
                        sources = []
                        st.error(answer)
                    else:
                        answer, sources = text, results
                        st.markdown(answer)
                        st.caption(f"Model: `{used_model}`  |  Best match: {best_score:.2f}")
            render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
