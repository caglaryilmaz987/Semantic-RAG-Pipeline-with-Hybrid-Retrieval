"""
app.py — Streamlit User Interface & RAG Application
Semantic RAG Pipeline with Hybrid Retrieval
(Dense + BM25 + Reciprocal Rank Fusion + Cross-Encoder Reranker)

Execution Instruction:
    streamlit run app.py
"""

import os
import warnings

# Suppress Python and HuggingFace warning messages
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import streamlit as st

from embedder import Embedder
from chat_model import ChatModel
from vector_store import VectorStore
from bm25_index import BM25Index
from reranker import Reranker
from retriever import HybridRetriever


# PAGE CONFIGURATION AND STYLING (CUSTOM CSS)

st.set_page_config(
    page_title="Semantic RAG Assistant | Hybrid Retrieval",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #8892b0;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .source-badge-dense {
        background-color: #1e3a8a;
        color: #93c5fd;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 5px;
    }
    .source-badge-bm25 {
        background-color: #065f46;
        color: #6ee7b7;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 5px;
    }
    .source-badge-score {
        background-color: #374151;
        color: #f3f4f6;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-family: monospace;
    }
    
    /* Animated Loading Spinner Ring */
    .loading-ring {
        display: inline-block;
        width: 18px;
        height: 18px;
        border: 3px solid rgba(79, 172, 254, 0.2);
        border-radius: 50%;
        border-top-color: #00f2fe;
        animation: spin 0.8s ease-in-out infinite;
        vertical-align: middle;
        margin-right: 8px;
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    .status-generating {
        color: #00f2fe;
        font-size: 0.88rem;
        font-weight: 500;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# RAG PIPELINE & RESOURCE MANAGEMENT

class RAGPipelineManager:
    """Caches and manages AI models and vector stores across Streamlit user sessions."""

    @st.cache_resource(show_spinner="🧠 Loading AI models and vector database...")
    def build_pipeline(_self):
        embedder = Embedder()
        chat = ChatModel()
        vector_store = VectorStore(embedding_dim=embedder.dimension)
        
        bm25 = BM25Index()
        bm25_path = "bm25_index.pkl"
        if not os.path.exists(bm25_path):
            st.error("❌ 'bm25_index.pkl' not found. Please run `python ingest_json.py` first.")
            st.stop()
        bm25.load(bm25_path)

        reranker = Reranker()
        retriever = HybridRetriever(
            emb_client=embedder.client,
            vector_store=vector_store,
            bm25_index=bm25,
            reranker=reranker,
        )
        return retriever, chat, vector_store.count()

    def __init__(self):
        try:
            self._retriever, self._chat, self.doc_count = self.build_pipeline()
        except Exception as e:
            st.error(f"❌ Initialization error: {e}")
            st.stop()

    def retrieve(self, query: str, top_k: int, use_reranker: bool, rrf_k: int) -> tuple[list[dict], float]:
        t0 = time.time()
        results = self._retriever.retrieve(
            query=query,
            top_k=top_k,
            use_reranker=use_reranker,
            rrf_k_override=rrf_k,
        )
        t_elapsed = (time.time() - t0) * 1000  # ms
        return results, t_elapsed

    def is_relevant(self, kaynaklar: list[dict], use_reranker: bool) -> bool:
        """
        Retrieval Guard (Confidence Threshold Gate):
        Rejects out-of-context queries instantly (0.00s) if retrieved document confidence is low.
        """
        if not kaynaklar:
            return False
        
        top_doc = kaynaklar[0]
        if use_reranker:
            rerank_score = top_doc.get("rerank_skor", -10.0)
            return rerank_score >= 0.10
        else:
            rrf_score = top_doc.get("rrf_skor", 0.0)
            return rrf_score >= 0.015

    def generate_stream(self, query: str, kaynaklar: list[dict]):
        """
        Generates real-time token streaming with strict grounding prompt instructions.
        """
        baglamlar = "\n\n---\n\n".join(k["metin"] for k in kaynaklar)

        sistem_promtu = (
            "You are a strict corporate AI assistant. Answer the user query using ONLY the provided CONTEXT.\n"
            "If the answer is NOT in CONTEXT, reply EXACTLY: 'Bu bilgi kurumsal veritabanında bulunmamaktadır.'\n"
            "Be direct, concise, and respond in clear Turkish. Do not add meta commentary.\n\n"
            f"CONTEXT:\n{baglamlar}"
        )

        messages = [
            {"role": "system", "content": sistem_promtu},
            {"role": "user", "content": query},
        ]

        if hasattr(self._chat, "complete_stream"):
            for token in self._chat.complete_stream(messages):
                if "Kullanıcı Sorusu:" in token or "Asistanın Cevabı:" in token:
                    break
                yield token
        else:
            resp = self._chat.complete(messages)
            yield resp


# MAIN APPLICATION UI

def main():
    pipeline = RAGPipelineManager()

    # SIDEBAR CONTROLS
    with st.sidebar:
        st.title("⚙️ Search Settings")
        st.markdown("---")

        top_k = st.slider("📌 Document Count (Top-K)", min_value=1, max_value=10, value=2)
        rrf_k = st.slider("🔀 RRF Constant (k)", min_value=10, max_value=100, value=60, step=5)
        use_reranker = st.toggle("🎯 Cross-Encoder Reranker", value=True)

        if st.button("🧹 Clear Cache & Reload"):
            st.cache_resource.clear()
            st.rerun()

        st.markdown("---")
        st.subheader("📊 System Specs")
        st.markdown(f"**Database Records:** `{pipeline.doc_count:,}` chunks")
        st.markdown("**Vector Store:** `SQLite (vector_store.db)`")
        st.markdown("**Dense Model:** `Qwen3-Embedding-0.6b`")
        st.markdown("**Sparse Search:** `BM25Okapi`")
        st.markdown("**Reranker Model:** `BAAI/bge-reranker-v2-m3` (CUDA: GPU)")
        st.markdown("**LLM Engine:** `Phi-3.5-mini`")

        st.markdown("---")
        st.subheader("💡 Sample Queries")
        ornek_sorular = [
            "Zitvatorok Antlaşması hangi devletler arasında imzalanmıştır?",
            "Kara Harp Okulu brövesinde zemin renginin kırmızı olması neyi simgeler?",
            "Hangi bitkiler sakinleştirici etki yapar?",
        ]
        selected_sample = None
        for q in ornek_sorular:
            if st.button(f"👉 {q}", use_container_width=True):
                selected_sample = q

    # MAIN HEADER
    col_title, col_logo = st.columns([4, 1])
    with col_title:
        st.markdown('<div class="main-header">Semantic RAG Pipeline 🔍</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Dense Embedding · BM25 Sparse · Reciprocal Rank Fusion · Cross-Encoder Reranker</div>',
            unsafe_allow_html=True,
        )

    # CHAT HISTORY SESSION STATE
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # RENDER PAST CHAT MESSAGES
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "stats" in msg:
                st.caption(msg["stats"])

    # USER PROMPT INPUT
    prompt = st.chat_input("Ask a question to the corporate database...")
    if selected_sample:
        prompt = selected_sample

    if prompt:
        # Render user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            # Stage 1: Hybrid Retrieval
            with st.spinner("🔍 Executing hybrid retrieval (Dense + BM25 + RRF + Rerank)..."):
                kaynaklar, retrieve_time = pipeline.retrieve(
                    query=prompt,
                    top_k=top_k,
                    use_reranker=use_reranker,
                    rrf_k=rrf_k,
                )

            # Render Source Chunks
            with st.expander(f"📚 {len(kaynaklar)} Retrieved Source Chunks ({retrieve_time:.1f} ms)", expanded=False):
                for i, k in enumerate(kaynaklar):
                    badge_str = ""
                    if "dense" in k["kaynaklar"]:
                        badge_str += '<span class="source-badge-dense">DENSE</span>'
                    if "bm25" in k["kaynaklar"]:
                        badge_str += '<span class="source-badge-bm25">BM25</span>'

                    rrf_score = k.get("rrf_skor", 0.0)
                    rerank_score = k.get("rerank_skor", 0.0)
                    score_str = f'<span class="source-badge-score">RRF: {rrf_score:.4f} | Rerank: {rerank_score:.2f}</span>'

                    st.markdown(
                        f"**[{i+1}] Document #{k['id']}** {badge_str} {score_str}\n\n"
                        f"> {k['metin']}",
                        unsafe_allow_html=True,
                    )
                    if i < len(kaynaklar) - 1:
                        st.divider()

            # Stage 2: Retrieval Guard & Real-Time Streaming Generation
            is_rel = pipeline.is_relevant(kaynaklar, use_reranker)

            if not is_rel:
                # Reject out-of-context queries instantly (0.00s)
                yanit = "Bu bilgi kurumsal veritabanında bulunmamaktadır."
                st.markdown(yanit)
                gen_time = 0.0
            else:
                # Call LLM with animated loading indicator for relevant queries
                status_placeholder = st.empty()
                status_placeholder.markdown(
                    '<div class="status-generating"><span class="loading-ring"></span>🤖 Generating response, please wait...</div>',
                    unsafe_allow_html=True,
                )

                t_gen_start = time.time()
                yanit = st.write_stream(pipeline.generate_stream(prompt, kaynaklar))
                gen_time = time.time() - t_gen_start

                status_placeholder.empty()

            stats_info = f"⚡ Search Time: {retrieve_time:.1f} ms | LLM Generation Time: {gen_time:.2f} sec"
            st.caption(stats_info)

            # Save assistant message to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": yanit,
                "stats": stats_info,
            })


if __name__ == "__main__":
    main()