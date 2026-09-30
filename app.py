"""Streamlit demo UI cho Local Enterprise RAG."""

from __future__ import annotations

import streamlit as st

from localrag.config import load_settings
from localrag.ollama_client import PrivacyError, ensure_local_host
from localrag.pipeline import IndexMismatchError, IndexMissingError, RAGPipeline


st.set_page_config(page_title="Local Enterprise RAG", page_icon="🧠", layout="wide")


@st.cache_resource

def get_pipeline() -> RAGPipeline:
    return RAGPipeline(load_settings())


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.evidence = []


pipeline = get_pipeline()
settings = pipeline.settings

st.title("🧠 Local Enterprise RAG")
st.caption("Multi-format document assistant · Hybrid retrieval · Grounded citations · Local-first")

with st.sidebar:
    st.subheader("System")
    try:
        ensure_local_host(settings.ollama_host, settings.allow_remote_ollama)
        st.success("Local-only Ollama")
    except PrivacyError as exc:
        st.error(str(exc))
    st.write(f"LLM: `{settings.llm_model}`")
    st.write(f"Embedding: `{settings.embed_model}`")
    st.write(f"Parser: `{settings.parser}`")
    st.write(f"Rerank: `{settings.rerank}`")
    st.write(f"Vision: `{settings.vision_enabled}`")

    stats = pipeline.stats()
    st.divider()
    st.subheader("Knowledge base")
    st.metric("Documents", stats["files"])
    st.metric("Chunks", stats["chunks"])

    if st.button("🔄 Ingest / Sync", use_container_width=True):
        try:
            with st.status("Đang đồng bộ tài liệu…", expanded=True) as status:
                report = pipeline.ingest(progress=st.write)
                status.update(label="Hoàn tất", state="complete")
            st.success(
                f"Added {len(report.added)} · Updated {len(report.updated)} · "
                f"Removed {len(report.removed)} · {report.chunks_added} chunks"
            )
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

    if st.button("🧹 Clear chat", use_container_width=True):
        reset_chat()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "evidence" not in st.session_state:
    st.session_state.evidence = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Hỏi về tài liệu…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = [
        (m["content"], st.session_state.messages[i + 1]["content"])
        for i, m in enumerate(st.session_state.messages[:-1])
        if m["role"] == "user"
        and i + 1 < len(st.session_state.messages)
        and st.session_state.messages[i + 1]["role"] == "assistant"
    ][-4:]

    try:
        with st.chat_message("assistant"):
            with st.spinner("Đang truy xuất evidence…"):
                result = pipeline.answer(question, history=history)
            st.markdown(result.answer)
            if result.latency_ms is not None:
                st.caption(
                    f"⏱ {result.latency_ms:.0f} ms · retrieval {result.retrieval_ms:.0f} ms · "
                    f"generation {result.generation_ms:.0f} ms"
                )
            if result.sources:
                with st.expander("📚 Evidence / Sources", expanded=True):
                    for i, src in enumerate(result.sources, 1):
                        location = f" · p.{src.page}" if src.page else ""
                        scores = (
                            f"hybrid {src.score:.3f} · vector {src.vector_score:.3f} · "
                            f"lexical {(src.lexical_score or 0):.3f}"
                        )
                        st.markdown(
                            f"**[{i}] {src.source}{location}**  \n"
                            f"{src.heading or ''}  \n"
                            f"`scores: {scores}`"
                        )
                        st.code(src.text[:1500], language="text")
    except (IndexMissingError, IndexMismatchError) as exc:
        st.error(str(exc))
        st.info("Hãy nhấn **Ingest / Sync** ở sidebar trước.")
        result = None
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        result = None

    if result is not None:
        st.session_state.messages.append({"role": "assistant", "content": result.answer})
