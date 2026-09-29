"""Private LightRAG service called by the authenticated Node application."""

import hashlib
import json
import os
import asyncio
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from lightrag import QueryParam

from setup_rag import (
    CHUNK_OVERLAP_TOKEN_SIZE,
    CHUNK_TOKEN_SIZE,
    EMBEDDING_MODEL,
    LLM_MODEL,
    OLLAMA_NUM_PREDICT,
    OLLAMA_THINK,
    WORKING_DIR,
    create_rag,
    load_json_documents,
)

app = FastAPI(title="Factory Chat RAG")
_instances: dict[str, object] = {}
_instance_lock = asyncio.Lock()
_roles = {"employee", "manager", "director"}
_role_rank = {"employee": 0, "manager": 1, "director": 2}
_json_path = Path(os.getenv("RAG_JSON_PATH", str(Path(__file__).with_name("data.json"))))
if not _json_path.is_absolute():
    _json_path = Path(__file__).parent / _json_path


class Document(BaseModel):
    title: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    mode: str = "hybrid"
    role: str
    documents: list[Document] = Field(default_factory=list)


async def _rag_for_documents(documents: list[Document], role: str):
    # Each distinct authorized corpus has its own LightRAG store. A user's query
    # can therefore never retrieve documents omitted by the Node permission check.
    corpus = [(doc.title, doc.content) for doc in documents]
    if _json_path.exists():
        json_documents = load_json_documents(_json_path)
        corpus.extend(
            (doc["title"], doc["content"])
            for doc in json_documents
            if _role_rank[role] >= _role_rank[doc["min_role"]]
        )
    corpus = sorted(set(corpus))
    # Keep indexes created with different models or chunk settings separate.
    # This also avoids reusing the old corpus that failed when qwen2.5 was missing.
    index_config = {
        "corpus": corpus,
        "llm_model": LLM_MODEL,
        "ollama_think": OLLAMA_THINK,
        "ollama_num_predict": OLLAMA_NUM_PREDICT,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": 768,
        "chunk_token_size": CHUNK_TOKEN_SIZE,
        "chunk_overlap_token_size": CHUNK_OVERLAP_TOKEN_SIZE,
    }
    digest = hashlib.sha256(
        json.dumps(index_config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    key = digest
    async with _instance_lock:
        if key not in _instances:
            store = Path(WORKING_DIR) / "authorized" / digest
            rag = create_rag(store)
            await rag.initialize_storages()
            if corpus:
                await rag.ainsert(
                    [f"Tiêu đề: {title}\nNội dung: {content}" for title, content in corpus]
                )
            _instances[key] = rag
        return _instances[key]


@app.post("/api/chat")
async def chat_endpoint(
    request: ChatRequest, x_rag_token: str | None = Header(default=None)
):
    expected_token = os.getenv("RAG_API_TOKEN")
    if expected_token and x_rag_token != expected_token:
        raise HTTPException(status_code=401, detail="Không được phép truy cập RAG.")
    if request.role not in _roles:
        raise HTTPException(status_code=400, detail="Cấp quyền không hợp lệ.")
    if request.mode not in {"naive", "local", "global", "hybrid"}:
        raise HTTPException(status_code=400, detail="Chế độ truy vấn không hợp lệ.")
    if not request.documents and not _json_path.exists():
        return {
            "answer": "Chưa có tài liệu bạn được phép xem. Hãy liên hệ quản lý để chia sẻ tài liệu."
        }
    try:
        rag = await _rag_for_documents(request.documents, request.role)
        answer = await rag.aquery(
            request.question.strip(), param=QueryParam(mode=request.mode)
        )
        if not isinstance(answer, str) or not answer.strip():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"RAG không tạo được câu trả lời. Kiểm tra Ollama đang chạy và đã có "
                    f"model chat '{LLM_MODEL}' cùng model embedding '{EMBEDDING_MODEL}'."
                ),
            )
        return {"answer": answer.strip()}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Không thể lập chỉ mục hoặc truy vấn RAG: {error}. Kiểm tra Ollama "
                f"và hai model '{LLM_MODEL}', '{EMBEDDING_MODEL}'."
            ),
        ) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("RAG_PORT", "8000")))
