"""Shared LightRAG configuration and optional JSON bootstrap."""

import json
import os
import asyncio
from pathlib import Path

from lightrag import LightRAG
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc

WORKING_DIR = Path(
    os.getenv("LIGHTRAG_WORKING_DIR", str(Path(__file__).with_name("lightrag_local_storage")))
)
LLM_MODEL = os.getenv("LIGHTRAG_LLM_MODEL", "gemma4:26b")
EMBEDDING_MODEL = os.getenv("LIGHTRAG_EMBEDDING_MODEL", "nomic-embed-text")
CHUNK_TOKEN_SIZE = int(os.getenv("LIGHTRAG_CHUNK_TOKEN_SIZE", "1200"))
CHUNK_OVERLAP_TOKEN_SIZE = int(os.getenv("LIGHTRAG_CHUNK_OVERLAP_TOKEN_SIZE", "100"))
OLLAMA_THINK = os.getenv("LIGHTRAG_OLLAMA_THINK", "false").lower() in {"1", "true", "yes"}
OLLAMA_NUM_PREDICT = int(os.getenv("LIGHTRAG_OLLAMA_NUM_PREDICT", "4096"))


def create_rag(working_dir: str | Path = WORKING_DIR) -> LightRAG:
    """Create a LightRAG instance using the same models for indexing and chat."""
    directory = Path(working_dir)
    directory.mkdir(parents=True, exist_ok=True)

    async def embed_texts(texts: list[str], embedding_dim: int) -> object:
        # ollama_embed is itself decorated with a 1024-dimension default.
        # Call its underlying implementation and pass Nomic's real 768 dimension.
        return await ollama_embed.func(
            texts, embed_model=EMBEDDING_MODEL, embedding_dim=embedding_dim
        )

    return LightRAG(
        working_dir=str(directory),
        llm_model_func=ollama_model_complete,
        llm_model_name=LLM_MODEL,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=embed_texts,
            send_dimensions=True,
        ),
        chunk_token_size=CHUNK_TOKEN_SIZE,
        chunk_overlap_token_size=CHUNK_OVERLAP_TOKEN_SIZE,
        llm_model_kwargs={
            "options": {"think": OLLAMA_THINK, "num_predict": OLLAMA_NUM_PREDICT}
        },
    )


def load_json_documents(json_file_path: str | Path) -> list[dict[str, str]]:
    """Read the current extraction JSON format into searchable document records."""
    with open(json_file_path, "r", encoding="utf-8") as file:
        raw_data = json.load(file)
    if isinstance(raw_data, dict):
        raw_data = raw_data.get("documents", [raw_data])
    if not isinstance(raw_data, list):
        raise ValueError("JSON phải là object tài liệu hoặc danh sách các object.")

    documents: list[dict[str, str]] = []
    for item in raw_data:
        if not isinstance(item, dict):
            continue

        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        title = (
            item.get("title")
            or metadata.get("source_file")
            or item.get("doc_id")
            or "Tài liệu không có tiêu đề"
        )
        sections = [f"Tiêu đề: {title}"]
        if item.get("doc_id"):
            sections.append(f"Mã tài liệu: {item['doc_id']}")
        for key in ("factory_code", "doc_type", "language", "access_level"):
            if metadata.get(key) is not None:
                sections.append(f"{key}: {metadata[key]}")

        # Support the current extraction format and the older title/content shape.
        text = item.get("cleaned_text") or item.get("content") or ""
        if text:
            sections.append(f"Nội dung:\n{text}")

        for table in item.get("tables") or []:
            if not isinstance(table, dict):
                continue
            caption = table.get("caption")
            markdown = table.get("markdown")
            if caption or markdown:
                sections.append("Bảng: " + "\n".join(part for part in (caption, markdown) if part))

        for image in item.get("images") or []:
            if not isinstance(image, dict):
                continue
            description = " ".join(
                str(value)
                for value in (
                    image.get("native_caption"),
                    image.get("vlm_caption"),
                    image.get("context_text"),
                )
                if value
            )
            if description:
                sections.append(f"Hình ảnh (trang {image.get('page', '?')}): {description}")

        if len(sections) > 1:
            access_level = metadata.get("access_level", 1)
            # JSON access levels 1/2/3 correspond to employee/manager/director.
            # Named role values are accepted as well; unknown values default to employee.
            level_to_role = {1: "employee", 2: "manager", 3: "director"}
            try:
                min_role = level_to_role.get(int(access_level), "employee")
            except (TypeError, ValueError):
                min_role = str(access_level).lower()
                if min_role not in {"employee", "manager", "director"}:
                    min_role = "employee"
            documents.append({
                "title": str(title),
                "content": "\n\n".join(sections),
                "min_role": min_role,
            })

    if not documents:
        raise ValueError("Không tìm thấy nội dung tài liệu có thể lập chỉ mục trong JSON.")
    return documents


def load_and_convert_json(json_file_path: str | Path) -> list[str]:
    """Compatibility helper for bootstrapping all JSON records into one LightRAG store."""
    return [doc["content"] for doc in load_json_documents(json_file_path)]


if __name__ == "__main__":
    data_file = Path("data.json")
    if not data_file.exists():
        raise SystemExit("Không tìm thấy data.json trong thư mục dự án.")
    documents = load_and_convert_json(data_file)

    async def ingest() -> None:
        rag = create_rag()
        await rag.initialize_storages()
        print(f"Đang nạp {len(documents)} tài liệu vào LightRAG...")
        await rag.ainsert(documents)
        await rag.finalize_storages()
        print("Đã nạp tài liệu thành công.")

    asyncio.run(ingest())
