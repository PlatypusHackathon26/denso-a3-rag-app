"""Kết nối Ollama cục bộ: embedding + sinh văn bản + vision, kèm chốt chặn quyền riêng tư."""

from __future__ import annotations

import base64
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import httpx
import ollama

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


class OllamaError(RuntimeError):
    """Lỗi giao tiếp Ollama, đã được diễn giải thành thông báo dễ hành động."""


class PrivacyError(RuntimeError):
    """Cấu hình sẽ khiến dữ liệu rời khỏi máy."""


def ensure_local_host(host: str, allow_remote: bool = False) -> None:
    parsed = urlparse(host if "://" in host else f"http://{host}")
    hostname = (parsed.hostname or "").lower()
    if hostname in _LOOPBACK_HOSTS or allow_remote:
        return
    raise PrivacyError(
        f"OLLAMA_HOST='{host}' không phải máy cục bộ: prompt và nội dung tài liệu sẽ được "
        "gửi qua mạng. Nếu đó là server nội bộ bạn tin cậy, đặt ALLOW_REMOTE_OLLAMA=true."
    )


def make_client(host: str, allow_remote: bool = False) -> ollama.Client:
    ensure_local_host(host, allow_remote)
    return ollama.Client(host=host)


@contextmanager
def _translate_errors(model: str, host: str = ""):
    try:
        yield
    except OllamaError:
        raise
    except (ConnectionError, httpx.HTTPError) as exc:
        raise OllamaError(
            f"Không kết nối được Ollama{f' tại {host}' if host else ''}. "
            "Hãy mở Ollama (chạy `ollama serve`) rồi thử lại."
        ) from exc
    except ollama.ResponseError as exc:
        if exc.status_code == 404:
            raise OllamaError(
                f"Model '{model}' chưa có trên máy. Chạy: ollama pull {model}"
            ) from exc
        raise OllamaError(f"Ollama báo lỗi ({exc.status_code}): {exc.error}") from exc


def _model_present(wanted: str, installed: list[str]) -> bool:
    if wanted in installed:
        return True
    if ":" not in wanted:
        return any(name == f"{wanted}:latest" or name.startswith(f"{wanted}:") for name in installed)
    return False


def missing_models(client: ollama.Client, wanted: list[str], host: str = "") -> list[str]:
    with _translate_errors("", host):
        response = client.list()
    installed = [getattr(m, "model", None) or m["model"] for m in response["models"]]
    return [w for w in dict.fromkeys(wanted) if w and not _model_present(w, installed)]


class OllamaEmbedder:
    def __init__(
        self,
        client: ollama.Client,
        model: str,
        query_prefix: str = "",
        doc_prefix: str = "",
        batch_size: int = 32,
        host: str = "",
    ):
        self._client = client
        self.model = model
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix
        self._batch_size = batch_size
        self._host = host

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        with _translate_errors(self.model, self._host):
            response = self._client.embed(model=self.model, input=inputs)
        vectors = [list(v) for v in response["embeddings"]]
        if len(vectors) != len(inputs):
            raise OllamaError(
                f"Embedding trả về {len(vectors)} vector cho {len(inputs)} đoạn văn bản."
            )
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = [self._doc_prefix + t for t in texts[i : i + self._batch_size]]
            vectors.extend(self._embed(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self._query_prefix + text])[0]


class ThinkFilter:
    """Loại khối <think>...</think> khỏi luồng token."""

    _OPEN, _CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    @staticmethod
    def _partial_suffix(buf: str, tag: str) -> int:
        for k in range(min(len(tag) - 1, len(buf)), 0, -1):
            if buf.endswith(tag[:k]):
                return k
        return 0

    def feed(self, text: str) -> str:
        self._buf += text
        out: list[str] = []
        while True:
            if self._in_think:
                end = self._buf.find(self._CLOSE)
                if end == -1:
                    keep = self._partial_suffix(self._buf, self._CLOSE)
                    self._buf = self._buf[len(self._buf) - keep :] if keep else ""
                    break
                self._buf = self._buf[end + len(self._CLOSE) :]
                self._in_think = False
            else:
                start = self._buf.find(self._OPEN)
                if start == -1:
                    keep = self._partial_suffix(self._buf, self._OPEN)
                    cut = len(self._buf) - keep
                    out.append(self._buf[:cut])
                    self._buf = self._buf[cut:]
                    break
                out.append(self._buf[:start])
                self._buf = self._buf[start + len(self._OPEN) :]
                self._in_think = True
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self._in_think else self._buf
        self._buf = ""
        return rest


def strip_think(text: str) -> str:
    f = ThinkFilter()
    return (f.feed(text) + f.flush()).strip()


class OllamaLLM:
    def __init__(
        self,
        client: ollama.Client,
        model: str,
        temperature: float = 0.1,
        num_ctx: int = 8192,
        think: bool = False,
        host: str = "",
    ):
        self._client = client
        self.model = model
        self._temperature = temperature
        self._num_ctx = num_ctx
        self._think = think
        self._host = host

    def _chat(self, system: str, user: str, stream: bool, images: list[str] | None = None):
        message = {"role": "user", "content": user}
        if images:
            message["images"] = images
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, message],
            "stream": stream,
            "options": {"temperature": self._temperature, "num_ctx": self._num_ctx},
        }
        if not self._think:
            kwargs["think"] = False
        try:
            return self._client.chat(**kwargs)
        except (TypeError, ollama.ResponseError) as exc:
            if "think" in kwargs and "think" in str(exc).lower():
                kwargs.pop("think")
                return self._client.chat(**kwargs)
            raise

    def generate(self, system: str, user: str) -> str:
        with _translate_errors(self.model, self._host):
            response = self._chat(system, user, stream=False)
            text = response["message"]["content"] or ""
        return strip_think(text)

    def stream(self, system: str, user: str) -> Iterator[str]:
        think_filter = ThinkFilter()
        started = False
        with _translate_errors(self.model, self._host):
            for part in self._chat(system, user, stream=True):
                piece = think_filter.feed(part["message"]["content"] or "")
                if not started:
                    piece = piece.lstrip()
                if piece:
                    started = True
                    yield piece
            tail = think_filter.flush()
            if not started:
                tail = tail.lstrip()
            if tail:
                yield tail

    def vision_extract(self, image_path: Path, prompt: str) -> str:
        """Đọc ảnh tại máy và chuyển nội dung quan sát được thành văn bản có cấu trúc."""
        payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
        system = (
            "You are a document understanding component. Extract only what is visible in the image. "
            "Do not invent values. Preserve exact numbers, identifiers, labels, and table cells when readable."
        )
        with _translate_errors(self.model, self._host):
            response = self._chat(system, prompt, stream=False, images=[payload])
            text = response["message"]["content"] or ""
        return strip_think(text)
