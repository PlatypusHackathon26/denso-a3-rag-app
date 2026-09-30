import pytest

from localrag.ollama_client import (
    OllamaEmbedder,
    OllamaLLM,
    PrivacyError,
    ThinkFilter,
    ensure_local_host,
    strip_think,
    _model_present,
)


@pytest.mark.parametrize("host", ["http://127.0.0.1:11434", "localhost:11434", "http://[::1]:11434"])
def test_local_hosts_allowed(host):
    ensure_local_host(host)


@pytest.mark.parametrize("host", ["http://192.168.1.10:11434", "https://ollama.example.com", "http://0.0.0.0:11434"])
def test_remote_hosts_blocked_unless_allowed(host):
    with pytest.raises(PrivacyError):
        ensure_local_host(host)
    ensure_local_host(host, allow_remote=True)


def test_strip_think_removes_block():
    assert strip_think("<think>nghĩ ngợi\nnhiều dòng</think>\n\nĐáp án là 42") == "Đáp án là 42"


def test_strip_think_drops_unterminated_block():
    assert strip_think("Trước <think>bị cắt giữa chừng") == "Trước"


def test_think_filter_handles_tags_split_across_chunks():
    f = ThinkFilter()
    parts = ["Xin ", "<thi", "nk>bí mật</th", "ink> chào", " bạn"]
    out = "".join(f.feed(p) for p in parts) + f.flush()
    assert out == "Xin  chào bạn"


def test_think_filter_keeps_lone_angle_bracket():
    f = ThinkFilter()
    out = f.feed("a < b và c <") + f.feed(" d") + f.flush()
    assert out == "a < b và c < d"


def test_model_matching():
    installed = ["qwen3:8b", "bge-m3:latest"]
    assert _model_present("qwen3:8b", installed)
    assert _model_present("bge-m3", installed)
    assert not _model_present("qwen3:14b", installed)
    assert not _model_present("qwen3", ["qwen3-embedding:0.6b"])


class FakeClient:
    def __init__(self):
        self.embed_inputs = []
        self.chat_kwargs = []

    def embed(self, model, input):
        self.embed_inputs.append(list(input))
        return {"embeddings": [[float(len(t)), 1.0] for t in input]}

    def chat(self, **kwargs):
        self.chat_kwargs.append(kwargs)
        if kwargs["stream"]:
            return iter([{"message": {"content": c}} for c in ["<think>x</think>", "\n", "Chào", " bạn"]])
        return {"message": {"content": "<think>x</think>\nChào bạn"}}


def test_embedder_batches_and_applies_prefixes():
    client = FakeClient()
    emb = OllamaEmbedder(client, "m", query_prefix="q: ", doc_prefix="d: ", batch_size=2)
    vectors = emb.embed_documents(["a", "b", "c"])
    assert len(vectors) == 3
    assert client.embed_inputs == [["d: a", "d: b"], ["d: c"]]
    emb.embed_query("hỏi")
    assert client.embed_inputs[-1] == ["q: hỏi"]


def test_llm_sets_num_ctx_and_disables_think_and_strips_output():
    client = FakeClient()
    llm = OllamaLLM(client, "qwen3:8b", temperature=0.2, num_ctx=4096, think=False)
    assert llm.generate("sys", "user") == "Chào bạn"
    kw = client.chat_kwargs[0]
    assert kw["options"] == {"temperature": 0.2, "num_ctx": 4096}
    assert kw["think"] is False
    assert "".join(llm.stream("sys", "user")) == "Chào bạn"


def test_llm_retries_without_think_when_unsupported():
    class OldClient(FakeClient):
        def chat(self, **kwargs):
            if "think" in kwargs:
                raise TypeError("chat() got an unexpected keyword argument 'think'")
            return super().chat(**kwargs)

    llm = OllamaLLM(OldClient(), "m", think=False)
    assert llm.generate("s", "u") == "Chào bạn"
