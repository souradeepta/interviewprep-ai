"""Offline representative checks for notebooks with external dependencies."""

import ast
import json
import sys
import types
from contextlib import nullcontext
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def code_cell(path: str, index: int) -> str:
    notebook = json.loads((ROOT / path).read_text(encoding="utf-8"))
    cell = notebook["cells"][index]
    assert cell["cell_type"] == "code"
    return "".join(cell["source"])


def test_provider_configuration_runs_with_stub_without_network(monkeypatch):
    calls = []
    anthropic = types.ModuleType("anthropic")
    anthropic.Anthropic = lambda *args, **kwargs: calls.append((args, kwargs))
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda: None
    monkeypatch.setitem(sys.modules, "anthropic", anthropic)
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    namespace = {}
    exec(compile(code_cell("agentic-ai/notebooks/01-what-is-an-agent.ipynb", 1), "provider_setup", "exec"), namespace)

    assert namespace["ANTHROPIC_API_KEY"] == ""
    assert calls == []  # setup code must not instantiate a live client

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    configured = {}
    exec(compile(code_cell("agentic-ai/notebooks/01-what-is-an-agent.ipynb", 1), "provider_setup_configured", "exec"), configured)
    assert configured["ANTHROPIC_API_KEY"] == "test-key"
    assert calls == []


def test_rag_retriever_logic_runs_on_local_toy_corpus():
    class Encoded(list):
        @property
        def shape(self):
            return (len(self), len(self[0]) if self else 0)

    class FakeModel:
        def __init__(self, _name):
            pass

        def encode(self, texts, convert_to_tensor=True):
            if isinstance(texts, str):
                return texts.lower().split()
            return Encoded([doc.lower().split() for doc in texts])

    class ScoreRow:
        def __init__(self, values):
            self.values = values

        def topk(self, count):
            indices = sorted(range(len(self.values)), key=self.values.__getitem__, reverse=True)[:count]
            return types.SimpleNamespace(indices=IndexList(indices))

        def __getitem__(self, index):
            return ScoreValue(self.values[index])

    class Matrix:
        def __init__(self, values):
            self.row = ScoreRow(values)

        def __getitem__(self, _index):
            return self.row

    class IndexList(list):
        def cpu(self):
            return self

        def numpy(self):
            return self

    class ScoreValue(float):
        def item(self):
            return float(self)

    class Similarity:
        @staticmethod
        def pytorch_cos_sim(query, docs):
            qset = set(query)
            return Matrix([len(qset & set(doc)) / max(len(qset), 1) for doc in docs])

    source = code_cell("papers/retrieval/notebooks/01-rag.ipynb", 4)
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SimpleRetriever")
    namespace = {"SentenceTransformer": FakeModel, "util": Similarity}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "simple_retriever", "exec"), namespace)
    retriever = namespace["SimpleRetriever"]("local-stub")
    retriever.index_documents(["France capital Paris", "Japan capital Tokyo"])
    docs, scores = retriever.retrieve("Paris France", top_k=1)

    assert docs == ["France capital Paris"]
    assert scores == [1.0]
    assert retriever.retrieve("unknown", top_k=1)[0] == ["France capital Paris"]


def test_codet5_embedding_wrapper_uses_stubbed_model_without_download():
    calls = []

    class FakeTensor:
        def __init__(self):
            self.slices = []
            self.cpu_called = False

        def to(self, _device):
            return self

        def __getitem__(self, key):
            self.slices.append(key)
            return self

        def cpu(self):
            self.cpu_called = True
            return self

        def numpy(self):
            return [[0.1, 0.2, 0.3]]

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_name):
            calls.append(("tokenizer", model_name))
            return cls()

        def __call__(self, *_args, **_kwargs):
            return {"input_ids": FakeTensor()}

    tensor = FakeTensor()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_name):
            calls.append(("model", model_name))
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return None

        def __call__(self, **_kwargs):
            return types.SimpleNamespace(last_hidden_state=tensor)

    source = code_cell("papers/code-systems/notebooks/01-codet5.ipynb", 3)
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_code_embedding")
    fake_torch = types.SimpleNamespace(no_grad=nullcontext)
    namespace = {
        "np": types.SimpleNamespace(ndarray=object),
        "AutoTokenizer": FakeTokenizer,
        "AutoModel": FakeModel,
        "torch": fake_torch,
        "device": "cpu",
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "codet5_wrapper", "exec"), namespace)

    vector = namespace["get_code_embedding"]("def add(a, b): return a + b", model_name="tiny-stub")

    assert vector == [0.1, 0.2, 0.3]
    assert calls == [("tokenizer", "tiny-stub"), ("model", "tiny-stub")]
    assert tensor.slices == [(slice(None), 0, slice(None))]
    assert tensor.cpu_called


def test_web_agent_http_call_is_mocked(monkeypatch):
    calls = []
    response = types.SimpleNamespace(text="fixture page")
    requests = types.ModuleType("requests")
    requests.get = lambda url: (calls.append(url), response)[1]
    monkeypatch.setitem(sys.modules, "requests", requests)
    namespace = {}
    exec(compile(code_cell("agentic-ai/notebooks/51-web-agents.ipynb", 1), "web_agent", "exec"), namespace)

    assert calls == ["https://example.com"]
    assert namespace["html"] == "fixture page"


def test_web_agent_timeout_retry_and_backoff_are_mocked(monkeypatch):
    calls = []
    sleeps = []
    attempts = iter([TimeoutError("temporary"), TimeoutError("temporary"), types.SimpleNamespace(json=lambda: {"ok": True})])
    requests = types.ModuleType("requests")

    def get(url, timeout=None):
        calls.append((url, timeout))
        outcome = next(attempts)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    requests.get = get
    fake_time = types.ModuleType("time")
    fake_time.sleep = sleeps.append
    monkeypatch.setitem(sys.modules, "requests", requests)
    monkeypatch.setitem(sys.modules, "time", fake_time)
    namespace = {"time": fake_time, "requests": requests}
    source = code_cell("agentic-ai/notebooks/51-web-agents.ipynb", 7)
    exec(compile(source, "web_agent_retry", "exec"), namespace)

    result = namespace["ErrorHandlingAgent"]().fetch_safe("https://fixture.test", retries=3)

    assert result == {"ok": True}
    assert calls == [("https://fixture.test", 5)] * 3
    assert sleeps == [1, 2]
