import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from ragbench.bench import make_embedder
from ragbench.cli import main
from ragbench.retrievers import EmbeddingError, VoyageEmbedder

ROOT = Path(__file__).parents[1]


Handler = Callable[[httpx.Request], httpx.Response]


def voyage(handler: Handler) -> VoyageEmbedder:
    embedder = VoyageEmbedder("vk", "voyage-3.5")
    embedder._client = httpx.Client(transport=httpx.MockTransport(handler))
    return embedder


def test_voyage_batches_and_restores_order() -> None:
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sizes.append(len(body["input"]))
        assert body["input_type"] == "query"
        data = [{"index": i, "embedding": [float(i)]} for i in range(len(body["input"]))]
        return httpx.Response(200, json={"data": data[::-1]})

    vectors = voyage(handler).embed(["t"] * 130, "query")

    assert sizes == [128, 2]
    assert vectors[:2] == [[0.0], [1.0]]


@pytest.mark.parametrize(
    ("handler", "message"),
    [
        (lambda r: httpx.Response(401, json={}), "check VOYAGE_API_KEY"),
        (lambda r: httpx.Response(500, json={}), "HTTP 500"),
        (lambda r: (_ for _ in ()).throw(httpx.ConnectError("down")), "unreachable"),
    ],
)
def test_voyage_failures_are_clear_errors(handler: Handler, message: str) -> None:
    with pytest.raises(EmbeddingError, match=message):
        voyage(handler).embed(["t"], "document")


def test_voyage_needs_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        make_embedder("voyage")
    monkeypatch.setenv("VOYAGE_API_KEY", "vk")
    assert make_embedder("voyage:voyage-3-lite").name == "voyage:voyage-3-lite"
    with pytest.raises(ValueError, match="unknown embedder"):
        make_embedder("nope")


def test_the_cli_reports_an_embedding_failure_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing(self, texts, kind):  # type: ignore[no-untyped-def]
        raise EmbeddingError("Voyage API returned HTTP 401 (check VOYAGE_API_KEY)")

    monkeypatch.setenv("VOYAGE_API_KEY", "bad")
    monkeypatch.setattr(VoyageEmbedder, "embed", failing)
    config = tmp_path / "bench.yaml"
    config.write_text(
        f"dataset: {ROOT / 'data/help-center'}\nchunkers: [{{name: d, type: document}}]\n"
        "retrievers: [dense:voyage]\n"
    )

    assert main(["run", str(config)]) == 2
    assert "check VOYAGE_API_KEY" in capsys.readouterr().err
