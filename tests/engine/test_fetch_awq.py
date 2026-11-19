import hashlib
import io
import os

import pytest

from ironmule_product import compare_tokenizer
from lab.portable_gpu import fetch_awq


class Response(io.BytesIO):
    def __init__(self, body, status=200, content_range=None):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Range": content_range} if content_range else {}


@pytest.fixture
def checkpoint(monkeypatch):
    data = {"model-00001.safetensors": b"quantized-weights", "config.json": b"config"}
    manifest = {name: (len(body), hashlib.sha256(body).hexdigest()) for name, body in data.items()}
    monkeypatch.setattr(fetch_awq, "FILES", manifest)
    return data


def test_resume_fetch_and_verify_changes(tmp_path, monkeypatch, checkpoint):
    shard = "model-00001.safetensors"
    weights = checkpoint[shard]
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / f".{shard}.incomplete").write_bytes(weights[:5])

    def serve(request, timeout):
        assert fetch_awq.REVISION in request.full_url
        assert not request.has_header("Authorization")
        if shard in request.full_url:
            assert request.get_header("Range") == "bytes=5-"
            return Response(weights[5:], 206, f"bytes 5-{len(weights)-1}/{len(weights)}")
        return Response(checkpoint["config.json"])

    monkeypatch.setattr(fetch_awq.urllib.request, "urlopen", serve)
    assert fetch_awq.fetch(folder) == folder
    receipt = fetch_awq.verify(folder)
    assert receipt["files"][shard]["sha256"] == hashlib.sha256(weights).hexdigest()
    assert receipt["shard_bytes"] == len(weights)
    assert receipt["total_bytes"] == sum(map(len, checkpoint.values()))
    assert fetch_awq.unchanged(folder, receipt)
    monkeypatch.setattr(fetch_awq.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("unnecessary HTTP request"))
    assert fetch_awq.fetch(folder) == folder
    assert fetch_awq.unchanged(folder, receipt)
    original = (folder / shard).stat()
    (folder / shard).write_bytes(b"x" * len(weights))
    os.utime(folder / shard, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert not fetch_awq.unchanged(folder, receipt)
    with pytest.raises(ValueError, match="SHA-256"):
        fetch_awq.verify(folder)
    with pytest.raises(ValueError, match="SHA-256"):
        fetch_awq.fetch(folder)


def test_missing_unknown_links_and_untrusted_receipts(tmp_path, checkpoint):
    for name, data in checkpoint.items():
        (tmp_path / name).write_bytes(data)
    receipt = fetch_awq.verify(tmp_path)
    assert not fetch_awq.unchanged(tmp_path, dict(receipt, manifest_sha256="0" * 64))
    extra = tmp_path / "adapter_config.json"
    extra.write_text("{}")
    assert not fetch_awq.unchanged(tmp_path, receipt)
    with pytest.raises(ValueError, match="unexpected"):
        fetch_awq.verify(tmp_path)
    extra.unlink()
    config = tmp_path / "config.json"
    config.unlink()
    assert not fetch_awq.unchanged(tmp_path, receipt)
    with pytest.raises(ValueError, match="missing"):
        fetch_awq.verify(tmp_path)
    config.symlink_to(tmp_path / "model-00001.safetensors")
    with pytest.raises(ValueError, match="regular file"):
        fetch_awq.verify(tmp_path)


@pytest.mark.parametrize("mode", ["short", "wrong-digest", "wrong-range", "ignore-range"])
def test_incomplete_transfers_never_publish_bad_file(tmp_path, monkeypatch, mode):
    content = b"0123456789"
    name = "model.safetensors"
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(fetch_awq, "FILES", {name: (len(content), digest)})
    calls = []

    def serve(request, timeout):
        calls.append(request.get_header("Range"))
        if len(calls) == 1:
            return Response(content[:3])
        assert request.get_header("Range") == "bytes=3-"
        if mode == "short":
            return Response(b"", 206, "bytes 3-9/10")
        if mode == "wrong-digest":
            return Response(b"X" * 7, 206, "bytes 3-9/10")
        if mode == "wrong-range":
            return Response(content[3:], 206, "bytes 0-6/10")
        return Response(content)

    monkeypatch.setattr(fetch_awq.urllib.request, "urlopen", serve)
    if mode == "ignore-range":
        assert fetch_awq.fetch(tmp_path) == tmp_path
        assert fetch_awq.verify(tmp_path)["files"][name]["sha256"] == digest
    else:
        with pytest.raises((ValueError, OSError)):
            fetch_awq.fetch(tmp_path)
        assert not (tmp_path / name).exists()
    assert calls[0] is None
    assert 2 <= len(calls) <= 3


def test_official_manifest_and_tokenizer_pin():
    repo, revision, config = compare_tokenizer.TOKENIZERS["qwen3-32b"]
    assert (repo, revision) == (fetch_awq.TOKENIZER_REPO, fetch_awq.TOKENIZER_COMMIT)
    assert config != fetch_awq.FILES["config.json"][1]
    for name, digest in compare_tokenizer._QWEN_FILES.items():
        assert fetch_awq.FILES[name][1] == digest
    assert sum(size for size, _ in fetch_awq.FILES.values()) == 19341507104
    assert sum(name.endswith(".safetensors") for name in fetch_awq.FILES) == 4
    assert fetch_awq._manifest_digest() == "00fd219734c5f6ff13b31c0038317ed7d6631e626619c8abf76c87412ea1395b"
