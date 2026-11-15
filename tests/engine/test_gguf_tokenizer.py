"""GG4: serving a GGUF without a registered Hugging Face snapshot (ironmule/gguf_tokenizer.py, `serve --gguf`)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ironmule.gguf_tokenizer import GgufDetokenizer, _complete_prefix, gguf_model_id, is_gguf_spec, render


class _Pieces:
    """A stand-in tokenizer whose tokens are byte pieces, some of them halves of one UTF-8 character."""

    def __init__(self, pieces: list[bytes]):
        self.pieces = pieces

    def piece(self, token: int) -> bytes:
        return self.pieces[token]


def test_the_detokenizer_streams_a_character_split_over_tokens_once_it_is_complete():
    smile = "😀".encode()  # four bytes, split over two tokens
    detokenizer = GgufDetokenizer(_Pieces([b"Gr", "ö".encode()[:1], "ö".encode()[1:] + b"sse ", smile[:2],
                                           smile[2:]]))
    out = []
    for token in range(5):
        detokenizer.add_token(token)
        out.append(detokenizer.last_segment)
    detokenizer.finalize()
    out.append(detokenizer.last_segment)
    assert out == ["Gr", "", "össe ", "", "😀", ""]
    assert "".join(out) == "Grösse 😀"


def test_an_incomplete_character_at_the_end_is_held_back():
    assert _complete_prefix(b"ab") == 2
    assert _complete_prefix("abö".encode()[:-1]) == 2
    assert _complete_prefix("😀".encode()[:3]) == 0
    assert _complete_prefix("😀".encode()) == 4


def test_the_template_renders_as_hugging_face_renders_it():
    template = ("{% for m in messages %}{% if m['role'] == 'bad' %}{{ raise_exception('no') }}{% endif %}"
                "<{{ m['role'] }}>{{ m['content'] }}\n{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
                "{{ bos_token }}")
    text = render(template, [{"role": "user", "content": "Hi"}], add_generation_prompt=True, bos_token="<s>")
    assert text == "<user>Hi\n<assistant><s>"
    from jinja2.exceptions import TemplateError

    with pytest.raises(TemplateError):
        render(template, [{"role": "bad", "content": ""}], add_generation_prompt=False, bos_token="")


def test_a_gguf_answers_to_its_file_name_and_names_itself(tmp_path: Path):
    gguf = tmp_path / "Qwen3-4B-Q8_0.gguf"
    gguf.write_bytes(b"GGUF")
    assert gguf_model_id(str(gguf)) == "gguf/Qwen3-4B-Q8_0"
    assert is_gguf_spec({"snapshot_path": str(gguf)})
    assert not is_gguf_spec({"snapshot_path": str(tmp_path)})  # a registered snapshot is a directory
    assert not is_gguf_spec({"snapshot_path": str(tmp_path / "absent.gguf")})


def test_serve_needs_a_model_no_model_or_a_gguf(tmp_path: Path, capsys):
    from ironmule_product.cli import serve

    with pytest.raises(SystemExit) as raised:
        serve(["--state-dir", str(tmp_path / "state")])
    assert raised.value.code == 2
    assert "one of --model, --no-model or --gguf" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        serve(["--state-dir", str(tmp_path / "state"), "--no-model", "--gguf", str(tmp_path / "x.gguf")])


GGUF = os.environ.get("IRONMULE_TEST_GGUF")


@pytest.mark.skipif(not GGUF, reason="IRONMULE_TEST_GGUF names no GGUF (and its llama.cpp library)")
def test_the_gguf_tokenizer_gives_the_engines_ids_and_its_own_end_tokens():
    from ironmule.ggml_engine import CHECK_TEXT, GgmlEngine
    from ironmule.gguf_tokenizer import GgufTokenizer

    tokenizer = GgufTokenizer(GGUF)
    engine = GgmlEngine(GGUF, speculate=False)
    try:
        assert tokenizer.encode(CHECK_TEXT) == engine.tokenize(CHECK_TEXT)
        assert tokenizer.eos_token_ids and all(engine.eog(t) for t in tokenizer.eos_token_ids)
        if tokenizer.chat_template:
            ids = tokenizer.apply_chat_template([{"role": "user", "content": "Hello!"}])
            assert ids and tokenizer.decode(ids).count("Hello!") == 1
    finally:
        engine.close()
        tokenizer.close()
