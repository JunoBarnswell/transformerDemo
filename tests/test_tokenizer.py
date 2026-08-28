from pathlib import Path

from chatdemo.tokenizer import CharTokenizer


def test_tokenizer_roundtrip(tmp_path: Path):
    tok = CharTokenizer.build_from_texts(["你好世界", "你好呀"], vocab_size=50)
    ids = tok.encode("你好世界", add_special_tokens=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    assert tok.decode(ids) == "你好世界"


def test_tokenizer_save_load(tmp_path: Path):
    tok = CharTokenizer.build_from_texts(["abcde"], vocab_size=10)
    path = tmp_path / "vocab.json"
    tok.save(str(path))
    loaded = CharTokenizer.load(str(path))
    assert loaded.token_to_id == tok.token_to_id
    assert loaded.id_to_token == tok.id_to_token
