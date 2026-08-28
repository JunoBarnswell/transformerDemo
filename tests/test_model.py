import torch

from chatdemo.model import TinyTransformerLM, TransformerConfig


def test_model_forward_and_generate_shape():
    cfg = TransformerConfig(vocab_size=100, max_seq_len=32, d_model=16, nhead=2, n_layers=1, ff_dim=32)
    model = TinyTransformerLM(cfg)

    x = torch.randint(low=0, high=100, size=(2, 12))
    logits = model(x)
    assert logits.shape == (2, 12, 100)

    out = model.generate(x[:, :4], max_new_tokens=6, temperature=1.0)
    assert out.shape[0] == 2
    assert out.shape[1] >= 4
