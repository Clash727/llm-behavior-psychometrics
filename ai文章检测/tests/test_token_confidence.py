import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_token_confidence import _validated_features

from b_text_detector.token_confidence import (
    FEATURE_NAMES, cache_contract, extract_split, score_batch_chunked, summarize_surprises,
    surprises_from_logits, _read_jsonl, tokenizer_fingerprint,
)


def test_shifted_true_token_logprobs_and_padding_special_mask():
    # t=0 predicts target 2 with p=.75; t=1 predicts target 3 with p=.25.
    # Position 3 is padding for row 1 and its logits are irrelevant.
    logits = torch.tensor([
        [[-1.0986123, -100., 0., -100.], [1.0986123, -100., -100., 0.], [100., -100., -100., -100.], [0., 0., 0., 0.]],
        [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]],
    ])
    ids = torch.tensor([[1, 2, 3, 0], [1, 2, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    values = surprises_from_logits(logits, ids, mask, special_token_ids=[1, 0])
    assert values[0].shape == (2,)
    assert np.allclose(values[0], [-np.log(0.75), -np.log(0.25)], atol=1e-6)
    assert values[1].shape == (1,)


def test_empty_and_single_token_boundaries_are_finite():
    empty = summarize_surprises([])
    assert set(empty) == set(FEATURE_NAMES)
    assert all(v == 0 for v in empty.values())
    logits = torch.randn(2, 1, 5)
    ids = torch.ones((2, 1), dtype=torch.long)
    mask = torch.ones_like(ids)
    assert all(len(x) == 0 for x in surprises_from_logits(logits, ids, mask))
    assert summarize_surprises([2.0])["adjacent_abs_delta_mean"] == 0.0


class DummyTokenizer:
    padding_side = "right"
    all_special_ids = [0]

    def __call__(self, texts, **kwargs):
        # Toy: first token is context-only BOS, remaining ids contain content.
        seqs = [[9] + [ord(ch) % 5 + 1 for ch in text] for text in texts]
        width = max(map(len, seqs))
        ids = torch.zeros((len(seqs), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(seqs):
            ids[i, :len(seq)] = torch.tensor(seq)
            mask[i, :len(seq)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class DummyBase(torch.nn.Module):
    def forward(self, input_ids, attention_mask, use_cache, return_dict):
        return type("Output", (), {"last_hidden_state": torch.nn.functional.one_hot(input_ids % 5, 5).float()})()


class DummyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.model = DummyBase()
        self.lm_head = torch.nn.Linear(5, 7, bias=False)


def test_cache_contract_resumption_and_mismatch(tmp_path):
    rows = [{"text": "a", "label": 0, "source": "x", "row_key": "ka"},
            {"text": "bb", "label": 1, "source": "y", "row_key": "kb"}]
    contract = cache_contract(split="train", input_hash="abc", model_name="m", revision="r",
                              tokenizer_name="t", precision="bf16", max_length=512)
    output = extract_split(rows, split="train", output_dir=tmp_path, tokenizer=DummyTokenizer(),
                           model=DummyLM(), contract=contract, batch_size=1, limit=1)
    assert len(output) == 1 and output[0]["row_key"] == "ka"
    assert json.loads((tmp_path / "train.progress.jsonl").read_text().splitlines()[-1])["completed"] == 1
    # Simulate an interrupted final feature write, then prove resume truncates
    # the partial bytes before appending the remaining valid feature row.
    with (tmp_path / "train.features.jsonl").open("ab") as f:
        f.write(b'{"row_key":"interrupted"')
    # Increased pilot cap resumes from row 2 and emits one result per row.
    output = extract_split(rows, split="train", output_dir=tmp_path, tokenizer=DummyTokenizer(),
                           model=DummyLM(), contract=contract, batch_size=2)
    assert [r["row_key"] for r in output] == ["ka", "kb"]
    assert all(np.isfinite([r[k] for k in FEATURE_NAMES]).all() for r in output)
    progress = json.loads((tmp_path / "train.progress.jsonl").read_text().splitlines()[-1])
    assert progress["completed"] == 2
    assert progress["examples_per_second"] * progress["elapsed_seconds"] == pytest.approx(1.0, rel=1e-3)
    assert len((tmp_path / "train.features.jsonl").read_text().splitlines()) == 2
    bad = dict(contract, precision="nf4")
    with pytest.raises(ValueError, match="stale/incompatible"):
        extract_split(rows, split="train", output_dir=tmp_path, tokenizer=DummyTokenizer(),
                      model=DummyLM(), contract=bad, batch_size=1)


def test_cache_rejects_corrupt_middle_jsonl_row(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_bytes(b'{"ok":1}\n{broken}\n{"ok":2}\n')
    with pytest.raises(ValueError, match="corrupt JSONL cache"):
        _read_jsonl(path, repair_partial_tail=True)


def test_report_cache_validation_binds_contract_and_rows(tmp_path):
    contract = {"split": "dev", "input_sha256": "frozen-hash", "feature_version": "v1"}
    rows = [{"row_key": "r1"}]
    (tmp_path / "dev.meta.json").write_text(json.dumps(contract))
    (tmp_path / "dev.features.jsonl").write_text(json.dumps({"row_key": "r1", **{k: 0.0 for k in FEATURE_NAMES}}) + "\n")
    assert _validated_features(tmp_path, "dev", rows, contract)[0]["row_key"] == "r1"
    with pytest.raises(ValueError, match="stale"):
        _validated_features(tmp_path, "dev", rows, dict(contract, input_sha256="other"))
    with pytest.raises(ValueError, match="exactly match"):
        _validated_features(tmp_path, "dev", [{"row_key": "r2"}], contract)


def test_tokenizer_hash_ignores_runtime_padding_and_truncation():
    from tokenizers import Tokenizer, models, pre_tokenizers

    class Wrapped:
        def __init__(self):
            self.backend_tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1}, unk_token="[UNK]"))
            self.backend_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    tok = Wrapped()
    baseline = tokenizer_fingerprint(tok)
    tok.backend_tokenizer.enable_truncation(max_length=4)
    tok.backend_tokenizer.enable_padding(length=8, pad_id=0, pad_token="[UNK]")
    assert tokenizer_fingerprint(tok) == baseline


def test_dummy_model_batch_padding_invariance():
    model = DummyLM()
    tok = DummyTokenizer()
    short = tok(["a"], padding=True)
    together = tok(["a", "long"], padding=True)
    solo = score_batch_chunked(model, short["input_ids"], short["attention_mask"], tok.all_special_ids)[0]
    batch = score_batch_chunked(model, together["input_ids"], together["attention_mask"], tok.all_special_ids)[0]
    assert np.allclose([solo[k] for k in FEATURE_NAMES], [batch[k] for k in FEATURE_NAMES], atol=1e-6)


def test_chunked_scoring_matches_small_random_qwen3_forward():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(71)
    config = Qwen3Config(vocab_size=31, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8,
                         max_position_embeddings=32, attention_dropout=0.0)
    model = Qwen3ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask,
                       use_cache=False).logits
        expected = [summarize_surprises(x) for x in
                    surprises_from_logits(logits, input_ids, attention_mask, special_token_ids=[0])]
        actual = score_batch_chunked(model, input_ids, attention_mask,
                                     special_token_ids=[0], time_chunk=2)
    for left, right in zip(expected, actual):
        assert np.allclose([left[k] for k in FEATURE_NAMES],
                           [right[k] for k in FEATURE_NAMES], rtol=1e-6, atol=1e-6)


def test_only_allowed_splits(tmp_path):
    with pytest.raises(ValueError, match="train/dev"):
        extract_split([], split="locked_test", output_dir=tmp_path, tokenizer=None,
                      model=None, contract={}, batch_size=1)
