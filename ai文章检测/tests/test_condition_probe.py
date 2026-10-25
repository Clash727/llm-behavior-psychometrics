from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from b_text_detector.condition_probe import (
    BODY_BUDGET, CONDITIONS, TEMPLATES, _append_jsonl, _read_jsonl, encode_body,
    encode_contract, extract_split, score_condition_batch, stable_row_key,
    surprises_from_logits, validate_split_path, validate_surprise_payload,
)


class TinyTokenizer:
    all_special_ids = [0, 2]

    def encode(self, text, add_special_tokens=False):
        # Simple whitespace-based IDs; template lexical counts are intentionally equal.
        words = text.replace("\n", " ").split()
        return [3 + (sum(word.encode()) % 40) for word in words]


def tiny_model():
    torch.manual_seed(3)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128,
                      attention_dropout=0.0)
    model = Qwen3ForCausalLM(cfg)
    model.eval()
    return model


def test_cpu_shift_and_special_target_mask_hand_calculation():
    # logits at t-1 predict input ID t; target at body offset zero is never scored.
    logits = torch.tensor([[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0],
                            [0.0, 3.0, 0.0], [0.0, 0.0, 2.0]]])
    got = surprises_from_logits(logits, [[1, 1, 2]], [0], special_token_ids=[2])[0]
    assert np.isnan(got[0])
    # body offset one is token 1 and predicted by logit position zero.
    expected = torch.logsumexp(logits[0, 0].float(), 0) - logits[0, 0, 1]
    assert got[1] == pytest.approx(float(expected), abs=1e-5)
    assert np.isnan(got[2])  # special-token target excluded


def test_chunked_projection_matches_tiny_qwen_forward_with_prefix_shift():
    model = tiny_model()
    bodies = [[5, 7, 9, 11, 2, 13], [8, 12, 15]]
    prefix = [4, 6]
    got = score_condition_batch(model, bodies, prefix, special_token_ids=[2],
                                logit_chunk=2, device="cpu")
    sequences = [prefix + b for b in bodies]
    width = max(map(len, sequences))
    input_ids = torch.zeros((2, width), dtype=torch.long)
    mask = torch.zeros_like(input_ids)
    for i, seq in enumerate(sequences):
        input_ids[i, :len(seq)] = torch.tensor(seq)
        mask[i, :len(seq)] = 1
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=mask, use_cache=False).logits.float()
    for row, body in enumerate(bodies):
        ref = np.full(len(body), np.nan, dtype=np.float32)
        for j in range(1, len(body)):
            if body[j] == 2:
                continue
            pos = len(prefix) + j - 1
            ref[j] = float(torch.logsumexp(logits[row, pos], dim=-1) - logits[row, pos, body[j]])
        np.testing.assert_allclose(got[row], ref, rtol=1e-5, atol=1e-5, equal_nan=True)


def test_equal_template_prefixes_and_label_independent_body_ids():
    tok = TinyTokenizer()
    prefixes, _ = encode_contract(tok)
    assert len({len(prefixes[n]) for n in CONDITIONS[1:]}) == 1
    original = "a body of ordinary text"
    changed_label = "a body of ordinary text"
    assert encode_body(tok, original) == encode_body(tok, changed_label)
    assert len(encode_body(tok, original)) <= BODY_BUDGET


def test_rows_use_legacy_row_key_and_retain_raw_text_hash():
    text = "example"
    assert stable_row_key("train", 4, text) == hashlib.sha256(
        f"train\0{4}\0{text}".encode("utf-8")).hexdigest()


def test_extract_roundtrip_resume_and_npz_contract(tmp_path):
    tok = TinyTokenizer()
    prefix_ids, _ = encode_contract(tok)
    model = tiny_model()
    rows = []
    for i, text in enumerate(("tiny text here", "a different longer body with words")):
        sha = hashlib.sha256(text.encode()).hexdigest()
        rows.append({"text": text, "label": i, "source": "s",
                     "original_index": 10 + i, "text_sha256": sha,
                     "row_key": stable_row_key("train", 10 + i, text)})
    manifest = {"schema_version": "test-v1", "input_sha256": "x", "special_token_ids": [0, 2],
                "scoring_batch_size": 1, "shard_rows": 2}
    observed_batch_sizes = []
    hook = model.model.register_forward_pre_hook(
        lambda _module, _args, kwargs: observed_batch_sizes.append(kwargs["input_ids"].shape[0]),
        with_kwargs=True)
    result = extract_split(rows=rows, split="train", out=tmp_path / "run",
                           tokenizer=tok, model=model, manifest=manifest,
                           prefix_ids=prefix_ids, shard_rows=2, scoring_batch_size=1)
    hook.remove()
    assert observed_batch_sizes and set(observed_batch_sizes) == {1}
    assert result["state"] == "complete" and result["failed"] is False
    row_records = _read_jsonl(tmp_path / "run/rows.jsonl")
    assert len(row_records) == 2
    assert [r["row_key"] for r in row_records] == [r["row_key"] for r in rows]
    assert set(row_records[0]["conditions"]) == set(CONDITIONS)
    assert set(row_records[0]["pairwise_surprise_deltas"]) == {
        "neutral_minus_raw", "human_minus_neutral", "machine_minus_neutral", "machine_minus_human"}
    shard_path = next((tmp_path / "run/shards/train").glob("*.npz"))
    with np.load(shard_path, allow_pickle=False) as z:
        assert z["surprise"].shape == (2, 4, BODY_BUDGET)
        assert z["body_token_ids"].shape == (2, BODY_BUDGET)
        assert z["valid_mask"].shape == z["surprise"].shape
        assert z["original_index"].tolist() == [10, 11]
        assert not z["valid_mask"][:, :, 0].any()
        assert np.isfinite(z["surprise"][z["valid_mask"]]).all()
    before = shard_path.read_bytes()
    # A second invocation verifies the existing hashes and reuses committed shards.
    result2 = extract_split(rows=rows, split="train", out=tmp_path / "run",
                            tokenizer=tok, model=model, manifest=manifest,
                            prefix_ids=prefix_ids, shard_rows=2, scoring_batch_size=1)
    assert result2["state"] == "complete"
    assert result2["newly_completed_rows"] == 0 and result2["reused_rows"] == 2
    assert result2["examples_per_second"] is None
    assert shard_path.read_bytes() == before


def test_resume_rejects_committed_shard_hash_mismatch(tmp_path):
    tok = TinyTokenizer(); prefix_ids, _ = encode_contract(tok)
    text = "simple body"
    row = {"text": text, "label": 0, "source": "s", "original_index": 0,
           "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
           "row_key": stable_row_key("train", 0, text)}
    out = tmp_path / "run"
    kwargs = dict(rows=[row], split="train", out=out, tokenizer=tok,
                  model=tiny_model(), manifest={"schema_version": "x", "special_token_ids": [0, 2],
                                                "scoring_batch_size": 1, "shard_rows": 1},
                  prefix_ids=prefix_ids, shard_rows=1, scoring_batch_size=1)
    extract_split(**kwargs)
    shard = next((out / "shards/train").glob("*.npz"))
    shard.write_bytes(shard.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        extract_split(**kwargs)


def test_only_frozen_train_dev_paths_are_accepted(tmp_path):
    root = tmp_path
    frozen = root / "artifacts/audited_train/splits/train.jsonl"
    frozen.parent.mkdir(parents=True); frozen.write_text("{}\n")
    assert validate_split_path(root, "train", frozen) == frozen.resolve()
    with pytest.raises(ValueError, match="only train and dev"):
        validate_split_path(root, "officialval", frozen)
    other = root / "locked.jsonl"; other.write_text("{}\n")
    with pytest.raises(ValueError, match="frozen audited path"):
        validate_split_path(root, "train", other)


def test_valid_unterminated_jsonl_tail_gets_delimiter_before_append(tmp_path):
    path = tmp_path / "entries.jsonl"
    path.write_text('{"a":1}')
    assert _read_jsonl(path, repair_tail=True) == [{"a": 1}]
    _append_jsonl(path, [{"b": 2}])
    assert _read_jsonl(path) == [{"a": 1}, {"b": 2}]


def test_expected_mask_is_id_derived_and_nan_in_scored_slot_is_fatal():
    body = [5, 6, 2, 7]
    surprise = np.full((1, 4, BODY_BUDGET), np.nan, dtype=np.float32)
    surprise[0, :, 1] = 1.25
    surprise[0, :, 3] = 0.75
    mask = validate_surprise_payload(surprise, [body], [2])
    assert mask[0, :, 1].all() and mask[0, :, 3].all()
    assert not mask[0, :, 0].any() and not mask[0, :, 2].any()
    surprise[0, 2, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite surprise at an expected"):
        validate_surprise_payload(surprise, [body], [2])
