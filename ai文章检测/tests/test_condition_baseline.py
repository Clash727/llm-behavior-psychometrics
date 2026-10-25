from b_text_detector.condition_baseline import (
    align_probe_rows,
    apply_human_reference,
    build_feature_matrices,
    build_feature_matrices_stream,
    fit_human_reference,
    make_role_split,
    probe_features,
    row_key,
    text_background_features,
)


def _row(label, article, body, text):
    return {"label": label, "article_id": article, "duplicate_article_ids": [article],
            "body_hash": body, "alias_body_hashes": [], "text": text, "_original_split": "train"}


def test_role_split_keeps_transitive_alias_components_together_and_reference_rows_human():
    rows = []
    for i in range(80):
        rows.append(_row(0, f"h{i}", f"hb{i}", f"Human article {i} " + "text " * 12))
    for i in range(120):
        rows.append(_row(1, f"m{i}", f"mb{i}", f"Machine article {i} " + "text " * 12))
    # Explicit alias edge across two records; both remain in one role.
    rows[0]["alias_body_hashes"] = ["link"]
    rows[100]["alias_body_hashes"] = ["link"]
    plan = make_role_split(rows)
    assert plan["cross_role_components"] == 0
    assert plan["row_roles"][0] == plan["row_roles"][100]
    r_rows = [i for i, role in enumerate(plan["row_roles"]) if role == "R"]
    assert {rows[i]["label"] for i in r_rows} == {0, 1}
    assert all(rows[i]["label"] == 0 for i in r_rows if rows[i]["label"] == 0)
    assert {rows[i]["label"] for i, role in enumerate(plan["row_roles"]) if role == "F"} == {0, 1}
    assert {rows[i]["label"] for i, role in enumerate(plan["row_roles"]) if role == "S"} == {0, 1}


def test_text_background_and_human_reference_bucket_fallback_are_deterministic():
    x = [[1.0, 4.0], [2.0, 4.0], [3.0, 4.0]]
    buckets = [("short", "latin")] * 3
    reference = fit_human_reference(x, buckets, min_cell_rows=200)
    transformed = apply_human_reference([[2.0, 4.0]], [("long", "cjk")], reference)
    assert transformed.shape == (1, 6)
    assert transformed[0, 0:2].tolist() == [2.0, 4.0]
    features, length, script = text_background_features("Hello 你好 42!")
    assert length == "0-63"
    assert script in {"latin", "cjk"}
    assert features["digit_ratio"] > 0
    assert features["cjk_letter_ratio"] > 0


def test_probe_alignment_and_four_condition_feature_contract():
    source = [{"text": "fixed text", "label": 0}]
    # Build the documented stable row key without importing source metadata.
    import hashlib
    key = row_key("train", 0, "fixed text")
    row = {"row_key": key, "text_sha256": hashlib.sha256(b"fixed text").hexdigest(),
           "original_index": 0, "label": 0,
           "conditions": {name: {"scored_token_count": 4, "mean": 2.0, "std": 0.1,
                                 "q10": 1.0, "q25": 1.5, "median": 2.0, "q75": 2.5, "q90": 3.0,
                                 "window16_means": [2.0]}
                          for name in ("raw", "neutral", "human_hypothesis", "machine_hypothesis")},
           "pairwise_surprise_deltas": {name: {"scored_token_count": 4, "mean": 0.2, "std": 0.1,
                                                "q10": 0.0, "q25": 0.1, "median": 0.2, "q75": 0.3,
                                                "q90": 0.4, "window16_means": [0.2]}
                                        for name in ("neutral_minus_raw", "human_minus_neutral",
                                                     "machine_minus_neutral", "machine_minus_human")}}
    aligned = align_probe_rows(source, [row], "train")
    a, c = probe_features(aligned[0])
    assert "raw_mean" in a and not any("window16" in key for key in a)
    assert "human_minus_neutral_mean" in c


def test_fixed_window_schema_matches_streaming_and_keeps_valid_counts(tmp_path):
    import json
    from b_text_detector.condition_baseline import row_key, text_sha256

    rows, probes = [], []
    for i, label in enumerate((0, 1)):
        text = f"stream row {i} " + "words " * 15
        rows.append({"text": text, "label": label, "_original_split": "train"})
        summary = {"scored_token_count": 10, "mean": float(label + 1), "std": .2,
                   "q10": 0., "q25": .5, "median": 1., "q75": 1.5, "q90": 2.,
                   "window16_means": [{"mean": 1.25, "count": 10, "start_body_offset": 0}]}
        probe = {"row_key": row_key("train", i, text), "text_sha256": text_sha256(text),
                 "original_index": i, "label": label, "body_token_count": 10 + i,
                 "conditions": {name: summary for name in
                     ("raw", "neutral", "human_hypothesis", "machine_hypothesis")},
                 "pairwise_surprise_deltas": {name: summary for name in
                     ("neutral_minus_raw", "human_minus_neutral", "machine_minus_neutral", "machine_minus_human")}}
        probes.append(probe)
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in probes), encoding="utf-8")
    memory, memory_names, _, _ = build_feature_matrices(rows, probes)
    streamed, stream_names, _, _ = build_feature_matrices_stream(rows, path, "train")
    assert memory_names == stream_names
    for name in memory:
        import numpy as np
        assert np.allclose(memory[name], streamed[name], rtol=1e-6, atol=1e-7)
    count_column = stream_names["C"].index("neutral_minus_raw_window16_count_000")
    assert streamed["C"][:, count_column].tolist() == [10.0, 10.0]
    assert streamed["C"][:, stream_names["C"].index("neutral_minus_raw_window16_count_001")].tolist() == [0.0, 0.0]


def test_zero_scored_token_rows_keep_fixed_schema_in_both_builders(tmp_path):
    import copy
    import hashlib
    import json
    import numpy as np
    import pytest
    from b_text_detector.condition_baseline import build_feature_matrices_stream

    text = "？"
    empty_summary = {"scored_token_count": 0, "mean": None, "std": None,
                     "q10": None, "q25": None, "median": None, "q75": None,
                     "q90": None, "window16_means": []}
    probe = {"row_key": row_key("train", 0, text),
             "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
             "original_index": 0, "label": 1, "body_token_count": 1,
             "conditions": {name: empty_summary for name in
                 ("raw", "neutral", "human_hypothesis", "machine_hypothesis")},
             "pairwise_surprise_deltas": {name: empty_summary for name in
                 ("neutral_minus_raw", "human_minus_neutral", "machine_minus_neutral",
                  "machine_minus_human")}}
    rows = [{"text": text, "label": 1, "_original_split": "train"}]
    path = tmp_path / "empty_probe_row.jsonl"
    path.write_text(json.dumps(probe) + "\n", encoding="utf-8")

    in_memory, in_memory_names, _, _ = build_feature_matrices(rows, [probe])
    streamed, streamed_names, _, _ = build_feature_matrices_stream(rows, path, "train")
    assert in_memory_names == streamed_names
    assert len(in_memory_names["A"]) == 8
    assert len(in_memory_names["C"]) == 8 + 14 + 4 * (8 + 2 * 28)
    assert in_memory["A"][0, in_memory_names["A"].index("raw_scored_token_count")] == 0
    assert in_memory["A"][0, in_memory_names["A"].index("raw_mean")] == 0
    assert in_memory["C"][0, in_memory_names["C"].index("machine_minus_human_mean")] == 0
    for name in in_memory:
        assert np.allclose(in_memory[name], streamed[name], rtol=1e-6, atol=1e-7)
        assert np.isfinite(streamed[name]).all()

    invalid = copy.deepcopy(probe)
    invalid["conditions"]["raw"]["scored_token_count"] = 1
    with pytest.raises(ValueError, match="missing/non-finite mean with scored_token_count=1"):
        build_feature_matrices(rows, [invalid])

    invalid_window = copy.deepcopy(probe)
    invalid_window["pairwise_surprise_deltas"]["machine_minus_human"] = {
        **empty_summary, "window16_means": [{"mean": None, "count": 1}]}
    with pytest.raises(ValueError, match="missing/non-finite window mean with count=1"):
        build_feature_matrices(rows, [invalid_window])

