# Condition probe artifact contract

This experiment records token-level teacher-forced negative log probabilities for each frozen train/dev article under four fixed contexts: `raw`, `neutral`, `human_hypothesis`, and `machine_hypothesis`. It is a measurement artifact, not a classifier or a prompt-selection run.

## Input and scoring contract

- Inputs are accepted only from `artifacts/audited_train/splits/train.jsonl` and `dev.jsonl`; locked/official validation paths are rejected.
- The text is tokenized once, independently, with `add_special_tokens=false`, then truncated to its first 448 token IDs. The exact resulting body IDs are reused unchanged in all four conditions.
- Prefixes are tokenized separately without special tokens. The initial templates are `raw=""`, `neutral="The following passage is ordinary written text:\n\n"`, `human_hypothesis="The following passage is human written text:\n\n"`, and `machine_hypothesis="The following passage is machine written text:\n\n"`. Before model loading, neutral/human/machine prefix token counts must match; any mismatch halts the run and requires a training-side template revision recorded in the manifest. Every prefix is limited to 64 tokens.
- Each model input is the literal ID concatenation `prefix_ids + body_ids`; no re-tokenization of rendered prefix-plus-text occurs. Only targets corresponding to body token offsets 1..N-1 are scored. Body offset zero is excluded in every condition, including raw, and special-token targets are excluded. Prefix tokens contribute context but are never targets. This makes the raw and prefixed conditions share the same target-token set.
- Model: `Qwen/Qwen3-4B`, revision `1cfa9a7208912126459214e8b04321603b3df60c`, BF16, SDPA, local files only, base causal LM with no classification adapter. The head projection and log-sum-exp are computed in bounded time chunks; full-vocabulary logits are not retained or persisted.

## Files and atomicity

The output root is a new run directory below `artifacts/condition_probe/`; existing artifact directories are never overwritten. Each split has its own directory with `manifest.json`, `rows.jsonl`, `shards.jsonl`, `progress.jsonl`, and `run_status.json`. Model scoring always uses exactly one article per forward call for all four conditions. Storage sharding is independent and bound as `shard_rows` in the manifest; the current singleton pilot uses one row per atomic NPZ file. Files are written as compressed NPZ (no pickle) through a temporary file, fsync, and atomic rename. A shard has:

- `surprise`: FP32 array `[rows, 4, 448]`, NaN outside valid scored offsets.
- `body_token_ids`: int32 `[rows, 448]`, zero-filled outside `body_length`.
- `valid_mask`: bool `[rows, 4, 448]`; derived from body length, target offset, and special-token IDs, then checked against FP32 values. It is never inferred from finite/NaN observations.
- `body_length`: int32 `[rows]`.
- `prefix_lengths`: int16 `[rows, 4]`.

Every committed shard has a SHA-256 entry in `shards.jsonl` and its own matching batch metadata. On resume, all committed files are hash-verified; a mismatch is fatal. An uncommitted temporary/tail file may be removed and its batch recomputed. JSONL records are append-only with newline-delimited atomic batch commits (truncate an unterminated final record only); malformed committed interior records are fatal.

`rows.jsonl` has one record per article with established `row_key = SHA256(split + NUL + original row index + NUL + raw text)`, raw-text `text_sha256`, original index, label/source metadata for downstream grouping, body token count, and four per-condition summaries. Summaries include mean/std/quantiles, scored count, local 16-token window means, and pairwise token-aligned surprise deltas (`neutral-raw`, `human_hypothesis-neutral`, `machine_hypothesis-neutral`, `machine_hypothesis-human_hypothesis`) with summary and 16-token window means. Summary rows are rebuilt by streaming committed shards into a temporary JSONL file, so memory does not grow with the complete nested-summary payload. No body text is duplicated in output; source data is pinned by input file hash and row key.

The per-split manifest binds the input file hash and selected row-key digest/count, model/revision, tokenizer fingerprint, precision/attention backend, exact template strings and prefix token IDs, body budget, `scoring_batch_size=1`, independent `shard_rows`, scoring/mask/concatenation policies, repeat tolerance, feature/schema version, Python/torch/transformers/numpy versions, and hashes of the probe engine and runner code. It is immutable after the first shard commit.

## Run safety and observability

Only one process may own an output run at a time. A root lock is acquired before model loading, and each split also has its own extraction lock. Locks contain PID, process start ticks, and start time; stale locks are reclaimable only when the PID/start-ticks pair is no longer the same process. A live run status contains PID, start ticks, split/batch, elapsed time, examples/sec, ETA, peak allocated GPU memory, and last failure. Each batch updates status and progress. Runs are resumable only when manifest and all committed shard hashes match. Complete status has `state=complete`, `failed=false`, and matching `completed`/`total` counts. Throughput counts only work newly completed in the current invocation; cached rows are reported as reused.

Pilot selection is deterministic and limited to train rows. Dev may be extracted after a run gate but is never used to choose templates. No template B, entropy, RL, perturbation, external data, or source-balanced-suite resume is part of this artifact contract.

## Numerical checks

Chunked projection is compared against ordinary causal-LM forward on tiny random Qwen weights and CPU hand-built logits verify target shift/masking. The preserved batch-2 diagnostic pilot at `artifacts/condition_probe/pilot_bf16_padding_audit/` found that mixed padding can affect the human-neutral response substantially (mean absolute padding error / solo response magnitude 0.567 on its shortest/longest example). Production scoring therefore uses singleton forward calls. The new train-only singleton pilot at `artifacts/condition_probe/pilot_singleton_v1/` repeated eight multilingual examples in forward and reverse article order. It compared every token's four-condition NLL and each of the four aligned pairwise deltas: all were bitwise exact, with maximum absolute difference 0.0 against a 1e-6 tolerance. The selected examples covered Latin, Cyrillic, Arabic, and CJK scripts with body lengths 60–448 tokens. Saved/reloaded shard hashes and ID-derived finite-mask checks are required.
