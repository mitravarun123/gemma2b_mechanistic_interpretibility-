# Gemma attention head specialization

A config-driven study of attention-head specialization in Gemma: generate disjoint discovery/validation prompts, extract attention features, select discovery candidates, test causal effects, and export eleven figures from saved measurements. Hooks act on individual query-head outputs before `o_proj`, preserving the model's native shared KV heads, sliding windows and soft caps.

**Implementation status:** all seven milestones are implemented. Offline tests run the complete workflow using a tiny randomly initialized Gemma model. Pretrained Gemma research results have **not** been produced: the configured run remains blocked by gated model access, and the latest attempt could not resolve `huggingface.co` from this environment. Test figures demonstrate the software, not learned head functions.

## Setup

Requires Python 3.10+. Run these commands from `gemma-head-specialization/`:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pytest -q
```

Dependencies are declared in `requirements.txt` and installed by `pyproject.toml`. Tests require no Hugging Face token or downloads. The verified local environment uses Python 3.10, PyTorch 2.11, Transformers 4.44.2, NumPy 1.26.4 and Matplotlib 3.8; the full declared dependency range has not been tested. The supported model interfaces are exercised with tiny Gemma and Gemma 2 configurations. Other Gemma variants must expose compatible decoder attention projections; incompatible dimensions fail explicitly.

For pretrained runs, accept the model's license, obtain access on [Hugging Face](https://huggingface.co/google/gemma-2b), ensure the host can reach Hugging Face, then make your authorized token available as `HF_TOKEN` in your shell. Do not put the token in YAML or source files. The loader uses eager attention and chooses CUDA, then MPS, then CPU when `model.device: auto`. Set an explicit device if needed. Default `float32` weights require substantial memory; `bfloat16` is supported when suitable for your device. Debug mode reduces prompt counts, **not** model size.

## Quick start: debug first, then full

After setup and model authorization, run all seven stages with one command:

```sh
python scripts/run_pipeline.py --config configs/debug.yaml
```

Omitting `--config` also selects `configs/debug.yaml`. Debug uses 20 prompts per task per split (80 discovery and 80 validation records, plus aligned behavioral twins and a separate diagnostic example). The runner streams each stage's logs and stops immediately on failure. `results/pipeline_run.json` records the requested stages/figures, config checksum, timestamps, stage status and exit codes. A successful run means the **requested** stages completed; it does not imply significant causal effects.

Figure 1 requires an actual causally validated head and a qualifying low-specialization control. If no head passes, stage 07 reports that figure as blocked while still attempting the others. An empty passing set is a valid scientific result. To explicitly request only the remaining figures after inspecting this outcome:

```sh
python scripts/run_pipeline.py --config configs/debug.yaml --from-stage 7 --figures 2 3 4 5 6 7 8 9 10 11
```

After checking debug calibration, memory use and outputs, run the full configuration:

```sh
python scripts/run_pipeline.py --config configs/default.yaml
```

Full mode uses 2,000 discovery and 1,000 validation prompts **per task** (8,000 and 4,000 records). All-head causal interventions and prompt-bootstrap clustering are substantially more expensive than feature extraction. No runtime estimate is claimed before a pretrained debug run succeeds on your hardware.

Both shipped configs write to the same `data/` and `results/` directories. A changed config invalidates affected caches and replaces generated outputs. To retain separate runs, copy the YAML, set distinct `data_dir` and `results_dir` values, and copy `data/raw/names.json` into the new data directory's `raw/` folder before running. Pin `model.revision` to an immutable Hugging Face revision for a reproducible research run. Relative data/results paths resolve against the project root; `--config` paths resolve from the invoking shell.

## Google Colab

A ready-to-upload source bundle and detailed notebook instructions are available in [`COLAB.md`](COLAB.md). In brief: upload the project ZIP to Drive, choose a Colab GPU runtime, install with `%pip install -e .`, authorize `HF_TOKEN` through Colab Secrets, then run `scripts/run_pipeline.py` with a config whose data and results folders point to Drive. Begin with debug; configure 75 examples per task in each split for 600 clean records total. See `COLAB.md` for persistent output setup and exact notebook cells.

## Commands and outputs

Each stage is also independently runnable. All commands below use the debug config; substitute `configs/default.yaml` consistently for a full run.

| Stage | Command | Main outputs |
| --- | --- | --- |
| 01 Dataset and calibration | `python scripts/01_generate_dataset.py --config configs/debug.yaml` | `data/processed/*.jsonl`, corrupted twins, tokenizer labels/manifest; `results/calibration/` |
| 02 Attention extraction | `python scripts/02_extract_attention.py --config configs/debug.yaml` | `results/attention/index.json`, selected-row NPZ shards, small set of full examples |
| 03 Features | `python scripts/03_compute_features.py --config configs/debug.yaml` | `results/features/` head/task/template tables, layer summaries and definitions |
| 04 Clustering | `python scripts/04_cluster_heads.py --config configs/debug.yaml` | `results/clusters/` assignments, PCA, stability and discovery shortlist |
| 05 Consistency and ablation | `python scripts/05_run_ablation.py --config configs/debug.yaml` | `results/consistency/`, `results/ablation/` paired effects and controls |
| 06 Patching and assessment | `python scripts/06_run_patching.py --config configs/debug.yaml` | `results/patching/` all-head effects, correlations and validated-head decisions |
| 07 Figures | `python scripts/07_make_figures.py --config configs/debug.yaml` | `results/figures/` 300 dpi PNG/PDF pairs, provenance and manifest |

`07_generate_figures.py` remains an equivalent compatibility entry point. The runner uses the specification's `07_make_figures.py` name.

Rerun the same command to resume: individual stages verify their own cached outputs and input/code/config fingerprints. The runner always invokes requested stages; it never trusts its previous run report to skip work. To restart at a known stage, stop after an intermediate stage, or inspect commands without executing them:

```sh
python scripts/run_pipeline.py --config configs/debug.yaml --from-stage 5
python scripts/run_pipeline.py --config configs/debug.yaml --through-stage 4
python scripts/run_pipeline.py --config configs/default.yaml --dry-run
```

Starting later requires intact upstream artifacts. `--figures` applies only when stage 07 is included. Stage failures preserve a nonzero exit code; interrupted runs preserve existing checkpoints. The latest invocation replaces `pipeline_run.json`; it is an execution report, not an archive of prior runs. Scientific output folders record package versions, model configuration and Git revision alongside their stage-specific provenance.

## Interpreting results and troubleshooting

- **Gated access or download failure:** stage 01 needs the configured tokenizer and pretrained weights. Confirm model authorization and `HF_TOKEN`, then rerun. `--dataset-only` still needs the configured tokenizer and explicitly defers calibration.
- **Memory exhaustion:** reduce `batch_size`, extraction's `max_attention_mb`, and validation's `max_activation_mb`/`max_forward_mb`; use a supported lower precision if appropriate. These budgets exclude model weights and some temporary buffers.
- **Missing or inconsistent artifacts:** rerun the producing stage and its dependents with the same configuration. Do not edit scientific CSVs in place; checksum mismatches intentionally fail.
- **Weak or saturated calibration:** inspect `results/calibration/report.json` before interpreting intervention effects. There is no universal pass threshold built into the runner.
- **No validated heads:** retain the empty result. Figure 1 remains unavailable; the atlas leaves unvalidated heads neutral. Attention specialization alone is not causal validation.
- **Undefined template correlations:** discovery/validation template IDs are disjoint; the default two templates per task cannot support the minimum three mapped pairs. Condition correlations are separately named and are not substitutes.
- **Recovery outside [0, 1]:** saved ratios are unclipped. Tiny baseline gaps are undefined; signed-gap and positive-gap summaries are distinct. See the patching definitions below.

## Verification

```sh
python -m pytest -q
# Only the actual offline seven-stage CLI workflow:
python -m pytest -q tests/test_pipeline_cli.py
```

The integration test creates a temporary random Gemma checkpoint and controlled tokenizer, runs the same one-command runner for all stages with Figures 2–11, and verifies saved numerical outputs, provenance, export DPI and resume behavior. Figure 1's renderer is tested on real saved attention with explicitly unvalidated labels; its scientific entry point must reject an empty passing set. Analytical feature tests, native Gemma/Gemma 2 hook tests, causal/statistical checks and fail-fast runner tests cover the remaining contracts. Temporary test results never populate the research result folders. See [milestone 7 verification](tests/milestone7-report.txt) and earlier `tests/milestone*-report.txt` files for run details and limits.

The sections below specify each stage's scientific and caching conventions.

## Dataset and calibration

```sh
python scripts/01_generate_dataset.py --config configs/debug.yaml
```

This writes `data/processed/discovery.jsonl`, `validation.jsonl`, matching files in `corrupted/`, filtered `name_pools.json`, tokenizer/config/code fingerprint manifest, and provenance. Each split contains 20 records per task in debug mode. The same command with `configs/default.yaml` uses 2,000 discovery and 1,000 validation records per task. Run from the project directory.

Calibration writes `results/calibration/baseline.jsonl` and `report.json` with actual next-token probabilities, correct-minus-distractor logit differences, forced-choice accuracy, and near-chance/saturation flags per template/task. Ties count as half correct. Chance is 0.5 for the two-name comparison, not vocabulary-wide prediction. Calibration checkpoints each completed batch; identical configurations resume completed work. Dataset caches validate file hashes and invalidate on tokenizer/config/code changes.

`--dataset-only` explicitly defers calibration but still requires access to the real configured tokenizer. Tests use a controlled fast tokenizer and random tiny Gemma weights to check contracts; these test results are not pretrained-model calibration.

Names are single space-prefixed tokens and disjoint across held-out pools. BOS is inserted once before span indexing. Induction uses distinct one-token words in a random sequence repeated twice, excluding the last item whose successor crosses the repeat boundary. Local context reuses both behavioral task prompts. Behavioral twins swap names for name tracking or gendered pronouns for coreference, invert correct/distractor targets, and must retain identical token length. Distractor names appear in an aside marking them absent. Distance and distractor variation increase difficulty, but baseline calibration must establish whether those variants are challenging for the model. Coreference templates ask explicitly about `she`/`he` to retain an unambiguous antecedent.

Behavioral correct-name gender is balanced within each task and template over complete four-record cycles, avoiding a fixed male-answer shortcut. Coreference twins reverse all relevant pronouns, including the final question. Pretrained calibration has not run successfully in this environment because gated model access was unavailable.

## Attention extraction and features

After generating the real configured-tokenizer dataset:

```sh
python scripts/02_extract_attention.py --config configs/debug.yaml
python scripts/03_compute_features.py --config configs/debug.yaml
```

Use `configs/default.yaml` in both commands for the full dataset. Stages read their inputs from disk and fail clearly when an earlier stage is missing or incomplete. `data_dir` and `results_dir` can point to separate directories for isolated runs.

Extraction writes `results/attention/index.json`, provenance, and compressed per-prompt `.npz` shards under `results/attention/prompts/`. Each shard contains selected query rows in both BOS modes, task-query and all-token per-head summary arrays, original BOS sink mass, valid query counts, token IDs and categories. `.json` sidecars checkpoint each prompt atomically. Cached shards are checksum-verified; missing/corrupt shards are recomputed. A completed verified run skips loading model weights. Resuming uses the recorded model snapshot: pin `model.revision` to an immutable checkpoint for reproducibility, or remove the attention output folder to request a fresh extraction.

Full matrices in both modes are saved only for `extraction.example_prompts` selected examples. The Figure 1 sentence is written by stage 01 to `data/processed/examples.jsonl`, extracted separately, and excluded from feature statistics. Rerun stage 01 if your dataset predates this file. Local-context queries cover every eligible previous-token position, so their retained query rows necessarily cover most of a matrix; other tasks retain only designated spans.

Batches have identical token lengths and explicit attention masks. `extraction.max_attention_mb` caps a conservative float32 estimate of returned attention tensors and reduces batch size as needed. The cap excludes model weights and temporary eager-attention workspace; reduce it and `batch_size` when device memory is tight. A prompt too large even for a single-item batch raises a clear error.

Feature tables under `results/features/`:

| File | Contents |
| --- | --- |
| `task_features.csv` | Per-head/split/task/BOS-mode summaries and valid prompt/query counts |
| `template_features.csv` | The same summaries separated by template |
| `head_features.csv` | Task scores, preferred token categories, overall features, z scores, dominant task and specialization |
| `layer_summaries.csv` | Layer means and specialization bootstrap intervals |
| `zscore_parameters.csv` | Per-split/BOS-mode task mean and population standard deviation |
| `feature_definitions.json` | Exact aggregation, normalization and missingness conventions |

Entropy uses natural logs; normalized entropy divides by log of the structurally available causal keys, respecting each layer's sliding window. Gini is finite-support normalized, giving zero for uniform rows and one for one-hot rows when more than one key is available. Singleton normalized entropy and Gini are zero. BOS removal preserves original token positions, sets BOS mass to zero, and renormalizes remaining mass. A row with no remaining mass is undefined (`NaN`); valid counts show which rows/prompts contributed. BOS sink mass always describes the original row.

Task-query metrics average queries within each prompt and then give prompts equal weight within a task. General head features give the four tasks equal weight. All-token reference statistics are saved separately. Specialization is the maximum task z score across heads within a split/BOS mode; zero-variance task scores get z = 0. Discovery z-score parameters are saved for future held-out transforms; validation results never enter discovery aggregation. The saved layer confidence band bootstraps heads within a layer and describes variation across heads; it is not a prompt-sampling confidence interval.

Tests include analytical metric checks, actual tiny Gemma/Gemma 2 forward passes with shared KV heads and sliding windows, and an offline CLI run of stages 01–03 at debug dataset sizes. All random-model artifacts live in temporary test directories and are separate from research results. See `tests/milestone3-report.txt` for verification and current gated-access limitations.

## Discovery clustering

```sh
python scripts/04_cluster_heads.py --config configs/debug.yaml
```

Run this after stages 01–03. Use `configs/default.yaml` for the full run. The stage loads saved CSV features and per-prompt extraction summaries; it does not load a language model. Pretrained Gemma clustering remains blocked until the gated dataset and feature stages can run.

The default clustering uses BOS-excluded discovery data. Configure `clustering.bos_mode` to use included data. Validation rows and example prompts are excluded from all fits, profiles, selection and bootstrap draws. Undefined discovery features produce a clear error rather than being silently imputed.

Four feature groups drive the fit: normalized entropy, mean distance, sparsity (Gini and top-1 mass), and best task z score. Features are standardized across discovery heads; each group has equal total weight by default, so the two sparsity columns each receive a factor of `1/sqrt(2)`. `group_weights` can change those weights.

KMeans and a regularized Gaussian mixture each choose k independently using the highest silhouette score over valid entries of `k_range`; ties prefer smaller k. Invalid sizes and collapsed candidate partitions are recorded. At least three distinct heads and a valid k below the head count are required. `primary_method` selects which partition drives profiles and candidates. PCA fits the same weighted standardized discovery space; both components and loadings are saved. UMAP is optional and not implemented.

Bootstrap draws resample discovery prompts within each task, preserving template counts by default. A single draw is shared across all heads. Each repeat recomputes task features and specialization, refits standardization and clustering, and compares with the original partition using adjusted Rand index at the discovery-selected k. This measures prompt-sampling stability while holding the clustering initialization seed fixed. Memory-mapped input arrays avoid keeping full prompt-feature tensors in RAM. Verified input arrays and individual bootstrap checkpoints are cached; completed outputs are checksum-verified and skipped.

Outputs under `results/clusters/`:

| File | Contents |
| --- | --- |
| `assignments.csv` | Discovery head features and both method assignments, plus the primary cluster |
| `cluster_profiles.csv` | Primary cluster means, dominant task, task agreement and alignment flags |
| `layer_distribution.csv` | Head counts and fractions for every layer/primary-cluster combination |
| `pca_coordinates.csv` | Two-dimensional coordinates and method labels |
| `pca_loadings.csv` | Component directions, loadings and feature group factors |
| `fit_parameters.npz` | Numeric scaler, PCA, KMeans and mixture parameters, without Python pickle objects |
| `k_selection.csv` | Silhouette scores, candidate status and selected k for both methods |
| `bootstrap_stability.csv` | Per-repeat/per-method ARI and realized cluster count |
| `stability_summary.csv` | Mean/std ARI and empirical bootstrap percentile range |
| `shortlist.csv` | Ranked discovery candidates, potentially empty |
| `selection_rules.json` | Applied configuration and statistical conventions |

The default shortlist requires a cluster's strongest mean task z score to be at least 0.5 and at least half its heads to prefer that task. Within an aligned cluster, it selects up to three heads matching that task with specialization at least 1.0. These thresholds and count are configurable. All candidates carry `causally_validated = false`. An empty shortlist is a legitimate result; no fallback heads are manufactured.

The ARI percentile range describes the distribution of bootstrap partitions; it is not a confidence interval on mean ARI. Test fixtures use tiny random models and remain separate from research outputs. See `tests/milestone4-report.txt` for verification.

## Held-out validation

```sh
python scripts/05_run_ablation.py --config configs/debug.yaml
python scripts/06_run_patching.py --config configs/debug.yaml
```

Run these after stages 01–04. Script 05 computes consistency and ablation; script 06 patches every head, correlates discovery features with causal effects, and assesses shortlisted candidates. Substitute `configs/default.yaml` for the full run. Tests exercise the one-command runner across stages 01–07 at debug dataset sizes using a temporary random Gemma checkpoint; pretrained Gemma validation still requires gated access and real upstream artifacts.

Behavioral evaluation uses only held-out name-tracking/coreference prompts. Correct-name probability is vocabulary-wide softmax probability; accuracy is forced choice between correct and distractor names (ties = 0.5). Corrupted-run measurements always use the **clean** answer IDs so directions and recovery remain comparable.

### Consistency

`results/consistency/consistency.csv` records each head/task in both BOS modes: fraction of valid held-out prompt scores above `consistency_threshold`, valid/total counts, template correlation, separate correlation over shared distractor-count conditions, and split-half reliability. Split halves are drawn within held-out template/distractor/query-count strata, and per-head stratum means are correlated. Valid split-half correlations are averaged, then Spearman–Brown corrected. Sparse/constant strata give NaN with status/count fields.

Discovery and validation template IDs are disjoint. The default template-wise correlation is therefore **undefined**, not an arbitrary pairing by row or template number. `validation.template_pairs` accepts explicit one-to-one mappings, each with `task`, `discovery`, and `validation` keys. The minimum is three finite, nonconstant pairs; the current two-template-per-task design cannot support this correlation even if two pairs are configured. Shared distractor-condition correlation is saved as a separate statistic and does not stand in for template correlation.

### Ablation

Mean ablation is the default. A seeded reference set of discovery behavioral prompts determines a token-pooled mean for every head before `o_proj`, including BOS. The reference mean broadcasts to all positions of a held-out prompt. `validation.ablation: zero` replaces those head slices with zero instead. All individual heads and available shortlisted top-k groups are evaluated. Requests larger than the shortlist are recorded as `insufficient_candidates`; no replacement candidates are invented.

Random controls match the target head count in each layer. They exclude target members whenever enough other heads exist; otherwise they sample the full layer and report the overlap. Repeated selections may recur. Identical control/head sets reuse actual measured intervention results.

Ablation outputs in `results/ablation/` include `baseline.csv`, `baseline_summary.csv`, `summary.csv`, streamed `per_prompt.csv`, `reference_set.json` for mean ablation, `groups.json`, `group_status.csv`, `random_controls.csv`, and `random_control_bands.csv`. Positive effects mean baseline minus ablated accuracy, logit difference or correct-name probability. Control bands are empirical percentile ranges over repeated matched groups, not bootstrap CIs.

### Patching and causal labels

Every head is tested in both directions: denoising runs corrupted input with its clean donor head activation; noising runs clean input with the corrupted donor activation. The default patch position is the final query predicting the answer. `validation.patch_positions: all` patches the same head at every aligned position. Patching all head outputs is not equivalent to replacing the full model state because residual streams also differ.

Denoising raw effect is patched minus corrupted; noising raw effect is clean minus patched. Both normalized scores divide their raw effect by the same clean-minus-corrupted baseline gap. Logit and probability ratios are both saved without clipping. Gaps at or below `recovery_epsilon` in magnitude are undefined. Signed ratios retain inverted gaps, and separate positive-gap summaries show the subset where clean input improves the corresponding baseline metric. Valid pair counts accompany every statistic.

`results/patching/` contains `baseline.csv`, `summary.csv`, streamed `per_prompt.csv`, `feature_correlations.csv`, and numeric head checkpoints. Correlations use discovery features and held-out denoising effects across all heads; constant or insufficient vectors remain undefined.

Script 06 also saves `validated_heads.csv` and `assessment_rules.json`. A discovery candidate passes only if its matching behavioral task meets the configured consistency fraction and has positive ablation probability loss and positive denoising logit effect, with both bootstrap lower bounds above zero and BH q values below `fdr_alpha`. Induction and local-context candidates remain unvalidated because the dataset has no matched behavioral targets for those tasks. Empty passing sets are legitimate.

### Statistics, memory and resume

Effects use paired bootstrap percentile intervals and paired Cohen's dz (mean difference / sample SD). Sign-flip p values are two-sided: exact for at most 12 pairs, Monte Carlo otherwise. BH corrects per-head/group ablation tests separately by metric/kind and patch tests by metric/direction, including task strata; feature correlations form their own family. Undefined statistics remain NaN, and a nonzero constant difference has infinite dz. Saved definitions specify each convention.

Reference means, baselines, ablation interventions and individual patch heads checkpoint completed batches atomically with checksums. Final tables can be rebuilt from intact numeric checkpoints without repeating model inference. Verified completed scripts skip loading weights. Donor activations move to CPU; only each replacement slice moves back to the model device. Per-prompt tables stream to disk. `max_activation_mb` and `max_forward_mb` reduce batch size for activation caches, logits and eager-attention workspace estimates; these budgets exclude model weights and other temporary buffers. Compatible newer model APIs request last-token logits only. Nonfinite model logits fail clearly.

See `tests/milestone5-report.txt` for verification and current limitations.


## Figures

After stages 01–06, export all eleven figures:

```bash
python scripts/07_make_figures.py --config configs/debug.yaml
# Full configuration, after its upstream stages finish:
python scripts/07_make_figures.py --config configs/default.yaml
# Explicitly render a subset:
python scripts/07_make_figures.py --config configs/debug.yaml --figures 2 3 4 5 6 7 8 9 10 11
```

Each module in `src/figures/` reads saved artifacts only. It does not load a model, refit PCA/clusters, or recompute scientific scores, effects or confidence intervals. Outputs are `results/figures/fig01_*.png` through `fig11_*.png`, at 300 dpi, with matching PDFs and embedded fonts. Figure 1 uses a viridis attention scale and annotates each finite cell with its probability; annotation color switches between black and white for contrast. Shared style uses a colorblind-safe palette. `manifest.json` records the requested subset, completion/blocked status, input/output SHA256 checksums and programmatically selected heads/examples. `metadata.json` records package versions, configuration, model configuration copied from extraction and Git revision. Verified figures skip rendering; changed inputs or code trigger rebuilding. Failed rebuilds remove stale exports for that figure, record the reason and exit nonzero after attempting the other requested figures. Missing inputs never produce placeholders.

`figures` config controls BOS mode, feature split, held-out behavioral task (`all`, `name_tracking`, `coreference`), top-head count, the maximum specialization of Figure 1's control, and the saved patch recovery metric. Optional `run_label` adds a visible figure footer, used by tests to mark random-model outputs. Ranking by an observed effect is descriptive; it is not a significance or validation claim.

Figure 1 requires a genuinely passing candidate from `patching/validated_heads.csv`, a measured low-specialization nonvalidated control, and the saved full attention matrix plus token labels for the exact diagnostic sentence. A missing passing head is a legitimate blocking result; no head is promoted to make the picture. Figure 10 similarly leaves unsupported functions neutral. Figure 9 prefers an observed wrong-to-correct patch example, but honestly shows the measured outcome if none exists. Recovery ratios remain unclipped and default to the saved positive-baseline-gap summary. Figure 11's band is explicitly a bootstrap across heads, not prompt-sampling uncertainty.

Stage 01 now saves tokenizer labels; stage 05 saves `ablation/baseline_summary.csv`; stage 06 includes measured patched accuracy and probability in `patching/per_prompt.csv`. Older artifacts without these fields need their generating stages rerun. Source fingerprints invalidate affected caches automatically.

Offline tests render Figures 2–11 from an actual tiny random Gemma run, separately exercise Figure 1's heatmap renderer without validated labels, and verify missing-input errors, source integrity, cache reuse, physical head grids, image DPI and PDF exports. These are software checks, not pretrained Gemma findings. See `tests/milestone6-report.txt` for the completed verification.
# gemma2b_mechanistic_interpretibility-
