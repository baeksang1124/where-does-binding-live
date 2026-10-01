# Where Does Binding Live?

Code for **"Where Does Binding Live? From UNet Cross-Attention Heads to the MM-DiT Text Stream"**
(ACCV 2026).

**Sangyeol Baek, Suan Lee** — School of Computer Science, Semyung University

The paper asks *which internal components of a text-to-image diffusion model are causal handles for attribute–object
binding* (a "red cube next to a blue sphere" rendered with the colors swapped), and answers it with causal activation
patching on clean ↔ attribute-swapped prompt pairs, graded by a VQA model, across three architectures:
**SD1.5** (UNet, isolated cross-attention), **PixArt-Σ** (DiT, isolated cross-attention) and **SD3.5-medium**
(MM-DiT, joint attention).

Main findings:

- In SD3.5, binding is not accessible through any screened single image-facing head (best head 0.03 held-out), but it is at
  **block-window granularity through the text stream**: replacing the text state entering blocks 9–11, with all other
  blocks pinned to the clean stream, flips 0.30 of held-out swaps, a 9–15 window 0.62 (0.44 above a matched placebo). A 2×2 control shows this is a
  granularity effect (clean-pinning), not a different channel.
- A prefix sweep shows a sharp **commitment cliff** between blocks 9 and 10 (color binding).
- The classic **single-head handle** is stable in the tested UNet (0.34 held-out) but does not generalize in either
  tested DiT (0.10 PixArt-Σ, 0.03 SD3.5).

## What is in this repository

| | |
|---|---|
| `prompts.py`, `prompts_ext.py`, `prompts_attr.py` | Prompt-pair sets: base set (ids 0–35), held-out set (ids ≥ 5000), attribute variants (material) |
| `grader.py` | VQA graders: Qwen2.5-VL-7B (primary), LLaVA-1.5-7B (second grader), CLIP (sanity only) |
| `binding_patch.py`, `binding_patch_pixart.py`, `binding_patch_sd3.py` | Per-head key/value patchers for SD1.5, PixArt-Σ and SD3.5 |
| `stream_sd3.py` | SD3.5 text-stream capture/injection engine (block pre-hooks, clean-pinning) |
| `scripts/00_setup_sanity/` | Patcher and grader correctness checks (no-swap / all-swap brackets) |
| `scripts/01_selection_sweeps/` | Pair qualification, head/block sweeps, top-k curves, key-vs-value decomposition |
| `scripts/02_heldout_generalization/` | Held-out top-1 head and block tests, dual-grader re-scoring |
| `scripts/03_stream_depth_controls/` | Text-stream depth curves, placebos, freeze controls, seed checks |
| `scripts/04_additional_controls/` | Image-trajectory hybrid capture, late read-out, CFG rows, material, shared gate, set construction |
| `scripts/05_followup_controls/` | 2×2 key/value factorial, image-pinned capture (`hyb2`; logged as "text-only" by the scripts), window length, prefix sweep, seeds, overlap sensitivity, unified metric, derived statistics (`derived_stats.py`), window controls (shifted windows, 9–15 placebo, clean-pinned head/block) |
| `scripts/figures/` | Figure generation (`cr_figures.py`: Fig. 1 with its numbers typed into the script, Figs. 2–4 read from the result JSONs; `f5_candidates.py`, `f5_sd15_tiles.py`, `f5_pixart_tiles.py` → `f5_compose.py`: Fig. 5) |
| `results/*.json` | Per-pair and aggregate results for every experiment in the paper |
| `SCRIPTS.md` | One row per script: what it measures, the JSON it writes, and how to run it |

Not included: generated images (every image is regenerated from the pair's fixed seed; on our setup regeneration was
pixel-identical on a checked pair, but bit-identity across GPUs or library versions is not guaranteed, since PyTorch
deterministic algorithms are not enabled), model weights (downloaded from Hugging Face on first use, at the pinned
revisions below) and the paper sources. Raw grader answers are stored only for some experiments (see the result-file
inventory in `SCRIPTS.md`).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Experiments were run on a single NVIDIA RTX 3090 (24 GB), fp16, Python 3.10, PyTorch 2.6.0 (CUDA 11.8 build),
diffusers 0.38.0 and transformers 4.57.1. SD3.5-medium is a gated model: accept its license on Hugging Face and log in
(`huggingface-cli login`) before running the SD3.5 scripts.

## Models

| Role | Hugging Face id (revision used for the paper) | Notes |
|---|---|---|
| SD1.5 | `sd-legacy/stable-diffusion-v1-5` (`451f4fe`) | DDIM scheduler |
| PixArt-Σ | transformer of `PixArt-alpha/PixArt-Sigma-XL-2-512-MS` (`76fb7eb`); T5, VAE and scheduler of `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` (`e102b35`) | DPM-Solver multistep |
| SD3.5 | `stabilityai/stable-diffusion-3.5-medium` (`b940f67`) | MMDiT-X variant: blocks 0–12 add an image-only self-attention (`attn2`), left unpatched; T5 encoder disabled (CLIP encoders only); FlowMatch Euler, shift 3 |
| Primary grader | `Qwen/Qwen2.5-VL-7B-Instruct` (`cc59489`) | per-object color question; 4-bit NF4 (bitsandbytes) |
| Second grader | `llava-hf/llava-1.5-7b-hf` (`b234b80`) | fp16 |
| Sanity grader | `openai/clip-vit-large-patch14` (`32bd642`) | setup checks only |

The loaders pin `revision=` to these commits (full hashes in `revisions.py`), so a default run fetches the paper's exact
checkpoints. `sd-legacy/stable-diffusion-v1-5` now redirects to `stable-diffusion-v1-5/stable-diffusion-v1-5` (same commit).

## Running

Every script is run **from the repository root** with the root on `PYTHONPATH` (the core modules are imported by
bare name). Each script that produces a result writes its own `results/*.json` (most also append to a `.log`);
re-running a script overwrites its JSON. This matters most for `headsweep_sd3.py` and `stream_ci_sd3.py`, whose default settings do not reproduce the
shipped files: see `SCRIPTS.md` for the settings that do. The shipped JSONs are version-controlled, so `git diff results/`
shows what a re-run changed and `git checkout -- results/` restores the released files.

```bash
export PY=python   # the interpreter of your environment
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY scripts/05_followup_controls/kv_factorial.py
```

Optional environment variables (subset sizes, sweep points, output suffixes) are listed per script in `SCRIPTS.md`.
The figure script `scripts/figures/cr_figures.py`, `scripts/05_followup_controls/unified_strict_o1.py` and
`scripts/05_followup_controls/derived_stats.py` run on CPU from the shipped JSONs.

### Where each result in the paper comes from

| Paper | Script(s) | Result file(s) |
|---|---|---|
| Tab. 1, per-model top-1 head, held-out (and SD1.5 in-sample) | `02_heldout_generalization/top1_sd15_ext.py`, `top1_pixart_ext.py`, `top1_sd3_ext.py` | `top1_sd15_ext.json`, `top1_pixart_ext.json`, `top1_sd3_ext.json` |
| Tab. 1, SD3.5 in-sample (0.08) | `01_selection_sweeps/top1_ci.py` (`MODEL=sd3`) | `top1_sd3.json` |
| Tab. 1, PixArt held-out / in-sample split (0.10 / 0.33) | `05_followup_controls/derived_stats.py`, from the `per_pair` values of `top1_pixart_ext.json` (ids ≥ 5000 are held-out; the file's own `mean`/`ci` are the combined value) | `derived_stats.json` |
| Tab. 1, unified strict metric; grader agreement; second grader | `02_heldout_generalization/dual_grade_full.py` → `05_followup_controls/unified_strict_o1.py` | `dual_grade_full.json`, `unified_strict_o1.json` |
| Fig. 2, in-sample top-k head curves | `01_selection_sweeps/run_sweep.py`, `single_pixart.py`, `single_sd3.py` | `sweep_results.json`, `pixart_single.json`, `sd3_single.json` |
| Abstract/§1, unpinned whole-block bound (≤ 0.11, in-sample); §2.2, all-head bracket (33/35 base pairs) | `01_selection_sweeps/sweep_sd3.py` | `sd3_coarse.json` |
| §4, 576-head sweep | `01_selection_sweeps/headsweep_sd3.py` (576-head setting in `SCRIPTS.md`) | `sd3_headsweep.json` |
| §4–5, all-block text stream (0.98, n = 30 in-sample) and in-sample single-block / prefix depth curves | `03_stream_depth_controls/stream_ci_sd3.py` run with `SD3_SUBSET=30`; CIs printed by `01_selection_sweeps/bootstrap_ci.py` | `sd3_stream_ci.json` |
| §4–5, all-block placebo (0.00) and its potency | `03_stream_depth_controls/placebo_potency.py` | `placebo_potency.json` |
| §4, all-block 1.00 on 40 held-out pairs; §5, latent-pin control (0.39) | `04_additional_controls/hybrid_stream.py` | `hybrid_stream.json` |
| §4–5, held-out block 9 and window 9–11 (text stream) | `02_heldout_generalization/block9_ext.py` | `block9_ext.json` |
| §4, image-facing key/value window 9–11 (0.17) | `03_stream_depth_controls/matched_late_controls.py` (part A) | `matched_late.json` (also arm A of `kv_factorial.json`) |
| Fig. 3a, 2×2 control (normalisation × pinning) | `05_followup_controls/kv_factorial.py` | `kv_factorial.json` |
| Fig. 3b, window length, joint-state vs. image-pinned stream | `05_followup_controls/window_extension.py` writes `window_extension.json` (40 pairs) and, with `WE_N=35 WE_OFFSET=40 WE_SUFFIX=_p2`, `window_extension_p2.json` (35 pairs); the n = 75 merge `window_extension_n75.json` is recomputed and checked against the shipped file by `05_followup_controls/derived_stats.py`; `hybrid_v2.py` | `window_extension_n75.json`, `hybrid_v2.json` |
| Fig. 4, prefix sweep / commitment cliff | `05_followup_controls/hyb2_prefix.py` | `hyb2_prefix.json`, `hyb2_prefix_fine.json` |
| §5, naive single-block pilot without pinning (≥ 0.95, n = 10) | `stream_sd3.py` | `sd3_stream.json` (`prop_mean`) |
| §5, window placebo and swap-specific excess | `03_stream_depth_controls/window_placebo_paired.py` | `window_placebo_paired.json` |
| §4, clean-pinned top head and block 9; §5, shifted 7-block windows and the 9–15 placebo (Supp. Tab. S2) | `05_followup_controls/window_controls.py`; distributions and complete-swap CIs from `05_followup_controls/window_controls_stats.py` (CPU) | `window_controls.json`, `window_controls.jsonl`, `window_controls_stats.json` |
| §8 and Supp. Tab. S3, second grader (LLaVA) on the regenerated 9–11 / 9–15 windows, both window placebos and the b = 9 / 10 prefix arms | `05_followup_controls/second_grader_windows.py` | `second_grader_windows.json`, `second_grader_windows_qwen.jsonl`, `second_grader_windows_llava.jsonl` |
| §5, late placebo (installation 0.00, 0/30; 0.72 from block 9) | `03_stream_depth_controls/matched_late_controls.py` (part B) | `matched_late.json` |
| §5, alien-prompt read-out | `04_additional_controls/late_readwindow.py` | `late_readwindow.json` |
| §5, CFG rows; material | `04_additional_controls/cfg_rows.py`, `material_ext.py`; material-vs-colour CI from `05_followup_controls/derived_stats.py` | `cfg_rows.json`, `material_ext.json`, `derived_stats.json` |
| §5, fresh seeds | `03_stream_depth_controls/multiseed_sd3_stream.py` (all-block 1.00 per seed, 14/15/15 pairs re-qualified), `03_stream_depth_controls/window_placebo_seeds.py`, `05_followup_controls/seeds_followup.py`; the seed SDs as reported (ddof = 1) from `05_followup_controls/derived_stats.py` | `multiseed_sd3_stream.json`, `window_placebo_seeds.json`, `seeds_followup.json`, `derived_stats.json` |
| §6, freeze control | `03_stream_depth_controls/freeze_depth_sd3.py`, `05_followup_controls/freeze_presence.py` | `sd3_freeze_depth.json`, `freeze_presence.json` |
| §8, prompt-overlap sensitivity | `05_followup_controls/holdout_overlap13.py` (per-model and window rows); `05_followup_controls/derived_stats.py` (image-pinned vs. joint-state at n = 75 → 63; unified PixArt 0.12 → 0.05) | `holdout_overlap13.json`, `derived_stats.json` |
| §8, shared-gate comparison | `04_additional_controls/shared_gate.py`, `holdout_purity.py` | `shared_gate.json`, `holdout_purity.json` |
| Derived statistics (merges, splits, SDs, sensitivity rows) | `05_followup_controls/derived_stats.py` (CPU, from the shipped JSONs) | `derived_stats.json` |
| Supplementary, key-only vs. value-only decomposition | `01_selection_sweeps/a4_decomp.py`, `a4_pixart.py` | `a4_decomp.json`, `pixart_a4.json` |
| Figures | `figures/cr_figures.py` (Fig. 1: numbers typed into the script; Figs. 2–4: read from the result JSONs); `figures/f5_candidates.py` (GPU; renders held-out candidate pairs at the measurement protocol and re-grades them) `figures/f5_sd15_tiles.py` (GPU; SD1.5 row, pair 5072) and `figures/f5_pixart_tiles.py` (GPU; PixArt-Σ row) then `figures/f5_compose.py` (Fig. 5 with its defaults: SD1.5 pair 5072, PixArt-Σ and SD3.5 pair 5029) | `results/figs/` |

## Citation

```bibtex
@inproceedings{baek2026binding,
  title     = {Where Does Binding Live? From {UNet} Cross-Attention Heads to the {MM-DiT} Text Stream},
  author    = {Baek, Sangyeol and Lee, Suan},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`). The pretrained models used here are distributed under their own licenses.
