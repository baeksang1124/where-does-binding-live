# SCRIPTS.md — script index

Index of every script in this repository for *Where Does Binding Live?
From UNet Cross-Attention Heads to the MM-DiT Text Stream*: what each one measures, the
result file it writes, and how to run it.

## Layout

```
.                              <- ALWAYS run from here (repo root)
├── prompts.py  prompts_ext.py  prompts_attr.py     core: prompt pair sets
├── grader.py                                       core: automated binding graders
├── binding_patch.py  binding_patch_sd3.py  binding_patch_pixart.py
│                                                   core: per-head attention patchers
├── stream_sd3.py                                   core: SD3.5 text-stream injection engine
├── scripts/
│   ├── 00_setup_sanity/               patcher + grader correctness brackets
│   ├── 01_selection_sweeps/           localization sweeps, head selection, OV/QK decomposition
│   ├── 02_heldout_generalization/     held-out top-1 / block-9 generalization + dual grading
│   ├── 03_stream_depth_controls/      text-stream, freeze, placebo and depth controls
│   ├── 04_additional_controls/        CFG rows, hybrid capture, late read window, material, shared gate
│   ├── 05_followup_controls/          per-block image pinning, K/V factorial, window length, seeds
│   └── figures/                       figure generation
└── results/                           result JSONs (shipped); every run writes here
```

Only the result JSONs are included in `results/`. When re-run, the scripts also write console
logs (`*.log`), images (PNG grids, per-pair images) and figures (`results/figs/`) into
`results/`; none of those are shipped.

**Two rules make this work, and both depend on running from the repo root:**

1. **Core modules stay at the root.** Every script uses bare imports
   (`from prompts import make_pairs`, `from binding_patch_sd3 import ...`).
   Those resolve only if the repo root is on `sys.path`. Running a script by
   path puts *its own* directory on `sys.path[0]`, not the cwd — so you must
   export `PYTHONPATH=.` yourself.
2. **Every output path is cwd-relative under `results/`.** No script uses
   `os.chdir`, an absolute path or a `../` output path, so a script in
   `scripts/03_stream_depth_controls/` still reads and writes `results/…` at the
   repo root — *provided the cwd is the repo root*.

So the universal invocation is:

```bash
cd where-does-binding-live              # the repo root
export PY=python                        # or the full path of your environment's interpreter
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/<folder>/<name>.py
```

`CUDA_VISIBLE_DEVICES=1` is only an example GPU index — set it to the GPU you want to use,
or drop it on a single-GPU machine. CPU-only scripts drop it. Many scripts read an
environment variable that sets the number of pairs (or blocks, seeds, output suffix); these are
listed per row with their defaults. Most result JSONs record the qualified pair ids and `n`
they were computed on.

### One cross-folder import

`scripts/05_followup_controls/holdout_overlap13.py` does
`from holdout_purity import ci, report, binom_upper`, and `holdout_purity.py` lives in
`scripts/04_additional_controls/`. The script inserts that sibling folder into `sys.path`
itself (relative to its own location), so the usual `PYTHONPATH=.` is enough; putting the
folder on the path explicitly is equivalent:

```bash
PYTHONPATH=.:scripts/04_additional_controls $PY scripts/05_followup_controls/holdout_overlap13.py
```

This is the only script-to-script import across folders. Within
`scripts/05_followup_controls/`, `hyb2_prefix.py`, `window_extension.py` and
`seeds_followup.py` import `hybrid_v2` from the same folder, which Python finds because the
script's own folder is on `sys.path`.

---

## Index

The *run* column assumes the repo root as cwd and `$PY` exported once (see above). Env knobs
in parentheses are optional overrides; the value after `=` is the default.

### Core modules (repo root — imported, not run)

| script | what it does | result file | how to run |
| --- | --- | --- | --- |
| `prompts.py` | Two-object colour-binding prompt pairs (selection set, ids 0–35): each pair shares two colours and two objects and swaps only which colour attaches to which object; fixed seed per pair. | — | imported (`from prompts import ...`) |
| `prompts_ext.py` | Extended colour-binding candidate pairs used as the held-out set (84 pairs, ids 5000–5083, separate seed space). | — | imported |
| `prompts_attr.py` | Non-colour attribute binding pairs (MATERIAL, SIZE), same structure as `prompts.py`. | — | imported |
| `grader.py` | Automated binding graders: `CLIPGrader` (caption similarity), `QwenGrader` (Qwen2.5-VL-7B per-object colour VQA) and `LlavaGrader` (LLaVA-1.5-7B, second grader). | — | imported |
| `binding_patch.py` | SD1.5 cross-attention patcher: recomputes the K and/or V of selected (layer, head) cells from the swapped prompt's text embedding while the clean prompt runs. Empty swap set is bit-identical to the clean generation; swapping all cells reproduces the swapped-prompt image. | — | imported |
| `binding_patch_sd3.py` | SD3.5-medium (MM-DiT) joint-attention patcher: per (block, head) swap of the text-token K and/or V, taken from a cached swapped-prompt run (activation patching, since the text stream evolves across blocks and steps). | — | imported |
| `binding_patch_pixart.py` | PixArt-Sigma cross-attention patcher: per (block, head) swap of the text K and/or V computed from the fixed swapped T5 embedding (28 blocks x 16 heads). | — | imported |
| `stream_sd3.py` | SD3.5 text-stream injection engine (`StreamHooks`, imported by the other stream scripts): replaces the full text stream (`encoder_hidden_states`) entering chosen blocks with the swapped run's, optionally pinning every other block to the clean stream. Its own `main()` is the 10-pair pilot: all-block reference, isolated (clean-pinned) and naive (unpinned) single-block arms, and prefix injection from block b onward; the naive arm (`prop_mean`) is the ≥ 0.95 pilot cited in §5. | `results/sd3_stream.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY stream_sd3.py` (`SD3_SUBSET=10`) |

### `scripts/00_setup_sanity/`

| script | what it measures | result file | how to run |
| --- | --- | --- | --- |
| `test_sanity.py` | SD1.5 patcher brackets: empty swap set vs vanilla clean (bit-identical), swap-all vs vanilla swapped prompt, single-head swap. | — (bracket images only) | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/00_setup_sanity/test_sanity.py` |
| `test_grader.py` | Whether a grader separates clean images from swap-all images. | — (bracket images only) | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/00_setup_sanity/test_grader.py` |
| `test_qwen.py` | Qwen grader on the bracket images saved by `test_grader.py`. | — (prints) | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/00_setup_sanity/test_qwen.py` |
| `sanity_sd3.py` | SD3.5 joint-attention patcher brackets (empty swap set, swap-all K&V, single head), pixel L1 and time per generation. | — (contact sheet only) | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/00_setup_sanity/sanity_sd3.py` |
| `sanity_pixart.py` | PixArt-Sigma cross-attention patcher brackets (same set of brackets). | — (contact sheet only) | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/00_setup_sanity/sanity_pixart.py` |
| `bracket_processor.py` | Whether installing the SD3.5 patch controller (no swap cells) leaves the clean generation bit-identical to the stock attention processor, on 2 held-out pairs. | — (prints) | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/00_setup_sanity/bracket_processor.py` |
| `validate_grader_sd3.py` | Second-grader check of the SD3.5 channel contrast (head swap vs text-stream injection): the same images graded by Qwen2.5-VL-7B and LLaVA-1.5-7B, with per-image agreement. | `results/validate_grader_sd3.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/00_setup_sanity/validate_grader_sd3.py` (`SD3_SUBSET=8`) |

### `scripts/01_selection_sweeps/`

| script | what it measures | result file | how to run |
| --- | --- | --- | --- |
| `qualify.py` | SD1.5 pair qualification: keeps (pair, seed) where the clean image renders an object in its clean colour and the swap-all image renders it in the swapped colour; records the graded object. | `results/qualified.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/01_selection_sweeps/qualify.py` |
| `run_sweep.py` | SD1.5 coarse-to-fine cross-attention localization: per-layer sweep, cumulative top-k layers, per-head sweep, cumulative top-k heads (recovery relative to clean / swap-all). | `results/sweep_results.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/01_selection_sweeps/run_sweep.py` (optional argument: qualified-pairs file, default `results/qualified.json`; `results/qual_small.json` is a 3-pair file in that format) |
| `sweep_sd3.py` | SD3.5 pair qualification (two-object gate) + block-level sweep (all 24 heads of one block, K&V), two-object flip and strict rebind. | `results/sd3_coarse.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/01_selection_sweeps/sweep_sd3.py` (`SD3_NPAIRS=0` = all) |
| `sweep_pixart.py` | PixArt-Sigma qualification, block-level sweep (28 blocks) and per-head sweep (448 heads) with a cumulative top-k head curve. | `results/pixart_sweep.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/01_selection_sweeps/sweep_pixart.py` (`PX_NPAIRS=0` = all, `PX_NPH=8`) |
| `headsweep_sd3.py` | SD3.5 per-head localization: single-head K&V swaps ranked by two-object flip, then a cumulative top-k head curve. | `results/sd3_headsweep.json` (shipped: all 24 blocks = 576 heads, pairs 0,2,3,8,10,11). The script always writes this name; `results/sd3_headsweep_144.json` is an earlier run with the default settings (6 blocks = 144 heads, 12 pairs), renamed. | Shipped 576-head file: `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 SD3_BLOCKS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 SD3_PIDS=0,2,3,8,10,11 SD3_KS=1,2,4,8,16,32,64,128 $PY scripts/01_selection_sweeps/headsweep_sd3.py` (`SD3_PIDS` overrides `SD3_SUBSET`). Defaults: `SD3_SUBSET=12`, `SD3_BLOCKS=9,5,11,2,4,13`, `SD3_KS=1,2,3,4,6,8,12,16,24,32`. **Any run overwrites `results/sd3_headsweep.json`**, whose head ranking is read by `top1_ci.py`, `single_sd3.py`, `a4_sd3.py`, `top1_sd3_ext.py`, `validate_top1_dual.py`, `dual_grade_full.py`, `validate_grader_sd3.py` and `f5_grids.py`; back it up first. |
| `greedy_sd3.py` | SD3.5 block-level greedy cumulative recovery: repeatedly adds the block that most increases the flip. | `results/sd3_greedy.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/01_selection_sweeps/greedy_sd3.py` (`SD3_SUBSET=12`, `SD3_MAXK=6`) |
| `single_sd3.py` | SD3.5 cumulative top-k head swap graded with the single-object-instance flip metric (same metric as SD1.5 / PixArt). | `results/sd3_single.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/01_selection_sweeps/single_sd3.py` (`SD3_SUBSET=12`) |
| `single_pixart.py` | PixArt per-head and cumulative top-k curves re-graded with the single-object-instance flip metric (head ranking from `pixart_sweep.json`). | `results/pixart_single.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/01_selection_sweeps/single_pixart.py` (`PX_TOPM=24`) |
| `a4_decomp.py` | SD1.5 OV/QK decomposition of the top heads: FULL (K,V swapped), PATTERN (K only), VALUE (V only), BASELINE. | `results/a4_decomp.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/01_selection_sweeps/a4_decomp.py` |
| `a4_sd3.py` | SD3.5 OV/QK decomposition on the top-ranked heads from `sd3_headsweep.json`, two-object metric, bootstrap CI on PATTERN − VALUE. | `results/sd3_a4.json` (not included) | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/01_selection_sweeps/a4_sd3.py` (`SD3_SUBSET=0` = all, `SD3_TOPK=16`) |
| `a4_pixart.py` | PixArt OV/QK decomposition on the top-ranked heads, single-object flip, bootstrap CI on PATTERN − VALUE. | `results/pixart_a4.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/01_selection_sweeps/a4_pixart.py` |
| `a4b_bleed.py` | SD1.5 colour-bleed control for the OV/QK decomposition: two-object grading that separates a proper two-object rebind from bleed / collapse (both objects one colour), on the same head sets and seeds as `a4_decomp.py`. | `results/a4b_bleed.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/01_selection_sweeps/a4b_bleed.py` |
| `bootstrap_ci.py` | Bootstrap 95% CIs over pairs for the SD3.5 depth curves and the matched-vs-placebo contrast, from the per-pair arrays in `sd3_stream_ci.json`. CPU only. | — (prints) | `PYTHONPATH=. $PY scripts/01_selection_sweeps/bootstrap_ci.py` |
| `top1_ci.py` | Per-pair top-1-head single-object flip for one model, with a pair-clustered bootstrap CI. | `results/top1_sd15.json`, `results/top1_pixart.json`, `results/top1_sd3.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 MODEL=sd3 $PY scripts/01_selection_sweeps/top1_ci.py` (`MODEL=sd15` \| `pixart` \| `sd3`, default `sd15`) |

### `scripts/02_heldout_generalization/`

| script | what it measures | result file | how to run |
| --- | --- | --- | --- |
| `top1_sd15_ext.py` | SD1.5 top-1 head single-object flip on the held-out pairs (`prompts_ext.py`), single-object protocol on o1. | `results/top1_sd15_ext.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/02_heldout_generalization/top1_sd15_ext.py` |
| `top1_sd3_ext.py` | SD3.5: qualifies the held-out pairs and measures the top-1 head single-object flip. | `results/top1_sd3_ext.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/02_heldout_generalization/top1_sd3_ext.py` |
| `top1_pixart_ext.py` | PixArt: qualifies candidate pairs and measures the top-1 head flip. Candidates are the held-out set plus the base pairs (deduplicated by clean prompt), so the 35 qualified pairs are 26 held-out + 9 in-sample, and the stored `mean` / `ci` are the combined value. The held-out / in-sample split reported in Tab. 1 (0.10 / 0.33) is computed from `per_pair` (ids ≥ 5000 = held-out) by `05_followup_controls/derived_stats.py`; the `held_out` / `in_sample` fields in the shipped JSON are not written by this script. | `results/top1_pixart_ext.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/02_heldout_generalization/top1_pixart_ext.py` |
| `block9_ext.py` | SD3.5 held-out, paired depth test: isolated text-stream injection at block 9, block 11 and the 9–11 window vs the image-facing whole-block-9 K/V swap on the same pairs, with a paired bootstrap CI. | `results/block9_ext.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/02_heldout_generalization/block9_ext.py` (`B9_NMAX=75`) |
| `validate_top1_dual.py` | Small dual-grader check of the top-1 head swap, graded by Qwen2.5-VL-7B and LLaVA-1.5-7B (single-object flip on o1): SD1.5 and PixArt on held-out pairs; the SD3.5 arm uses the first `N_PAIRS` base (in-sample) pairs of `sd3_coarse.json`, not held-out pairs. The full held-out dual grading is `dual_grade_full.py`. | `results/validate_top1_dual.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/02_heldout_generalization/validate_top1_dual.py` (`N_PAIRS=8`) |
| `dual_grade_full.py` | Full held-out dual-grader re-scoring: regenerates the top-1 head swap image for every held-out qualified pair of all three models and grades each with both graders; per-model metric, a unified o1 single-object metric, between-model bootstrap CIs, grader agreement and raw grader answers. Skips images already on disk. | `results/dual_grade_full.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/02_heldout_generalization/dual_grade_full.py` |

### `scripts/03_stream_depth_controls/`

| script | what it measures | result file | how to run |
| --- | --- | --- | --- |
| `stream_ci_sd3.py` | SD3.5 isolated single-block and prefix text-stream injection with per-pair storage (for bootstrap CIs), plus an all-block placebo that injects the stream of a third-colour prompt (same objects and template, colours c3/c4 disjoint from c1/c2). | `results/sd3_stream_ci.json` (n = 30; `results/sd3_stream_ci.n10.bak.json`: an earlier 10-pair run) | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 SD3_SUBSET=30 $PY scripts/03_stream_depth_controls/stream_ci_sd3.py`. The default `SD3_SUBSET=10` reproduces only the 10-pair pilot and overwrites the shipped n = 30 file. |
| `stream_attr_sd3.py` | Pilot of the SD3.5 text-stream injection (single-block, prefix) and per-block image-facing K/V sweep for a non-colour attribute. The shipped material run is small (n = 4); the paper's material result comes from `material_ext.py`. | `results/sd3_stream_material.json` (`ATTR=size` writes `sd3_stream_size.json`, not included) | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 ATTR=material $PY scripts/03_stream_depth_controls/stream_attr_sd3.py` (`ATTR=material` \| `size`, `ATTR_NPAIRS=0` = all) |
| `freeze_sd3.py` | SD3.5 freeze-text control: the text stream is frozen to its first-block value at every block; object presence and correct binding, normal vs frozen. | `results/sd3_freeze.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/03_stream_depth_controls/freeze_sd3.py` (`SD3_SUBSET=10`) |
| `freeze_depth_sd3.py` | SD3.5 freeze-depth sweep: the text stream is frozen at its block-b value from block b onward; correct binding and "presence" vs b. Presence here counts objects for which the grader returned a colour word; `freeze_presence.py` re-measures presence with yes/no questions. | `results/sd3_freeze_depth.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/03_stream_depth_controls/freeze_depth_sd3.py` (`SD3_SUBSET=10`) |
| `matched_late_controls.py` | Two controls in one run: (A) image-facing K/V swap of all heads in blocks 9–11 on the same 75 held-out pairs as `block9_ext.py`, a paired comparison with the text window at equal block granularity; (B) late-block placebo potency, third-colour placebo stream injected from b ∈ {9, 15, 18}. | `results/matched_late.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/03_stream_depth_controls/matched_late_controls.py` |
| `placebo_potency.py` | Whether the third-colour placebo installs its own colours (o1 = c3, o2 = c4) while not flipping c1/c2, and a re-measurement of the matched all-block and placebo flips with whole-word colour parsing. | `results/placebo_potency.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/03_stream_depth_controls/placebo_potency.py` (`SD3_SUBSET=30`) |
| `window_placebo_paired.py` | Third-colour placebo at the block 9–11 window on the same 75 held-out pairs and seeds as `block9_ext.py`; paired bootstrap CI on swap − placebo. | `results/window_placebo_paired.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/03_stream_depth_controls/window_placebo_paired.py` |
| `window_placebo_seeds.py` | (A) Window-level placebo: third-colour stream injected at the isolated 9–11 window (cross-flip and potency); (B) window and prefix (b = 9, 15) flips over 3 fresh seed offsets with re-qualification. | `results/window_placebo_seeds.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/03_stream_depth_controls/window_placebo_seeds.py` (`WP_NA=30`, `WP_NB=15`) |
| `multiseed_sd3_stream.py` | Seed robustness of the SD3.5 all-block text-stream flip: each pair is re-qualified at fresh generation seeds and the matched all-block injection is re-measured; per-seed and pooled mean ± SD. | `results/multiseed_sd3_stream.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/03_stream_depth_controls/multiseed_sd3_stream.py` (`MS_NPAIRS=15`) |

### `scripts/04_additional_controls/`

| script | what it measures | result file | how to run |
| --- | --- | --- | --- |
| `hybrid_stream.py` | Hybrid capture (v1): the text stream is recomputed under the swapped prompt while the step-end image latent is pinned to the clean trajectory; injected at the 9–11 window and at all blocks, paired against the standard capture on the same pairs. `hybrid_v2.py` pins the image state at every block instead. | `results/hybrid_stream.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/04_additional_controls/hybrid_stream.py` (`HYB_N=40`) |
| `late_readwindow.py` | Whether late text injection still changes the image: prefix injection of swap, third-colour placebo and "alien" (different colours and objects) streams, measured by pixel L1, with the swap arm also graded for flip. | `results/late_readwindow.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/04_additional_controls/late_readwindow.py` (`LRW_N=20`) |
| `holdout_purity.py` | Construction of the selection and held-out prompt sets: object and (colour, object) overlap statistics, and every held-out result recomputed without the 6/84 held-out pairs whose ordered (clean, swapped) text also appears in the selection set. Uses stored per-pair arrays; CPU only. | `results/holdout_purity.json` | `PYTHONPATH=. $PY scripts/04_additional_controls/holdout_purity.py` |
| `cfg_rows.py` | Block 9–11 window injection applied to the conditional CFG row only, the unconditional row only, or both; the both-row arm is checked against `block9_ext.json`. | `results/cfg_rows.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/04_additional_controls/cfg_rows.py` (`CFG_N=40`) |
| `material_ext.py` | Material binding with a wider candidate pool (8-object set): qualification funnel, then block 9–11 window injection plus an all-block reference; pilot-sized. | `results/material_ext.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/04_additional_controls/material_ext.py` (`MAT_N=48`) |
| `shared_gate.py` | Shared-gate comparison: re-grades SD1.5's held-out qualified pairs under the two-object gate used for PixArt / SD3.5, then compares the models on the resulting common set. | `results/shared_gate.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/04_additional_controls/shared_gate.py` |

### `scripts/05_followup_controls/`

| script | what it measures | result file | how to run |
| --- | --- | --- | --- |
| `hybrid_v2.py` | Hybrid capture v2: the image hidden state entering every block is pinned to the clean run's value (per block, per step) while the text stream is recomputed under the swapped prompt; tensor-level brackets, then injection at the 9–11 window and at all blocks. | `results/hybrid_v2.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/05_followup_controls/hybrid_v2.py` (`HYB2_N=40`) |
| `hyb2_prefix.py` | Prefix injection (blocks b..23) of the text-only (v2) stream and of the standard joint-state stream, paired on the same pairs. | `results/hyb2_prefix.json` (b ∈ {0,3,6,9,12,15}); `results/hyb2_prefix_fine.json` (b ∈ {7,8,10,11}) | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/05_followup_controls/hyb2_prefix.py` (`HP_N=40`, `HP_BS=0,3,6,9,12,15`, `HP_SUFFIX`); fine file: `HP_BS=7,8,10,11 HP_SUFFIX=_fine` |
| `window_extension.py` | Isolated injection window 9..e for e ∈ {11, 13, 15, 17, 23}, joint-state vs text-only stream, paired. | `results/window_extension.json` (first 40 of the 75 `block9_ext.py` pairs), `results/window_extension_p2.json` (the remaining 35), `results/window_extension_n75.json` (the two runs combined, n = 75; `derived_stats.py` recomputes the merge and checks it against this file) | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/05_followup_controls/window_extension.py` (`WE_N=40`, `WE_OFFSET=0`, `WE_SUFFIX`); second run: `WE_N=35 WE_OFFSET=40 WE_SUFFIX=_p2` |
| `kv_factorial.py` | 2x2 factorial explaining the gap between the text window and the image-facing K/V window: pooled-CLIP `temb` (swapped vs clean) x text stream in non-window blocks (unpinned vs pinned clean), plus the raw text-stream window, on the same 75 held-out pairs. | `results/kv_factorial.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/05_followup_controls/kv_factorial.py` (`KV_N=75`) |
| `seeds_followup.py` | Fresh-seed robustness: 40 held-out pairs x 3 seed offsets, re-qualified, windows 9–11 and 9–15, both streams, per-seed pair ids stored. | `results/seeds_followup.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/05_followup_controls/seeds_followup.py` (`SF_N=40`) |
| `freeze_presence.py` | Re-grades the `freeze_depth_sd3.py` sweep on the same pairs with explicit yes/no presence questions and a distinct-object count, alongside the original colour questions. | `results/freeze_presence.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/05_followup_controls/freeze_presence.py` (`FP_SUBSET=10`) |
| `holdout_overlap13.py` | Recomputes every held-out result without the 13/84 held-out pairs whose unordered {clean, swapped} prompt set appears in the selection set (vs 6/84 ordered in `holdout_purity.py`), including the matched image-facing 9–11 window. CPU only. | `results/holdout_overlap13.json` | `PYTHONPATH=. $PY scripts/05_followup_controls/holdout_overlap13.py` (see *One cross-folder import*) |
| `unified_strict_o1.py` | Unified strict o1 metric (1 iff o1 reads the swapped colour c2) for all three models from the raw grader answers in `dual_grade_full.json`; exact binomial bootstrap CIs per model, 1e5-resample bootstrap for differences (seed 1234). CPU only. | `results/unified_strict_o1.json` | `PYTHONPATH=. $PY scripts/05_followup_controls/unified_strict_o1.py` |
| `derived_stats.py` | Derived statistics computed from the shipped result JSONs: the n = 75 window-length merge (`window_extension.json` + `window_extension_p2.json`, checked against the shipped `window_extension_n75.json`, which it does not rewrite), the PixArt held-out / in-sample split of `top1_pixart_ext.json` (checked against its stored `held_out` / `in_sample` fields), the seed SDs as reported (ddof = 1), the material-vs-colour difference CI, and the prompt-overlap sensitivity rows not covered by `holdout_overlap13.py` (text-only vs joint-state at n = 75 → 63, unified PixArt 0.12 → 0.05). CPU only. | `results/derived_stats.json` | `PYTHONPATH=. $PY scripts/05_followup_controls/derived_stats.py` |

### `scripts/figures/`

Figures are written to `results/figs/` (created on first run) and are not shipped.

| script | what it does | output | how to run |
| --- | --- | --- | --- |
| `cr_figures.py` | Figures 1–4 from the result JSONs: F1 teaser, F2 top-k localization curves, F3 granularity, F4 commitment (fixed categorical palette, 95% bootstrap CIs). Supersedes `plot_figures.py` and `f1_teaser.py`, and overwrites their `F1_teaser` / `F2_localization_curves` files. | `results/figs/F1_teaser`, `F2_localization_curves`, `F3_granularity`, `F4_commitment` (`.pdf` + `.png`) | `PYTHONPATH=. $PY scripts/figures/cr_figures.py` (CPU only) |
| `plot_figures.py` | Earlier versions of F2 (localization curves), F3 (channel contrast + single-block depth curve) and F4 (prefix depth-commitment, colour + material). | `results/figs/F2_localization_curves.pdf`, `F3_channel_depth.pdf`, `F4_depth_commitment.pdf` | `PYTHONPATH=. $PY scripts/figures/plot_figures.py` (CPU only) |
| `f1_teaser.py` | Earlier version of the F1 teaser schematic. | `results/figs/F1_teaser.pdf` | `PYTHONPATH=. $PY scripts/figures/f1_teaser.py` (CPU only) |
| `f5_grids.py` | Earlier version of the F5 grid (not the published figure): PixArt single top-head swap vs SD3.5 single top-head swap and all-block text-stream injection. Default PixArt pair 27 (`F5_PIXART_PID`), output path `F5_OUT`. | `results/figs/F5_grids.png` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/figures/f5_grids.py` |
| `f5_vectorize.py` | Earlier F5 version: wraps the `f5_grids.py` grid in a PDF with vector text labels (the generated samples stay raster). Needs the output of `f5_grids.py` first. | `results/figs/F5_grids.pdf` | `PYTHONPATH=. $PY scripts/figures/f5_vectorize.py` (CPU only; `F5_IN`, `F5_PDF`) |
| `f5_candidates.py` | Fig. 5 candidates: for held-out pairs where the SD1.5 top-1 head flips, the SD3.5 top-1 head does not, and the SD3.5 blocks 9–15 window flips, renders clean / top-1 head swap / window 9–15 / swapped-prompt tiles with the measurement scripts' own code paths and re-grades every tile with Qwen. | `results/figs/f5_candidates/` (tiles, contact sheets, `candidates.json`) | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY scripts/figures/f5_candidates.py` (`F5C_PHASE`, `F5C_N`, `F5C_OUT`) |
| `f5_pixart_tiles.py` | Fig. 5 PixArt-Σ row: clean / top-1 head swap / swapped-prompt tiles with the `top1_pixart_ext.py` code path, re-graded with Qwen. | `results/figs/f5_candidates/<id>_pixart_*.png`, `pixart_tiles.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY scripts/figures/f5_pixart_tiles.py` (`F5P_IDS`) |
| `f5_sd15_tiles.py` | Fig. 5 SD1.5 row: clean / top-1 head swap / swapped-prompt tiles for any pair ids (base or held-out) with the `top1_sd15_ext.py` code path, re-graded with Qwen (o1 rule). | `results/figs/f5_candidates/<id>_sd15_*.png`, `<id>_sd15_grades.json` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY scripts/figures/f5_sd15_tiles.py` (`F5S_IDS`) |
| `f5_compose.py` | Composes the published Fig. 5 from the candidate tiles (defaults: SD1.5 pair 5072, PixArt-Σ and SD3.5 pair 5029, three rows, 8.5 pt labels; column 2 is the target = swapped-prompt reference); CPU only. | `results/figs/F5_grids.pdf` | `PYTHONPATH=. $PY scripts/figures/f5_compose.py` (`F5_ROWS`, `F5_SD15_ID`, `F5_PIX_ID`, `F5_SD3_ID`, `F5_FS`, `F5_OUT`) |
| `make_figs.py` | SD1.5 localization curves and example image grids from `sweep_results.json`; run after `run_sweep.py`. | `results/localization_curves.png`, `results/top_head_grids.png` | `PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY scripts/figures/make_figs.py` |

### Other files in `results/`

`pixart_sweep_smoke.json`, `sd3_stream_smoke.json` and `sd3_stream_material_smoke.json` are
smoke-size runs of `sweep_pixart.py`, `stream_sd3.py` and `stream_attr_sd3.py`; they are kept
for reference and are not used by any other script.

### `verdict` fields

Some result JSONs store a `verdict` string (e.g. `sd3_stream.json`, `sd3_freeze.json`,
`sd3_freeze_depth.json`, `a4b_bleed.json`; also `sd3_headsweep.json`, `sd3_greedy.json`,
`pixart_sweep.json`, `sd3_stream_material.json`). These are run-time heuristics written by the scripts
during exploration and are superseded by the paper's analysis. In particular, the freeze control does
not separate binding from object loss (see `freeze_presence.json` and the paper's discussion of the
freeze experiment as a failed control), whatever the freeze verdicts say.
