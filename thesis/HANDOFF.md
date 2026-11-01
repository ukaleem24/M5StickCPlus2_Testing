# Thesis handoff (written 2026-10-29)

Working note for the next session. Read this first, then `src/chapter3.tex` and
`src/chapter4.tex` in full before writing anything.

## 1. Structure (agreed with the author)

| Ch. | Title | File | Status |
|---|---|---|---|
| 1 | Introduction | `src/Introduction.tex` | written; "Thesis Structure" section still "[To be added later]". **Template.tex still inputs `src/intro` (blindtext)** -> switch to `src/Introduction` |
| 2 | State of the Art | `src/chapter2.tex` | rewritten 2026-10-30 after supervisor feedback (topic-led, industry section, new comparison table). v3 (2026-10-31): plain wording, author named only for Ward et al. (one mention), Figs. 2.1-2.3 (sensing overview, timeline, hardware tiers) and Tabs. 2.2-2.3 (embedded deployments, evaluation protocol) added. Open: `[VERIFY]` markers for papers whose PDFs are not yet in the Literature folder (Ward 2006, Tao 2018, Bhattacharya 2022, Mahfuz 2026), 3 `{[}CITATION{]}` placeholders, bib Group B/C entries in `bibliography.bib`. |
| 3 | Data Acquisition and Dataset Construction | `src/chapter3.tex` | updated to final dataset (6 sessions, 2565 windows, 9 classes); style pass still pending (see 4) |
| 4 | Model Design, Training and Deployment | `src/chapter4.tex` | done, reviewed by supervisor once; deployment merged in as Sec. 4.8 |
| 5 | Results and Evaluation | -- | **to write** (`\label{ch:results}`) |
| 6 | Discussion | -- | **to write** (`\label{ch:discussion}`) |
| 7 | Conclusion | -- | **to write** (`\label{ch:conclusion}`) |

Also to write at the end: Chapter 1 "Thesis Structure" section, abstract.
Author needs >= 80 pages excluding title page and ToC. Current PDF: 53 pages (Ch. 2, 5-7 empty).

Compile: `export PATH="/c/Users/ukale/AppData/Local/Programs/MiKTeX/miktex/bin/x64:$PATH"; pdflatex -interaction=nonstopmode Template.tex` (twice) in `thesis/`.
Render pages to check figures: `pdftoppm -f N -l N -r 80 -png Template.pdf <out>`.

## 2. Style rules from supervisor / author (apply to every new chapter)

- No colons or semicolons in the middle of sentences ("…else: the same", "x; y; z" lists). Colons only directly before a list. Supervisor calls these "AI slop".
- References written into the sentence: "as described in Section~\ref{..}", never "(Section~\ref{..})".
- Explain every symbol/term for a new reader (e.g. what p_c is, what coordinate ascent means, what SpecAugment is).
- Natural, readable academic English (German Master thesis tone), no buzzwords.
- Chapter 4 describes what was done and why; results belong to Chapter 5. No "turned out", no performance claims in Ch. 4.
- Sessions are "sessions 1-3" / "sessions 4-6", never "old/new" sessions.
- Citations as `{[}CITATION{]}` placeholders (author fills them; his literature list is not available yet). Keep the Discussion's comparisons to prior work generic + placeholder.
- TikZ figures, tabularx tables, same `blk`/`grp`/`flow` styles as Ch. 3/4. Render and visually check every figure.
- Ask the author for approval of ToC + draft content before editing LaTeX; he reviews and then says "go ahead".

## 3. Tooling traps

- **Edit LaTeX only with the Edit/Write tools.** Python-in-Bash heredocs halve `\\` (broke a table and a script twice).
- Export scripts segfault inside the TFLite converter when TensorFlow sees the GPU (DirectML). Run them CPU-only: `tf.config.set_visible_devices([], "GPU")` before running the script (a tiny runpy wrapper works).
- venv: `.venv/Scripts/python` (TF 2.10 + DirectML, Python 3.10). Training scripts in `training/`, run from that folder.
- PlatformIO: `~/.platformio/penv/Scripts/pio.exe run -e <env> [-t upload --upload-port COM5]`. Envs: `m5stick-c-inference` (early fusion), `m5stick-c-decision-fusion`, `m5stick-c-embedding-fusion`.

## 4. Open fixes in Chapters 3/4 (from review on 2026-10-22, not yet applied)

1. Ch. 4 Sec. 4.3.2: "roughly **triples** the amount of acoustic training data" -> pooled audio is 2565 -> 4829 windows, i.e. **almost doubles**; extra placements come from sessions 4-6 only.
2. Ch. 4 Sec. 4.8.3: "IMU delivered about 250 samples per second" -> Ch. 3 says approx. 220-250 Hz.
3. Ch. 4 Sec. 4.1: "All models are trained and evaluated on the same windows" contradicts the standalone-audio exception -> qualify.
4. Ch. 3 Sec. 3.2: "the screwdriving class dominates the dataset" -> "is the largest class" (23 %).
5. Ch. 4 Table 4.8: delete LaTeX comment "Arena sizes ... to be confirmed" (confirmed on device, values unchanged).
6. Ch. 4 Sec. 4.1 fusion bullet: "Fusion models combine ... They are combined at the input level ..." -> one sentence.
7. Ch. 3 style pass (written before supervisor feedback): ~8 bracketed refs, ~10 prose semicolons (worst: the 32-step protocol sentence in 3.2), ~8 mid-sentence colons (e.g. "four sensor devices were worn: one on ..."). Same rules as Sec. 2.

## 5. Promises Chapter 4 makes to Chapter 5 (must be delivered or the sentence removed)

| Promise (Ch. 4) | Status |
|---|---|
| Performance of the 3 audio pooling variants (4.3.2) | only hybrid run on new split (85.8 %). **Run GAP and GRU standalone** (`train_audio_cnn.py`, change `MODEL_BUILDER`). |
| Per-class weights used to interpret results (4.4.3) | available (Sec. 6 below) |
| **Effect of the best-weights callback** (4.7.2) | **not on new split**. Old-split evidence only (Sec. 7). Either run one model without the callback on the new split, or report the old-split observation as such / drop the sentence. |
| Float vs. int8 accuracy (4.9.3) | available |
| Memory, timing, self-test (4.8, 4.9.3) | available, `results/on_device/` |
| **Effect of wearing orientation** (4.8.5) | qualitative observation available (Sec. 8) |

## 6. Results on the final (new) split -- headline numbers

Split: `training/data_split.py` (blocks of <= 20 windows, overlap purge). Train 1979 / val 530 / discarded 56; every class 20-22 % val. Single run per model, best-epoch weights.

| Pipeline | Script | Val. accuracy | Macro-F1 |
|---|---|---|---|
| Early fusion, small (deployed) | `train_early_fusion_concat_small_cnn.py` | **93.2 %** | **0.92** |
| Early fusion, standard | `train_early_fusion_concat_cnn.py` | 92.8 % | 0.91 |
| Decision fusion, standard, per-class | `train_decision_fusion_cnn.py` | 90.8 % | 0.90 |
| Decision fusion, standard, global (w = 0.60) | same | 88.5 % | 0.87 |
| Decision fusion, small, per-class (deployed) | `train_decision_fusion_small.py` | 90.2 % | 0.90 |
| Decision fusion, small, global (w = 0.55) | same | 88.1 % | 0.87 |
| Embedding fusion, standard | `train_fusion_cnn.py` | 90.0 % | 0.89 |
| Embedding fusion, small (deployed) | `train_embedding_fusion_small.py` | 90.0 % | 0.89 |
| Early fusion, unified spectrogram | `train_early_fusion_spectrogram_cnn.py` | 89.8 % | 0.87 |
| Random forest (116 hand-crafted features) | `train_classical_feature_fusion.py` | 87.9 % | 0.86 |
| IMU CNN, standard (inside decision fusion) | `train_decision_fusion_cnn.py` | 87.7 % | 0.86 |
| IMU CNN, standard (inside embedding fusion) | `train_fusion_cnn.py` | 87.2 % | 0.86 |
| IMU backbone, small | `train_fusion_backbones_small.py` | 87.2 % | 0.86 |
| Audio CNN GAP, standard (inside decision fusion) | `train_decision_fusion_cnn.py` | 78.9 % | 0.73 |
| Audio CNN GAP, standard (inside embedding fusion) | `train_fusion_cnn.py` | 77.5 % | 0.73 |
| Audio backbone, small | `train_fusion_backbones_small.py` | 77.9 % | 0.75 |
| Standalone audio, hybrid pooling, 3 placements pooled (976 val windows) | `train_audio_cnn.py` | 85.8 % | 0.76 (report together with accuracy; class mix differs) |

`place_container` F1 (n_val = 15): IMU 0.83-0.94, audio 0.50-0.55, decision small per-class 0.97, early small 0.86, embedding small 0.90, RF 0.75.

Per-class IMU weights, decision fusion small (deployed, `models/decision_fusion_small_export_data.npz`), class order assemble_parts, carry_container, hammering, idle, insert_screws, measuring, place_container, screw_tightening, walk:
`0.55, 0.20, 0.55, 0.55, 0.55, 0.20, 0.65, 0.95, 0.55` (global 0.55). Note: measuring leans on audio (0.20) and screw_tightening on IMU (0.95), against intuition; the search optimises validation accuracy and only affects near-tie windows. Ch. 4 text was already softened accordingly; interpret carefully, and mention the optimistic bias (weights tuned on the validation set).

**Detailed logs (per-class reports, confusion matrices) were lost** with the scratchpad. Regenerate before writing Ch. 5:
- Deployed models: do NOT retrain (firmware was built from them). Evaluate the saved models in `models/` on the val split (load .keras + export_data .npz normalisation stats, predict, `classification_report` + `confusion_matrix`).
- Non-deployed pipelines: re-run the scripts with logs saved to `results/training_logs_new_split/` and with `EXPORT_DIR` redirected so `models/` is not overwritten. Numbers will differ slightly from the table (GPU non-determinism); use the re-run as the reported values and keep the deployed-model values from evaluation of `models/`.

## 7. Quantization (int8, full validation split, `models/*_quantization_report.json`)

| Model | Float | int8 |
|---|---|---|
| Early fusion small | 93.2 % | 93.2 % |
| Decision fusion small (per-class) | 90.2 % | 89.1 % |
| Embedding fusion small | 90.0 % | 89.2 % |
| IMU backbone small | 87.2 % | 87.5 % |
| Audio backbone small | 77.9 % | 77.2 % |

Old-split facts that may be useful in the Discussion (label clearly as earlier/old-split experiments if used): restoring best-epoch weights instead of last-epoch weights improved decision fusion 87.5 -> 89.9 % and 88.2 -> 88.7 %, audio backbone by ~10 pp. The old split had 37 % validation share, 31 overlapping train/val window pairs, and put 99 % of carry_container (pooled audio) into validation -- motivation for the new split (already described in Ch. 4.9.1).

Parameter counts (9 classes): IMU CNN 36,297; audio GAP 23,945; GRU 153,353; hybrid 157,449; early fusion 49,097; early fusion small 18,729; spectrogram early fusion 25,673; IMU small 12,329; audio small 7,689; fusion head 8,841.
int8 model sizes: early small 29.9 KB; decision 23.5 + 14.2 KB; embedding 22.2 + 12.9 + 11.1 KB.

## 8. On-device results -- see `results/on_device/SUMMARY.md` (+ raw logs)

Inference 1417 ms (early fusion) / 1066 ms (decision) / 1068 ms (embedding); one prediction every ~3.5 / ~3.2 / ~3.2 s. RAM 36-37 %, flash 53-54 %. Self-test 6/9, 7/9, 7/9 vs. label but **9/9 agreement with the Python int8 reference** for all three (that is the meaningful result). Live tests not logged: activities recognised when worn normally; with strap reversed (device rotated 180 deg on the back of the wrist, embedding-fusion firmware) only hammering was still recognised, carry_container / walk / idle mostly wrong. No quantified live accuracy -> state as limitation; labelled live study = future work.

## 9. Things the author may still provide

Photos of the device on the wrist / workbench (optional). Literature list for Discussion comparisons (later, after his Ch. 2 rewrite).

## 10. Session 2026-10-31 (done / open)

Done: title page (author, programme, supervisors; date still "[submission date]"),
draft abstract (numbers from Sec. 6-8, re-check after Ch. 5), real acronym list,
Introduction integrated (`src/Introduction.tex`, label `ch:intro`) with citations,
British spelling everywhere (label keys `sec:quantization`/`eq:quantization` unchanged),
Sec. 4 items 1-4 fixed, 20 placeholders filled (bib "Group D").

Open:
- Thesis Structure (1.4) -- after Ch. 5-7.
- 5 placeholders: orientation/placement source (2.3), Arduino/Jetson data sheets
  (Fig. 2.3), recurrent-layer conversion on MCUs (2.6; NOTE the GRU unrolling in
  4.7.1 was for DirectML, not the MCU), Fraunhofer IPA Future Work Lab (3.2),
  80 Hz choice (3.6.2).
- DATA ISSUE: pooled audio folder (`data/windowed_data_combined_no_ci/right_hand_dominant/audio`)
  has 4829 wavs = 2565 dominant-wrist + 2224 non-dominant/waist + **40 `remove_screws`
  windows filed under insert_screws** (no IMU pair). Ch. 3 says remove screws was excluded.
  Affects only the standalone audio model; decide whether to remove and re-run.
- Supervisor-style review (chat 2026-10-31): B1 validation set used for epoch
  selection + fusion-weight tuning + reporting; B2 single run per model; B3 two
  participants; B4 "real-time" undefined; B5-B8 Ch. 3 sync/decimation/70 %/null class;
  C1 fusion terminology in Ch. 1 objective 2; C3 Ch. 3 style pass; E1 bib formatting.
