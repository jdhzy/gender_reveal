# Gender Reveal

Pipeline to study how simple skin-tone normalization affects gender classification accuracy and equity on the FairFace dataset. The repo downloads/cleans FairFace, applies a light normalization transform, runs the HuggingFace FairFace gender model, and fine-tunes the model on normalized images to compare baseline vs debiased behavior.

## Repo layout
- `script/data_processing/`: download FairFace, filter to front-facing photos, build balanced mini-eval splits, and image normalization utilities.
- `script/apis/`: wrappers around the HuggingFace FairFace gender classifier (`HFFairFaceGenderModel`) and a base API interface.
- `script/train/`: fine-tuning the FairFace classifier on normalized images.
- `script/eval/`: run inference and analyze accuracy/bias across race/gender; plotting helpers.
- `batchJob/`: SGE (qsub) scripts for SCC-style clusters.
- `metadata/`: expected location for models and result CSVs/plots (not versioned here).
- `local_test/`: quick smoke test for the normalization transform.

## Setup
1) Python: tested with Python 3.10+. Create and activate a venv:  
   `python -m venv venv && source venv/bin/activate`
2) Install deps: `pip install -r requirements.txt`  
   (On SCC, load the provided PyTorch module instead of pip-installing torch/torchvision.)
3) Get the base HF model: download the FairFace gender checkpoint (e.g., from the HuggingFace repo) into `metadata/models/fairface_gender_image_detection_pt2/` so it contains `config.json` and `pytorch_model.bin`. The code loads the model from this local path to avoid network calls.

## Data folder setup
- Create the data root if it does not exist: `mkdir -p data`.
- Download the FairFace dataset into `data/fairface/` (the download script below will do this automatically), so you end up with `data/fairface/train/` and `data/fairface/validation/` plus their `labels.csv` files.

## Data pipeline
1) **Download FairFace (from HuggingFaceM4/FairFace):**  
   `python script/data_processing/fetch_FF_data.py --version 1.25`  
   - Versions: `--version` can be `1.25` (default) or `0.25`.  
   - Optional speed cap: add `--max_per_split 500` (or any int) for quick smoke tests.  
   The script writes JPEGs and `labels.csv` to `data/fairface/{train,validation}/`.
2) **Keep front-facing images only:**  
   `python script/data_processing/get_frontish_faces.py`  
   Uses Haar cascades to keep images with a detectable face + two eyes. Output: `data/cleaned/frontish/{train,validation}/`.
3) **Optional mini eval split:**  
   `python script/data_processing/make_mini_eval.py --split validation --per_group 20 --out_root data/mini_eval`  
   Balances across (race, gender) groups for fast experiments.

## Running the model
- **Baseline vs. normalized inference:**  
  `python script/eval/run_fairface.py --data_root data/cleaned/frontish --split validation`  
  Add `--use_norm` to apply `normalize_skintone` before inference. Use `--model_dir` to switch between the base checkpoint and any fine-tuned checkpoint.
- Results are written to `metadata/results/{model_name}_{frontish|mini}_{split}[_norm].csv`.
- **Quick visualization of the transform:** edit `local_test/test_one.py` with two image paths and run it to write before/after JPEGs to `local_test/output/`.

## Fine-tuning on normalized images
- Train the classifier head on normalized `data/cleaned/frontish` images:  
  `python script/train/fine_tune_fairface_norm.py --epochs 3 --batch_size 32 --data_root data/cleaned/frontish --model_dir metadata/models/fairface_gender_image_detection_pt2 --out_dir metadata/models/fairface_gender_image_detection_norm_ft_full`
- The script freezes the transformer backbone and saves the fine-tuned checkpoint + feature extractor into `out_dir`.

## Evaluating and analyzing bias
- Compare two result CSVs (e.g., baseline vs normalized) and produce summaries/plots:  
  `python script/eval/analyze_fairface_results.py --baseline_csv metadata/results/hf_fairface_gender_frontish_validation.csv --normalized_csv metadata/results/hf_fairface_gender_frontish_validation_norm.csv`
- For baseline-only summaries, use `script/eval/analyze_baseline_only.py`.  
- To inspect class balance of the cleaned data, run `script/eval/cleaned_distribution.py --split train`.
- Plots and summary CSVs land in `metadata/results/`.

## Cluster (SGE) jobs
- Submit with `qsub batchJob/<jobname>.job`. Notable jobs:
  - `fetch_ff.job`: download FairFace.
  - `get_frontish.job`: filter to front-facing subset.
  - `run_ff_val_base_cpu.job` / `run_ff_val_norm_cpu.job`: baseline vs normalized validation.
  - `train_ff_norm_gpu.job`: fine-tune on GPU.  
- Logs are written to `batchJob/logs/*.txt`. Jobs assume `venv` is already created and the model checkpoint is in `metadata/models/`.

## Expected data/model layout
- `data/fairface/{train,validation}/labels.csv` and images (raw download).
- `data/cleaned/frontish/{train,validation}/labels.csv` and filtered images.
- `metadata/models/fairface_gender_image_detection_pt2/` (base HF checkpoint).
- `metadata/models/fairface_gender_image_detection_norm_ft*/` (fine-tuned checkpoints).
- `metadata/results/` (evaluation CSVs and plots).
