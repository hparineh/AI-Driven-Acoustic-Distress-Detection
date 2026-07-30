# Acoustic Distress Detection for Crowd-Safety Monitoring

This repository provides the source code used to support the computational analyses presented in the study:

> **Acoustic signatures enable occlusion-resistant early warning for crowd crushes**

The project investigates whether environmental audio can provide an additional sensing layer for identifying distress-related acoustic conditions in dense crowds. It includes tools for machine-learning benchmarking, frequency-domain analysis, and retrospective temporal evaluation on continuous video recordings.

The proposed framework is intended to complement—not replace—established crowd-safety measures such as visual monitoring, density estimation, operational supervision, risk assessment, and emergency management.

---

## Repository overview

This repository contains code for:

- Binary classification of **background/non-distress** and **distress-related** crowd audio
- Benchmarking deep-learning and classical machine-learning models using log-mel spectrogram images
- Source-disjoint train, validation, and test splitting
- Dominant frequency–magnitude analysis
- Model-performance aggregation and visualisation
- External temporal validation on continuous video audio
- Causal persistence filtering across overlapping prediction windows
- Extraction of acoustic descriptors and learned embeddings
- Generation of publication-ready tables and figures

The repository is structured to avoid exposing private local paths, copyrighted media, restricted raw recordings, or other machine-specific information.

---

## Repository structure

```text
AI-Driven-Acoustic-Distress-Detection/
├── Codes/
│   ├── metadata/
│   ├── baseline_benchmark.py
│   ├── external_video_validation.py
│   ├── frequency_magnitude_analysis.py
│   ├── requirements.txt
│   ├── LICENSE_NOTE.txt
│   └── README.md
├── Dataset/
│   └── README.md
├── External-Validation-Video/
│   └── README.md
└── README.md
```

The exact repository structure may differ slightly depending on the final release.

---

## Main analyses

### 1. Baseline classification benchmark

The baseline benchmark evaluates non-attention deep-learning and classical machine-learning models using log-mel spectrogram images.

Implemented deep-learning models include:

- Baseline CNN
- Wide Baseline CNN
- MobileNetV2
- EfficientNet-B0
- DenseNet121
- ResNet50
- VGG16
- CRNN-GRU

Implemented classical baselines include:

- PCA + Logistic Regression
- PCA + RBF-SVM
- PCA + Random Forest
- PCA + k-Nearest Neighbours
- PCA + XGBoost, when installed

The script supports:

- Multiple random seeds
- Source-disjoint splitting
- Class weighting
- Spectrogram-aware augmentation
- Sensitivity and specificity analysis
- False-alarm and missed-detection rates
- ROC-AUC and PR-AUC
- Inference-time measurement
- Parameter-count reporting
- Prediction-level exports
- Confusion matrices
- Publication-ready summary tables

---

### 2. Frequency–magnitude analysis

The frequency-analysis script extracts the top dominant spectral components from each audio sample.

For every recording, it:

1. Loads the audio
2. Removes the DC offset
3. Computes the short-time Fourier transform
4. Averages spectral magnitude over time
5. Detects dominant spectral peaks
6. Extracts the top-ranked frequency–magnitude components
7. Exports the results to CSV
8. Generates pooled and rank-wise figures

The resulting figures are intended as descriptive evidence of differences in the joint spectral-energy distributions of the two classes.

They should not be interpreted independently as proof of universal acoustic separability, causality, or operational reliability.

---

### 3. External temporal validation

The temporal-validation script applies a trained classifier to overlapping windows extracted from a continuous video soundtrack.

It supports:

- Mono audio extraction using FFmpeg
- Explicit resampling to the training sample rate
- Two-second analysis windows
- Configurable temporal overlap
- Log-mel spectrogram generation
- Per-window distress probabilities
- Multiple decision thresholds
- Causal persistence filtering
- Event merging
- Acoustic descriptor extraction
- Learned embedding export
- Temporal probability plots
- Optional annotated-video generation

The external-video analysis is retrospective. It should be described as an external case study rather than prospective operational validation.

---

## Installation

Python 3.10 is recommended.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

FFmpeg must also be installed and available on the system path.

To check:

```bash
ffmpeg -version
```

---

## Dataset organisation

The analysis code does not rely on absolute local file paths.

Dataset records should be declared through a manifest.

### LMS benchmark manifest

Example:

```csv
relative_path,label,source_id
BG/sample_0001.png,BG,recording_001
BG/sample_0002.png,BG,recording_001
Distress/sample_0003.png,Distress,recording_002
```

Required columns:

| Column | Description |
|---|---|
| `relative_path` | Path relative to the dataset root |
| `label` | `BG` or `Distress` |
| `source_id` | Identifier of the original recording or source |

All segments derived from the same original recording must have the same `source_id`.

This requirement applies even when one recording contains both background and distress-labelled segments.

### Recommended dataset structure

```text
dataset/
├── BG/
│   ├── sample_0001.png
│   └── sample_0002.png
└── Distress/
    ├── sample_0003.png
    └── sample_0004.png
```

---

## Running the baseline benchmark

Example:

```bash
python src/publication_safe_baseline_benchmark.py \
  --data-root /path/to/dataset \
  --manifest metadata/dataset_manifest.csv \
  --output-dir outputs/baseline_benchmark
```

The default public-release configuration uses a fixed binary decision threshold of `0.50`.

Validation-based threshold optimisation should only be enabled when that procedure is explicitly reported in the manuscript and analysis protocol.

Generated outputs include:

```text
outputs/baseline_benchmark/
├── dataset_summary/
├── splits/
├── models/
├── final_comparison/
├── paper_figures/
└── run_metadata/
```

---

## Running the frequency analysis

Example:

```bash
python src/frequency_magnitude_analysis.py \
  --data-root /path/to/audio_dataset \
  --manifest metadata/audio_manifest.csv \
  --output-dir outputs/frequency_analysis
```

Optional fixed-rate resampling:

```bash
python src/frequency_magnitude_analysis.py \
  --data-root /path/to/audio_dataset \
  --manifest metadata/audio_manifest.csv \
  --output-dir outputs/frequency_analysis \
  --target-sr 48000
```

Important parameters include:

```text
n_fft                  = 2048
hop_length             = 128
frequency range        = 20–5000 Hz
top-k components       = 10
minimum peak distance  = 30 Hz
peak prominence ratio  = 0.02
```

These values should remain unchanged when reproducing the reported analysis unless an alternative configuration is being evaluated explicitly.

---

## Running external temporal validation

Example:

```bash
python src/external_video_validation.py \
  --video /path/to/source_video.mp4 \
  --model /path/to/trained_model.keras \
  --output-dir outputs/external_validation \
  --training-sr 48000
```

The `--training-sr` value must match the sample rate actually used when generating the model’s training inputs.

Do not select a value solely because it appears in a manuscript draft. The code, model inputs, metadata, and final Methods section must report the same sample rate.

Default temporal settings:

```text
window duration        = 2.0 seconds
window hop             = 1.0 second
primary threshold      = 0.90
sensitivity thresholds = 0.80, 0.90, 0.95
persistence rule       = causal 2-of-3
```

Optional outputs:

```bash
--save-lms-images
--save-annotated-video
```

These options are disabled by default because generated images and derivative videos may be subject to copyright, licensing, privacy, or research-governance restrictions.

---

## Reproducibility safeguards

The public-release scripts include several safeguards intended to improve reproducibility:

- Source-aware dataset manifests
- Rejection of missing and duplicate records
- Prevention of paths escaping the dataset root
- Train, validation, and test leakage checks
- Saved split CSV files
- Explicit random seeds
- Saved configurations
- Software-version metadata
- Model and source-file hashes
- Prediction-level result files
- No silent fallback from pretrained to randomly initialised models
- No absolute local paths in public result tables
- Explicit declaration of thresholds and sampling rates

For exact reproduction, the split files used in the paper should be preserved and released with the repository.

---

## Data availability and redistribution

This repository does not automatically grant permission to redistribute third-party audio or video.

Do not upload:

- Copyrighted source videos
- Commercial stock audio
- Restricted online recordings
- Extracted audio from third-party videos
- Annotated derivatives of restricted footage
- Files containing identifiable speech where redistribution is not permitted
- Model weights trained on data that cannot legally be redistributed, unless permitted

Where direct redistribution is not possible, a data-access record may include:

- Source identifiers
- Original URLs where appropriate
- Access dates
- Temporal boundaries
- Labels
- Processing parameters
- Source-level metadata
- Instructions for reconstructing the research dataset lawfully

Users are responsible for complying with the relevant copyright, privacy, platform, ethics, and institutional requirements.

---

## Interpretation and intended use

The models in this repository classify short acoustic windows into operational sound-event categories.

They do not:

- Diagnose panic
- Infer psychological state
- Estimate crowd density
- Identify speakers
- Transcribe speech
- Localise individuals
- Replace human supervision
- Guarantee detection of hazardous crowd conditions
- Establish that an alert would have prevented a historical disaster

The system should be considered a potential complementary sensing layer within a broader crowd-safety framework.

Any real-world deployment would require prospective validation, calibrated alert thresholds, environmental testing, hardware evaluation, governance procedures, and integration with trained human operators.

---

## Privacy considerations

The proposed analysis operates on short-duration acoustic representations and does not require speaker identification or speech transcription.

Nevertheless, environmental audio may still contain sensitive or identifiable information.

Responsible deployment should consider:

- On-device or edge processing
- Immediate deletion of raw audio
- Storage of non-reconstructable features rather than recordings
- Restricted access
- Retention limits
- Encryption
- Public notification and governance
- Compliance with applicable surveillance and privacy law

---

## Citation

When using this repository, please cite the associated paper:

```bibtex
@article{parineh_acoustic_distress,
  title   = {Acoustic signatures enable occlusion-resistant early warning for crowd crushes},
  author  = {Parineh, Hossein and Haghani, Milad and Rajabifard, Abbas},
  journal = {To be updated},
  year    = {2026},
  doi     = {To be updated}
}
```

Update the journal, year, volume, article number, and DOI after publication.

A machine-readable citation should also be provided through `CITATION.cff`.

---

## Licence

The source code is released under the licence provided in the `LICENSE` file.

The code licence applies only to the original source code in this repository. It does not apply to third-party datasets, audio recordings, source videos, pretrained weights, or other externally owned material.

---

## Contact

For questions regarding the methodology, dataset documentation, or reproducibility, please contact:

**Hossein Parineh**  
Centre for Spatial Data Infrastructures and Land Administration  
The University of Melbourne  
Melbourne, Australia

Repository issues should be used for:

- Reproducibility problems
- Code defects
- Unclear instructions
- Missing dependencies
- Documentation corrections

Please do not use public issues to share private data, restricted media, credentials, or copyrighted source material.

---

## Acknowledgements

This repository supports research into privacy-conscious and multimodal sensing approaches for crowd-safety monitoring.

The authors acknowledge the original creators and rights holders of publicly accessible source materials used under the applicable research, licensing, and institutional conditions.
