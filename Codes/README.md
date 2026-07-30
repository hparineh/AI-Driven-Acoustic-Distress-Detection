# Acoustic Distress Analysis — Public Release Tools

This package contains two sanitised research scripts:

1. `src/frequency_magnitude_analysis.py`  
   Extracts the top-*k* dominant frequency–magnitude components from a
   provenance-controlled audio manifest and creates pooled and rank-wise plots.

2. `src/external_video_validation.py`  
   Runs retrospective temporal inference on an external video, calculates
   acoustic descriptors, exports embeddings, applies causal persistence, and
   optionally creates an annotated derivative video.

## Important scientific limitation

These scripts are components of the analysis, not the complete model-training
repository. They do not reproduce the attention-model benchmark or train the
classifier used for external validation.

The frequency figures are descriptive. The external-video procedure is a
retrospective case study, not prospective operational validation. Do not claim
that the script demonstrates that an alert would have prevented a historical
event.

## Data and copyright

Do not commit raw audio, source videos, extracted audio, generated LMS images,
model files, or annotated derivative videos unless you hold explicit rights to
redistribute them.

The external script therefore:

- does not retain LMS images by default;
- does not create an annotated video unless explicitly requested;
- records hashes rather than local source paths;
- excludes source filenames from metadata by default.

## Frequency-analysis manifest

Create a CSV based on `metadata/audio_manifest_template.csv`:

```csv
relative_path,label,source_id
BG/example_001.wav,BG,source_recording_001
Distress/example_002.wav,Distress,source_recording_002
```

Every segment derived from the same original recording must use the same
`source_id`, including recordings that contain both classes.

Run:

```bash
python src/frequency_magnitude_analysis.py \
  --data-root /path/to/audio \
  --manifest metadata/audio_manifest.csv \
  --output-dir outputs/frequency_analysis
```

To reproduce a fixed-rate analysis, add `--target-sr RATE`. Omitting it retains
each file's native sample rate and records the rate per record.

## External temporal validation

The sample rate is intentionally required rather than silently assumed:

```bash
python src/external_video_validation.py \
  --video /path/to/source_video.mp4 \
  --model /path/to/best_model.keras \
  --output-dir outputs/external_validation \
  --training-sr 48000
```

Use `48000` only when 48 kHz was actually used to generate the training inputs.
If the historical experiment used another rate, declare that exact rate and
align the manuscript accordingly.

Optional, rights-sensitive outputs:

```bash
--save-lms-images
--save-annotated-video
```

The last option creates a derivative of the source video and should not be used
for a public repository without redistribution permission.

## Model assumptions

The external-validation script expects:

- one image input of shape `(163, 279, 3)` by default;
- one sigmoid output or two class outputs;
- class order `BG`, then `Distress` for a two-output model.

Override image dimensions through command-line arguments where needed. Use
`--embedding-layer NAME` when the desired embedding is not the input to the
final output layer.

## Reproducibility note

The LMS renderer intentionally follows the original figure-rendering workflow.
A stronger future repository should centralise LMS generation in one shared
module used by both training and validation so pixel-level preprocessing cannot
drift.

## Installation

```bash
python -m pip install -r requirements.txt
```

FFmpeg must also be installed and available on `PATH`, or `imageio-ffmpeg` must
be installed.

## Licensing

Add a code licence only after all contributors approve it. A permissive licence
such as BSD-3-Clause or MIT is common for research software, but it does not
grant rights to redistribute third-party data, videos, recordings, or model
weights.
