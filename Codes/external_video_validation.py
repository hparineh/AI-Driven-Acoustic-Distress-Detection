#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
External temporal validation for an acoustic-distress classifier.

Public-release safeguards:
  * all paths and scientific parameters are command-line arguments;
  * the audio sampling rate must be declared explicitly;
  * generated LMS images are not retained by default;
  * absolute source/model paths are not written to public output tables;
  * annotated video export is opt-in because source-video redistribution may
    be restricted;
  * model input/output dimensions and threshold settings are validated;
  * metadata records exact parameters and file hashes.

This script performs model inference. It does not establish that an alert would
have prevented a historical event, and its outputs must be interpreted as an
external case study rather than prospective operational validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


@dataclass(frozen=True)
class ValidationConfig:
    training_sr_hz: int
    image_height: int
    image_width: int
    channels: int
    n_fft: int
    lms_hop_length: int
    n_mels: int
    audio_window_sec: float
    audio_hop_sec: float
    thresholds: tuple[float, ...]
    primary_threshold: float
    smoothing_window: int
    smoothing_min_positive: int
    batch_size: int


def parse_thresholds(value: str) -> tuple[float, ...]:
    try:
        thresholds = tuple(float(x.strip()) for x in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Thresholds must be comma-separated numbers.") from exc
    if not thresholds or any(not 0 < x < 1 for x in thresholds):
        raise argparse.ArgumentTypeError("Every threshold must be between 0 and 1.")
    return thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run external temporal validation on a video."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--training-sr",
        type=int,
        required=True,
        help=(
            "Exact sample rate used to create the training LMS data. "
            "Use 48000 only if this matches the actual training pipeline."
        ),
    )
    parser.add_argument("--image-height", type=int, default=163)
    parser.add_argument("--image-width", type=int, default=279)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--lms-hop-length", type=int, default=128)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--hop-sec", type=float, default=1.0)
    parser.add_argument("--thresholds", type=parse_thresholds,
                        default=(0.80, 0.90, 0.95))
    parser.add_argument("--primary-threshold", type=float, default=0.90)
    parser.add_argument("--smoothing-window", type=int, default=3)
    parser.add_argument("--smoothing-min-positive", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--embedding-layer",
        type=str,
        default=None,
        help=(
            "Optional named embedding layer. If omitted, the input to the "
            "final output layer is used."
        ),
    )
    parser.add_argument(
        "--save-lms-images",
        action="store_true",
        help="Retain generated LMS PNGs. Off by default.",
    )
    parser.add_argument(
        "--save-annotated-video",
        action="store_true",
        help=(
            "Export an annotated derivative video. Use only when redistribution "
            "and publication permissions allow it."
        ),
    )
    parser.add_argument("--display-video", action="store_true")
    parser.add_argument(
        "--include-source-filenames-in-metadata",
        action="store_true",
        help=(
            "Include only source basenames, not absolute paths. Off by default "
            "for public metadata."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def get_ffmpeg_executable() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg is required. Install ffmpeg or imageio-ffmpeg."
        ) from exc


def validate_inputs(args: argparse.Namespace) -> ValidationConfig:
    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if args.training_sr <= 0:
        raise ValueError("training-sr must be positive.")
    if args.n_fft <= 0 or args.lms_hop_length <= 0 or args.n_mels <= 0:
        raise ValueError("LMS parameters must be positive.")
    if args.window_sec <= 0 or args.hop_sec <= 0:
        raise ValueError("Window and hop durations must be positive.")
    if not 0 < args.primary_threshold < 1:
        raise ValueError("primary-threshold must be between 0 and 1.")
    if not any(np.isclose(args.primary_threshold, t) for t in args.thresholds):
        raise ValueError("primary-threshold must be included in --thresholds.")
    if not 1 <= args.smoothing_min_positive <= args.smoothing_window:
        raise ValueError(
            "smoothing-min-positive must be between 1 and smoothing-window."
        )

    return ValidationConfig(
        training_sr_hz=args.training_sr,
        image_height=args.image_height,
        image_width=args.image_width,
        channels=args.channels,
        n_fft=args.n_fft,
        lms_hop_length=args.lms_hop_length,
        n_mels=args.n_mels,
        audio_window_sec=args.window_sec,
        audio_hop_sec=args.hop_sec,
        thresholds=tuple(args.thresholds),
        primary_threshold=args.primary_threshold,
        smoothing_window=args.smoothing_window,
        smoothing_min_positive=args.smoothing_min_positive,
        batch_size=args.batch_size,
    )


def extract_audio(video_path: Path, wav_path: Path, sample_rate: int) -> None:
    command = [
        get_ffmpeg_executable(),
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(command, check=True)


def normalize_audio(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    peak_to_peak = float(np.ptp(array))
    if peak_to_peak == 0:
        return array
    return 2.0 * ((array - float(np.min(array))) / peak_to_peak) - 1.0


def create_windows(audio: np.ndarray, sr: int, config: ValidationConfig):
    window_length = int(round(config.audio_window_sec * sr))
    hop_length = int(round(config.audio_hop_sec * sr))
    if window_length <= 0 or hop_length <= 0:
        raise ValueError("Window/hop lengths resolve to zero samples.")

    windows: list[np.ndarray] = []
    rows: list[dict] = []
    total = len(audio)

    for window_id, start in enumerate(range(0, total, hop_length)):
        end = start + window_length
        segment = audio[start:end]
        valid_end = min(end, total)
        if len(segment) < window_length:
            segment = np.pad(segment, (0, window_length - len(segment)))
        windows.append(np.asarray(segment, dtype=np.float32))
        rows.append({
            "window_id": window_id,
            "start_time_sec": start / sr,
            "end_time_sec": valid_end / sr,
            "center_time_sec": ((start / sr) + (valid_end / sr)) / 2.0,
            "is_padded_final_window": bool(end > total),
        })
    return windows, pd.DataFrame(rows)


def lms_rgb_array(segment: np.ndarray, sr: int, config: ValidationConfig) -> np.ndarray:
    normalized = normalize_audio(segment)
    mel = librosa.feature.melspectrogram(
        y=normalized,
        sr=sr,
        n_fft=config.n_fft,
        hop_length=config.lms_hop_length,
        n_mels=config.n_mels,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Reproduce the original figure-rendering route without retaining files.
    fig = plt.figure(figsize=(5, 3), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    librosa.display.specshow(
        log_mel,
        sr=sr,
        hop_length=config.lms_hop_length,
        x_axis=None,
        y_axis="mel",
        ax=ax,
    )
    ax.axis("off")
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[:, :, :3].copy()
    plt.close(fig)

    resized = cv2.resize(
        rgb,
        (config.image_width, config.image_height),
        interpolation=cv2.INTER_AREA,
    )
    if config.channels == 1:
        resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)[..., None]
    elif config.channels != 3:
        raise ValueError("Only 1 or 3 channels are supported.")
    return resized.astype(np.float32) / 255.0


def compute_descriptors(segment: np.ndarray, sr: int, config: ValidationConfig) -> dict:
    eps = 1e-12
    rms = float(np.sqrt(np.mean(np.square(segment), dtype=np.float64)))
    power = np.abs(
        librosa.stft(
            segment,
            n_fft=config.n_fft,
            hop_length=config.lms_hop_length,
        )
    ) ** 2

    return {
        "rms_db": float(20.0 * np.log10(rms + eps)),
        "spectral_centroid_hz": float(
            librosa.feature.spectral_centroid(S=power, sr=sr).mean()
        ),
        "spectral_bandwidth_hz": float(
            librosa.feature.spectral_bandwidth(S=power, sr=sr).mean()
        ),
        "spectral_rolloff_hz": float(
            librosa.feature.spectral_rolloff(
                S=power, sr=sr, roll_percent=0.85
            ).mean()
        ),
        "spectral_flatness": float(
            librosa.feature.spectral_flatness(S=power).mean()
        ),
        "zero_crossing_rate": float(
            librosa.feature.zero_crossing_rate(segment).mean()
        ),
    }


def load_model(model_path: Path, config: ValidationConfig) -> keras.Model:
    model = keras.models.load_model(model_path, compile=False)
    input_shape = model.input_shape
    output_shape = model.output_shape

    if isinstance(input_shape, list) or len(input_shape) != 4:
        raise ValueError(f"Expected one 4D image input, received {input_shape}.")
    expected = (
        config.image_height,
        config.image_width,
        config.channels,
    )
    actual = tuple(input_shape[1:])
    if actual != expected:
        raise ValueError(
            f"Model input shape {actual} does not match configured {expected}."
        )

    if isinstance(output_shape, list) or len(output_shape) != 2:
        raise ValueError(f"Unexpected model output shape: {output_shape}")
    if output_shape[-1] not in (1, 2):
        raise ValueError(
            "Model output must contain one sigmoid score or two class scores."
        )
    return model


def build_embedding_model(
    model: keras.Model, embedding_layer: str | None
) -> keras.Model:
    if embedding_layer:
        try:
            output = model.get_layer(embedding_layer).output
        except ValueError as exc:
            names = [layer.name for layer in model.layers]
            raise ValueError(
                f"Embedding layer {embedding_layer!r} not found. "
                f"Available layers: {names}"
            ) from exc
    else:
        if len(model.layers) < 2:
            raise ValueError("Model has no penultimate representation.")
        output = model.layers[-1].input

    embedding_model = keras.Model(model.inputs, output)
    if len(embedding_model.output_shape) != 2:
        raise ValueError(
            "Selected embedding output must be a two-dimensional matrix."
        )
    return embedding_model


def apply_causal_persistence(
    flags: np.ndarray, window: int, minimum: int
) -> np.ndarray:
    flags = np.asarray(flags, dtype=bool)
    smoothed = np.zeros_like(flags, dtype=bool)
    for index in range(len(flags)):
        start = max(0, index - window + 1)
        smoothed[index] = int(flags[start:index + 1].sum()) >= minimum
    return smoothed


def merge_events(
    results: pd.DataFrame,
    output_path: Path,
    max_gap_sec: float,
) -> pd.DataFrame:
    detected = results[results["smoothed_distress_flag"]].copy()
    columns = [
        "event_id", "event_start_sec", "event_end_sec",
        "event_duration_sec", "max_proba_distress",
        "mean_proba_distress", "num_positive_windows",
    ]
    if detected.empty:
        empty = pd.DataFrame(columns=columns)
        empty.to_csv(output_path, index=False)
        return empty

    detected = detected.sort_values("start_time_sec").reset_index(drop=True)
    events: list[dict] = []
    start = float(detected.loc[0, "start_time_sec"])
    end = float(detected.loc[0, "end_time_sec"])
    scores = [float(detected.loc[0, "proba_Distress"])]

    for row in detected.iloc[1:].itertuples(index=False):
        row_start = float(row.start_time_sec)
        row_end = float(row.end_time_sec)
        score = float(row.proba_Distress)
        if row_start - end <= max_gap_sec:
            end = max(end, row_end)
            scores.append(score)
        else:
            events.append({
                "event_id": len(events) + 1,
                "event_start_sec": start,
                "event_end_sec": end,
                "event_duration_sec": end - start,
                "max_proba_distress": float(np.max(scores)),
                "mean_proba_distress": float(np.mean(scores)),
                "num_positive_windows": len(scores),
            })
            start, end, scores = row_start, row_end, [score]

    events.append({
        "event_id": len(events) + 1,
        "event_start_sec": start,
        "event_end_sec": end,
        "event_duration_sec": end - start,
        "max_proba_distress": float(np.max(scores)),
        "mean_proba_distress": float(np.mean(scores)),
        "num_positive_windows": len(scores),
    })
    event_df = pd.DataFrame(events, columns=columns)
    event_df.to_csv(output_path, index=False)
    return event_df


def plot_probability(results: pd.DataFrame, output_dir: Path, config: ValidationConfig) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(results["center_time_sec"], results["proba_Distress"],
            linewidth=1.0, label="P(Distress)")
    for threshold in config.thresholds:
        ax.axhline(threshold, linestyle="--", linewidth=0.8,
                   label=f"Threshold {threshold:.2f}")
    ax.set(
        xlabel="Time (s)",
        ylabel="Distress probability",
        ylim=(0, 1),
        title="Temporal distress probability",
    )
    ax.legend(ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "distress_probability.png", dpi=300)
    fig.savefig(output_dir / "distress_probability.pdf")
    plt.close(fig)


def annotate_video(
    video_path: Path,
    results: pd.DataFrame,
    output_path: Path,
    display: bool,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Could not open source video.")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Source video has an invalid frame rate.")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    silent_path = output_path.with_name(output_path.stem + "_silent.mp4")
    writer = cv2.VideoWriter(
        str(silent_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not initialise annotated-video writer.")

    centers = results["center_time_sec"].to_numpy()
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            result_index = int(np.argmin(np.abs(centers - timestamp)))
            row = results.iloc[result_index]
            distress = bool(row["smoothed_distress_flag"])
            probability = float(row["proba_Distress"])
            label = "DISTRESS DETECTED" if distress else "BG"
            colour = (0, 0, 255) if distress else (0, 150, 0)
            text = f"{label} | P(Distress)={probability:.2f}"

            cv2.rectangle(frame, (20, 20), (min(width - 20, 900), 85),
                          colour, -1)
            cv2.putText(frame, text, (35, 63), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 3, cv2.LINE_AA)
            writer.write(frame)

            if display:
                cv2.imshow("External validation", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            frame_index += 1
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    command = [
        get_ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(silent_path), "-i", str(video_path),
        "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest", str(output_path),
    ]
    subprocess.run(command, check=True)
    silent_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    config = validate_inputs(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model.resolve(), config)
    embedding_model = build_embedding_model(model, args.embedding_layer)

    with tempfile.TemporaryDirectory() as temp_directory:
        temp_dir = Path(temp_directory)
        wav_path = temp_dir / "extracted_audio.wav"
        extract_audio(args.video.resolve(), wav_path, config.training_sr_hz)

        audio, actual_sr = librosa.load(wav_path, sr=None, mono=True)
        audio = np.asarray(audio, dtype=np.float32)
        if actual_sr != config.training_sr_hz:
            raise RuntimeError(
                f"Extracted sample rate {actual_sr} differs from declared "
                f"training rate {config.training_sr_hz}."
            )

        windows, results = create_windows(audio, actual_sr, config)
        images: list[np.ndarray] = []
        descriptors: list[dict] = []

        retained_lms_dir = output_dir / "generated_lms_images"
        if args.save_lms_images:
            retained_lms_dir.mkdir(parents=True, exist_ok=True)

        for index, segment in enumerate(windows):
            image = lms_rgb_array(segment, actual_sr, config)
            images.append(image)
            descriptors.append(compute_descriptors(segment, actual_sr, config))

            if args.save_lms_images:
                png = (np.clip(image, 0, 1) * 255).astype(np.uint8)
                if config.channels == 3:
                    png = cv2.cvtColor(png, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(retained_lms_dir / f"window_{index:05d}.png"), png
                )

            if (index + 1) % 100 == 0 or index + 1 == len(windows):
                print(f"Prepared {index + 1}/{len(windows)} windows")

        input_tensor = np.asarray(images, dtype=np.float32)
        probabilities = model.predict(
            input_tensor, batch_size=config.batch_size, verbose=1
        )
        embeddings = embedding_model.predict(
            input_tensor, batch_size=config.batch_size, verbose=1
        )
        np.save(output_dir / "window_embeddings.npy", embeddings)

        if probabilities.ndim != 2:
            raise ValueError(f"Unexpected prediction shape: {probabilities.shape}")
        if probabilities.shape[1] == 2:
            bg_scores = probabilities[:, 0]
            distress_scores = probabilities[:, 1]
        elif probabilities.shape[1] == 1:
            distress_scores = probabilities[:, 0]
            bg_scores = 1.0 - distress_scores
        else:
            raise ValueError(f"Unexpected prediction shape: {probabilities.shape}")

        descriptor_df = pd.DataFrame(descriptors)
        results = pd.concat([results, descriptor_df], axis=1)
        results["proba_BG"] = bg_scores
        results["proba_Distress"] = distress_scores

        for threshold in config.thresholds:
            tag = f"{int(round(threshold * 100)):02d}"
            raw = distress_scores >= threshold
            results[f"raw_above_{tag}"] = raw
            results[f"persistent_{config.smoothing_min_positive}of"
                    f"{config.smoothing_window}_{tag}"] = apply_causal_persistence(
                        raw,
                        config.smoothing_window,
                        config.smoothing_min_positive,
                    )

        primary_tag = f"{int(round(config.primary_threshold * 100)):02d}"
        primary_column = (
            f"persistent_{config.smoothing_min_positive}of"
            f"{config.smoothing_window}_{primary_tag}"
        )
        results["smoothed_distress_flag"] = results[primary_column]
        results["pred_label"] = np.where(
            results["smoothed_distress_flag"], "Distress", "BG"
        )

        results.to_csv(
            output_dir / "video_distress_predictions_enriched.csv", index=False
        )
        merge_events(
            results,
            output_dir / "merged_distress_events.csv",
            max_gap_sec=config.audio_hop_sec * 1.5,
        )
        plot_probability(results, output_dir, config)

        metadata = {
            "analysis": "external temporal validation",
            "config": asdict(config),
            "video_sha256": sha256_file(args.video.resolve()),
            "model_sha256": sha256_file(args.model.resolve()),
            "model_parameters": int(model.count_params()),
            "audio_duration_sec": float(len(audio) / actual_sr),
            "n_windows": int(len(results)),
            "embedding_dimension": int(embeddings.shape[1]),
            "software": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "librosa": librosa.__version__,
                "tensorflow": tf.__version__,
                "opencv": cv2.__version__,
            },
            "interpretation_note": (
                "This is a retrospective external case study, not prospective "
                "operational validation or evidence that an alert would have "
                "prevented the historical event."
            ),
        }
        if args.include_source_filenames_in_metadata:
            metadata["video_filename"] = args.video.name
            metadata["model_filename"] = args.model.name

        (output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        if args.save_annotated_video:
            annotate_video(
                args.video.resolve(),
                results,
                output_dir / "annotated_distress_video.mp4",
                args.display_video,
            )
        elif args.display_video:
            raise ValueError(
                "--display-video requires --save-annotated-video."
            )

    print(f"Completed. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
