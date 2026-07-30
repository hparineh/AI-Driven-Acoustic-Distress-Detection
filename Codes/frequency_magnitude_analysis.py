#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Source-aware frequency–magnitude analysis for acoustic distress research.

This public-release script:
  * reads audio through a provenance manifest;
  * extracts the top-k dominant frequency–magnitude components;
  * exports only repository-relative identifiers, never local absolute paths;
  * records all analysis parameters and software versions;
  * generates pooled and rank-wise figures from the exported CSV.

The figures are descriptive. They do not, by themselves, establish statistical
significance, causal effects, or universal class separability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ALLOWED_LABELS = ("BG", "Distress")
REQUIRED_MANIFEST_COLUMNS = ("relative_path", "label", "source_id")
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


@dataclass(frozen=True)
class AnalysisConfig:
    target_sr: int | None
    mono: bool
    n_fft: int
    hop_length: int
    window: str
    freq_min_hz: float
    freq_max_hz: float
    top_k: int
    min_peak_distance_hz: float
    peak_prominence_ratio: float
    normalize_audio: bool
    histogram_bins: int
    dpi: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and plot dominant frequency–magnitude components."
    )
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Root directory containing the audio files.")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="CSV with relative_path,label,source_id.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-sr", type=int, default=None,
                        help="Optional resampling rate. Omit to retain native rates.")
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--freq-min", type=float, default=20.0)
    parser.add_argument("--freq-max", type=float, default=5000.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-peak-distance-hz", type=float, default=30.0)
    parser.add_argument("--peak-prominence-ratio", type=float, default=0.02)
    parser.add_argument("--normalize-audio", action="store_true")
    parser.add_argument("--histogram-bins", type=int, default=80)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--fail-on-error", action="store_true",
                        help="Stop on the first unreadable audio file.")
    return parser.parse_args()


def resolved_under_root(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Manifest path escapes data root: {relative_path!r}"
        ) from exc
    return candidate


def validate_manifest(manifest_path: Path, data_root: Path) -> pd.DataFrame:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root not found: {data_root}")

    df = pd.read_csv(manifest_path, dtype=str)
    missing = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    df = df.loc[:, REQUIRED_MANIFEST_COLUMNS].copy()
    for col in REQUIRED_MANIFEST_COLUMNS:
        df[col] = df[col].fillna("").str.strip()

    if (df["relative_path"] == "").any():
        raise ValueError("Manifest contains empty relative_path values.")
    if (df["source_id"] == "").any():
        raise ValueError("Manifest contains empty source_id values.")

    invalid_labels = sorted(set(df["label"]) - set(ALLOWED_LABELS))
    if invalid_labels:
        raise ValueError(f"Unsupported labels: {invalid_labels}")

    duplicated = df["relative_path"].duplicated(keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, "relative_path"].head(10).tolist()
        raise ValueError(f"Duplicate relative_path entries found, e.g. {examples}")

    absolute_paths: list[Path] = []
    missing_files: list[str] = []
    invalid_extensions: list[str] = []

    for rel in df["relative_path"]:
        path = resolved_under_root(data_root, rel)
        absolute_paths.append(path)
        if not path.is_file():
            missing_files.append(rel)
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            invalid_extensions.append(rel)

    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} manifest files are missing. "
            f"Examples: {missing_files[:10]}"
        )
    if invalid_extensions:
        raise ValueError(
            f"Unsupported audio extensions. Examples: {invalid_extensions[:10]}"
        )

    df["_absolute_path"] = absolute_paths
    return df


def stable_record_id(relative_path: str, source_id: str) -> str:
    payload = f"{source_id}\n{relative_path}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def load_audio(path: Path, target_sr: int | None, normalize: bool) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise ValueError("Audio file is empty.")
    if not np.isfinite(y).all():
        raise ValueError("Audio contains NaN or infinite values.")

    y = y - float(np.mean(y))
    if normalize:
        peak = float(np.max(np.abs(y)))
        if peak > 0:
            y = y / peak
    return y, int(sr)


def extract_top_k(
    y: np.ndarray,
    sr: int,
    config: AnalysisConfig,
) -> list[tuple[int, float, float]]:
    if len(y) < 2:
        raise ValueError("Audio is too short for spectral analysis.")

    stft = librosa.stft(
        y=y,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        window=config.window,
    )
    spectrum = np.mean(np.abs(stft), axis=1)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=config.n_fft)

    mask = (
        (frequencies >= config.freq_min_hz)
        & (frequencies <= config.freq_max_hz)
    )
    frequencies = frequencies[mask]
    spectrum = spectrum[mask]

    if frequencies.size < config.top_k:
        raise ValueError(
            "Selected frequency range contains fewer bins than top_k."
        )

    resolution = (
        float(frequencies[1] - frequencies[0])
        if frequencies.size > 1 else 1.0
    )
    distance_bins = max(
        1, int(round(config.min_peak_distance_hz / resolution))
    )
    max_magnitude = float(np.max(spectrum))
    prominence = config.peak_prominence_ratio * max_magnitude

    peak_indices, _ = find_peaks(
        spectrum,
        distance=distance_bins,
        prominence=prominence,
    )

    # Preserve the documented fallback but make it explicit in output metadata.
    if peak_indices.size < config.top_k:
        candidate_indices = np.argsort(spectrum)[::-1]
    else:
        ordered = np.argsort(spectrum[peak_indices])[::-1]
        candidate_indices = peak_indices[ordered]

    selected = candidate_indices[: config.top_k]
    selected = selected[np.argsort(spectrum[selected])[::-1]]

    return [
        (rank, float(frequencies[idx]), float(spectrum[idx]))
        for rank, idx in enumerate(selected, start=1)
    ]


def extract_dataset(
    manifest_df: pd.DataFrame,
    config: AnalysisConfig,
    fail_on_error: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    errors: list[dict] = []

    for index, record in manifest_df.reset_index(drop=True).iterrows():
        rel = str(record["relative_path"])
        label = str(record["label"])
        source_id = str(record["source_id"])
        path = Path(record["_absolute_path"])
        record_id = stable_record_id(rel, source_id)

        try:
            y, sr = load_audio(path, config.target_sr, config.normalize_audio)
            peaks = extract_top_k(y, sr, config)
            duration = len(y) / sr

            for rank, frequency, magnitude in peaks:
                rows.append({
                    "record_id": record_id,
                    "relative_path": rel.replace("\\", "/"),
                    "source_id": source_id,
                    "class": label,
                    "rank": rank,
                    "frequency_hz": frequency,
                    "magnitude": magnitude,
                    "sampling_rate_hz": sr,
                    "duration_sec": duration,
                })
        except Exception as exc:
            error = {
                "record_id": record_id,
                "relative_path": rel.replace("\\", "/"),
                "source_id": source_id,
                "class": label,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            errors.append(error)
            if fail_on_error:
                raise RuntimeError(f"Failed to process {rel}: {exc}") from exc

        if (index + 1) % 100 == 0 or index + 1 == len(manifest_df):
            print(f"Processed {index + 1}/{len(manifest_df)} files")

    feature_df = pd.DataFrame(rows)
    error_df = pd.DataFrame(errors)

    if feature_df.empty:
        raise RuntimeError("No frequency features were extracted.")
    return feature_df, error_df


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig: plt.Figure, output_stem: Path, dpi: int) -> None:
    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def class_subsets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bg = df[df["class"] == "BG"]
    distress = df[df["class"] == "Distress"]
    if bg.empty or distress.empty:
        raise ValueError("Both BG and Distress classes must be present.")
    return bg, distress


def plot_pooled(df: pd.DataFrame, output_dir: Path, config: AnalysisConfig) -> None:
    configure_plot_style()
    bg, distress = class_subsets(df)
    bins = np.linspace(
        config.freq_min_hz, config.freq_max_hz, config.histogram_bins + 1
    )

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.scatter(bg["frequency_hz"], bg["magnitude"], s=10, alpha=0.10,
               edgecolors="none", label="BG")
    ax.scatter(distress["frequency_hz"], distress["magnitude"], s=10,
               alpha=0.10, edgecolors="none", label="Distress")
    ax.set(xlabel="Frequency (Hz)", ylabel="Magnitude",
           title="Frequency–magnitude distribution")
    ax.set_xlim(config.freq_min_hz, config.freq_max_hz)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir / "frequency_magnitude_scatter", config.dpi)

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.hist(bg["frequency_hz"], bins=bins, alpha=0.50, label="BG")
    ax.hist(distress["frequency_hz"], bins=bins, alpha=0.50,
            label="Distress")
    ax.set(xlabel="Frequency (Hz)", ylabel="Count",
           title="Dominant-frequency occurrence")
    ax.set_xlim(config.freq_min_hz, config.freq_max_hz)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir / "frequency_histogram", config.dpi)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    axes[0].scatter(bg["frequency_hz"], bg["magnitude"], s=10, alpha=0.10,
                    edgecolors="none", label="BG")
    axes[0].scatter(distress["frequency_hz"], distress["magnitude"], s=10,
                    alpha=0.10, edgecolors="none", label="Distress")
    axes[0].set(xlabel="Frequency (Hz)", ylabel="Magnitude",
                title="Frequency–magnitude distribution")
    axes[0].set_xlim(config.freq_min_hz, config.freq_max_hz)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].hist(bg["frequency_hz"], bins=bins, alpha=0.50, label="BG")
    axes[1].hist(distress["frequency_hz"], bins=bins, alpha=0.50,
                 label="Distress")
    axes[1].set(xlabel="Frequency (Hz)", ylabel="Count",
                title="Dominant-frequency occurrence")
    axes[1].set_xlim(config.freq_min_hz, config.freq_max_hz)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[0].text(0.5, -0.19, "(a)", transform=axes[0].transAxes,
                 ha="center", va="center")
    axes[1].text(0.5, -0.19, "(b)", transform=axes[1].transAxes,
                 ha="center", va="center")
    fig.tight_layout()
    save_figure(fig, output_dir / "combined_frequency_analysis", config.dpi)


def plot_rankwise(df: pd.DataFrame, output_dir: Path, config: AnalysisConfig) -> None:
    rank_dir = output_dir / "rankwise"
    rank_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    for rank in sorted(df["rank"].unique()):
        rank_df = df[df["rank"] == rank]
        bg, distress = class_subsets(rank_df)
        fig, ax = plt.subplots(figsize=(8.5, 6.2))
        ax.scatter(bg["frequency_hz"], bg["magnitude"], s=12, alpha=0.15,
                   edgecolors="none", label="BG")
        ax.scatter(distress["frequency_hz"], distress["magnitude"], s=12,
                   alpha=0.15, edgecolors="none", label="Distress")
        ax.set(
            xlabel="Frequency (Hz)",
            ylabel="Magnitude",
            title=f"Rank {rank}: frequency–magnitude distribution",
        )
        ax.set_xlim(config.freq_min_hz, config.freq_max_hz)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, rank_dir / f"rank_{rank:02d}", config.dpi)


def save_metadata(
    output_dir: Path,
    config: AnalysisConfig,
    manifest_df: pd.DataFrame,
    features_df: pd.DataFrame,
    errors_df: pd.DataFrame,
) -> None:
    metadata = {
        "analysis": "top-k dominant frequency–magnitude extraction",
        "config": asdict(config),
        "n_manifest_records": int(len(manifest_df)),
        "n_processed_records": int(features_df["record_id"].nunique()),
        "n_failed_records": int(len(errors_df)),
        "class_record_counts": {
            str(k): int(v)
            for k, v in (
                features_df[["record_id", "class"]]
                .drop_duplicates()["class"].value_counts().items()
            )
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "librosa": librosa.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "interpretation_note": (
            "Outputs are descriptive distributions and do not alone establish "
            "statistical significance, causality, or universal separability."
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.n_fft <= 0 or args.hop_length <= 0 or args.top_k <= 0:
        raise ValueError("n_fft, hop_length, and top_k must be positive.")
    if not 0 <= args.peak_prominence_ratio <= 1:
        raise ValueError("peak-prominence-ratio must be between 0 and 1.")
    if args.freq_min < 0 or args.freq_max <= args.freq_min:
        raise ValueError("Invalid frequency range.")

    config = AnalysisConfig(
        target_sr=args.target_sr,
        mono=True,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        window="hann",
        freq_min_hz=args.freq_min,
        freq_max_hz=args.freq_max,
        top_k=args.top_k,
        min_peak_distance_hz=args.min_peak_distance_hz,
        peak_prominence_ratio=args.peak_prominence_ratio,
        normalize_audio=args.normalize_audio,
        histogram_bins=args.histogram_bins,
        dpi=args.dpi,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = validate_manifest(args.manifest.resolve(), args.data_root.resolve())
    features_df, errors_df = extract_dataset(
        manifest_df, config, args.fail_on_error
    )

    features_df.to_csv(output_dir / "top_k_frequency_magnitude.csv", index=False)
    errors_df.to_csv(output_dir / "processing_errors.csv", index=False)
    plot_pooled(features_df, output_dir, config)
    plot_rankwise(features_df, output_dir, config)
    save_metadata(output_dir, config, manifest_df, features_df, errors_df)

    print(f"Completed. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
