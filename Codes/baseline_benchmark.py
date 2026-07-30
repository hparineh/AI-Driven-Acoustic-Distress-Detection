# -*- coding: utf-8 -*-
"""Publication-ready baseline LMS image-classification benchmark.

Purpose:
    Test non-attention baseline approaches for binary classification between
    BG and Distress classes from Log-Mel Spectrogram (LMS) images.

Important:
    This script intentionally EXCLUDES attention models because those results
    are assumed to already exist in a separate attention benchmark directory.

Baseline groups:
    A) Deep non-attention image models
        1. BaselineCNN
        2. BaselineCNN_Wide
        3. MobileNetV2
        4. EfficientNetB0
        5. DenseNet121
        6. ResNet50
        7. VGG16
        8. CRNN_GRU

    B) Classical LMS-image baselines
        1. PCA_LMS_LogisticRegression
        2. PCA_LMS_SVM_RBF
        3. PCA_LMS_RandomForest
        4. PCA_LMS_KNN
        5. PCA_LMS_XGBoost, if xgboost is installed

Outputs:
    - Per-model/per-seed metrics
    - Per-class metrics
    - Binary sensitivity/specificity and ROC-AUC/PR-AUC
    - Raw and normalized confusion matrices
    - Training curves for deep models
    - Prediction CSVs
    - Final comparison tables
    - Publication-ready plots
    - Excel workbook

Recommended environment:
    Python 3.10
    TensorFlow 2.10.x
    scikit-learn
    pandas
    numpy
    matplotlib
    opencv-python
    openpyxl

Public-release notes:
    - Dataset and output paths are supplied through command-line arguments.
    - A source manifest is required by default to enforce source-disjoint splits.
    - ImageNet weight-loading failures stop the run rather than silently changing the experiment.
    - Benchmark metrics use the fixed 0.50 decision threshold by default.
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================

import os
import re
import sys
import argparse
import platform
import gc
import json
import time
import random
import warnings
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)
from sklearn.utils.class_weight import compute_class_weight

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


warnings.filterwarnings("ignore")


# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

REPOSITORY_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("ADD_DATA_DIR", REPOSITORY_ROOT / "data" / "lms"))
OUTPUT_DIR = Path(os.environ.get("ADD_OUTPUT_DIR", REPOSITORY_ROOT / "outputs" / "baseline_benchmark"))
MANIFEST_PATH = Path(os.environ.get("ADD_MANIFEST", REPOSITORY_ROOT / "metadata" / "dataset_manifest.csv"))
REQUIRE_SOURCE_MANIFEST = True
PUBLICATION_FAIL_FAST = True

CLASS_NAMES = ["BG", "Distress"]

CLASS_FOLDER_MAP = {
    "BG": "BG",
    "Distress": "Distress"
}

IMAGE_SIZE = (163, 279)
CHANNELS = 3
NUM_CLASSES = len(CLASS_NAMES)

TEST_SIZE = 0.10
VAL_SIZE = 0.10

RANDOM_SEEDS = [42, 7, 21]

BATCH_SIZE = 32
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 12
REDUCE_LR_PATIENCE = 6

INITIAL_LR = 1e-4
MIN_LR = 1e-7

USE_CLASS_WEIGHTS = True
USE_AUGMENTATION = True

# =============================================================================
# IMBALANCE-AWARE SETTINGS FOR FULL DATASET
# =============================================================================
# Full dataset example: BG ≈ 19,784 and Distress ≈ 2,100.
# Do NOT downsample BG. Keep all data and compensate during training/evaluation.
#
# Options:
#   "class_weight_ce" : weighted categorical cross-entropy, recommended first run.
#   "focal_loss"      : focal loss + class weighting effect, useful if minority recall is poor.
IMBALANCE_LOSS_MODE = "class_weight_ce"

# For imbalanced binary testing, do not rely only on the default 0.50 threshold.
# The manuscript benchmark should use a fixed 0.50 threshold unless threshold
# optimisation is explicitly documented as part of the reported method.
TUNE_BINARY_THRESHOLD = False
THRESHOLD_OPTIMIZATION_METRIC = "macro_f1"   # "macro_f1", "distress_f1", "balanced_accuracy", "youden"
THRESHOLD_GRID = np.linspace(0.05, 0.95, 181)

SAVE_MODELS = True
RUN_QUICK_TEST = False
QUICK_TEST_MAX_PER_CLASS = 80

# Deep models without attention.
DEEP_MODEL_NAMES = [
    "BaselineCNN",
    "BaselineCNN_Wide",
    "MobileNetV2",
    "EfficientNetB0",
    "DenseNet121",
    "ResNet50",
    "VGG16",
    "CRNN_GRU"
]

# Classical baselines based on LMS images.
CLASSICAL_MODEL_NAMES = [
    "PCA_LMS_LogisticRegression",
    "PCA_LMS_SVM_RBF",
    "PCA_LMS_RandomForest",
    "PCA_LMS_KNN",
]

if XGBOOST_AVAILABLE:
    CLASSICAL_MODEL_NAMES.append("PCA_LMS_XGBoost")

RUN_DEEP_MODELS = True
RUN_CLASSICAL_MODELS = True

# Classical feature settings.
CLASSICAL_IMAGE_SIZE = (96, 160)   # smaller for memory efficiency: H x W
PCA_COMPONENTS = 256

# -----------------------------------------------------------------------------
# Split-integrity controls
# -----------------------------------------------------------------------------
# IMPORTANT:
# For audio-window datasets, random image-level splitting can overestimate
# performance if neighbouring windows from the same original recording appear in
# both train and test. Keep GROUP_AWARE_SPLIT=True for publication-level results.
#
# The script infers a group_id from each image filename. All images with the same
# group_id are forced into only one split: train, validation, or test.
#
# After the first run, inspect:
#   OUTPUT_DIR/dataset_summary/group_id_audit.csv
# to verify that group_id correctly corresponds to the original recording/source.
GROUP_AWARE_SPLIT = True
STRICT_SPLIT_INTEGRITY_CHECK = True

# If your filenames have a known pattern, adjust infer_group_id() below.
# Conservative default patterns remove common window/segment suffixes only.
GROUP_ID_PATTERNS_TO_REMOVE = [
    r"(_|-)?win(dow)?(_|-)?\d+.*$",
    r"(_|-)?seg(ment)?(_|-)?\d+.*$",
    r"(_|-)?chunk(_|-)?\d+.*$",
    r"(_|-)?frame(_|-)?\d+.*$",
    r"(_|-)?slice(_|-)?\d+.*$",
    r"(_|-)?start(_|-)?\d+.*$",
    r"(_|-)?from(_|-)?\d+.*$",
    r"(_|-)?t\d+(_|-)?to(_|-)?\d+.*$",
]

# Spectrogram-aware augmentation.
# Important: no horizontal flip and no vertical flip.
AUG_BRIGHTNESS_DELTA = 0.08
AUG_CONTRAST_LOW = 0.90
AUG_CONTRAST_HIGH = 1.10
AUG_NOISE_STD = 0.015
AUG_TIME_MASK_PROB = 0.50
AUG_FREQ_MASK_PROB = 0.50

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


# =============================================================================
# 2. ENVIRONMENT AND REPRODUCIBILITY
# =============================================================================

def set_all_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_environment():
    print("=" * 80)
    print("ENVIRONMENT")
    print("=" * 80)
    print("Timestamp:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Python executable:", os.sys.executable)
    print("TensorFlow version:", tf.__version__)
    print("XGBoost available:", XGBOOST_AVAILABLE)

    gpus = tf.config.list_physical_devices("GPU")
    print("Available GPUs:", gpus)

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled.")
        except Exception as e:
            print("Could not enable GPU memory growth:", e)

    print("=" * 80)


def make_dirs():
    dirs = [
        OUTPUT_DIR,
        os.path.join(OUTPUT_DIR, "dataset_summary"),
        os.path.join(OUTPUT_DIR, "splits"),
        os.path.join(OUTPUT_DIR, "models"),
        os.path.join(OUTPUT_DIR, "final_comparison"),
        os.path.join(OUTPUT_DIR, "paper_figures"),
        os.path.join(OUTPUT_DIR, "logs")
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_config():
    config = {
        "DATA_DIR": "<provided at runtime>",
        "OUTPUT_DIR": "<provided at runtime>",
        "MANIFEST_FILE": MANIFEST_PATH.name,
        "REQUIRE_SOURCE_MANIFEST": REQUIRE_SOURCE_MANIFEST,
        "CLASS_NAMES": CLASS_NAMES,
        "CLASS_FOLDER_MAP": CLASS_FOLDER_MAP,
        "IMAGE_SIZE": IMAGE_SIZE,
        "CHANNELS": CHANNELS,
        "NUM_CLASSES": NUM_CLASSES,
        "TEST_SIZE": TEST_SIZE,
        "VAL_SIZE": VAL_SIZE,
        "RANDOM_SEEDS": RANDOM_SEEDS,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "REDUCE_LR_PATIENCE": REDUCE_LR_PATIENCE,
        "INITIAL_LR": INITIAL_LR,
        "USE_CLASS_WEIGHTS": USE_CLASS_WEIGHTS,
        "USE_AUGMENTATION": USE_AUGMENTATION,
        "IMBALANCE_LOSS_MODE": IMBALANCE_LOSS_MODE,
        "TUNE_BINARY_THRESHOLD": TUNE_BINARY_THRESHOLD,
        "THRESHOLD_OPTIMIZATION_METRIC": THRESHOLD_OPTIMIZATION_METRIC,
        "SAVE_MODELS": SAVE_MODELS,
        "RUN_QUICK_TEST": RUN_QUICK_TEST,
        "DEEP_MODEL_NAMES": DEEP_MODEL_NAMES,
        "CLASSICAL_MODEL_NAMES": CLASSICAL_MODEL_NAMES,
        "CLASSICAL_IMAGE_SIZE": CLASSICAL_IMAGE_SIZE,
        "PCA_COMPONENTS": PCA_COMPONENTS,
        "GROUP_AWARE_SPLIT": GROUP_AWARE_SPLIT,
        "STRICT_SPLIT_INTEGRITY_CHECK": STRICT_SPLIT_INTEGRITY_CHECK,
        "GROUP_ID_PATTERNS_TO_REMOVE": GROUP_ID_PATTERNS_TO_REMOVE
    }

    with open(os.path.join(OUTPUT_DIR, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


# =============================================================================
# 3. DATASET LOADING AND SPLITS
# =============================================================================

def is_image_file(path):
    path = str(path).lower()
    return path.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))


def infer_group_id(file_path):
    """
    Infer the original recording/source ID from an LMS image filename.

    Why this matters:
        In audio-window datasets, many LMS images can be generated from the same
        original recording using overlapping windows. If related windows are
        randomly split across train/test, the model can see near-duplicate
        acoustic context during training, causing optimistic test accuracy.

    Default logic:
        1) Take filename stem.
        2) Remove common segment/window suffixes such as _win12, _seg003, etc.
        3) Prefix with the class folder name to avoid accidental collisions
           between BG and Distress files with identical stems.

    You should inspect group_id_audit.csv after the first run. If your filename
    convention is different, edit this function to return the true source ID.
    """
    path = Path(file_path)
    stem = path.stem

    group_stem = stem
    for pattern in GROUP_ID_PATTERNS_TO_REMOVE:
        group_stem = re.sub(pattern, "", group_stem, flags=re.IGNORECASE)

    group_stem = group_stem.strip("_- .")

    # Never prefix the class name: a source containing both classes must remain
    # one indivisible group. Manifest-based source IDs are preferred.
    if not group_stem:
        raise ValueError(f"Could not infer a source ID from filename: {file_path}")
    return group_stem


def _safe_resolve_under_root(root, relative_path):
    """Resolve a manifest path and reject directory traversal."""
    root = Path(root).expanduser().resolve()
    candidate = (root / str(relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Manifest path escapes DATA_DIR: {relative_path}") from exc
    return candidate


def load_dataset_index():
    """Load the dataset index from a provenance manifest or, optionally, folders.

    Required manifest columns:
        relative_path, label, source_id

    `source_id` must identify the original recording or source video, not an
    individual two-second segment. All segments sharing a source_id are held in
    the same train/validation/test partition.
    """
    root = Path(DATA_DIR).expanduser().resolve()
    manifest = Path(MANIFEST_PATH).expanduser().resolve()

    if not root.exists():
        raise RuntimeError(f"DATA_DIR does not exist: {root}")

    label_encoder = LabelEncoder()
    label_encoder.fit(CLASS_NAMES)

    if manifest.exists():
        df = pd.read_csv(manifest)
        required = {"relative_path", "label", "source_id"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")

        df = df.copy()
        df["label"] = df["label"].astype(str).str.strip()
        df["source_id"] = df["source_id"].astype(str).str.strip()

        invalid_labels = sorted(set(df["label"]) - set(CLASS_NAMES))
        if invalid_labels:
            raise ValueError(f"Manifest contains unsupported labels: {invalid_labels}")
        if (df["source_id"] == "").any() or df["source_id"].isna().any():
            raise ValueError("Manifest contains an empty source_id.")

        df["file_path"] = [str(_safe_resolve_under_root(root, p)) for p in df["relative_path"]]
        missing_files = [p for p in df["file_path"] if not Path(p).is_file()]
        if missing_files:
            preview = "\n".join(missing_files[:10])
            raise FileNotFoundError(f"Manifest references missing files (first 10):\n{preview}")

        non_images = [p for p in df["file_path"] if not is_image_file(p)]
        if non_images:
            raise ValueError(f"Manifest contains unsupported image files: {non_images[:10]}")

        df["folder"] = [Path(p).parent.name for p in df["file_path"]]
        df["label_id"] = label_encoder.transform(df["label"]).astype(int)
        df["group_id"] = df["source_id"]
    else:
        if REQUIRE_SOURCE_MANIFEST:
            raise FileNotFoundError(
                f"Required source manifest not found: {manifest}\n"
                "Create metadata/dataset_manifest.csv with columns: "
                "relative_path,label,source_id."
            )

        rows = []
        for folder_name, clean_label in CLASS_FOLDER_MAP.items():
            folder_path = root / folder_name
            if not folder_path.exists():
                continue
            for image_path in folder_path.rglob("*"):
                if image_path.is_file() and is_image_file(image_path):
                    rows.append({
                        "file_path": str(image_path.resolve()),
                        "relative_path": str(image_path.resolve().relative_to(root)),
                        "folder": folder_name,
                        "label": clean_label,
                        "label_id": int(label_encoder.transform([clean_label])[0]),
                        "source_id": infer_group_id(str(image_path)),
                        "group_id": infer_group_id(str(image_path)),
                    })
        df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No LMS images were indexed.")
    if df["file_path"].duplicated().any():
        duplicates = df.loc[df["file_path"].duplicated(keep=False), "file_path"].tolist()
        raise ValueError(f"Duplicate file paths found in manifest: {duplicates[:10]}")

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(df["label"].value_counts())
    print("Total images:", len(df))
    print("Unique source groups:", df["group_id"].nunique())
    print("=" * 80)

    public_index = df.drop(columns=["file_path"], errors="ignore")
    public_index.to_csv(Path(OUTPUT_DIR) / "dataset_summary" / "dataset_index.csv", index=False)

    audit = df[["group_id", "label", "relative_path"]].sort_values(["group_id", "relative_path"])
    audit.to_csv(Path(OUTPUT_DIR) / "dataset_summary" / "group_id_audit.csv", index=False)

    group_counts_df = (
        df.groupby(["group_id", "label"]).size().reset_index(name="n_images")
        .sort_values(["label", "group_id"])
    )
    group_counts_df.to_csv(Path(OUTPUT_DIR) / "dataset_summary" / "group_counts.csv", index=False)

    class_counts = df["label"].value_counts().reindex(CLASS_NAMES).fillna(0).astype(int)
    class_counts_df = class_counts.rename_axis("class").reset_index(name="count")
    class_counts_df.to_csv(Path(OUTPUT_DIR) / "dataset_summary" / "class_counts.csv", index=False)
    plot_class_distribution(class_counts_df)
    return df


def maybe_reduce_for_quick_test(df):
    if not RUN_QUICK_TEST:
        return df

    reduced = []
    for label, sub in df.groupby("label"):
        n = min(len(sub), QUICK_TEST_MAX_PER_CLASS)
        reduced.append(sub.sample(n=n, random_state=42))

    reduced_df = pd.concat(reduced, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print("=" * 80)
    print("QUICK TEST DATASET SUMMARY")
    print("=" * 80)
    print(reduced_df["label"].value_counts())
    print("=" * 80)

    return reduced_df


def plot_class_distribution(class_counts_df):
    plt.figure(figsize=(8, 5))
    plt.bar(class_counts_df["class"], class_counts_df["count"])
    plt.xlabel("Class")
    plt.ylabel("Number of LMS images")
    plt.title("Dataset Class Distribution")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "dataset_summary", "class_distribution.png"))
    plt.savefig(os.path.join(OUTPUT_DIR, "paper_figures", "Fig_dataset_class_distribution.pdf"))
    plt.close()


def _can_stratify(labels):
    counts = pd.Series(labels).value_counts()
    return len(counts) >= 2 and counts.min() >= 2


def _group_table_from_df(df):
    group_rows = []

    for group_id, sub in df.groupby("group_id"):
        label_counts = sub["label_id"].value_counts()
        majority_label_id = int(label_counts.idxmax())
        majority_label = CLASS_NAMES[majority_label_id]

        group_rows.append({
            "group_id": group_id,
            "label_id": majority_label_id,
            "label": majority_label,
            "n_images": len(sub),
            "n_unique_labels_inside_group": int(sub["label_id"].nunique())
        })

    return pd.DataFrame(group_rows)


def _split_groups(group_df, test_size, val_size, seed):
    """
    Split group IDs into train/val/test. Stratification is attempted at group
    level using the majority class of each group. If a class has too few groups,
    the function falls back to non-stratified group splitting with a warning.
    """
    stratify_test = group_df["label_id"] if _can_stratify(group_df["label_id"]) else None
    if stratify_test is None:
        print("WARNING: Group-level stratification for test split is not possible.")
        print("Reason: At least one class has fewer than 2 groups.")
        print("Falling back to non-stratified group split.")

    train_val_groups, test_groups = train_test_split(
        group_df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_test
    )

    val_relative_size = val_size / (1.0 - test_size)

    stratify_val = (
        train_val_groups["label_id"]
        if _can_stratify(train_val_groups["label_id"])
        else None
    )
    if stratify_val is None:
        print("WARNING: Group-level stratification for validation split is not possible.")
        print("Falling back to non-stratified validation group split.")

    train_groups, val_groups = train_test_split(
        train_val_groups,
        test_size=val_relative_size,
        random_state=seed,
        stratify=stratify_val
    )

    return train_groups, val_groups, test_groups


def check_split_integrity(train_df, val_df, test_df, split_dir):
    """
    Save and optionally enforce leakage checks.

    Checks:
        1) Exact file_path overlap between train/val/test.
        2) group_id overlap between train/val/test.
    """
    train_files = set(train_df["file_path"])
    val_files = set(val_df["file_path"])
    test_files = set(test_df["file_path"])

    train_groups = set(train_df["group_id"])
    val_groups = set(val_df["group_id"])
    test_groups = set(test_df["group_id"])

    report = {
        "n_train_files": len(train_files),
        "n_val_files": len(val_files),
        "n_test_files": len(test_files),
        "n_train_groups": len(train_groups),
        "n_val_groups": len(val_groups),
        "n_test_groups": len(test_groups),
        "file_overlap_train_val": sorted(list(train_files & val_files)),
        "file_overlap_train_test": sorted(list(train_files & test_files)),
        "file_overlap_val_test": sorted(list(val_files & test_files)),
        "group_overlap_train_val": sorted(list(train_groups & val_groups)),
        "group_overlap_train_test": sorted(list(train_groups & test_groups)),
        "group_overlap_val_test": sorted(list(val_groups & test_groups)),
    }

    report["file_leakage_detected"] = (
        len(report["file_overlap_train_val"]) > 0
        or len(report["file_overlap_train_test"]) > 0
        or len(report["file_overlap_val_test"]) > 0
    )

    report["group_leakage_detected"] = (
        len(report["group_overlap_train_val"]) > 0
        or len(report["group_overlap_train_test"]) > 0
        or len(report["group_overlap_val_test"]) > 0
    )

    with open(os.path.join(split_dir, "split_integrity_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("-" * 80)
    print("SPLIT INTEGRITY CHECK")
    print("-" * 80)
    print("File leakage detected:", report["file_leakage_detected"])
    print("Group leakage detected:", report["group_leakage_detected"])
    print("Train groups:", report["n_train_groups"])
    print("Validation groups:", report["n_val_groups"])
    print("Test groups:", report["n_test_groups"])
    print("-" * 80)

    if STRICT_SPLIT_INTEGRITY_CHECK:
        if report["file_leakage_detected"]:
            raise RuntimeError(
                "File-level data leakage detected. Check split_integrity_report.json."
            )
        if GROUP_AWARE_SPLIT and report["group_leakage_detected"]:
            raise RuntimeError(
                "Group-level data leakage detected. Check split_integrity_report.json."
            )

    return report


def create_splits(df, seed):
    split_dir = os.path.join(OUTPUT_DIR, "splits", f"seed_{seed}")
    os.makedirs(split_dir, exist_ok=True)

    if GROUP_AWARE_SPLIT:
        print("Using GROUP-AWARE split. Test set is unseen at inferred recording/source level.")

        if "group_id" not in df.columns:
            raise RuntimeError("GROUP_AWARE_SPLIT=True but df has no group_id column.")

        group_df = _group_table_from_df(df)

        ambiguous_groups = group_df[group_df["n_unique_labels_inside_group"] > 1]
        if not ambiguous_groups.empty:
            ambiguous_groups.to_csv(
                os.path.join(split_dir, "groups_with_multiple_labels.csv"),
                index=False
            )
            print("WARNING: Some group_id values contain multiple labels.")
            print("Review groups_with_multiple_labels.csv.")
            print("This may be valid if a source recording contains both BG and Distress windows.")

        group_df.to_csv(os.path.join(split_dir, "groups_before_split.csv"), index=False)

        train_groups, val_groups, test_groups = _split_groups(
            group_df=group_df,
            test_size=TEST_SIZE,
            val_size=VAL_SIZE,
            seed=seed
        )

        train_group_ids = set(train_groups["group_id"])
        val_group_ids = set(val_groups["group_id"])
        test_group_ids = set(test_groups["group_id"])

        train_df = df[df["group_id"].isin(train_group_ids)].copy()
        val_df = df[df["group_id"].isin(val_group_ids)].copy()
        test_df = df[df["group_id"].isin(test_group_ids)].copy()

    else:
        print("Using image-level random stratified split.")
        print("WARNING: This does not guarantee recording-level unseen testing.")

        train_val_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=df["label_id"]
        )

        val_relative_size = VAL_SIZE / (1.0 - TEST_SIZE)

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_relative_size,
            random_state=seed,
            stratify=train_val_df["label_id"]
        )

    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        public_split = split_df.drop(columns=["file_path"], errors="ignore")
        public_split.to_csv(os.path.join(split_dir, f"{split_name}.csv"), index=False)

    check_split_integrity(train_df, val_df, test_df, split_dir)

    print("=" * 80)
    print(f"SPLIT SUMMARY | SEED {seed}")
    print("=" * 80)
    print("Train:", len(train_df))
    print(train_df["label"].value_counts())
    print("Validation:", len(val_df))
    print(val_df["label"].value_counts())
    print("Test:", len(test_df))
    print(test_df["label"].value_counts())

    if "group_id" in train_df.columns:
        print("Train groups:", train_df["group_id"].nunique())
        print("Validation groups:", val_df["group_id"].nunique())
        print("Test groups:", test_df["group_id"].nunique())

    print("=" * 80)

    return train_df, val_df, test_df


# =============================================================================
# 4. TF.DATA PIPELINE
# =============================================================================

def decode_image_tf(file_path):
    img_bytes = tf.io.read_file(file_path)
    img = tf.image.decode_image(img_bytes, channels=CHANNELS, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def augment_lms_image(img):
    img = tf.image.random_brightness(img, max_delta=AUG_BRIGHTNESS_DELTA)
    img = tf.image.random_contrast(img, lower=AUG_CONTRAST_LOW, upper=AUG_CONTRAST_HIGH)

    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=AUG_NOISE_STD)
    img = tf.clip_by_value(img + noise, 0.0, 1.0)

    def apply_time_mask():
        height = tf.shape(img)[0]
        width = tf.shape(img)[1]
        channels = tf.shape(img)[2]

        max_mask_width = tf.maximum(6, tf.cast(tf.cast(width, tf.float32) * 0.12, tf.int32))
        mask_width = tf.random.uniform([], minval=5, maxval=max_mask_width, dtype=tf.int32)

        max_start = tf.maximum(1, width - mask_width)
        start = tf.random.uniform([], minval=0, maxval=max_start, dtype=tf.int32)

        left = tf.ones((height, start, channels), dtype=img.dtype)
        middle = tf.zeros((height, mask_width, channels), dtype=img.dtype)
        right = tf.ones((height, width - start - mask_width, channels), dtype=img.dtype)

        mask = tf.concat([left, middle, right], axis=1)
        return img * mask

    img = tf.cond(
        tf.random.uniform([]) < AUG_TIME_MASK_PROB,
        apply_time_mask,
        lambda: img
    )

    def apply_frequency_mask():
        height = tf.shape(img)[0]
        width = tf.shape(img)[1]
        channels = tf.shape(img)[2]

        max_mask_height = tf.maximum(6, tf.cast(tf.cast(height, tf.float32) * 0.10, tf.int32))
        mask_height = tf.random.uniform([], minval=5, maxval=max_mask_height, dtype=tf.int32)

        max_start = tf.maximum(1, height - mask_height)
        start = tf.random.uniform([], minval=0, maxval=max_start, dtype=tf.int32)

        top = tf.ones((start, width, channels), dtype=img.dtype)
        middle = tf.zeros((mask_height, width, channels), dtype=img.dtype)
        bottom = tf.ones((height - start - mask_height, width, channels), dtype=img.dtype)

        mask = tf.concat([top, middle, bottom], axis=0)
        return img * mask

    img = tf.cond(
        tf.random.uniform([]) < AUG_FREQ_MASK_PROB,
        apply_frequency_mask,
        lambda: img
    )

    return tf.clip_by_value(img, 0.0, 1.0)


def make_tf_dataset(df, batch_size, shuffle_data, augment):
    file_paths = df["file_path"].values.astype(str)
    labels = df["label_id"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if shuffle_data:
        ds = ds.shuffle(buffer_size=len(df), seed=42, reshuffle_each_iteration=True)

    @tf.autograph.experimental.do_not_convert
    def _map(path, label):
        img = decode_image_tf(path)

        if augment:
            img = augment_lms_image(img)

        label_oh = tf.one_hot(label, depth=NUM_CLASSES)
        return img, label_oh

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def get_y_true_from_df(df):
    return df["label_id"].values.astype(np.int32)


# =============================================================================
# 5. MODEL DEFINITIONS: DEEP NON-ATTENTION BASELINES
# =============================================================================

def conv_bn_relu(x, filters, kernel_size=3, dropout=0.0, name=None):
    x = layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(1e-4),
        name=None if name is None else f"{name}_conv"
    )(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_bn")(x)
    x = layers.Activation("relu", name=None if name is None else f"{name}_relu")(x)

    if dropout > 0:
        x = layers.SpatialDropout2D(dropout, name=None if name is None else f"{name}_sdrop")(x)

    return x


def cnn_block_no_attention(x, filters, block_id):
    x = conv_bn_relu(x, filters, 3, dropout=0.05, name=f"block{block_id}_conv1")
    x = conv_bn_relu(x, filters, 3, dropout=0.05, name=f"block{block_id}_conv2")
    x = layers.MaxPooling2D(pool_size=(3, 3), name=f"block{block_id}_pool")(x)
    return x


def build_baseline_cnn():
    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS), name="input_lms")

    x = cnn_block_no_attention(inputs, 32, 1)
    x = cnn_block_no_attention(x, 64, 2)
    x = cnn_block_no_attention(x, 128, 3)
    x = cnn_block_no_attention(x, 256, 4)

    x = layers.GlobalAveragePooling2D(name="gap_features")(x)

    x = layers.Dense(
        1024,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.001),
        name="dense_1024"
    )(x)
    x = layers.Dropout(0.20, name="dropout_1024")(x)

    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.001),
        name="embedding_64"
    )(x)
    x = layers.Dropout(0.20, name="dropout_embedding")(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="softmax_output")(x)

    return keras.Model(inputs, outputs, name="BaselineCNN")


def build_baseline_cnn_wide():
    """
    Wider non-attention CNN. This is useful as a parameter-budget control.
    If it performs below attention models despite similar/larger capacity,
    the argument for attention becomes stronger.
    """
    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS), name="input_lms")

    x = cnn_block_no_attention(inputs, 48, 1)
    x = cnn_block_no_attention(x, 96, 2)
    x = cnn_block_no_attention(x, 192, 3)
    x = cnn_block_no_attention(x, 384, 4)

    x = layers.GlobalAveragePooling2D(name="gap_features")(x)

    x = layers.Dense(
        1024,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.001),
        name="dense_1024"
    )(x)
    x = layers.Dropout(0.30)(x)

    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.001),
        name="embedding_128"
    )(x)
    x = layers.Dropout(0.25)(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="softmax_output")(x)

    return keras.Model(inputs, outputs, name="BaselineCNN_Wide")


def add_classifier_head(base, model_name):
    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS), name="input_lms")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap_features")(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001), name="embedding_64")(x)
    x = layers.Dropout(0.20)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="softmax_output")(x)
    return keras.Model(inputs, outputs, name=model_name)


def build_transfer_model(model_name):
    """
    Pretrained ImageNet transfer baselines.
    Weight-loading failure is fatal because random initialization would define
    a different experiment.
    """
    base_kwargs = {
        "include_top": False,
        "weights": "imagenet",
        "input_shape": (IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS)
    }

    model_class = {
        "MobileNetV2": keras.applications.MobileNetV2,
        "EfficientNetB0": keras.applications.EfficientNetB0,
        "DenseNet121": keras.applications.DenseNet121,
        "ResNet50": keras.applications.ResNet50,
        "VGG16": keras.applications.VGG16
    }[model_name]

    try:
        base = model_class(**base_kwargs)
        print(f"{model_name}: loaded ImageNet weights.")
    except Exception as e:
        raise RuntimeError(
            f"{model_name}: required ImageNet weights could not be loaded. "
            "Install/cache the weights before reproducing the benchmark."
        ) from e

    # Fine-tune the top part only.
    base.trainable = True
    n_layers = len(base.layers)
    freeze_until = int(n_layers * 0.70)
    for layer in base.layers[:freeze_until]:
        layer.trainable = False

    return add_classifier_head(base, model_name)


def build_crnn_gru():
    """
    Non-attention convolutional recurrent baseline.
    The CNN extracts local spectro-temporal patterns; GRU models temporal evolution.
    """
    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS), name="input_lms")

    x = conv_bn_relu(inputs, 32, 3, dropout=0.05, name="crnn_conv1")
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = conv_bn_relu(x, 64, 3, dropout=0.05, name="crnn_conv2")
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = conv_bn_relu(x, 128, 3, dropout=0.05, name="crnn_conv3")
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Treat image width as time axis.
    # Shape after pooling is approximately H/8 x W/8 x C.
    h = int(x.shape[1])
    w = int(x.shape[2])
    c = int(x.shape[3])

    x = layers.Permute((2, 1, 3), name="time_axis_width")(x)
    x = layers.Reshape((w, h * c), name="sequence_features")(x)

    x = layers.Bidirectional(layers.GRU(64, return_sequences=False), name="bigru")(x)
    x = layers.Dropout(0.30)(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001), name="embedding_64")(x)
    x = layers.Dropout(0.20)(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="softmax_output")(x)

    return keras.Model(inputs, outputs, name="CRNN_GRU")


def build_deep_model(model_name):
    if model_name == "BaselineCNN":
        return build_baseline_cnn()
    if model_name == "BaselineCNN_Wide":
        return build_baseline_cnn_wide()
    if model_name in ["MobileNetV2", "EfficientNetB0", "DenseNet121", "ResNet50", "VGG16"]:
        return build_transfer_model(model_name)
    if model_name == "CRNN_GRU":
        return build_crnn_gru()

    raise ValueError(f"Unknown deep model name: {model_name}")


# =============================================================================
# 6. METRICS AND PLOTS
# =============================================================================

def calculate_metrics(y_true, y_pred, y_proba=None):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        labels=list(range(NUM_CLASSES)),
        zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1
    }

    if NUM_CLASSES == 2:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        metrics.update({
            "tn_BG": int(tn),
            "fp_BG_as_Distress": int(fp),
            "fn_Distress_as_BG": int(fn),
            "tp_Distress": int(tp),
            "distress_sensitivity_recall": sensitivity,
            "bg_specificity": specificity,
            "distress_precision_ppv": precision_pos,
            "negative_predictive_value": npv,
            "false_alarm_rate_BG_to_Distress": false_alarm_rate,
            "miss_rate_Distress_to_BG": miss_rate
        })

        if y_proba is not None:
            try:
                if len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
                    distress_scores = y_proba[:, 1]
                else:
                    distress_scores = y_proba
                metrics["roc_auc_Distress"] = roc_auc_score(y_true, distress_scores)
            except Exception:
                metrics["roc_auc_Distress"] = np.nan
            try:
                if len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
                    distress_scores = y_proba[:, 1]
                else:
                    distress_scores = y_proba
                metrics["pr_auc_Distress"] = average_precision_score(y_true, distress_scores)
            except Exception:
                metrics["pr_auc_Distress"] = np.nan

    per_class_df = pd.DataFrame({
        "class": CLASS_NAMES,
        "precision": per_class_precision,
        "recall": per_class_recall,
        "f1": per_class_f1,
        "support": support
    })

    return metrics, per_class_df


def save_classification_report(y_true, y_pred, out_dir):
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(out_dir, "classification_report.csv"))

    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(
            classification_report(
                y_true,
                y_pred,
                target_names=CLASS_NAMES,
                zero_division=0
            )
        )


def plot_confusion_matrices(y_true, y_pred, out_dir, title_prefix):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(os.path.join(out_dir, "confusion_matrix_raw.csv"))

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{title_prefix} - Raw Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(tick_marks, CLASS_NAMES)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, "confusion_matrix_raw.png"))
    plt.savefig(os.path.join(out_dir, "confusion_matrix_raw.pdf"))
    plt.close()

    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    cm_norm_df = pd.DataFrame(cm_norm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_norm_df.to_csv(os.path.join(out_dir, "confusion_matrix_normalized.csv"))

    plt.figure(figsize=(8, 7))
    plt.imshow(cm_norm, interpolation="nearest", vmin=0, vmax=1)
    plt.title(f"{title_prefix} - Normalized Confusion Matrix")
    plt.colorbar()

    plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(tick_marks, CLASS_NAMES)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, "confusion_matrix_normalized.png"))
    plt.savefig(os.path.join(out_dir, "confusion_matrix_normalized.pdf"))
    plt.close()


def plot_training_history(history, out_dir, title_prefix):
    hist = pd.DataFrame(history.history)
    hist.to_csv(os.path.join(out_dir, "training_history.csv"), index=False)

    if "accuracy" in hist.columns:
        plt.figure(figsize=(7, 5))
        plt.plot(hist["accuracy"], label="Training Accuracy")
        if "val_accuracy" in hist.columns:
            plt.plot(hist["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{title_prefix} - Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "training_accuracy_curve.png"))
        plt.savefig(os.path.join(out_dir, "training_accuracy_curve.pdf"))
        plt.close()

    if "loss" in hist.columns:
        plt.figure(figsize=(7, 5))
        plt.plot(hist["loss"], label="Training Loss")
        if "val_loss" in hist.columns:
            plt.plot(hist["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{title_prefix} - Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "training_loss_curve.png"))
        plt.savefig(os.path.join(out_dir, "training_loss_curve.pdf"))
        plt.close()


def save_predictions(test_df, y_true, y_pred, y_proba, out_dir):
    pred_df = test_df.copy()
    pred_df["true_id"] = y_true
    pred_df["pred_id"] = y_pred
    pred_df["true_label"] = [CLASS_NAMES[i] for i in y_true]
    pred_df["pred_label"] = [CLASS_NAMES[i] for i in y_pred]
    pred_df["correct"] = pred_df["true_id"] == pred_df["pred_id"]

    if y_proba is not None:
        if len(y_proba.shape) == 2 and y_proba.shape[1] == NUM_CLASSES:
            for i, class_name in enumerate(CLASS_NAMES):
                pred_df[f"proba_{class_name}"] = y_proba[:, i]
        else:
            pred_df[f"proba_{CLASS_NAMES[1]}"] = y_proba

    pred_df = pred_df.drop(columns=["file_path"], errors="ignore")
    pred_df.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)


def benchmark_inference_time(model, test_ds, warmup_batches=2, measured_batches=10):
    for x_batch, _ in test_ds.take(warmup_batches):
        _ = model.predict(x_batch, verbose=0)

    total_images = 0
    total_time = 0.0

    for x_batch, _ in test_ds.take(measured_batches):
        start = time.time()
        _ = model.predict(x_batch, verbose=0)
        end = time.time()

        total_time += end - start
        total_images += int(x_batch.shape[0])

    if total_images == 0:
        return np.nan

    return (total_time / total_images) * 1000.0


# =============================================================================
# 7. DEEP MODEL TRAINING AND EVALUATION
# =============================================================================

def compute_class_weights(train_df):
    y = train_df["label_id"].values.astype(np.int32)
    classes = np.arange(NUM_CLASSES)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )

    return {int(c): float(w) for c, w in zip(classes, weights)}


def save_class_weight_report(train_df, out_dir):
    class_weight_dict = compute_class_weights(train_df)
    rows = []
    counts = train_df["label_id"].value_counts().to_dict()
    total = len(train_df)

    for class_id, class_name in enumerate(CLASS_NAMES):
        n = int(counts.get(class_id, 0))
        rows.append({
            "class_id": class_id,
            "class": class_name,
            "train_count": n,
            "train_percent": 100.0 * n / max(total, 1),
            "class_weight": float(class_weight_dict[class_id])
        })

    report_df = pd.DataFrame(rows)
    report_df.to_csv(os.path.join(out_dir, "imbalance_class_weight_report.csv"), index=False)
    return class_weight_dict, report_df


def weighted_focal_loss_from_class_weights(class_weight_dict, gamma=2.0):
    alpha = tf.constant([class_weight_dict[i] for i in range(NUM_CLASSES)], dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1.0 - keras.backend.epsilon())
        ce = -y_true * tf.math.log(y_pred)
        focal = tf.pow(1.0 - y_pred, gamma)
        weighted = alpha * focal * ce
        return tf.reduce_sum(weighted, axis=-1)

    return loss_fn


def get_imbalance_aware_loss(class_weight_dict):
    if IMBALANCE_LOSS_MODE == "focal_loss":
        return weighted_focal_loss_from_class_weights(class_weight_dict, gamma=2.0)
    if IMBALANCE_LOSS_MODE == "class_weight_ce":
        return "categorical_crossentropy"
    raise ValueError(f"Unknown IMBALANCE_LOSS_MODE: {IMBALANCE_LOSS_MODE}")


def predict_with_threshold(y_proba, threshold):
    if NUM_CLASSES != 2:
        return np.argmax(y_proba, axis=1)
    distress_scores = y_proba[:, 1] if len(y_proba.shape) == 2 else y_proba
    return (distress_scores >= threshold).astype(np.int32)


def tune_binary_threshold(y_true_val, y_proba_val, out_dir):
    if NUM_CLASSES != 2 or not TUNE_BINARY_THRESHOLD:
        return 0.50, pd.DataFrame()

    distress_scores = y_proba_val[:, 1] if len(y_proba_val.shape) == 2 else y_proba_val
    rows = []

    for thr in THRESHOLD_GRID:
        y_pred_thr = (distress_scores >= thr).astype(np.int32)
        cm = confusion_matrix(y_true_val, y_pred_thr, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_val, y_pred_thr, labels=[0, 1], zero_division=0
        )
        bal_acc = balanced_accuracy_score(y_true_val, y_pred_thr)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        youden = sensitivity + specificity - 1.0
        macro_f1 = float(np.mean(f1))

        rows.append({
            "threshold": float(thr),
            "macro_f1": macro_f1,
            "distress_f1": float(f1[1]),
            "balanced_accuracy": float(bal_acc),
            "youden": float(youden),
            "distress_precision": float(precision[1]),
            "distress_recall": float(recall[1]),
            "bg_specificity": float(specificity),
            "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "miss_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        })

    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(os.path.join(out_dir, "validation_threshold_sweep.csv"), index=False)

    metric = THRESHOLD_OPTIMIZATION_METRIC
    if metric not in threshold_df.columns:
        raise ValueError(f"Unknown THRESHOLD_OPTIMIZATION_METRIC: {metric}")

    best_row = threshold_df.sort_values([metric, "distress_recall", "bg_specificity"], ascending=False).iloc[0]
    best_threshold = float(best_row["threshold"])

    best_row.to_frame().T.to_csv(os.path.join(out_dir, "selected_validation_threshold.csv"), index=False)
    return best_threshold, threshold_df


def get_callbacks(out_dir):
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            os.path.join(out_dir, "keras_training_log.csv")
        )
    ]

    if SAVE_MODELS:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                os.path.join(out_dir, "best_model.h5"),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        )

    return callbacks


def run_single_deep_model(model_name, seed, train_df, val_df, test_df):
    print("\n" + "=" * 100)
    print(f"DEEP BASELINE MODEL: {model_name} | SEED: {seed}")
    print("=" * 100)

    set_all_seeds(seed)

    out_dir = os.path.join(OUTPUT_DIR, "models", model_name, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    train_ds = make_tf_dataset(
        train_df,
        batch_size=BATCH_SIZE,
        shuffle_data=True,
        augment=USE_AUGMENTATION
    )

    val_ds = make_tf_dataset(
        val_df,
        batch_size=BATCH_SIZE,
        shuffle_data=False,
        augment=False
    )

    test_ds = make_tf_dataset(
        test_df,
        batch_size=BATCH_SIZE,
        shuffle_data=False,
        augment=False
    )

    model = build_deep_model(model_name)

    class_weight_dict, class_weight_report = save_class_weight_report(train_df, out_dir)
    loss_fn = get_imbalance_aware_loss(class_weight_dict)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    with open(os.path.join(out_dir, "model_summary.txt"), "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    params_total = model.count_params()
    params_trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    params_non_trainable = int(params_total - params_trainable)

    fit_class_weight = class_weight_dict if (USE_CLASS_WEIGHTS and IMBALANCE_LOSS_MODE == "class_weight_ce") else None

    start_train = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=fit_class_weight,
        callbacks=get_callbacks(out_dir),
        verbose=1
    )

    train_time_sec = time.time() - start_train

    plot_training_history(history, out_dir, title_prefix=f"{model_name} Seed {seed}")

    y_true = get_y_true_from_df(test_df)
    y_val_true = get_y_true_from_df(val_df)

    val_proba = model.predict(val_ds, verbose=0)
    selected_threshold, threshold_df = tune_binary_threshold(y_val_true, val_proba, out_dir)

    start_pred = time.time()
    y_proba = model.predict(test_ds, verbose=1)
    prediction_time_sec = time.time() - start_pred

    # Save default 0.50-threshold outputs for auditability, then use validation-tuned threshold
    # for the main imbalanced-test metrics.
    default_dir = os.path.join(out_dir, "default_threshold_0p50")
    os.makedirs(default_dir, exist_ok=True)
    y_pred_default = predict_with_threshold(y_proba, 0.50)
    save_classification_report(y_true, y_pred_default, default_dir)
    plot_confusion_matrices(y_true, y_pred_default, default_dir, title_prefix=f"{model_name} Seed {seed} Default 0.50")
    save_predictions(test_df, y_true, y_pred_default, y_proba, default_dir)

    y_pred = predict_with_threshold(y_proba, selected_threshold)

    inference_ms_per_image = benchmark_inference_time(model, test_ds)

    metrics, per_class_df = calculate_metrics(y_true, y_pred, y_proba=y_proba)

    extra_metrics = {
        "model": model_name,
        "model_group": "Deep_non_attention",
        "seed": seed,
        "train_time_sec": train_time_sec,
        "prediction_time_sec": prediction_time_sec,
        "inference_ms_per_image": inference_ms_per_image,
        "params_total": params_total,
        "params_trainable": params_trainable,
        "params_non_trainable": params_non_trainable,
        "epochs_ran": len(history.history["loss"]),
        "best_val_loss": float(np.min(history.history["val_loss"])),
        "best_val_accuracy": float(np.max(history.history["val_accuracy"])),
        "imbalance_loss_mode": IMBALANCE_LOSS_MODE,
        "threshold_used": selected_threshold,
        "threshold_source": "validation_tuned" if TUNE_BINARY_THRESHOLD else "default_0.50",
        "threshold_optimization_metric": THRESHOLD_OPTIMIZATION_METRIC
    }

    metrics.update(extra_metrics)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)

    per_class_df["model"] = model_name
    per_class_df["model_group"] = "Deep_non_attention"
    per_class_df["seed"] = seed
    per_class_df.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)

    save_classification_report(y_true, y_pred, out_dir)
    plot_confusion_matrices(y_true, y_pred, out_dir, title_prefix=f"{model_name} Seed {seed}")
    save_predictions(test_df, y_true, y_pred, y_proba, out_dir)

    print("\nTest metrics:")
    display_cols = [
        "model",
        "seed",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "distress_sensitivity_recall",
        "bg_specificity",
        "roc_auc_Distress",
        "pr_auc_Distress",
        "inference_ms_per_image"
    ]
    display_cols = [c for c in display_cols if c in metrics_df.columns]
    print(metrics_df[display_cols])

    keras.backend.clear_session()
    del model, train_ds, val_ds, test_ds
    gc.collect()

    return metrics_df, per_class_df


# =============================================================================
# 8. CLASSICAL LMS-IMAGE BASELINES
# =============================================================================

def read_image_for_classical(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")

    img = cv2.resize(img, (CLASSICAL_IMAGE_SIZE[1], CLASSICAL_IMAGE_SIZE[0]))
    img = img.astype(np.float32) / 255.0
    return img.reshape(-1)


def load_classical_features(df):
    X = []
    y = df["label_id"].values.astype(np.int32)

    for p in df["file_path"].values:
        X.append(read_image_for_classical(p))

    X = np.asarray(X, dtype=np.float32)
    return X, y


def build_classical_model(model_name, seed):
    if model_name == "PCA_LMS_LogisticRegression":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced" if USE_CLASS_WEIGHTS else None,
            random_state=seed
        )
    elif model_name == "PCA_LMS_SVM_RBF":
        clf = SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            class_weight="balanced" if USE_CLASS_WEIGHTS else None,
            random_state=seed
        )
    elif model_name == "PCA_LMS_RandomForest":
        clf = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced" if USE_CLASS_WEIGHTS else None,
            random_state=seed,
            n_jobs=-1
        )
    elif model_name == "PCA_LMS_KNN":
        clf = KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            n_jobs=-1
        )
    elif model_name == "PCA_LMS_XGBoost":
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("xgboost is not installed.")
        clf = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.90,
            colsample_bytree=0.90,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown classical model name: {model_name}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_COMPONENTS, random_state=seed)),
        ("clf", clf)
    ])

    return pipeline


def get_classical_proba(model, X_test):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        if proba.shape[1] == 2:
            return proba
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        scores = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1.0 - scores, scores])

    pred = model.predict(X_test)
    return np.column_stack([1 - pred, pred]).astype(float)


def run_single_classical_model(model_name, seed, train_df, val_df, test_df):
    print("\n" + "=" * 100)
    print(f"CLASSICAL BASELINE MODEL: {model_name} | SEED: {seed}")
    print("=" * 100)

    set_all_seeds(seed)

    out_dir = os.path.join(OUTPUT_DIR, "models", model_name, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading classical LMS features...")
    X_train, y_train = load_classical_features(train_df)
    X_val, y_val = load_classical_features(val_df)
    X_test, y_true = load_classical_features(test_df)

    class_weight_dict, class_weight_report = save_class_weight_report(train_df, out_dir)

    model = build_classical_model(model_name, seed)

    # Give XGBoost explicit imbalance information because it does not use class_weight.
    if model_name == "PCA_LMS_XGBoost" and XGBOOST_AVAILABLE:
        n_bg = max(int(np.sum(y_train == 0)), 1)
        n_distress = max(int(np.sum(y_train == 1)), 1)
        model.named_steps["clf"].set_params(scale_pos_weight=n_bg / n_distress)

    start_train = time.time()
    model.fit(X_train, y_train)
    train_time_sec = time.time() - start_train

    val_proba = get_classical_proba(model, X_val)
    selected_threshold, threshold_df = tune_binary_threshold(y_val, val_proba, out_dir)

    start_pred = time.time()
    y_proba = get_classical_proba(model, X_test)
    y_pred_default = predict_with_threshold(y_proba, 0.50)
    y_pred = predict_with_threshold(y_proba, selected_threshold)
    prediction_time_sec = time.time() - start_pred

    default_dir = os.path.join(out_dir, "default_threshold_0p50")
    os.makedirs(default_dir, exist_ok=True)
    save_classification_report(y_true, y_pred_default, default_dir)
    plot_confusion_matrices(y_true, y_pred_default, default_dir, title_prefix=f"{model_name} Seed {seed} Default 0.50")
    save_predictions(test_df, y_true, y_pred_default, y_proba, default_dir)

    inference_ms_per_image = (prediction_time_sec / len(test_df)) * 1000.0 if len(test_df) > 0 else np.nan

    metrics, per_class_df = calculate_metrics(y_true, y_pred, y_proba=y_proba)

    extra_metrics = {
        "model": model_name,
        "model_group": "Classical_LMS_image",
        "seed": seed,
        "train_time_sec": train_time_sec,
        "prediction_time_sec": prediction_time_sec,
        "inference_ms_per_image": inference_ms_per_image,
        "params_total": np.nan,
        "params_trainable": np.nan,
        "params_non_trainable": np.nan,
        "epochs_ran": np.nan,
        "best_val_loss": np.nan,
        "best_val_accuracy": np.nan,
        "imbalance_loss_mode": "class_weight_or_model_weight",
        "threshold_used": selected_threshold,
        "threshold_source": "validation_tuned" if TUNE_BINARY_THRESHOLD else "default_0.50",
        "threshold_optimization_metric": THRESHOLD_OPTIMIZATION_METRIC
    }

    metrics.update(extra_metrics)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)

    per_class_df["model"] = model_name
    per_class_df["model_group"] = "Classical_LMS_image"
    per_class_df["seed"] = seed
    per_class_df.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)

    save_classification_report(y_true, y_pred, out_dir)
    plot_confusion_matrices(y_true, y_pred, out_dir, title_prefix=f"{model_name} Seed {seed}")
    save_predictions(test_df, y_true, y_pred, y_proba, out_dir)

    print("\nTest metrics:")
    display_cols = [
        "model",
        "seed",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "distress_sensitivity_recall",
        "bg_specificity",
        "roc_auc_Distress",
        "pr_auc_Distress",
        "inference_ms_per_image"
    ]
    display_cols = [c for c in display_cols if c in metrics_df.columns]
    print(metrics_df[display_cols])

    del model, X_train, X_val, X_test
    gc.collect()

    return metrics_df, per_class_df


# =============================================================================
# 9. FINAL AGGREGATION AND PUBLICATION FIGURES
# =============================================================================

def collect_all_results():
    metric_files = []
    per_class_files = []

    for root, _, files in os.walk(os.path.join(OUTPUT_DIR, "models")):
        if "metrics_summary.csv" in files:
            metric_files.append(os.path.join(root, "metrics_summary.csv"))
        if "per_class_metrics.csv" in files:
            per_class_files.append(os.path.join(root, "per_class_metrics.csv"))

    metrics_df = pd.concat([pd.read_csv(f) for f in metric_files], ignore_index=True) if metric_files else pd.DataFrame()
    per_class_df = pd.concat([pd.read_csv(f) for f in per_class_files], ignore_index=True) if per_class_files else pd.DataFrame()

    return metrics_df, per_class_df


def aggregate_results(metrics_df, per_class_df):
    final_dir = os.path.join(OUTPUT_DIR, "final_comparison")
    paper_dir = os.path.join(OUTPUT_DIR, "paper_figures")

    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(paper_dir, exist_ok=True)

    metrics_df.to_csv(os.path.join(final_dir, "all_runs_metrics.csv"), index=False)
    per_class_df.to_csv(os.path.join(final_dir, "all_runs_per_class_metrics.csv"), index=False)

    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "distress_sensitivity_recall",
        "bg_specificity",
        "distress_precision_ppv",
        "negative_predictive_value",
        "false_alarm_rate_BG_to_Distress",
        "miss_rate_Distress_to_BG",
        "roc_auc_Distress",
        "pr_auc_Distress",
        "train_time_sec",
        "prediction_time_sec",
        "inference_ms_per_image",
        "params_total",
        "params_trainable",
        "params_non_trainable",
        "epochs_ran",
        "best_val_loss",
        "best_val_accuracy"
    ]

    existing_metric_cols = [c for c in metric_cols if c in metrics_df.columns]

    group_cols = ["model"]
    if "model_group" in metrics_df.columns:
        model_group_map = metrics_df.groupby("model")["model_group"].first().reset_index()
    else:
        model_group_map = pd.DataFrame({"model": metrics_df["model"].unique(), "model_group": "Unknown"})

    summary_mean = metrics_df.groupby("model")[existing_metric_cols].mean().add_suffix("_mean")
    summary_std = metrics_df.groupby("model")[existing_metric_cols].std().add_suffix("_std")

    summary = pd.concat([summary_mean, summary_std], axis=1).reset_index()
    summary = summary.merge(model_group_map, on="model", how="left")

    summary = summary.sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
    summary.insert(0, "rank_by_macro_f1", range(1, len(summary) + 1))

    summary.to_csv(os.path.join(final_dir, "final_model_ranking_mean_std.csv"), index=False)

    publication_cols = [
        "rank_by_macro_f1",
        "model_group",
        "model",
        "accuracy_mean",
        "accuracy_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "macro_precision_mean",
        "macro_precision_std",
        "macro_recall_mean",
        "macro_recall_std",
        "macro_f1_mean",
        "macro_f1_std",
        "weighted_f1_mean",
        "weighted_f1_std",
        "distress_sensitivity_recall_mean",
        "distress_sensitivity_recall_std",
        "bg_specificity_mean",
        "bg_specificity_std",
        "false_alarm_rate_BG_to_Distress_mean",
        "false_alarm_rate_BG_to_Distress_std",
        "miss_rate_Distress_to_BG_mean",
        "miss_rate_Distress_to_BG_std",
        "roc_auc_Distress_mean",
        "roc_auc_Distress_std",
        "pr_auc_Distress_mean",
        "pr_auc_Distress_std",
        "inference_ms_per_image_mean",
        "inference_ms_per_image_std",
        "params_total_mean",
        "epochs_ran_mean"
    ]

    publication_cols = [c for c in publication_cols if c in summary.columns]
    publication_table = summary[publication_cols].copy()
    publication_table.to_csv(os.path.join(final_dir, "publication_main_results_table.csv"), index=False)

    try:
        xlsx_path = os.path.join(final_dir, "Baseline_Only_Benchmark_AllResults.xlsx")

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            metrics_df.to_excel(writer, sheet_name="All Runs Metrics", index=False)
            per_class_df.to_excel(writer, sheet_name="Per Class Metrics", index=False)
            summary.to_excel(writer, sheet_name="Mean Std Ranking", index=False)
            publication_table.to_excel(writer, sheet_name="Publication Table", index=False)

    except Exception as e:
        print("Could not save Excel workbook:", e)

    plot_final_comparisons(summary, per_class_df)

    return summary, publication_table


def plot_bar_metric(summary, metric_mean, metric_std, xlabel, title, filename):
    final_dir = os.path.join(OUTPUT_DIR, "final_comparison")
    paper_dir = os.path.join(OUTPUT_DIR, "paper_figures")

    plot_df = summary.sort_values(metric_mean, ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(plot_df["model"], plot_df[metric_mean])

    if metric_std in plot_df.columns:
        plt.errorbar(
            plot_df[metric_mean],
            plot_df["model"],
            xerr=plot_df[metric_std],
            fmt="none",
            capsize=3
        )

    plt.xlabel(xlabel)
    plt.ylabel("Model")
    plt.title(title)
    plt.tight_layout()

    plt.savefig(os.path.join(final_dir, f"{filename}.png"))
    plt.savefig(os.path.join(final_dir, f"{filename}.pdf"))
    plt.savefig(os.path.join(paper_dir, f"Fig_{filename}.pdf"))
    plt.close()


def plot_final_comparisons(summary, per_class_df):
    final_dir = os.path.join(OUTPUT_DIR, "final_comparison")
    paper_dir = os.path.join(OUTPUT_DIR, "paper_figures")

    if summary.empty:
        return

    if "macro_f1_mean" in summary.columns:
        plot_bar_metric(
            summary,
            "macro_f1_mean",
            "macro_f1_std",
            "Macro-F1",
            "Baseline Model Comparison by Macro-F1",
            "baseline_model_comparison_macro_f1"
        )

    if "accuracy_mean" in summary.columns:
        plot_bar_metric(
            summary,
            "accuracy_mean",
            "accuracy_std",
            "Accuracy",
            "Baseline Model Comparison by Accuracy",
            "baseline_model_comparison_accuracy"
        )

    if "balanced_accuracy_mean" in summary.columns:
        plot_bar_metric(
            summary,
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "Balanced Accuracy",
            "Baseline Model Comparison by Balanced Accuracy",
            "baseline_model_comparison_balanced_accuracy"
        )

    if "weighted_f1_mean" in summary.columns:
        plot_bar_metric(
            summary,
            "weighted_f1_mean",
            "weighted_f1_std",
            "Weighted-F1",
            "Baseline Model Comparison by Weighted-F1",
            "baseline_model_comparison_weighted_f1"
        )

    if "macro_f1_mean" in summary.columns and "inference_ms_per_image_mean" in summary.columns:
        plot_df = summary.dropna(subset=["macro_f1_mean", "inference_ms_per_image_mean"])

        plt.figure(figsize=(8, 6))
        plt.scatter(plot_df["inference_ms_per_image_mean"], plot_df["macro_f1_mean"], s=90)

        for _, row in plot_df.iterrows():
            plt.text(
                row["inference_ms_per_image_mean"],
                row["macro_f1_mean"],
                row["model"],
                fontsize=8
            )

        plt.xlabel("Inference Time (ms/image)")
        plt.ylabel("Macro-F1")
        plt.title("Macro-F1 vs Inference Time")
        plt.tight_layout()

        plt.savefig(os.path.join(final_dir, "macro_f1_vs_inference_time.png"))
        plt.savefig(os.path.join(final_dir, "macro_f1_vs_inference_time.pdf"))
        plt.savefig(os.path.join(paper_dir, "Fig_baseline_macro_f1_vs_inference_time.pdf"))
        plt.close()

    if "macro_f1_mean" in summary.columns and "params_total_mean" in summary.columns:
        plot_df = summary.dropna(subset=["macro_f1_mean", "params_total_mean"])
        plot_df = plot_df[plot_df["params_total_mean"] > 0]

        if not plot_df.empty:
            plt.figure(figsize=(8, 6))
            plt.scatter(plot_df["params_total_mean"], plot_df["macro_f1_mean"], s=90)

            for _, row in plot_df.iterrows():
                plt.text(
                    row["params_total_mean"],
                    row["macro_f1_mean"],
                    row["model"],
                    fontsize=8
                )

            plt.xscale("log")
            plt.xlabel("Number of Parameters")
            plt.ylabel("Macro-F1")
            plt.title("Macro-F1 vs Model Size")
            plt.tight_layout()

            plt.savefig(os.path.join(final_dir, "macro_f1_vs_params.png"))
            plt.savefig(os.path.join(final_dir, "macro_f1_vs_params.pdf"))
            plt.savefig(os.path.join(paper_dir, "Fig_baseline_macro_f1_vs_params.pdf"))
            plt.close()

    if not per_class_df.empty:
        pc_summary = per_class_df.groupby(["model", "class"])["f1"].mean().reset_index()
        pivot = pc_summary.pivot(index="class", columns="model", values="f1")
        pivot = pivot.reindex(CLASS_NAMES)

        pivot.to_csv(os.path.join(final_dir, "per_class_f1_model_comparison.csv"))

        plt.figure(figsize=(12, 6))
        pivot.plot(kind="bar", figsize=(12, 6))
        plt.xlabel("Class")
        plt.ylabel("Mean F1-score")
        plt.title("Per-Class F1 Comparison Across Baseline Models")
        plt.xticks(rotation=30, ha="right")
        plt.legend(fontsize=8)
        plt.tight_layout()

        plt.savefig(os.path.join(final_dir, "per_class_f1_model_comparison.png"))
        plt.savefig(os.path.join(final_dir, "per_class_f1_model_comparison.pdf"))
        plt.savefig(os.path.join(paper_dir, "Fig_baseline_per_class_f1_model_comparison.pdf"))
        plt.close()


# =============================================================================
# 10. COMMAND-LINE CONFIGURATION AND VALIDATION
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the publication baseline benchmark on LMS images."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Root directory containing LMS images.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH,
                        help="CSV with relative_path,label,source_id columns.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Directory for generated outputs.")
    parser.add_argument("--quick-test", action="store_true",
                        help="Run a small smoke test; never use for reported results.")
    parser.add_argument("--allow-filename-source-inference", action="store_true",
                        help="Permit filename-derived source IDs. Not recommended for publication.")
    parser.add_argument("--tune-threshold", action="store_true",
                        help="Tune threshold on validation data. Use only if documented in the paper.")
    parser.add_argument("--no-save-models", action="store_true",
                        help="Do not save trained model files.")
    return parser.parse_args()


def apply_runtime_config(args):
    global DATA_DIR, OUTPUT_DIR, MANIFEST_PATH
    global RUN_QUICK_TEST, REQUIRE_SOURCE_MANIFEST, TUNE_BINARY_THRESHOLD, SAVE_MODELS
    DATA_DIR = args.data_dir.expanduser().resolve()
    OUTPUT_DIR = args.output_dir.expanduser().resolve()
    MANIFEST_PATH = args.manifest.expanduser().resolve()
    RUN_QUICK_TEST = bool(args.quick_test)
    REQUIRE_SOURCE_MANIFEST = not bool(args.allow_filename_source_inference)
    TUNE_BINARY_THRESHOLD = bool(args.tune_threshold)
    SAVE_MODELS = not bool(args.no_save_models)


def validate_publication_configuration():
    if not GROUP_AWARE_SPLIT:
        raise ValueError("Publication runs require GROUP_AWARE_SPLIT=True.")
    if not STRICT_SPLIT_INTEGRITY_CHECK:
        raise ValueError("Publication runs require STRICT_SPLIT_INTEGRITY_CHECK=True.")
    if TEST_SIZE <= 0 or VAL_SIZE <= 0 or TEST_SIZE + VAL_SIZE >= 1:
        raise ValueError("TEST_SIZE and VAL_SIZE must be positive and sum to less than 1.")
    if len(set(RANDOM_SEEDS)) != len(RANDOM_SEEDS):
        raise ValueError("RANDOM_SEEDS contains duplicate values.")
    if RUN_QUICK_TEST:
        print("WARNING: quick-test mode is not suitable for manuscript results.")


def save_environment_manifest():
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "opencv": cv2.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "xgboost_available": XGBOOST_AVAILABLE,
    }
    with open(Path(OUTPUT_DIR) / "environment.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

def main():
    validate_publication_configuration()
    setup_environment()
    make_dirs()
    save_config()
    save_environment_manifest()

    df = load_dataset_index()
    df = maybe_reduce_for_quick_test(df)

    all_metrics = []
    all_per_class = []

    for seed in RANDOM_SEEDS:
        print("\n" + "#" * 100)
        print(f"STARTING SEED {seed}")
        print("#" * 100)

        train_df, val_df, test_df = create_splits(df, seed)

        if RUN_DEEP_MODELS:
            for model_name in DEEP_MODEL_NAMES:
                try:
                    metrics_df, per_class_df = run_single_deep_model(
                        model_name=model_name,
                        seed=seed,
                        train_df=train_df,
                        val_df=val_df,
                        test_df=test_df
                    )

                    all_metrics.append(metrics_df)
                    all_per_class.append(per_class_df)

                except Exception as e:
                    print("\n" + "!" * 100)
                    print(f"ERROR while running deep model {model_name} seed {seed}")
                    print("Error:", e)
                    print("!" * 100)

                    keras.backend.clear_session()
                    gc.collect()
                    if PUBLICATION_FAIL_FAST:
                        raise

        if RUN_CLASSICAL_MODELS:
            for model_name in CLASSICAL_MODEL_NAMES:
                try:
                    metrics_df, per_class_df = run_single_classical_model(
                        model_name=model_name,
                        seed=seed,
                        train_df=train_df,
                        val_df=val_df,
                        test_df=test_df
                    )

                    all_metrics.append(metrics_df)
                    all_per_class.append(per_class_df)

                except Exception as e:
                    print("\n" + "!" * 100)
                    print(f"ERROR while running classical model {model_name} seed {seed}")
                    print("Error:", e)
                    print("!" * 100)
                    gc.collect()
                    if PUBLICATION_FAIL_FAST:
                        raise

    print("\n" + "=" * 80)
    print("FINAL AGGREGATION")
    print("=" * 80)

    metrics_df, per_class_df = collect_all_results()

    if metrics_df.empty:
        print("No metrics were collected. Please check earlier errors.")
        return

    summary, publication_table = aggregate_results(metrics_df, per_class_df)

    print("\nFinal baseline model ranking by Macro-F1:")
    ranking_cols = [
        "rank_by_macro_f1",
        "model_group",
        "model",
        "macro_f1_mean",
        "macro_f1_std",
        "accuracy_mean",
        "balanced_accuracy_mean",
        "weighted_f1_mean",
        "distress_sensitivity_recall_mean",
        "bg_specificity_mean",
        "false_alarm_rate_BG_to_Distress_mean",
        "miss_rate_Distress_to_BG_mean",
        "roc_auc_Distress_mean",
        "pr_auc_Distress_mean",
        "inference_ms_per_image_mean"
    ]
    ranking_cols = [c for c in ranking_cols if c in summary.columns]
    print(summary[ranking_cols])

    print("\nPublication table:")
    print(publication_table)

    print("\nAll baseline-only outputs saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    runtime_args = parse_args()
    apply_runtime_config(runtime_args)
    main()

