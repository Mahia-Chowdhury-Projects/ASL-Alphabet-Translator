"""
ASL Alphabet Classification Pipeline
=====================================
A computer vision pipeline to classify 3,000+ ASL alphabet gesture images
using NumPy, OpenCV, and Scikit-Learn.

Pipeline stages:
  1. Data loading & directory scanning
  2. Preprocessing  – resize, grayscale, CLAHE equalization
  3. Augmentation   – flips, rotations, brightness jitter
  4. Feature extraction – HOG descriptors
  5. Normalization   – StandardScaler
  6. Model training  – SVM (RBF kernel) + Random Forest
  7. Evaluation      – accuracy, confusion matrix, classification report
  8. Bias mitigation – per-class balancing via augmentation

Usage
-----
    python asl_pipeline.py --data_dir /path/to/dataset --model svm
    python asl_pipeline.py --data_dir /path/to/dataset --model rf
    python asl_pipeline.py --demo          # runs on synthetic data (no dataset needed)

Expected dataset layout
-----------------------
    dataset/
        A/  img1.jpg  img2.jpg ...
        B/  img1.jpg ...
        ...
        Z/  img1.jpg ...

The Kaggle "ASL Alphabet" dataset (87,000 images, 29 classes) and the
"Sign Language MNIST" dataset both match this layout.
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
IMG_SIZE     = 64          # resize to IMG_SIZE × IMG_SIZE
HOG_CELL     = 8           # HOG pixels per cell
HOG_BLOCK    = 2           # HOG cells per block
HOG_BINS     = 9           # HOG orientation bins
RANDOM_STATE = 42
ASL_CLASSES  = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ══════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════

def scan_dataset(data_dir: str) -> dict[str, list[str]]:
    """
    Walk ``data_dir`` and return {class_label: [image_paths]}.
    Skips subdirectories that are not single-uppercase-letter ASL classes
    so the pipeline stays focused on A–Z.
    """
    data_dir = Path(data_dir)
    dataset: dict[str, list[str]] = {}

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name.upper()
        if label not in ASL_CLASSES:
            continue
        paths = [
            str(p) for p in class_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXT
        ]
        if paths:
            dataset[label] = paths

    return dataset


def load_image(path: str) -> np.ndarray | None:
    """Load an image, convert to grayscale, resize, apply CLAHE."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    # CLAHE: Contrast Limited Adaptive Histogram Equalization
    # Improves generalisation across lighting conditions
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    return img


# ══════════════════════════════════════════════
# 2. DATA AUGMENTATION
# ══════════════════════════════════════════════

def augment_image(img: np.ndarray, n: int = 3) -> list[np.ndarray]:
    """
    Return ``n`` augmented variants of ``img``.
    Transformations: horizontal flip, random rotation (±15°),
    brightness jitter, and Gaussian blur.
    """
    h, w = img.shape
    augmented = []
    for _ in range(n):
        aug = img.copy()

        # Random horizontal flip
        if np.random.rand() > 0.5:
            aug = cv2.flip(aug, 1)

        # Random rotation –15° to +15°
        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Brightness jitter
        beta = np.random.randint(-30, 30)
        aug = np.clip(aug.astype(np.int16) + beta, 0, 255).astype(np.uint8)

        # Occasional Gaussian blur (simulates motion / focus variation)
        if np.random.rand() > 0.7:
            aug = cv2.GaussianBlur(aug, (3, 3), 0)

        augmented.append(aug)
    return augmented


# ══════════════════════════════════════════════
# 3. FEATURE EXTRACTION — HOG
# ══════════════════════════════════════════════

def extract_hog(img: np.ndarray) -> np.ndarray:
    """
    Compute a Histogram of Oriented Gradients (HOG) descriptor.
    HOG captures local shape/edge structure — robust to minor
    lighting and position differences.
    """
    win_size   = (IMG_SIZE, IMG_SIZE)
    block_size = (HOG_BLOCK * HOG_CELL, HOG_BLOCK * HOG_CELL)
    block_step = (HOG_CELL, HOG_CELL)
    cell_size  = (HOG_CELL, HOG_CELL)

    hog = cv2.HOGDescriptor(win_size, block_size, block_step, cell_size, HOG_BINS)
    descriptor = hog.compute(img).flatten()
    return descriptor


# ══════════════════════════════════════════════
# 4. BIAS MITIGATION — PER-CLASS BALANCING
# ══════════════════════════════════════════════

def balance_classes(
    images_by_class: dict[str, list[np.ndarray]],
    target_per_class: int | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    """
    Ensure each class has ``target_per_class`` samples.
    Under-represented classes are upsampled via augmentation;
    over-represented classes are randomly downsampled.

    This directly addresses dataset bias, which degrades generalisation
    to signers who were underrepresented in the original collection.
    """
    if target_per_class is None:
        counts = [len(v) for v in images_by_class.values()]
        target_per_class = int(np.median(counts))

    all_images, all_labels = [], []

    for label, imgs in images_by_class.items():
        n = len(imgs)

        if n >= target_per_class:
            # Downsample
            chosen = [imgs[i] for i in np.random.choice(n, target_per_class, replace=False)]
        else:
            # Upsample via augmentation
            chosen = imgs.copy()
            while len(chosen) < target_per_class:
                src = imgs[np.random.randint(len(imgs))]
                chosen.extend(augment_image(src, n=1))
            chosen = chosen[:target_per_class]

        all_images.extend(chosen)
        all_labels.extend([label] * target_per_class)

    return all_images, all_labels


# ══════════════════════════════════════════════
# 5. FULL PREPROCESSING PIPELINE
# ══════════════════════════════════════════════

def build_feature_matrix(
    dataset: dict[str, list[str]],
    augment: bool = True,
    balance: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Load images → preprocess → (augment) → (balance) → HOG features.
    Returns X (feature matrix), y (encoded labels), label_encoder.
    """
    images_by_class: dict[str, list[np.ndarray]] = {}

    total_files = sum(len(v) for v in dataset.values())
    loaded = 0

    for label, paths in dataset.items():
        imgs = []
        for path in paths:
            img = load_image(path)
            if img is not None:
                imgs.append(img)
                if augment:
                    imgs.extend(augment_image(img))
                loaded += 1
        images_by_class[label] = imgs

        if verbose:
            print(f"  [{label}] loaded {len(paths)} files → {len(imgs)} samples "
                  f"({'augmented' if augment else 'raw'})")

    if balance:
        if verbose:
            print("\n[bias mitigation] balancing classes …")
        all_images, all_labels = balance_classes(images_by_class)
    else:
        all_images, all_labels = [], []
        for label, imgs in images_by_class.items():
            all_images.extend(imgs)
            all_labels.extend([label] * len(imgs))

    if verbose:
        print(f"\nExtracting HOG features from {len(all_images):,} samples …")

    X = np.array([extract_hog(img) for img in all_images], dtype=np.float32)
    le = LabelEncoder()
    y  = le.fit_transform(all_labels)

    return X, y, le


# ══════════════════════════════════════════════
# 6. MODEL DEFINITIONS
# ══════════════════════════════════════════════

def build_svm() -> SVC:
    """
    Support Vector Machine with RBF kernel.
    Strong performance on HOG feature spaces; probability=True
    enables confidence scores.
    """
    return SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )


def build_random_forest() -> RandomForestClassifier:
    """
    Random Forest: ensemble of 300 decision trees.
    More interpretable than SVM; naturally handles multi-class.
    """
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ══════════════════════════════════════════════
# 7. TRAINING & EVALUATION
# ══════════════════════════════════════════════

def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
    model_name: str = "svm",
    cv_folds: int = 5,
    output_dir: str = ".",
) -> dict:
    """
    Normalise → split → cross-validate → fit → evaluate.
    Saves confusion matrix and metric plots to ``output_dir``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Normalisation ──────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train / test split (80/20, stratified) ─
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # ── Model selection ────────────────────────
    if model_name == "svm":
        model = build_svm()
    elif model_name == "rf":
        model = build_random_forest()
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose 'svm' or 'rf'.")

    # ── Cross-validation ───────────────────────
    print(f"\n[{model_name.upper()}] {cv_folds}-fold stratified cross-validation …")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Final fit ─────────────────────────────
    print(f"  Training on {len(X_train):,} samples …")
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")

    # ── Evaluation ────────────────────────────
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    cm     = confusion_matrix(y_test, y_pred)

    print(f"\n  Test accuracy : {acc:.4f}")
    print("\n  Classification Report:\n")
    print(report)

    # ── Plots ─────────────────────────────────
    _plot_confusion_matrix(cm, le.classes_, model_name, output_dir)
    _plot_cv_scores(cv_scores, model_name, output_dir)

    return {
        "model":      model,
        "scaler":     scaler,
        "le":         le,
        "accuracy":   acc,
        "cv_scores":  cv_scores,
        "report":     report,
        "cm":         cm,
    }


# ══════════════════════════════════════════════
# 8. VISUALISATION HELPERS
# ══════════════════════════════════════════════

def _plot_confusion_matrix(
    cm: np.ndarray,
    classes: list[str],
    model_name: str,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title(f"Confusion Matrix — {model_name.upper()}", fontsize=14, pad=12)
    plt.tight_layout()
    out = output_dir / f"confusion_matrix_{model_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {out}")


def _plot_cv_scores(
    scores: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    folds = list(range(1, len(scores) + 1))
    ax.bar(folds, scores, color="#4C72B0", edgecolor="white", width=0.6)
    ax.axhline(scores.mean(), color="crimson", linestyle="--", label=f"Mean: {scores.mean():.4f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(max(0, scores.min() - 0.05), 1.0)
    ax.set_title(f"Cross-Validation Accuracy — {model_name.upper()}")
    ax.legend()
    plt.tight_layout()
    out = output_dir / f"cv_scores_{model_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  CV score plot  saved → {out}")


def visualize_samples(
    dataset: dict[str, list[str]],
    n_per_class: int = 2,
    output_dir: str = ".",
) -> None:
    """Plot a grid of sample images (one row per class)."""
    classes = sorted(dataset.keys())
    n_cls   = len(classes)
    fig, axes = plt.subplots(n_cls, n_per_class, figsize=(n_per_class * 2, n_cls * 1.5))

    for r, label in enumerate(classes):
        paths = dataset[label][:n_per_class]
        for c, path in enumerate(paths):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            ax = axes[r][c] if n_per_class > 1 else axes[r]
            ax.imshow(img if img is not None else np.zeros((IMG_SIZE, IMG_SIZE)), cmap="gray")
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(label, fontsize=8, rotation=0, labelpad=18, va="center")

    fig.suptitle("ASL Dataset Samples (grayscale)", fontsize=13)
    plt.tight_layout()
    out = Path(output_dir) / "dataset_samples.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Dataset sample grid saved → {out}")


# ══════════════════════════════════════════════
# 9. DEMO MODE (synthetic data — no dataset needed)
# ══════════════════════════════════════════════

def run_demo(model_name: str = "svm", output_dir: str = "asl_output") -> None:
    """
    Generate synthetic hand-like images for all 26 ASL classes,
    run the full pipeline, and save evaluation plots.
    Use this to verify the pipeline without a real dataset.
    """
    print("=" * 60)
    print("  ASL PIPELINE — DEMO MODE (synthetic data)")
    print("=" * 60)

    np.random.seed(RANDOM_STATE)
    n_per_class = 120   # 26 classes × 120 = 3,120 samples (mirrors the project scope)
    all_X, all_y_str = [], []

    print("\n[1/4] Generating synthetic ASL gesture images …")
    for label in ASL_CLASSES:
        seed_val = ord(label)
        for i in range(n_per_class):
            # Simulate a hand silhouette: white blobs on dark background
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            rng  = np.random.RandomState(seed_val + i)

            # Palm
            cx, cy = IMG_SIZE // 2 + rng.randint(-5, 5), IMG_SIZE // 2 + rng.randint(5, 10)
            cv2.ellipse(img, (cx, cy), (14 + rng.randint(0, 4), 18 + rng.randint(0, 4)),
                        0, 0, 360, 255, -1)

            # Fingers (number & angle vary per class for class-specific structure)
            n_fingers = (seed_val % 5) + 1
            for f in range(n_fingers):
                angle = (seed_val * 7 + f * 35) % 360
                rad   = np.deg2rad(angle)
                length = 20 + rng.randint(0, 8)
                x2 = int(cx + length * np.cos(rad))
                y2 = int(cy - length * np.sin(rad))
                cv2.line(img, (cx, cy), (x2, y2), 255, 5 + rng.randint(0, 2))

            # Slight noise to simulate real image variation
            noise = rng.randint(0, 30, img.shape).astype(np.uint8)
            img   = cv2.add(img, noise)

            hog = extract_hog(img)
            all_X.append(hog)
            all_y_str.append(label)

    X  = np.array(all_X, dtype=np.float32)
    le = LabelEncoder()
    y  = le.fit_transform(all_y_str)

    print(f"  Feature matrix: {X.shape}  ({X.shape[0]:,} samples × {X.shape[1]:,} HOG features)")
    print(f"  Classes: {list(le.classes_)}")

    print("\n[2/4] Normalising features with StandardScaler …")
    print("\n[3/4] Training & evaluating …")
    results = train_and_evaluate(X, y, le, model_name=model_name, cv_folds=5, output_dir=output_dir)

    print("\n[4/4] Summary")
    print(f"  Model         : {model_name.upper()}")
    print(f"  Samples       : {X.shape[0]:,}")
    print(f"  Feature dim   : {X.shape[1]:,}")
    print(f"  CV accuracy   : {results['cv_scores'].mean():.4f} ± {results['cv_scores'].std():.4f}")
    print(f"  Test accuracy : {results['accuracy']:.4f}")
    print(f"\n  Output files saved to: {Path(output_dir).resolve()}")
    print("=" * 60)


# ══════════════════════════════════════════════
# 10. REAL DATASET MODE
# ══════════════════════════════════════════════

def run_pipeline(
    data_dir: str,
    model_name: str = "svm",
    augment: bool = True,
    balance: bool = True,
    output_dir: str = "asl_output",
    visualize: bool = True,
) -> None:
    """End-to-end pipeline on a real dataset directory."""
    print("=" * 60)
    print("  ASL ALPHABET CLASSIFICATION PIPELINE")
    print("=" * 60)

    # ── Scan dataset ──────────────────────────
    print(f"\n[1/5] Scanning dataset: {data_dir}")
    dataset = scan_dataset(data_dir)
    if not dataset:
        print(f"  ERROR: No valid ASL class folders found in '{data_dir}'")
        print("  Expected subdirectories named A–Z containing image files.")
        sys.exit(1)

    total = sum(len(v) for v in dataset.values())
    print(f"  Found {len(dataset)} classes, {total:,} images")

    if visualize:
        print("\n[2/5] Saving sample image grid …")
        visualize_samples(dataset, n_per_class=3, output_dir=output_dir)
    else:
        print("\n[2/5] Skipping sample visualisation (--no-vis)")

    # ── Feature extraction ────────────────────
    print("\n[3/5] Preprocessing + feature extraction …")
    X, y, le = build_feature_matrix(dataset, augment=augment, balance=balance)
    print(f"\n  Feature matrix shape : {X.shape}")
    print(f"  Classes              : {list(le.classes_)}")

    # ── Train & evaluate ──────────────────────
    print("\n[4/5] Training & evaluating …")
    results = train_and_evaluate(X, y, le, model_name=model_name, output_dir=output_dir)

    # ── Summary ───────────────────────────────
    print("\n[5/5] Pipeline complete")
    print(f"  Model         : {model_name.upper()}")
    print(f"  Samples       : {X.shape[0]:,}")
    print(f"  Feature dim   : {X.shape[1]:,}")
    print(f"  CV accuracy   : {results['cv_scores'].mean():.4f} ± {results['cv_scores'].std():.4f}")
    print(f"  Test accuracy : {results['accuracy']:.4f}")
    print(f"\n  Output saved to: {Path(output_dir).resolve()}")
    print("=" * 60)


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ASL Alphabet CV Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data_dir", type=str, default=None,
                   help="Root directory of the ASL dataset (A–Z subdirs).")
    p.add_argument("--model", choices=["svm", "rf"], default="svm",
                   help="Classifier: 'svm' (SVC/RBF) or 'rf' (Random Forest). Default: svm.")
    p.add_argument("--output_dir", type=str, default="asl_output",
                   help="Directory to save plots and reports. Default: asl_output/")
    p.add_argument("--no-aug", action="store_true",
                   help="Disable data augmentation.")
    p.add_argument("--no-balance", action="store_true",
                   help="Disable per-class balancing (bias mitigation).")
    p.add_argument("--no-vis", action="store_true",
                   help="Skip sample image visualisation.")
    p.add_argument("--demo", action="store_true",
                   help="Run in demo mode using synthetic data (no dataset required).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.demo or args.data_dir is None:
        run_demo(model_name=args.model, output_dir=args.output_dir)
    else:
        run_pipeline(
            data_dir=args.data_dir,
            model_name=args.model,
            augment=not args.no_aug,
            balance=not args.no_balance,
            output_dir=args.output_dir,
            visualize=not args.no_vis,
        )
