"""Preprocessing / cleaning pipeline for the Stress Level dataset.

``StressLevelDataset.csv`` (Student Stress Factors) has 1,100 rows and 21
integer columns: 20 self-reported factor features (e.g. ``anxiety_level``,
``self_esteem``, ``sleep_quality``, ``academic_performance``, ``bullying``)
and a ``stress_level`` target with three already-encoded classes (0/1/2).

This module:
  1. Loads the raw CSV from ``data/raw`` (auto-detected).
  2. Cleans it: normalises column names, replaces +/-inf with NaN, removes
     exact duplicate rows, and median-imputes any missing values (statistics
     learned on TRAIN only).
  3. Drops constant / zero-variance feature columns.
  4. Stratified 80/20 train/test split (the source is a single file).
  5. Standard-scales the features (scaler fit on TRAIN only -> no leakage).
  6. Writes cleaned CSVs and the fitted scaler to ``data/processed``.

The target is already integer-encoded, so no label encoding is applied; we
only validate the class set. Scaling is fit on the training split exclusively
to avoid leaking test statistics into the model.

Run from the repo root or anywhere:

    python ml_pipeline/src/preprocess.py
    python ml_pipeline/src/preprocess.py --no-scale --test-size 0.25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Paths -----------------------------------------------------------------
# .../ml_pipeline/src/preprocess.py -> ml_pipeline/
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PIPELINE_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PIPELINE_ROOT / "data" / "processed"

TARGET_COLUMN = "stress_level"
RANDOM_STATE = 42


def find_dataset(raw_dir: Path) -> Path:
    """Locate the source CSV in ``raw_dir``.

    Prefers a file whose name mentions 'stress'; otherwise the only CSV present.
    """
    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSV found in {raw_dir}. Place StressLevelDataset.csv there."
        )
    preferred = [p for p in csvs if "stress" in p.name.lower()]
    chosen = preferred[0] if preferred else csvs[0]
    print(f"[load] dataset: {chosen.name}")
    return chosen


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Non-fitted cleaning: strip column names, inf->NaN, drop duplicate rows."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    numeric = df.select_dtypes(include=[np.number])
    n_inf = int(np.isinf(numeric).sum().sum())
    if n_inf:
        df = df.replace([np.inf, -np.inf], np.nan)
        print(f"[clean] replaced {n_inf} inf values with NaN")

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if before != len(df):
        print(f"[clean] removed {before - len(df)} duplicate rows")

    n_missing = int(df.isna().sum().sum())
    print(f"[clean] rows={len(df)} cols={df.shape[1]} missing_cells={n_missing}")
    return df


def preprocess(
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    scale: bool = True,
    test_size: float = 0.2,
) -> dict:
    """Run the full pipeline and write artifacts. Returns a small summary dict."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = clean_frame(pd.read_csv(find_dataset(raw_dir)))

    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' missing. Columns: {list(df.columns)}"
        )

    # --- Split features / target ------------------------------------------
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()

    # Target is already integer-encoded (0/1/2); just report the distribution.
    class_counts = {int(k): int(v) for k, v in y.value_counts().sort_index().items()}
    print(f"[target] '{TARGET_COLUMN}' class distribution -> {class_counts}")

    # --- Drop constant / zero-variance feature columns --------------------
    nunique = X.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        X = X.drop(columns=constant_cols)
        feature_cols = [c for c in feature_cols if c not in constant_cols]
        print(f"[features] dropped {len(constant_cols)} constant columns: {constant_cols}")

    # --- Stratified train/test split --------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[split] train={len(X_train)} test={len(X_test)} (test_size={test_size})")

    # --- Median imputation (fit on train only) ----------------------------
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    # --- Scale features (fit on train only) -------------------------------
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test), columns=feature_cols, index=X_test.index
        )
        print(f"[scale] StandardScaler applied to {len(feature_cols)} features")

    # --- Write outputs ----------------------------------------------------
    train_out = X_train.copy()
    train_out[TARGET_COLUMN] = y_train.values
    test_out = X_test.copy()
    test_out[TARGET_COLUMN] = y_test.values

    train_out.to_csv(processed_dir / "train_processed.csv", index=False)
    test_out.to_csv(processed_dir / "test_processed.csv", index=False)
    if scaler is not None:
        joblib.dump(scaler, processed_dir / "scaler.joblib")

    summary = {
        "n_features": len(feature_cols),
        "features": feature_cols,
        "target": TARGET_COLUMN,
        "class_distribution": class_counts,
        "train_rows": int(len(train_out)),
        "test_rows": int(len(test_out)),
        "test_size": test_size,
        "scaled": scale,
        "constant_dropped": constant_cols,
        "random_state": RANDOM_STATE,
    }
    with open(processed_dir / "preprocess_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[done] wrote train_processed.csv ({summary['train_rows']} rows), "
        f"test_processed.csv ({summary['test_rows']} rows) to {processed_dir}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean & preprocess the Stress Level dataset.")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument(
        "--no-scale",
        dest="scale",
        action="store_false",
        help="Skip StandardScaler (e.g. for tree models that don't need it).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        scale=args.scale,
        test_size=args.test_size,
    )
