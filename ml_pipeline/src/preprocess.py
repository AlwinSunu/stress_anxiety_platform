"""Preprocessing / cleaning pipeline for the SWELL-KW HRV stress dataset.

The SWELL Heart-Rate-Variability dataset ships as two files (``train.csv`` /
``test.csv``) with 36 columns: 34 HRV features, a ``datasetId`` recording
identifier, and a ``condition`` target with three classes
(``no stress``, ``interruption``, ``time pressure``).

This module:
  1. Loads the raw train/test CSVs from ``data/raw`` (auto-detected).
  2. Cleans them: normalises column names, drops the ``datasetId`` identifier,
     replaces +/-inf with NaN, removes exact duplicate rows, and median-imputes
     any remaining missing values (statistics learned on TRAIN only).
  3. Drops constant / zero-variance columns.
  4. Label-encodes the ``condition`` target (mapping learned on TRAIN).
  5. Standard-scales the features (scaler fit on TRAIN only -> no leakage).
  6. Writes cleaned CSVs and the fitted artifacts to ``data/processed``.

We intentionally do NOT replicate the original Kaggle notebook's
"keep only positively correlated features" step: it silently discards
informative negatively-correlated features and is not a sound selection method.
Per the project decision, the dataset's original train/test split is kept as-is.

Run from the repo root or anywhere:

    python ml_pipeline/src/preprocess.py
    python ml_pipeline/src/preprocess.py --raw-dir path/to/raw --no-scale
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Paths -----------------------------------------------------------------
# .../ml_pipeline/src/preprocess.py -> ml_pipeline/
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PIPELINE_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PIPELINE_ROOT / "data" / "processed"

TARGET_COLUMN = "condition"
# Identifier column: a recording id, not a physiological feature -> drop it.
ID_COLUMNS = ["datasetId"]


def find_csv(raw_dir: Path, keyword: str) -> Path | None:
    """Return the first CSV in ``raw_dir`` whose name contains ``keyword``."""
    matches = sorted(p for p in raw_dir.glob("*.csv") if keyword in p.name.lower())
    return matches[0] if matches else None


def load_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test frames.

    Supports either a pre-split pair (``train*.csv`` / ``test*.csv``) or a
    single combined CSV, which is then split 80/20 (stratified) as a fallback.
    """
    train_path = find_csv(raw_dir, "train")
    test_path = find_csv(raw_dir, "test")

    if train_path and test_path:
        print(f"[load] train: {train_path.name}")
        print(f"[load] test : {test_path.name}")
        return pd.read_csv(train_path), pd.read_csv(test_path)

    # Fallback: a single CSV that we split ourselves.
    csvs = list(sorted(raw_dir.glob("*.csv")))
    if len(csvs) == 1:
        from sklearn.model_selection import train_test_split

        print(f"[load] single file {csvs[0].name} -> stratified 80/20 split")
        df = _normalise_columns(pd.read_csv(csvs[0]))
        strat = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None
        train, test = train_test_split(
            df, test_size=0.2, random_state=42, stratify=strat
        )
        return train.reset_index(drop=True), test.reset_index(drop=True)

    raise FileNotFoundError(
        f"No train/test CSVs found in {raw_dir}. "
        "Place 'train.csv' and 'test.csv' (or one combined CSV) there."
    )


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names (SWELL variants sometimes pad them)."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Apply non-fitted cleaning steps: column norm, drop ids, inf->NaN, dedupe."""
    df = _normalise_columns(df)

    dropped = [c for c in ID_COLUMNS if c in df.columns]
    if dropped:
        df = df.drop(columns=dropped)
        print(f"[clean:{name}] dropped identifier columns: {dropped}")

    # +/-inf can appear in ratio features (e.g. LF_HF); treat as missing.
    numeric = df.select_dtypes(include=[np.number])
    n_inf = int(np.isinf(numeric).sum().sum())
    if n_inf:
        df = df.replace([np.inf, -np.inf], np.nan)
        print(f"[clean:{name}] replaced {n_inf} inf values with NaN")

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if before != len(df):
        print(f"[clean:{name}] removed {before - len(df)} duplicate rows")

    n_missing = int(df.isna().sum().sum())
    print(f"[clean:{name}] rows={len(df)} cols={df.shape[1]} missing_cells={n_missing}")
    return df


def preprocess(
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    scale: bool = True,
) -> dict:
    """Run the full pipeline and write artifacts. Returns a small summary dict."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw = load_raw(raw_dir)
    train = clean_frame(train_raw, "train")
    test = clean_frame(test_raw, "test")

    if TARGET_COLUMN not in train.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' missing from train data.")

    # --- Split into features / target -------------------------------------
    feature_cols = [c for c in train.columns if c != TARGET_COLUMN and c in test.columns]

    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()
    y_train_raw = train[TARGET_COLUMN].astype(str).str.strip()
    y_test_raw = test[TARGET_COLUMN].astype(str).str.strip()

    # --- Drop constant / zero-variance columns (learned on train) ---------
    nunique = X_train.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        X_train = X_train.drop(columns=constant_cols)
        X_test = X_test.drop(columns=constant_cols)
        feature_cols = [c for c in feature_cols if c not in constant_cols]
        print(f"[features] dropped {len(constant_cols)} constant columns: {constant_cols}")

    # --- Median imputation (fit on train only) ----------------------------
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    # --- Encode target (fit on train) -------------------------------------
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    unseen = set(y_test_raw.unique()) - set(le.classes_)
    if unseen:
        raise ValueError(f"Test set has labels unseen in train: {unseen}")
    y_test = le.transform(y_test_raw)
    label_mapping = {cls: int(i) for i, cls in enumerate(le.classes_)}
    print(f"[target] classes -> {label_mapping}")

    # --- Scale features (fit on train) ------------------------------------
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
    train_out[TARGET_COLUMN] = y_train
    test_out = X_test.copy()
    test_out[TARGET_COLUMN] = y_test

    train_out.to_csv(processed_dir / "train_processed.csv", index=False)
    test_out.to_csv(processed_dir / "test_processed.csv", index=False)

    # Fitted artifacts for inference-time reuse.
    joblib.dump(le, processed_dir / "label_encoder.joblib")
    if scaler is not None:
        joblib.dump(scaler, processed_dir / "scaler.joblib")

    summary = {
        "n_features": len(feature_cols),
        "features": feature_cols,
        "label_mapping": label_mapping,
        "train_rows": int(len(train_out)),
        "test_rows": int(len(test_out)),
        "scaled": scale,
        "constant_dropped": constant_cols,
        "id_dropped": ID_COLUMNS,
    }
    with open(processed_dir / "preprocess_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[done] wrote train_processed.csv ({summary['train_rows']} rows), "
        f"test_processed.csv ({summary['test_rows']} rows) to {processed_dir}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean & preprocess the SWELL HRV dataset.")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument(
        "--no-scale",
        dest="scale",
        action="store_false",
        help="Skip StandardScaler (e.g. for tree models that don't need it).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(raw_dir=args.raw_dir, processed_dir=args.processed_dir, scale=args.scale)
