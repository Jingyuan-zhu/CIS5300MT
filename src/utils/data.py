"""Data loading helpers for translation datasets stored in parquet format."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd


def load_translation_parquet(
    parquet_path: str | Path,
    source_lang: str = "en",
    target_lang: str = "es",
    source_column: str = "source",
    target_column: str = "target",
) -> pd.DataFrame:
    """Load a translation dataset into a DataFrame with standardized columns.

    Parameters
    ----------
    parquet_path:
        Path to the parquet file containing translation examples.
    source_lang, target_lang:
        Expected language keys when examples are stored as dictionaries
        (e.g. Hugging Face `translation` column).
    source_column, target_column:
        Column names to fall back on if the parquet already includes flat
        columns for each language.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with two columns: ``source`` and ``target``.

    Raises
    ------
    FileNotFoundError
        If the provided path does not exist.
    ValueError
        If the parquet file does not share a recognizable schema.
    """

    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(path)

    if "translation" in df.columns:
        translations = df["translation"].apply(_ensure_mapping)
        source_series = translations.apply(lambda row: row.get(source_lang))
        target_series = translations.apply(lambda row: row.get(target_lang))
        return pd.DataFrame({"source": source_series, "target": target_series})

    available_columns = {col.lower(): col for col in df.columns}
    src_key = available_columns.get(source_column.lower()) or available_columns.get(source_lang.lower())
    tgt_key = available_columns.get(target_column.lower()) or available_columns.get(target_lang.lower())

    if src_key and tgt_key:
        renamed_df = df.rename(columns={src_key: "source", tgt_key: "target"})
        return renamed_df.loc[:, ["source", "target"]]

    raise ValueError(
        "Unable to derive translation columns. Expected either a `translation` column "
        f"with keys `{source_lang}` and `{target_lang}` or flat columns named "
        f"`{source_column}`/`{target_column}`."
    )


def make_parallel_corpus(
    df: pd.DataFrame,
    source_column: str = "source",
    target_column: str = "target",
) -> List[Tuple[str, str]]:
    """Convert a DataFrame into a list of (source, target) sentence tuples."""
    if source_column not in df.columns or target_column not in df.columns:
        missing = {column for column in (source_column, target_column) if column not in df.columns}
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    return list(df[[source_column, target_column]].itertuples(index=False, name=None))


def sample_dataframe(
    df: pd.DataFrame,
    n_samples: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Randomly sample n rows from a DataFrame with fixed seed for reproducibility.
    
    Parameters
    ----------
    df:
        Input DataFrame to sample from.
    n_samples:
        Number of samples to draw. If >= len(df), returns full DataFrame.
    random_seed:
        Random seed for reproducibility across models (default: 42).
    
    Returns
    -------
    pandas.DataFrame
        Sampled DataFrame with n_samples rows (or fewer if df is smaller).
    """
    if n_samples >= len(df):
        return df
    return df.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)


def _ensure_mapping(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, MappingABC):
        return dict(value)
    raise TypeError(f"Expected a mapping for translation column, received: {type(value)!r}")

