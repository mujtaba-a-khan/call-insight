# core/storage.py
# Simple local storage for dataframes (no SQL).
# - In-memory by default
# - Optional snapshots to disk (Parquet if available, else CSV)
# - Atomic writes to avoid partial files
#
# You can safely delete snapshot files; the app will recreate them.

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .schema import ensure_columns, normalize_calls_df
from .utils import ensure_dir


# ---------- Helpers for atomic writes ----------

def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False


# ---------- Storage class ----------

@dataclass
class SnapshotPaths:
    root: Path = Path("./artifacts/state")
    calls_parquet: str = "calls.parquet"
    calls_csv: str = "calls.csv"
    excluded_parquet: str = "excluded.parquet"
    excluded_csv: str = "excluded.csv"

    @property
    def calls_pq_path(self) -> Path:
        return self.root / self.calls_parquet

    @property
    def calls_csv_path(self) -> Path:
        return self.root / self.calls_csv

    @property
    def excluded_pq_path(self) -> Path:
        return self.root / self.excluded_parquet

    @property
    def excluded_csv_path(self) -> Path:
        return self.root / self.excluded_csv


class LocalStorage:
    """
    Keeps DataFrames in memory and optionally snapshots them to disk.
    No external services; no DB. Parquet preferred if pyarrow is present.
    """

    def __init__(self, paths: SnapshotPaths | None = None) -> None:
        self.paths = paths or SnapshotPaths()
        ensure_dir(self.paths.root)
        self.calls_df: pd.DataFrame = pd.DataFrame()
        self.excluded_df: pd.DataFrame = pd.DataFrame(columns=["call_id", "reason"])

    # ----- Load / Save -----

    def load_snapshots(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load snapshots if present; otherwise keep empty frames."""
        calls = self._load_calls()
        excluded = self._load_excluded()
        # Normalize columns/dtypes just in case
        self.calls_df = normalize_calls_df(ensure_columns(calls))
        self.excluded_df = excluded if not excluded.empty else pd.DataFrame(columns=["call_id", "reason"])
        return self.calls_df, self.excluded_df

    def save_snapshots(self) -> None:
        """Persist current in-memory frames to disk."""
        self._save_calls(self.calls_df)
        self._save_excluded(self.excluded_df)

    # ----- Public mutators -----

    def set_calls(self, df: pd.DataFrame, *, snapshot: bool = False) -> None:
        """Replace calls DataFrame and optionally snapshot to disk."""
        self.calls_df = normalize_calls_df(ensure_columns(df))
        if snapshot:
            self._save_calls(self.calls_df)

    def append_calls(self, df: pd.DataFrame, *, snapshot: bool = False) -> None:
        """Append rows to calls DataFrame (de-duplicates by call_id if present)."""
        incoming = normalize_calls_df(ensure_columns(df))
        if self.calls_df.empty:
            self.calls_df = incoming
        else:
            self.calls_df = pd.concat([self.calls_df, incoming], ignore_index=True)
            # Optional: de-duplicate by call_id + start_time (if both exist)
            if "call_id" in self.calls_df.columns and "start_time" in self.calls_df.columns:
                self.calls_df.drop_duplicates(subset=["call_id", "start_time"], keep="last", inplace=True)
        if snapshot:
            self._save_calls(self.calls_df)

    def set_excluded(self, df: pd.DataFrame, *, snapshot: bool = False) -> None:
        """Replace excluded rows DataFrame."""
        cols = ["call_id", "reason"]
        self.excluded_df = df[cols] if all(c in df.columns for c in cols) else pd.DataFrame(columns=cols)
        if snapshot:
            self._save_excluded(self.excluded_df)

    def append_excluded(self, df: pd.DataFrame, *, snapshot: bool = False) -> None:
        """Append excluded rows."""
        cols = ["call_id", "reason"]
        incoming = df[cols] if all(c in df.columns for c in cols) else pd.DataFrame(columns=cols)
        if self.excluded_df.empty:
            self.excluded_df = incoming
        else:
            self.excluded_df = pd.concat([self.excluded_df, incoming], ignore_index=True)
        if snapshot:
            self._save_excluded(self.excluded_df)

    # ----- Export helpers -----

    def export_visible_csv(self, df: pd.DataFrame, file_name: str = "export.csv") -> bytes:
        """
        Return CSV bytes for a given DataFrame (caller decides which columns/rows).
        Streamlit can pass this directly to download_button.
        """
        return df.to_csv(index=False).encode("utf-8")

    # ----- Internal IO -----

    def _load_calls(self) -> pd.DataFrame:
        if _parquet_available() and self.paths.calls_pq_path.exists():
            try:
                return pd.read_parquet(self.paths.calls_pq_path)
            except Exception:
                pass
        if self.paths.calls_csv_path.exists():
            try:
                return pd.read_csv(self.paths.calls_csv_path)
            except Exception:
                pass
        return pd.DataFrame()

    def _load_excluded(self) -> pd.DataFrame:
        if _parquet_available() and self.paths.excluded_pq_path.exists():
            try:
                return pd.read_parquet(self.paths.excluded_pq_path)
            except Exception:
                pass
        if self.paths.excluded_csv_path.exists():
            try:
                return pd.read_csv(self.paths.excluded_csv_path)
            except Exception:
                pass
        return pd.DataFrame(columns=["call_id", "reason"])

    def _save_calls(self, df: pd.DataFrame) -> None:
        if df is None:
            return
        if _parquet_available():
            # Parquet path
            tmp = io.BytesIO()
            df.to_parquet(tmp, index=False)
            _atomic_write_bytes(self.paths.calls_pq_path, tmp.getvalue())
        else:
            _atomic_write_bytes(self.paths.calls_csv_path, df.to_csv(index=False).encode("utf-8"))

    def _save_excluded(self, df: pd.DataFrame) -> None:
        if df is None:
            return
        if _parquet_available():
            tmp = io.BytesIO()
            df.to_parquet(tmp, index=False)
            _atomic_write_bytes(self.paths.excluded_pq_path, tmp.getvalue())
        else:
            _atomic_write_bytes(self.paths.excluded_csv_path, df.to_csv(index=False).encode("utf-8"))
