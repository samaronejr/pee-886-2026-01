"""Loader for the 3W dataset (Petrobras well event classification).

Discovers and loads Parquet files organised by class label, with optional
filtering by source type (real / simulated / drawn).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATA_PATH = _PROJECT_ROOT / "data" / "samarone_junior" / "3w"


class ThreeWLoader:
    """Load 3W Parquet instances by class label and source type."""

    SENSORS: list[str] = [
        "P-PDG",
        "P-TPT",
        "T-TPT",
        "P-MON-CKP",
        "T-JUS-CKP",
    ]

    EVENT_NAMES: dict[int, str] = {
        0: "Normal",
        1: "Abrupt BSW Increase",
        2: "Spurious DHSV Closure",
        3: "Severe Slugging",
        4: "Flow Instability",
        5: "Rapid Productivity Loss",
        6: "Quick Restriction in PCK",
        7: "Scaling in PCK",
        8: "Hydrate in Production Line",
        9: "Hydrate in Service Line",
    }

    # Mapping from source type name to file-name prefix
    _SOURCE_PREFIXES: dict[str, str] = {
        "real": "WELL-",
        "simulated": "SIMULATED_",
        "drawn": "DRAWN_",
    }

    def __init__(self, data_path: Optional[str] = None) -> None:
        if data_path is not None:
            self.data_path = Path(data_path)
        else:
            env = os.environ.get("QML_DATA_PATH")
            if env:
                self.data_path = Path(env)
            else:
                self.data_path = _DEFAULT_DATA_PATH

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_instances(
        self,
        class_label: int,
        source_types: Optional[List[str]] = None,
    ) -> list[Path]:
        """Return sorted list of Parquet file paths for *class_label*.

        Parameters
        ----------
        class_label:
            Integer label (0-9) corresponding to the event class directory.
        source_types:
            Optional subset of ``{'real', 'simulated', 'drawn'}``.  When
            ``None`` every file in the class directory is returned.
        """
        class_dir = self.data_path / str(class_label)
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Class directory not found: {class_dir}"
            )

        files = sorted(class_dir.glob("*.parquet"))

        if source_types is not None:
            prefixes = tuple(
                self._SOURCE_PREFIXES[st] for st in source_types
            )
            files = [f for f in files if f.name.startswith(prefixes)]

        return files

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_instances(
        self,
        class_label: int,
        source_types: Optional[List[str]] = None,
    ) -> list[pd.DataFrame]:
        """Load Parquet files and return DataFrames with SENSORS + class."""
        paths = self.list_instances(class_label, source_types=source_types)
        keep_cols = self.SENSORS + ["class"]
        frames: list[pd.DataFrame] = []
        for p in paths:
            df = pd.read_parquet(p, columns=keep_cols)
            frames.append(df)
        return frames
