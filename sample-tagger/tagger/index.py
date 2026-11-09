"""SQLite index of analysed files. Stores raw measurements and model outputs;
categories are computed from them on read, so tuning rules needs no re-scan."""

import json
import sqlite3
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).resolve().parent.parent / "data/index.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    features TEXT,
    panns_scores BLOB,
    panns_embedding BLOB,
    error TEXT
);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


def is_current(db: sqlite3.Connection, file: Path) -> bool:
    st = file.stat()
    row = db.execute("SELECT mtime, size FROM files WHERE path = ?", (str(file),)).fetchone()
    return row is not None and row["mtime"] == st.st_mtime and row["size"] == st.st_size


def save(db: sqlite3.Connection, file: Path, features: dict | None,
         scores: np.ndarray | None, embedding: np.ndarray | None,
         error: str | None = None) -> None:
    st = file.stat()
    db.execute(
        "INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?, ?, ?)",
        (str(file), st.st_mtime, st.st_size,
         json.dumps(features) if features is not None else None,
         scores.astype(np.float16).tobytes() if scores is not None else None,
         embedding.astype(np.float16).tobytes() if embedding is not None else None,
         error),
    )
    db.commit()


def save_features(db: sqlite3.Connection, file: Path, features: dict) -> None:
    db.execute("UPDATE files SET features = ? WHERE path = ?",
               (json.dumps(features), str(file)))
    db.commit()


def features_version(features_json: str) -> int:
    return json.loads(features_json).get("v", 1)


def load_row(row: sqlite3.Row) -> tuple[dict, np.ndarray]:
    return (json.loads(row["features"]),
            np.frombuffer(row["panns_scores"], dtype=np.float16).astype(np.float32))
