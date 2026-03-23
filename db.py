"""
filemap database layer.

all sqlite operations live here. the rest of filemap talks to these
functions, never to sqlite directly.
"""

import json
import sqlite3
from datetime import datetime


SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id          INTEGER PRIMARY KEY,
    directory   TEXT    NOT NULL,
    scanned_at  TEXT    NOT NULL,
    entry_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS entries (
    id            INTEGER PRIMARY KEY,
    scan_id       INTEGER NOT NULL REFERENCES scans(id),
    name          TEXT    NOT NULL,
    path          TEXT    NOT NULL UNIQUE,
    relative_path TEXT    NOT NULL,
    is_folder     INTEGER NOT NULL DEFAULT 0,
    category      TEXT    NOT NULL DEFAULT 'other',
    extension     TEXT,
    size_bytes    INTEGER NOT NULL DEFAULT 0,
    modified      TEXT,
    created       TEXT,
    accessed      TEXT,
    sha256        TEXT,
    extra         TEXT,
    ai_summary    TEXT,
    ai_category   TEXT,
    tag           TEXT,
    note          TEXT
);

CREATE TABLE IF NOT EXISTS categories (
    name        TEXT PRIMARY KEY,
    description TEXT,
    extensions  TEXT
);
"""


def init_db(db_path, categories=None):
    """open (or create) the database and ensure tables exist.

    categories: dict of {name: {description, extensions}} to seed.
    returns a connection with row_factory set to sqlite3.Row.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)

    if categories:
        for name, info in categories.items():
            conn.execute(
                """INSERT OR REPLACE INTO categories (name, description, extensions)
                   VALUES (?, ?, ?)""",
                (name, info["description"], json.dumps(sorted(info["extensions"]))),
            )
        conn.commit()

    return conn


def insert_scan(conn, directory):
    """record a new scan. returns the scan id."""
    cur = conn.execute(
        "INSERT INTO scans (directory, scanned_at) VALUES (?, ?)",
        (directory, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    return cur.lastrowid


def upsert_entry(conn, scan_id, entry):
    """insert a file/folder entry, or update it if the path already exists.

    entry is a dict with keys matching the entries columns.
    on conflict (same path), we update everything except user-set fields
    (tag, note, ai_summary, ai_category) so re-scans refresh metadata
    without clobbering manual annotations.
    """
    conn.execute(
        """INSERT INTO entries
               (scan_id, name, path, relative_path, is_folder,
                category, extension, size_bytes,
                modified, created, accessed, sha256, extra)
           VALUES
               (:scan_id, :name, :path, :relative_path, :is_folder,
                :category, :extension, :size_bytes,
                :modified, :created, :accessed, :sha256, :extra)
           ON CONFLICT(path) DO UPDATE SET
               scan_id       = excluded.scan_id,
               name          = excluded.name,
               relative_path = excluded.relative_path,
               is_folder     = excluded.is_folder,
               category      = excluded.category,
               extension     = excluded.extension,
               size_bytes    = excluded.size_bytes,
               modified      = excluded.modified,
               created       = excluded.created,
               accessed      = excluded.accessed,
               sha256        = excluded.sha256,
               extra         = excluded.extra
        """,
        {
            "scan_id": scan_id,
            "name": entry.get("name"),
            "path": entry.get("path"),
            "relative_path": entry.get("relative_path"),
            "is_folder": entry.get("is_folder", 0),
            "category": entry.get("category", "other"),
            "extension": entry.get("extension"),
            "size_bytes": entry.get("size_bytes", 0),
            "modified": entry.get("modified"),
            "created": entry.get("created"),
            "accessed": entry.get("accessed"),
            "sha256": entry.get("sha256"),
            "extra": json.dumps(entry["extra"]) if entry.get("extra") else None,
        },
    )


def finalize_scan(conn, scan_id, count):
    """update the scan record with the final entry count."""
    conn.execute(
        "UPDATE scans SET entry_count = ? WHERE id = ?",
        (count, scan_id),
    )
    conn.commit()


def get_entries(conn, scan_id=None):
    """read entries back as a list of dicts.

    if scan_id is given, only entries from that scan.
    otherwise, all entries (latest scan_id per path).
    """
    if scan_id:
        rows = conn.execute(
            "SELECT * FROM entries WHERE scan_id = ? ORDER BY category, name",
            (scan_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM entries ORDER BY category, name"
        ).fetchall()

    return [dict(row) for row in rows]
