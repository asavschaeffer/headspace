import sqlite3
import json
from typing import Optional, Tuple

DB_NAME = "file_index.db"

class Database:
    def __init__(self, db_path: str = DB_NAME):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_conn() as conn:
            # Added extra_metadata for flexibility as requested
            conn.execute('''CREATE TABLE IF NOT EXISTS files
              (path TEXT PRIMARY KEY, 
               summary TEXT, 
               type TEXT, 
               topics TEXT, 
               action TEXT,
               extra_metadata TEXT)''')
            conn.commit()

    def upsert_file(self, path: str, summary: str, type_: str = "", topics: str = "", action: str = "", extra_metadata: Optional[dict] = None):
        extra_json = json.dumps(extra_metadata) if extra_metadata else "{}"
        with self._get_conn() as conn:
            conn.execute("INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?)",
                         (path, summary, type_, topics, action, extra_json))
            conn.commit()

    def get_file(self, path: str) -> Optional[Tuple]:
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT * FROM files WHERE path = ?", (path,))
            return cursor.fetchone()

    def close(self):
        pass
