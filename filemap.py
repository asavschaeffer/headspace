"""
filemap - see what you have before you organize it.

Run it on any folder. It looks at the folder's immediate contents —
files and subfolders — and builds an index of what's there.

Usage:
    python filemap.py                    # scans your Desktop
    python filemap.py ~/Documents        # scans Documents
    python filemap.py ~/Desktop -o my.db # custom database name
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import db

# ── file categories ──────────────────────────────────────────────────
# add to these as needed. anything not listed goes under "other".

CATEGORIES = {
    "document": {
        "extensions": {
            ".pdf", ".docx", ".doc", ".txt", ".rtf", ".odt",
            ".pptx", ".ppt", ".xlsx", ".xls", ".csv",
            ".md", ".mht", ".mhtml", ".epub",
        },
        "description": "written documents, spreadsheets, presentations",
    },
    "image": {
        "extensions": {
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg",
            ".webp", ".tiff", ".ico", ".xcf", ".psd", ".ai",
            ".raw", ".cr2", ".nef",
        },
        "description": "photos, artwork, graphics",
    },
    "video": {
        "extensions": {
            ".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv",
            ".webm", ".m4v",
        },
        "description": "video recordings and clips",
    },
    "audio": {
        "extensions": {
            ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a",
            ".wma",
        },
        "description": "music, voice memos, podcasts",
    },
    "code": {
        "extensions": {
            ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css",
            ".rs", ".go", ".java", ".c", ".cpp", ".h", ".cs",
            ".rb", ".php", ".sh", ".bat", ".ps1", ".sql",
            ".json", ".yaml", ".yml", ".toml", ".xml", ".ini",
            ".cfg", ".conf",
        },
        "description": "source code and config files",
    },
    "archive": {
        "extensions": {
            ".zip", ".tar", ".gz", ".7z", ".rar", ".bz2",
            ".xz", ".tgz",
        },
        "description": "compressed archives",
    },
    "executable": {
        "extensions": {
            ".exe", ".msi", ".lnk", ".appimage", ".dmg",
        },
        "description": "programs and shortcuts",
    },
    "3d-and-game": {
        "extensions": {
            ".blend", ".fbx", ".obj", ".stl", ".unitypackage",
            ".godot", ".tscn", ".tres", ".import",
        },
        "description": "3D models and game assets",
    },
}

# build a fast lookup: extension → category name
EXT_TO_CATEGORY = {}
for cat_name, cat_info in CATEGORIES.items():
    for ext in cat_info["extensions"]:
        EXT_TO_CATEGORY[ext] = cat_name


def categorize(filename):
    """return the category for a file based on its extension."""
    ext = Path(filename).suffix.lower()
    return EXT_TO_CATEGORY.get(ext, "other")


def human_size(num_bytes):
    """turn bytes into something readable."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def format_time(timestamp):
    """unix timestamp → readable date."""
    try:
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
    except (OSError, ValueError):
        return None


def folder_stats(folder_path):
    """quick stats for a subfolder: count items and total size."""
    total_size = 0
    child_count = 0
    try:
        for item in os.scandir(folder_path):
            child_count += 1
            try:
                if item.is_file(follow_symlinks=False):
                    total_size += item.stat().st_size
            except (PermissionError, OSError):
                pass
    except (PermissionError, OSError):
        pass
    return child_count, total_size


def scan_directory(target_dir):
    """scan a folder's immediate children — files and subfolders."""
    target = Path(target_dir).resolve()
    entries = []

    try:
        children = sorted(os.scandir(target), key=lambda e: e.name.lower())
    except PermissionError:
        print(f"\n  no permission to read {target}\n")
        return []

    for item in children:
        fpath = Path(item.path)

        if item.is_file(follow_symlinks=False):
            try:
                stat = item.stat()
                entries.append({
                    "name": item.name,
                    "path": str(fpath),
                    "relative_path": item.name,
                    "is_folder": 0,
                    "category": categorize(item.name),
                    "extension": fpath.suffix.lower() or None,
                    "size_bytes": stat.st_size,
                    "modified": format_time(stat.st_mtime),
                    "created": format_time(stat.st_ctime),
                    "accessed": format_time(stat.st_atime),
                })
            except (PermissionError, OSError):
                entries.append({
                    "name": item.name,
                    "path": str(fpath),
                    "relative_path": item.name,
                    "is_folder": 0,
                    "category": categorize(item.name),
                    "extension": fpath.suffix.lower() or None,
                    "size_bytes": 0,
                    "extra": {"error": "could not read file metadata"},
                })

        elif item.is_dir(follow_symlinks=False):
            try:
                stat = item.stat()
                child_count, total_size = folder_stats(fpath)
                entries.append({
                    "name": item.name,
                    "path": str(fpath),
                    "relative_path": item.name,
                    "is_folder": 1,
                    "category": "folder",
                    "extension": None,
                    "size_bytes": total_size,
                    "modified": format_time(stat.st_mtime),
                    "created": format_time(stat.st_ctime),
                    "accessed": format_time(stat.st_atime),
                    "extra": {"child_count": child_count},
                })
            except (PermissionError, OSError):
                entries.append({
                    "name": item.name,
                    "path": str(fpath),
                    "relative_path": item.name,
                    "is_folder": 1,
                    "category": "folder",
                    "extension": None,
                    "size_bytes": 0,
                    "extra": {"error": "could not read folder metadata"},
                })

    return entries


def build_summary(entries):
    """crunch the entries into a human-friendly summary."""
    by_category = {}
    for e in entries:
        cat = e["category"]
        if cat not in by_category:
            by_category[cat] = {"count": 0, "total_bytes": 0}
        by_category[cat]["count"] += 1
        by_category[cat]["total_bytes"] += e.get("size_bytes", 0)

    return by_category


def print_report(target_dir, entries, summary):
    """print a clear, friendly report to the console."""
    files = [e for e in entries if not e.get("is_folder")]
    folders = [e for e in entries if e.get("is_folder")]

    print()
    print(f"  filemap results for: {target_dir}")
    print(f"  {'─' * 50}")
    print(f"  {len(files)} files, {len(folders)} folders")
    print()

    # sort categories by file count, descending
    sorted_cats = sorted(summary.items(), key=lambda x: x[1]["count"], reverse=True)

    for cat_name, cat_data in sorted_cats:
        if cat_name == "folder":
            continue  # show folders separately
        desc = CATEGORIES.get(cat_name, {}).get("description", "uncategorized files")
        total_size = human_size(cat_data["total_bytes"])
        print(f"  {cat_name:<16} {cat_data['count']:>5} files   {total_size:>10}   ({desc})")

    print()

    if folders:
        print(f"  {'─' * 50}")
        print(f"  folders:")
        for f in sorted(folders, key=lambda x: x["size_bytes"], reverse=True):
            extra = f.get("extra", {})
            if isinstance(extra, str):
                extra = json.loads(extra)
            child_count = extra.get("child_count", "?")
            size = human_size(f["size_bytes"])
            print(f"    {f['name']:<36} {child_count:>4} items   {size:>10}")

    print()
    print(f"  {'─' * 50}")
    total_size = human_size(sum(e.get("size_bytes", 0) for e in entries))
    print(f"  total size: {total_size}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="filemap — see what you have before you organize it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python filemap.py                    scan your Desktop
  python filemap.py ~/Documents        scan Documents
  python filemap.py ~/Desktop -o my.db custom database name
        """,
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=str(Path.home() / "Desktop"),
        help="folder to scan (default: your Desktop)",
    )
    parser.add_argument(
        "-o", "--output",
        default="filemap.db",
        help="database file to write to (default: filemap.db)",
    )

    args = parser.parse_args()
    target = Path(args.directory).resolve()

    if not target.is_dir():
        print(f"\n  '{target}' is not a folder. check the path and try again.\n")
        sys.exit(1)

    print(f"\n  scanning: {target}")
    print(f"  ...")

    entries = scan_directory(target)

    if not entries:
        print(f"\n  no files or folders found in {target}\n")
        sys.exit(0)

    summary = build_summary(entries)
    print_report(str(target), entries, summary)

    # write to database
    conn = db.init_db(args.output, categories=CATEGORIES)
    scan_id = db.insert_scan(conn, str(target))

    for entry in entries:
        db.upsert_entry(conn, scan_id, entry)

    db.finalize_scan(conn, scan_id, len(entries))
    conn.close()

    print(f"  index saved to: {args.output}")
    print(f"  (query it with: sqlite3 {args.output} \"SELECT name, category FROM entries\")")
    print()


if __name__ == "__main__":
    main()
