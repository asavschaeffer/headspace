"""
filemap - see what you have before you organize it.

Run it on any folder. It reads nothing inside your files — just names,
sizes, dates, and types. Produces a simple report you can review.

Usage:
    python filemap.py                    # scans your Desktop
    python filemap.py ~/Documents        # scans Documents
    python filemap.py ~/Documents -d 2   # only go 2 folders deep
    python filemap.py ~/Desktop -o report.json   # custom output name
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

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
        return "unknown"


def scan_directory(target_dir, max_depth=None):
    """walk a directory and collect metadata for every file."""
    target = Path(target_dir).resolve()
    entries = []

    for root, dirs, files in os.walk(target):
        # calculate current depth
        depth = Path(root).relative_to(target).parts
        if max_depth is not None and len(depth) >= max_depth:
            dirs.clear()  # don't go deeper
            continue

        for fname in files:
            fpath = Path(root) / fname
            try:
                stat = fpath.stat()
                entries.append({
                    "name": fname,
                    "path": str(fpath),
                    "relative_path": str(fpath.relative_to(target)),
                    "category": categorize(fname),
                    "extension": fpath.suffix.lower() or "(none)",
                    "size_bytes": stat.st_size,
                    "size": human_size(stat.st_size),
                    "modified": format_time(stat.st_mtime),
                    "created": format_time(stat.st_ctime),
                })
            except (PermissionError, OSError):
                entries.append({
                    "name": fname,
                    "path": str(fpath),
                    "relative_path": str(fpath.relative_to(target)),
                    "category": categorize(fname),
                    "extension": fpath.suffix.lower() or "(none)",
                    "size_bytes": 0,
                    "size": "?",
                    "modified": "?",
                    "created": "?",
                    "error": "could not read file metadata",
                })

    return entries


def build_summary(entries):
    """crunch the entries into a human-friendly summary."""
    by_category = {}
    for e in entries:
        cat = e["category"]
        if cat not in by_category:
            by_category[cat] = {"count": 0, "total_bytes": 0, "files": []}
        by_category[cat]["count"] += 1
        by_category[cat]["total_bytes"] += e["size_bytes"]
        by_category[cat]["files"].append(e)

    return by_category


def print_report(target_dir, entries, summary):
    """print a clear, friendly report to the console."""
    print()
    print(f"  filemap results for: {target_dir}")
    print(f"  {'─' * 50}")
    print(f"  total files found: {len(entries)}")
    print()

    # sort categories by file count, descending
    sorted_cats = sorted(summary.items(), key=lambda x: x[1]["count"], reverse=True)

    for cat_name, cat_data in sorted_cats:
        desc = CATEGORIES.get(cat_name, {}).get("description", "uncategorized files")
        total_size = human_size(cat_data["total_bytes"])
        print(f"  {cat_name:<16} {cat_data['count']:>5} files   {total_size:>10}   ({desc})")

    print()
    print(f"  {'─' * 50}")
    total_size = human_size(sum(e["size_bytes"] for e in entries))
    print(f"  total size: {total_size}")
    print()


def save_index(entries, summary, target_dir, output_path):
    """save the full index as JSON."""
    index = {
        "filemap_version": "0.1.0",
        "scanned": target_dir,
        "scanned_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_files": len(entries),
        "summary": {
            cat: {"count": data["count"], "total_size": human_size(data["total_bytes"])}
            for cat, data in summary.items()
        },
        "files": entries,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"  index saved to: {output_path}")
    print(f"  (open this file to see every file with its category, size, and dates)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="filemap — see what you have before you organize it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python filemap.py                    scan your Desktop
  python filemap.py ~/Documents        scan Documents
  python filemap.py . -d 1            scan current folder, top level only
  python filemap.py ~/Desktop -o my_index.json
        """,
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=str(Path.home() / "Desktop"),
        help="folder to scan (default: your Desktop)",
    )
    parser.add_argument(
        "-d", "--depth",
        type=int,
        default=None,
        help="how many folders deep to go (default: unlimited)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="where to save the index JSON (default: filemap_<foldername>.json)",
    )

    args = parser.parse_args()
    target = Path(args.directory).resolve()

    if not target.is_dir():
        print(f"\n  '{target}' is not a folder. check the path and try again.\n")
        sys.exit(1)

    # default output name based on the folder
    if args.output is None:
        safe_name = target.name.lower().replace(" ", "_")
        args.output = f"filemap_{safe_name}.json"

    print(f"\n  scanning: {target}")
    if args.depth:
        print(f"  depth limit: {args.depth} levels")
    print(f"  ...")

    entries = scan_directory(target, max_depth=args.depth)

    if not entries:
        print(f"\n  no files found in {target}\n")
        sys.exit(0)

    summary = build_summary(entries)
    print_report(str(target), entries, summary)
    save_index(entries, summary, str(target), args.output)


if __name__ == "__main__":
    main()
