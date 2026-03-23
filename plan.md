# filemap

see what you have before you organize it.

## what it is

a tool that helps anyone organize their computer, one folder at a time.
it scans a directory, builds an index of everything in it, optionally
uses ai to understand what each file actually is, and then helps you
put things where they belong.

no programming knowledge required. run one command, get a clear picture.

## principles

- **start simple, add layers.** metadata first. ai second. reorganization third.
- **the index is the contract.** every step reads from and writes to the same JSON index. scan populates it, ai enriches it, you review it, organize executes it.
- **never destructive by default.** moves are previewed before they happen. deletes require confirmation. originals can be preserved.
- **accessible to everybody.** plain language in all output. no jargon. the tool teaches you what it's doing as it does it.

## versions

### v0.1 — metadata scan (done)
- walk a directory, collect file metadata (name, size, dates, extension)
- categorize by extension (document, image, video, audio, code, archive, etc.)
- output a summary to console and a full index to JSON
- configurable depth limit
- zero dependencies, just python stdlib

### v0.2 — interactive review
- `filemap review <index.json>` opens a simple interactive session
- walk through files one at a time (or by category)
- for each file, user can tag: `keep`, `move`, `archive`, `delete`, `skip`
- user can add a one-line note ("this is my visa application")
- tags and notes are written back into the index JSON
- support filtering: `filemap review index.json --category document`
- at the end, print a summary of decisions made

### v0.3 — ai enrichment (gemini multimodal)
- `filemap enrich <index.json>` sends files to gemini for classification
- for text/pdf/docx: extract content, send to model, get back a summary + smarter category
- for images: send the image, get a description
- for audio/video: send filename + metadata (content later if api supports it)
- results written back into the index as `ai_summary` and `ai_category` fields
- rate limiting and cost awareness built in
- works without api key — just skips enrichment gracefully

### v0.4 — organize
- `filemap organize <index.json>` reads tags and moves files
- preview mode by default: shows what would happen, asks for confirmation
- builds a target folder structure based on categories or user-defined rules
- moves files, updates the index with new paths
- generates a move log so nothing is lost

### future ideas (not committed)
- web ui for review instead of terminal
- watch mode: monitor a folder and auto-index new files
- deduplication (hash-based)
- cross-folder index merging (the "larger index" idea)
- tagging system beyond categories
- integration with cloud storage (gdrive, onedrive)
- local model support (ollama) as alternative to gemini

## usage

```
python filemap.py                        # scan Desktop (default)
python filemap.py ~/Documents            # scan any folder
python filemap.py ~/Desktop -d 1         # top-level only
python filemap.py ~/Desktop -o my.json   # custom output name
```

## file structure

```
filemap/
  filemap.py        # the whole tool (single file for now)
  plan.md           # this file
  filemap_*.json    # generated indexes (gitignored)
```
