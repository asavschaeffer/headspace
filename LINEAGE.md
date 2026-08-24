# The headspace lineage

Thirteen attempts at one idea, June 2024 to August 2026, preserved in one
repository.

Headspace is a new user experience for organizing ideas on a computer. The
pitch was written down in August 2024:

> A user would type an entry into the system, containing any idea that they
> have had… Typically labeling it as "blob" status, the contents of that entry
> are linked by contents, whether they are hashtags by the user or concepts
> identified by the A.I.. Then those contents are linked together with other
> pre-existing blobs in the system to form larger documents.

Everything under that sentence was thrown away and rebuilt eleven times, in
five languages, under eleven names. The sentence did not move. This file is the
map. Nothing was reconstructed that could be recovered intact.

```
git log --first-parent lineage      the story, fourteen commits
git log lineage                     the complete braided archive, 509 commits
git log --graph --oneline lineage   the braid
git checkout blobs-to-docs          stand at the beginning
```

## The eras

| era | when | what it was | stack | commits |
|---|---|---|---|---|
| [`blobs-to-docs`](../../tree/blobs-to-docs) | Jun–Nov 2024 | The pitch, the blobs it describes, and two manual reorganizations. No code. | Obsidian | 20 |
| [`dirsnap`](../../tree/dirsnap) | Mar–Apr 2025 | Directory to text map, and the map back. Shipped as an installer with a right-click entry. | Python, Tkinter | 26 |
| [`loom`](../../tree/loom) | May 2025 | Experiential convergence with data. A keylogger for yourself. | C#, Unity | 8 |
| [`jarvis`](../../tree/jarvis) | Jun–Jul 2025 | The first real pipeline. Embeddings, clustering, a map you can look at. | Go, Python, HDBSCAN | 6 |
| [`globule`](../../tree/globule) | Jul–Aug 2025 | Capture first, organize later. Plus a [42-doc design wiki](../../tree/globule-wiki). | Python | 171 |
| [`brain`](../../tree/brain) | Sep 2025 | Living memory galaxy. A weekend, and the design document it was built against. | Python | 3 |
| [`thoughtspace`](../../tree/thoughtspace) | Sep 2025 | Cosmic document explorer. Text becomes a star. | JS, MiniLM in-browser | 6 |
| [`cosmic-diary`](../../tree/cosmic-diary) | Oct–Nov 2025 | Headspace, the Cosmic Knowledge System. Shipped on Render. | Python, ChromaDB, Three.js | 129 |
| [`ai-os`](../../tree/ai-os) | Nov–Dec 2025 | Proposals, rollback, safety guardrails. The review gate is born. | Python | 12 |
| [`cortex`](../../tree/cortex) | Feb 2026 | A filesystem daemon for AI agents. Content-addressed, inode identity, review-gated. | Rust, SQLite | 10 |
| [`filemap`](../../tree/filemap) | Mar 2026 | See what you have before you organize it. The index is the contract. | Python, SQLite | 2 |
| [`substrate`](../../tree/substrate) | Jul 2026 | The kernel, isolated. Five nouns, three verbs. | JS, no dependencies | 22 |
| [`main`](../../tree/main) | Jul–Aug 2026 | **Headspace 0.0.1.** The first deliberately releasable text-kernel slice. | TypeScript | 16 |

No two era trunks share a commit ancestor. Each was started from an empty
directory, not forked from the one before it. Preserved attic work contributes
further independent roots, so the repository holds twenty-two roots in total.
What carried between eras was the idea, restated from memory each time.

## What the idea kept insisting on

Across twelve unrelated codebases, the same convictions reappear without ever
being copied forward:

- **Filing is the machine's job.** Every era refuses hierarchical folders as
  the primary structure. The 2024 pitch and the current pitch are the same
  pitch.
- **Space, not lists.** Documents as stars, relationships as nebulae, a memory
  palace you navigate. The word *nebula* is written down in July 2024 and then
  independently re-derived in Brain, Thoughtspace, cosmic-diary and the current
  TypeScript era — in codebases that share no line of code.
- **The machine proposes; the human disposes.** AI-OS built a proposal engine
  with rollback in December 2025. The Rust era built a review queue before
  embedding anything in February. The current era made proposals inert objects
  carrying their own provenance. None of them knew about the others.
- **There is a line the data does not cross without being asked.** Named
  explicitly in the substrate era as the API line: what can be computed offline
  for free, versus what costs money and leaves the machine.
- **Smaller each time.** 2025 adds 3D, multiplayer and a shared cosmos. 2026
  strips them back out — filemap keeps only an honest index, substrate keeps
  only a kernel, and 0.0.1 ships a text kernel with a markdown editor.

## How the dates work

Seven eras had real git history and are preserved untouched: original
commits, original authors, original dates, original merge structure.

Four eras were never under version control. Their commits were built one per
file, each dated to that file's modification time on disk:

| era | reconstructed from | commits |
|---|---|---|
| `blobs-to-docs` | `Documents/Obsidian/avs/{reorg,Zettelkasten}/…` | 12 |
| `loom` | `Documents/Obsidian/renaissance/loom try/` | 8 |
| `thoughtspace` | `Projects/another-try-for-headspace/` | 6 |
| `substrate` | `Projects/substrate/` | 22 |

Ordering within those four reflects when each file was **last written**, not
the order it was authored in. Contents went in as they were found — old
manifests, old dead ends, nothing tidied in hindsight.

Nothing was invented. There are no commits on dates where no work happened,
and no era contains a file that did not exist when its commits are dated.

One file was recovered from an agent session rather than from disk:
`brain`'s `endgame.txt`, the 1,942-line design document the era was built
against. It exists nowhere on this machine; its full text was inlined into a
Gemini CLI session on 2025-09-26, and its commit is dated 13:37:55 that day
— the moment it is provably attested, which falls between the era's two
original commits. Its date is therefore earlier than its parent's.

Agent transcripts were also searched for lost file versions in the `globule`
era. Every payload found was an intermediate state of a file that was later
committed in fuller form, so nothing was taken from them.

## What is independently verifiable

- **`jarvis`'s first commit is 20 June 2025** — original, not reconstructed —
  and matches this repository's `created_at` on GitHub to the day. It is the
  repo's own original content, left behind when the repo was re-rooted in
  October 2025 and recovered from a directory named JARVIS in August 2026.
- **The 2024 mtimes** are on the original Obsidian files, untouched.
- **`globule`, `cosmic-diary`, `ai-os`, `cortex` and `filemap`** carry their
  own unmodified histories.

What this repository does **not** claim is a continuous publication history.
GitHub records pushes server-side; `gh api repos/asavs/headspace` will show
when this archive was actually pushed, and that is the honest answer. The claim
here is priority of thought, evidenced, not priority of publication.

## The essay

A narrative version of all this, written for a reader rather than for the
archive, is at [docs/the-headspace-lineage.md](docs/the-headspace-lineage.md).

## Elsewhere

- **[`asavs/globule`](https://github.com/asavs/globule)** — the globule era's
  own repository. Its history is included here; that repo remains its home.
- **`attic/`** — 20 branches of work never merged anywhere, including
  `attic/cosmic-diary/visual-and-perf-improvements`: twelve days from November
  2025 that survived only because a second clone happened to still be on disk.
- **`globule-genesis`** — the week of research before globule's first commit:
  a vision document, a high level design, an architecture diagram drawn
  around ChromaDB and a daily passdown sheet, and two of the six logo drafts.
- **`hopper`** — written the night before the substrate kernel and finished
  the same morning. Design notes for an unattended folder reorganizer, the
  conversation that produced them, and a real ingestion run over 125 entries
  in `Projects`. Its line "how much of each file you read is a knob" becomes
  the extraction terraces the next afternoon.
- **`head-space-site`** — ten essays published under the name on my personal
  site in 2025, titled Game Design Brain Dump. Not the tool, but it is where
  the name went while nothing was being built.
- **`docs/lineage/`** — planning documents belonging to no particular era.
- Not checked in: an Obsidian vault of personal notes, and the live datastore
  the `brain` era wrote, still readable by the code on that branch.
- Agent session transcripts survive for five eras, across Gemini CLI, Codex,
  opencode and Claude Code stores. The earliest is 2025-07-28. No agent
  transcript of any kind exists for `blobs-to-docs`, `loom`, `jarvis`,
  `thoughtspace` or `ai-os`; their source survives on disk instead.
