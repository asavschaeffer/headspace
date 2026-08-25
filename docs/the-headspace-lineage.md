# The Headspace Lineage

*Thirteen attempts at one idea, and the twelve I threw away. June 2024 to August 2026.*

In August 2024 I wrote down what I wanted to build, in a file in my notes folder
called `headspace; blobs to docs software`. The whole idea fits in one paragraph:

> A user would type an entry into the system, containing any idea that they have
> had, a response to something, a note, a story point, whatever. Typically
> labeling it as "blob" status, the contents of that entry are linked by
> contents, whether they are hashtags by the user or concepts identified by the
> A.I.. Then those contents are linked together with other pre-existing blobs in
> the system to form larger documents.

Under the next heading, which reads `## detailed`, the file says, in full: "idk
ask chatgpt".

Then I did not build any of it for months, which I had assumed meant the idea
went quiet. Going through my own activity logs to write this, it had not. In
November I spent an afternoon in a File Explorer window titled
`MASSIVEREORGLETSGOOOOO` and wrote two notes called Knowledge Organization and
REORG INDEX, which is the same idea with the ambition removed: I was doing the
filing by hand because nothing would do it for me.

Four months after that I built the thing and did not notice. DirSnap turns a
directory into a text map and a text map back into a directory, and I shipped it
with an installer and a right-click menu entry. I filed it under developer
tooling. It is the same round trip the current version does to markdown,
arriving sixteen months early under a name that kept it hidden from me until I
went looking for this essay.

When I did start building on purpose, I did not stop at one attempt. I built
this twelve more times, in five languages, and every version began in an empty
directory. Not one of them is a fork of the one before it. I would abandon a
codebase, wait, and then sit down and derive the whole thing again from scratch,
usually because some new model had shipped and I wanted to see what the idea
looked like if I argued it out from first principles instead of dragging the old
code forward.

What follows is the record. I am not embarrassed by the count. The interesting
part is not that I quit twelve times. It is what I rebuilt without meaning to.

## The eras

| when | era | what it was | stack | commits |
|---|---|---|---|---|
| Jun–Nov 2024 | `blobs-to-docs` | The pitch, the blobs it describes, and two manual reorganizations. No code. | Obsidian | 20 |
| Mar–Apr 2025 | `dirsnap` | Directory to text map, and the map back. Shipped as a Windows installer, released three times. | Python, Tkinter | 26 |
| May 2025 | `loom` | A keylogger for myself. If the material worth organizing is what I make without noticing, record all of it. | C#, Unity | 8 |
| Jun–Jul 2025 | `jarvis` | The first pipeline that ran end to end. Markdown in, embeddings out, clustered, drawn as a map. | Go, Python, HDBSCAN | 6 |
| Jul–Aug 2025 | `globule` | Capture first, organize later. A week of research, then 149 commits and a 42 document design wiki. | Python | 213 |
| Sep 2025 | `brain` | A living memory galaxy. Type anything, find it later through any mental pathway. | Python | 3 |
| Sep 2025 | `thoughtspace` | Started over the following afternoon. Text becomes a star in semantic space. | JS, MiniLM in browser | 6 |
| Oct–Nov 2025 | `cosmic-diary` | The only version anyone else could open. 3D constellations, deployed on Render. | Python, ChromaDB, Three.js | 129 |
| Nov–Dec 2025 | `ai-os` | The product is the reorganization, not the picture. Proposals with rollback and safety guards. | Python | 12 |
| Feb 2026 | `cortex` | A filesystem daemon for AI agents. Content addressed, inode identity, review gated. | Rust, SQLite | 10 |
| Mar 2026 | `filemap` | See what you have before you organize it. No embeddings, no clustering, no 3D. | Python, SQLite | 2 |
| Jul 2026 | `substrate` | The kernel. Five nouns, three verbs, and the API line. | JS, no dependencies | 22 |
| Jul–Aug 2026 | `main` | **Headspace 0.1.0.** A markdown editor over a chunk kernel. | TypeScript | 17 |

Between the 2024 notes and DirSnap there are four months. Between DirSnap and
loom, five weeks, ending on 23 May 2025 with fourteen hours in a single sitting
on agentic operating systems. Brain and Thoughtspace are one day apart.
Substrate and the current version are one day apart.

## What kept coming back

None of these codebases share a line. Twenty-three independent root commits
across thirteen eras, no forks, no copied files. So anything that shows up twice
got there because I reached the same conclusion again, having forgotten I
reached it the first time.

**The word "nebula", written down before anything could render one.** It is in
the 2024 notes, next to a line about syncing it with a graph view. Then it turns
up again in four codebases that share no code, always as the name for the same
picture: documents as stars, relationships as the cloud between them, a space
you fly through instead of a list you scroll.

**Filing is the machine's job, not mine.** Every version refuses folders as the
primary structure, worded almost identically in 2024 and in 2026. In September
2025 I wrote it out at length in a design document for a version that lasted a
weekend: "The core tension in your system is between 'just type whatever' and
'find it later through any mental pathway.' This isn't a UI problem, it's a
fundamental indexing challenge." That file survived by accident. I had pasted it
into a chat session, and the copy on disk is gone.

**The machine proposes, I dispose.** This one is the most convincing to me,
because the three versions of it are months apart and none of them knew about
the others. In December I built a proposal engine with rollback and safety
guards. In February I built a two pass crawl with a human review gate, on the
reasoning that a filesystem reorganizer has no business embedding everything it
sees unasked. In August I made generated work an inert object that arrives
carrying its author, its inputs, and the revisions it was based on, and goes
stale rather than overwriting you.

**There is a line the data does not cross without being asked.** In the July
2026 kernel I finally gave it a name. Everything that can be computed from a
file locally, its name, its type, its size, its hash, is free, offline and
private. Everything past that point, summaries and embeddings and entity
extraction, costs money and leaves the machine. That boundary is the API line.
It is enforced by a gate rather than respected as a convention, which is the
only version of a privacy rule I actually trust.

## The shape of it

Reading the ledger back, the two years split cleanly down the middle, and not
the way I expected.

In 2025 I added: 3D constellations, a multiplayer shared cosmos, a visual diary,
procedural shapes, a hosted deployment, a dozen parallel worktrees.

In 2026 I took it back out: filemap keeps only an honest index, substrate keeps
only a kernel, 0.1.0 ships a text editor.

## Where it ended up

Headspace 0.1.0 is a markdown editor. That is the whole release, and after
everything above it is a strange thing to type, but it is the first version I
have been willing to number.

Underneath it is the kernel from July, which I can now state without hedging.
Five nouns are the anatomy: kernel, driver, index, binding, store. Three verbs
are the physiology: select, reduce, generate. Everything the system does is
those three verbs in some order. Search is select. Opening a document is reduce.
Asking a model to continue it is generate. Ingesting your filesystem is not a
second kernel, it is a driver. The model is not kernel code either, it is also a
driver.

It runs on loopback only. The filesystem stays authoritative, so nothing gets
moved out from under you. Generated work never applies itself. Nothing leaves
the machine without the egress being labelled first.

I picked TypeScript over Rust and Python for an honest reason, which is that it
is a language I actually read. That is the sort of decision I would not have
made in 2025, and it is probably why this one shipped.

Twenty six months, thirteen tries, and one paragraph that never changed. I built
the round trip in 2025 and filed it under developer tooling. Whatever this
record shows, it is not that I lacked the idea.

## On the record itself

Commit counts are a bad proxy for effort and I have left them in anyway, because
they are the only number every era has. My window title logs put globule at
somewhere over a hundred hours and put brain, thoughtspace, loom and jarvis
together at under five, which the ledger above does not show at all. Those
figures are rough. Anything built inside an agent CLI logs under a conversation
name rather than a project name, so every era after mid 2025 is undercounted by
an amount I cannot measure. Read the ledger as a list of attempts, not as a
distribution of effort.

Nothing was invented. There are no commits on dates where no work happened, and
no era holds a file that did not exist when its commits are dated. Seven eras
kept their original version control and are untouched. Four were never under
version control, so their commits were built one per file, each dated to that
file's modification time on disk. Ordering inside those four reflects when a
file was last written, not the order I wrote it in. Contents went in as they
were found, old manifests and dead ends included, because tidying them in
hindsight would be the fastest way to make the whole thing worthless.

One file was recovered rather than found. The September 2025 design document
exists nowhere on my disk. Its full text survived because I had pasted it into
an agent session on 26 September 2025, and its commit is dated to that
afternoon, the moment it is provably attested. That date falls between the two
original commits of the era it belongs to, so it is older than its own parent.
That is accurate rather than tidy, and I would rather have it accurate.

For the branch layout, the attic, and the per era provenance, see
[LINEAGE.md](../LINEAGE.md).
