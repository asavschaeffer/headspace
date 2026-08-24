// The kernel vocabulary. Truth is graph-shaped: chunks are continuing identities,
// revisions are immutable historical states, blobs are immutable payloads,
// occurrences place chunks in containers, derivations record ancestry, links
// record explicit connections. Everything else in Substrate is built over these.

export type ChunkId = string;
export type RevisionId = string;
export type BlobHash = string; // sha-256 hex of (mediaType + "\0" + text)
export type OccurrenceId = string;
export type LinkId = string;
export type DerivationId = string;
export type OperationId = string;
export type ProposalId = string;
export type CommitId = string;
export type ActorId = string; // "human:asa", "agent:<model>", "driver:fs", "external:<layer>"

export const MEDIA_MARKDOWN = 'text/markdown';
export const MEDIA_TEXT = 'text/plain';
// A composite's content is its child occurrences; ordering lives in occurrence
// positions, never in the blob, so rearrangement is not a content edit.
export const MEDIA_COMPOSITE = 'application/x-substrate-composite';

export interface Chunk {
  id: ChunkId;
  currentRevisionId: RevisionId;
  tombstoned: boolean;
}

export interface Revision {
  id: RevisionId;
  chunkId: ChunkId;
  blobHash: BlobHash;
  mediaType: string;
  parentRevisionIds: RevisionId[]; // >1 parent = merge
  createdBy: ActorId;
  createdAt: string;
  operationId: OperationId;
  redacted?: boolean; // content hidden from display/index; identity and shape remain
}

export interface Blob {
  hash: BlobHash;
  mediaType: string;
  text: string; // v1 payloads are text; binary media arrives behind the same seam later
}

// A span inside the decoded text of one immutable revision, as produced by a
// specific, versioned decomposition method. Offsets are UTF-16 code units.
export interface SpanAddress {
  revisionId: RevisionId;
  method: string; // e.g. "md/blocks@1", "sent/icu@1", "word/icu@1", "raw@1"
  start: number;
  end: number;
}

export type OccurrenceMode = 'contain' | 'transclude';

export interface Occurrence {
  id: OccurrenceId;
  containerId: ChunkId;
  chunkId: ChunkId;
  position: string; // fractional index key; sibling order is lexicographic
  mode: OccurrenceMode;
  pin: 'current' | RevisionId; // transclusions default to a pinned revision
  watch: boolean; // watched transclusion: pinned render + update proposals on source change
}

export interface ExternalRef {
  layer: string; // e.g. "wikipedia", "wikidata", "web"
  key: string;
  url?: string;
  snapshotAt?: string;
}

export interface Link {
  id: LinkId;
  fromChunkId: ChunkId;
  fromSpan?: SpanAddress;
  toChunkId?: ChunkId;
  toRevisionId?: RevisionId;
  toExternal?: ExternalRef;
  role: string; // "references", "replies-to", "candidate-referent", ...
  operationId: OperationId;
}

export interface Derivation {
  id: DerivationId;
  childChunkId: ChunkId;
  sourceRevisionId: RevisionId;
  sourceSpan?: SpanAddress;
  via: 'copy' | 'fork' | 'generate' | 'extract';
  operationId: OperationId;
}

export type OperationKind =
  | 'create'
  | 'revise'
  | 'place'
  | 'move'
  | 'sever'
  | 'relate'
  | 'unrelate'
  | 'copy'
  | 'reference'
  | 'transclude'
  | 'promote'
  | 'propose'
  | 'accept'
  | 'reject'
  | 'tombstone'
  | 'redact'
  | 'import'
  | 'reconcile';

export interface Operation {
  id: OperationId;
  kind: OperationKind;
  actorId: ActorId;
  at: string;
  inputRevisionIds: RevisionId[];
  outputRevisionIds: RevisionId[];
  params?: unknown;
}

// A proposed change is inert data. Nothing here touches truth until an accept
// operation applies the payload atomically.
export type ProposedChange =
  | { op: 'revise'; chunkId: ChunkId; text: string; mediaType?: string; mergeParentRevisionIds?: RevisionId[] }
  | {
      op: 'create';
      tempId: string; // resolved to a real ChunkId at accept time
      text: string;
      mediaType?: string;
      derivedFrom?: { sourceRevisionId: RevisionId; via: Derivation['via']; sourceSpan?: SpanAddress };
    }
  | {
      op: 'place';
      containerId: ChunkId;
      chunkId: ChunkId | { tempId: string };
      // Anchoring: after a sibling, at the container start, or unanchored —
      // which follows the previous place into the same container within the
      // same accept (so a run of new blocks keeps its order), else the end.
      after?: OccurrenceId;
      at?: 'start';
      mode?: OccurrenceMode;
      watch?: boolean;
    }
  | { op: 'repin'; occurrenceId: OccurrenceId; revisionId: RevisionId }
  | { op: 'relate'; fromChunkId: ChunkId; toChunkId?: ChunkId; toExternal?: ExternalRef; role: string }
  | { op: 'sever'; occurrenceId: OccurrenceId };

export type ProposalKind =
  | 'generation'
  | 'source-update'
  | 'suggested-edit'
  | 'detected-relation'
  | 'merge'
  | 'reconciliation';

export type ProposalStatus = 'open' | 'accepted' | 'rejected' | 'superseded';

export interface OccurrencePrecondition {
  id: OccurrenceId;
  containerId: ChunkId;
  chunkId: ChunkId;
  position: string;
  mode: OccurrenceMode;
  pin: 'current' | RevisionId;
  watch: boolean;
}

export interface ContextStructurePrecondition {
  // Exact child sets/order for every rendered or selection-bearing container.
  containers: Array<{ containerId: ChunkId; occurrences: OccurrencePrecondition[] }>;
  // Exact placements of selected items, including the focus itself.
  placements: Array<{ chunkId: ChunkId; occurrences: OccurrencePrecondition[] }>;
}

export interface ContextRevisionPrecondition {
  chunkId: ChunkId;
  revisionId: RevisionId;
  // Pinned revisions remain valid if their chunk head advances; current-following
  // revisions do not. Both still carry their visibility state below.
  followsCurrent: boolean;
  redacted: boolean;
  chunkTombstoned: boolean;
}

export interface ProducerRef {
  id: string;
  version: string;
  implementation?: string;
  receiptId?: string;
}

export interface Proposal {
  id: ProposalId;
  // The exact propose operation carries the complete input revision set. This
  // stays optional only so pre-release snapshots written before 0.1.0 can be
  // inspected; every newly created proposal records it.
  operationId?: OperationId;
  kind: ProposalKind;
  status: ProposalStatus;
  basisRevisionIds: RevisionId[]; // what the proposal was computed against
  // Revisions whose heads must remain current for this proposal to be honest.
  // Generation records every context revision here: a proposal may still be
  // structurally applicable after a context edit, but it is no longer fresh.
  // Optional only for pre-release snapshots created before this field existed.
  freshnessRevisionIds?: RevisionId[];
  freshnessRevisionStates?: ContextRevisionPrecondition[];
  freshnessStructure?: ContextStructurePrecondition;
  producer?: ProducerRef;
  targetChunkIds: ChunkId[];
  payload: ProposedChange[];
  note?: string; // short human-readable account of what is proposed and why
  createdBy: ActorId;
  createdAt: string;
  resolution?: { by: ActorId; at: string; operationId?: OperationId; reason?: string };
}

// The atomic unit of change: one operation, its resulting facts, one log append.
export interface Facts {
  blobs?: Blob[];
  chunks?: Chunk[];
  revisions?: Revision[];
  setCurrent?: { chunkId: ChunkId; revisionId: RevisionId }[];
  occurrences?: Occurrence[];
  occurrenceUpdates?: { id: OccurrenceId; position?: string; pin?: 'current' | RevisionId; watch?: boolean }[];
  removeOccurrences?: OccurrenceId[];
  links?: Link[];
  removeLinks?: LinkId[];
  derivations?: Derivation[];
  proposals?: Proposal[];
  proposalUpdates?: { id: ProposalId; status: ProposalStatus; resolution?: Proposal['resolution'] }[];
  tombstone?: ChunkId[];
  redactRevisions?: RevisionId[];
}

export interface Commit {
  id: CommitId;
  parentIds: CommitId[];
  at: string;
  actorId: ActorId;
  operation: Operation;
  facts: Facts;
}

export interface Actor {
  id: ActorId;
  kind: 'human' | 'agent' | 'driver' | 'external';
  name: string;
}
