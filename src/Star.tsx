import { useEffect, useMemo, useRef, useState } from 'react';
import type { WorkspaceSession } from './App';
import {
  currentText,
  focusTarget,
  labelOf,
  leafBlocks,
  proposalHistoryForDoc,
  revisionCount,
  type LeafBlock,
} from './client/helpers';
import { dispatchToLocalCollaborator, OFFLINE_COLLABORATOR } from './collaboration/stub';
import type { CollaboratorCapability } from './collaboration/types';
import { buildIndexes, duplicatesOf, echoesOf, searchChunks } from './index/indexes';
import { generateProposal, reduce, select, type CompletionOutput, type ContextRole } from './kernel/select';
import { isComposite, occurrencesOfChunk, revisionText, type WorkspaceGraph } from './kernel/state';
import {
  acceptProposal,
  moveOccurrence,
  promoteCopy,
  promoteExtract,
  promoteSpanAnchor,
  rejectProposal,
  revise,
  severOccurrence,
  staleReason,
  transclude,
  type TxCtx,
} from './kernel/tx';
import { METHOD_RAW } from './kernel/decompose';
import type { ChunkId, Proposal, ProposedChange, SpanAddress } from './kernel/types';

type Span = SpanAddress & { chunkId: ChunkId };

export class HostActionError extends Error {
  override name = 'HostActionError';

  constructor(
    readonly code: string | null,
    readonly hostMessage: string,
    readonly status: number,
  ) {
    super(code ? `${code} — ${hostMessage}` : hostMessage);
  }
}

export function actionErrorMessage(label: string, error: unknown): string {
  return `${label}: ${error instanceof Error ? error.message : String(error)}`;
}

export async function projectSource(
  relPath: string,
  fetchImpl: typeof globalThis.fetch = globalThis.fetch,
): Promise<void> {
  const response = await fetchImpl('/api/project', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ relPath }),
  });
  if (response.ok) return;

  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    // A structured host diagnostic is preferred, with an HTTP fallback for
    // intermediaries or malformed responses.
  }
  const record = body && typeof body === 'object' && !Array.isArray(body)
    ? body as Record<string, unknown>
    : null;
  const code = typeof record?.code === 'string' ? record.code : null;
  const hostMessage = typeof record?.error === 'string'
    ? record.error
    : `projection request failed: HTTP ${response.status}`;
  throw new HostActionError(code, hostMessage, response.status);
}

const CONTEXT_REASON: Record<ContextRole, string> = {
  focus: 'the document you are working in',
  child: 'a directly contained part',
  parent: 'a container that locates the focus',
  sibling: 'material beside the focus',
  search: 'matched the current instruction',
};

// The focused work surface: compose, promote, dispatch, integrate.
export function Star({
  session,
  docId,
  onFocusDoc,
  onBack,
  backLabel,
}: {
  session: WorkspaceSession;
  docId: ChunkId;
  onFocusDoc: (id: ChunkId) => void;
  onBack: () => void;
  backLabel?: string;
}) {
  const { state, bindings, adapters } = session.ws!;
  const ctx = session.ctx!;
  const [instruction, setInstruction] = useState('');
  const [notice, setNotice] = useState<string | null>(null);
  const [span, setSpan] = useState<Span | null>(null);
  const [fatesFor, setFatesFor] = useState<ChunkId | null>(null);
  const [collaboratorId, setCollaboratorId] = useState(OFFLINE_COLLABORATOR.id);
  const [dispatching, setDispatching] = useState(false);
  const noticeTimer = useRef<number | null>(null);

  // Selection and fates belong to the doc they were made in.
  useEffect(() => {
    setSpan(null);
    setFatesFor(null);
    setInstruction('');
  }, [docId]);

  const blocks = useMemo(() => leafBlocks(state, docId), [docId, session.version]); // eslint-disable-line react-hooks/exhaustive-deps
  const indexes = useMemo(() => buildIndexes(state), [session.version]); // eslint-disable-line react-hooks/exhaustive-deps
  const proposalHistory = useMemo(() => proposalHistoryForDoc(state, docId), [docId, session.version]); // eslint-disable-line react-hooks/exhaustive-deps
  const contextSearchHits = useMemo(
    () => (instruction.trim() ? searchChunks(state, indexes, instruction).slice(0, 5) : []),
    [instruction, indexes, state],
  );
  const contextPreview = useMemo(
    () => reduce(select(state, docId, contextSearchHits)),
    [state, docId, contextSearchHits, session.version], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const collaborators = useMemo(() => {
    const advertised = session.ws?.collaborators ?? [];
    const byId = new Map<string, CollaboratorCapability>([[OFFLINE_COLLABORATOR.id, OFFLINE_COLLABORATOR]]);
    for (const capability of advertised) byId.set(capability.id, capability);
    return [...byId.values()];
  }, [session.ws?.collaborators]);
  const collaborator = collaborators.find((candidate) => candidate.id === collaboratorId) ?? OFFLINE_COLLABORATOR;
  const binding = bindings.find((b) => b.docChunkId === docId);
  const canProject = Boolean(
    binding &&
      adapters
        .find(
          (capability) =>
            capability.id === binding.adapterId && capability.version === binding.adapterVersion,
        )
        ?.outputs.some(
          (output) => output.mediaType === binding.mediaType && output.writeback === 'round-trip',
        ),
  );

  const flash = (msg: string) => {
    setNotice(msg);
    if (noticeTimer.current != null) clearTimeout(noticeTimer.current);
    noticeTimer.current = window.setTimeout(() => {
      noticeTimer.current = null;
      setNotice(null);
    }, 4000);
  };

  const focusVia = (id: ChunkId) => onFocusDoc(focusTarget(state, id));

  const guard = async (label: string, fn: () => Promise<unknown> | unknown) => {
    try {
      await fn();
    } catch (e) {
      flash(actionErrorMessage(label, e));
    }
  };

  const dispatch = () => {
    if (dispatching) return Promise.resolve();
    setDispatching(true);
    return session.runDispatch((dispatchCtx) => guard('dispatch', async () => {
      if (collaborator.availability.status !== 'ready') {
        throw new Error(collaborator.availability.diagnostic.message);
      }
      const complete = async (context: typeof contextPreview, nextInstruction: string): Promise<CompletionOutput> => {
        const request = {
          collaboratorId: collaborator.id,
          instruction: nextInstruction,
          context,
        };
        const result = collaborator.execution === 'local'
          ? await dispatchToLocalCollaborator(request)
          : await session.complete(request);
        if (
          result.collaboratorId !== collaborator.id ||
          result.collaboratorVersion !== collaborator.version ||
          result.actorId !== collaborator.actorId
        ) {
          throw new Error('collaborator identity changed during dispatch; no proposal was created');
        }
        return {
          text: result.text,
          producer: {
            id: result.collaboratorId,
            version: result.collaboratorVersion,
            implementation: result.model,
            receiptId: result.providerResponseId,
          },
        };
      };
      await generateProposal(dispatchCtx, {
        focusChunkId: docId,
        instruction: instruction.trim() || 'continue this',
        searchHits: contextSearchHits,
        complete,
        modelActorId: collaborator.actorId,
      });
      setInstruction('');
    })).finally(() => setDispatching(false));
  };

  const onAccept = (p: Proposal) =>
    guard('accept', async () => {
      const r = await acceptProposal(ctx, { proposalId: p.id });
      if (!r.applied) flash(`proposal superseded — ${r.reason}`);
    });

  const projectToFile = () =>
    guard('project', async () => {
      if (!binding) return;
      await projectSource(binding.relPath);
      flash(`projected to ${binding.relPath}`);
    });

  return (
    <div className="doc">
      <header>
        <button onClick={onBack}>← {backLabel ?? 'nebula'}</button>
        <h2>{labelOf(state, bindings, docId)}</h2>
        <span className="meta">
          v{revisionCount(state, docId)} · {blocks.length} blocks
        </span>
        {canProject && <button onClick={projectToFile}>project → file</button>}
      </header>

      {notice && <div className="notice">{notice}</div>}

      {blocks.length === 0 && !isComposite(state, docId) && (
        <LeafDocView key={docId} state={state} ctx={ctx} docId={docId} onError={flash} />
      )}

      {blocks.map((b, i) => (
        <BlockView
          key={b.occurrence.id}
          block={b}
          state={state}
          ctx={ctx}
          prevSibling={blocks[i - 1]}
          nextSibling={blocks[i + 1]}
          span={span?.chunkId === b.chunkId ? span : null}
          onSpan={setSpan}
          fatesOpen={fatesFor === b.chunkId}
          onToggleFates={() => setFatesFor(fatesFor === b.chunkId ? null : b.chunkId)}
          onError={flash}
        />
      ))}

      {fatesFor && <FatesPanel state={state} indexes={indexes} chunkId={fatesFor} bindings={bindings} onFocusDoc={focusVia} />}

      {span && (
        <div className="toolbar">
          <span className="meta">
            span [{span.start},{span.end})
          </span>
          <button onClick={() => guard('extract', () => promoteExtract(ctx, { span })).then(() => setSpan(null))}>
            promote: extract
          </button>
          <button
            onClick={() => guard('copy', () => promoteCopy(ctx, { span, containerId: docId, at: 'end' })).then(() => setSpan(null))}
          >
            promote: copy to end
          </button>
          <button onClick={() => guard('anchor', () => promoteSpanAnchor(ctx, { span })).then(() => setSpan(null))}>
            promote: anchor
          </button>
        </div>
      )}

      <section className="collaborator">
        <div className="collaborator-heading">
          <label>
            collaborator{' '}
            <select
              aria-label="Active collaborator"
              value={collaborator.id}
              disabled={dispatching}
              onChange={(event) => setCollaboratorId(event.target.value)}
            >
              {collaborators.map((candidate) => (
                <option
                  key={candidate.id}
                  value={candidate.id}
                  disabled={candidate.availability.status !== 'ready'}
                >
                  {candidate.label}{candidate.model ? ` · ${candidate.model}` : ''}
                  {candidate.availability.status !== 'ready' ? ' · unavailable' : ''}
                </option>
              ))}
            </select>
          </label>
          <span className="meta">
            {collaborator.actorId} · {collaborator.execution} · proposal-only
          </span>
        </div>
        <div className="collaborator-roster">
          {collaborators.map((candidate) => (
            <div key={candidate.id} className="meta">
              {candidate.label} · {candidate.execution}
              {candidate.execution === 'remote'
                ? ' · displayed context leaves this machine on dispatch'
                : ' · stays on this machine'}
              {candidate.availability.status === 'unavailable'
                ? ` · ${candidate.availability.diagnostic.message}`
                : ' · ready'}
            </div>
          ))}
        </div>
        <details className="context" open>
          <summary>
            bounded context · {contextPreview.items.length} item(s) · {contextPreview.chars}/6000 characters
            {contextPreview.dropped > 0 ? ` · ${contextPreview.dropped} dropped` : ''}
          </summary>
          <div className="context-items">
            {contextPreview.items.map((item) => (
              <div className="context-item" key={`${item.role}:${item.chunkId}`}>
                <div className="meta">
                  <span className="context-role">{item.role}</span> · {CONTEXT_REASON[item.role]} · {item.chunkId} · {item.revisionId}
                  {item.occurrenceId ? ` · occurrence ${item.occurrenceId}` : ''}
                </div>
                <details className="context-dependencies">
                  <summary>{item.dependencies.length} exact input revision(s)</summary>
                  {item.dependencies.map((dependency) => (
                    <div className="meta" key={`${dependency.chunkId}:${dependency.revisionId}`}>
                      {dependency.chunkId} · {dependency.revisionId} · {dependency.followsCurrent ? 'follows current' : 'pinned'}
                      {dependency.redacted ? ' · redacted' : ''}
                    </div>
                  ))}
                </details>
                {item.text && <pre>{item.text}</pre>}
              </div>
            ))}
          </div>
        </details>
      </section>

      <div className="dispatch">
        <input
          placeholder="instruct an agent…"
          value={instruction}
          onChange={(e) => setInstruction(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && !dispatching && void dispatch()}
          disabled={dispatching}
        />
        <button onClick={() => void dispatch()} disabled={dispatching}>
          {dispatching ? 'thinking…' : 'dispatch'}
        </button>
      </div>

      <AttachBox state={state} ctx={ctx} indexes={indexes} docId={docId} bindings={bindings} onError={flash} />

      {proposalHistory.length > 0 && (
        <div className="proposals">
          <div className="meta">
            {proposalHistory.filter((proposal) => proposal.status === 'open').length} open · {proposalHistory.length} total proposal(s)
          </div>
          {proposalHistory.map((p) => (
            <ProposalCard key={p.id} p={p} state={state} onAccept={() => onAccept(p)} onReject={() => guard('reject', () => rejectProposal(ctx, { proposalId: p.id }))} />
          ))}
        </div>
      )}
    </div>
  );
}

function BlockView({
  block,
  state,
  ctx,
  prevSibling,
  nextSibling,
  span,
  onSpan,
  fatesOpen,
  onToggleFates,
  onError,
}: {
  block: LeafBlock;
  state: WorkspaceGraph;
  ctx: TxCtx;
  prevSibling?: LeafBlock;
  nextSibling?: LeafBlock;
  span: Span | null;
  onSpan: (s: Span | null) => void;
  fatesOpen: boolean;
  onToggleFates: () => void;
  onError: (msg: string) => void;
}) {
  const [buffer, setBuffer] = useState(block.text);
  const revId = block.revision.id;
  useEffect(() => setBuffer(block.text), [revId]); // eslint-disable-line react-hooks/exhaustive-deps
  const ref = useRef<HTMLTextAreaElement>(null);

  const rows = (text: string) => Math.min(text.split('\n').reduce((n, line) => n + 1 + Math.floor(line.length / 75), 0), 14);

  const commitEdit = async () => {
    if (block.transcluded || buffer === block.text) return;
    try {
      await revise(ctx, { chunkId: block.chunkId, text: buffer });
    } catch (e) {
      onError(`edit: ${e instanceof Error ? e.message : String(e)}`);
      setBuffer(block.text);
    }
  };

  const captureSpan = () => {
    const el = ref.current;
    if (!el || block.transcluded || buffer !== block.text) return onSpan(null);
    const { selectionStart, selectionEnd } = el;
    if (selectionStart != null && selectionEnd != null && selectionEnd > selectionStart) {
      onSpan({ chunkId: block.chunkId, revisionId: revId, method: METHOD_RAW, start: selectionStart, end: selectionEnd });
    } else if (span) {
      onSpan(null);
    }
  };

  const sameContainer = (other?: LeafBlock) => other && other.occurrence.containerId === block.occurrence.containerId;
  // Every mutation from this surface reports through onError. An arrangement
  // op can still refuse (a sibling severed under us between render and click),
  // and React does not route event-handler throws to an error boundary — the
  // click would look ignored, with the reason only in the console.
  const guarded = (label: string, fn: () => unknown) => () => {
    try {
      fn();
    } catch (e) {
      onError(`${label}: ${e instanceof Error ? e.message : String(e)}`);
    }
  };
  // Swapping with a neighbor: push the previous sibling after me (up), or push
  // myself after the next sibling (down). Arrangement only — never a revision.
  const moveUp = guarded('move up', () => moveOccurrence(ctx, { occurrenceId: prevSibling!.occurrence.id, at: { after: block.occurrence.id } }));
  const moveDown = guarded('move down', () => moveOccurrence(ctx, { occurrenceId: block.occurrence.id, at: { after: nextSibling!.occurrence.id } }));
  const sever = guarded('sever', () => severOccurrence(ctx, { occurrenceId: block.occurrence.id }));

  return (
    <div className={`block ${block.transcluded ? 'transcluded' : ''}`} style={{ marginLeft: block.depth * 18 }}>
      <textarea
        ref={ref}
        value={buffer}
        rows={rows(buffer)}
        readOnly={block.transcluded}
        onChange={(e) => setBuffer(e.target.value)}
        onBlur={commitEdit}
        onSelect={captureSpan}
      />
      <div className="block-side">
        <span className="meta">
          {block.chunkId.slice(0, 12)} · v{revisionCount(state, block.chunkId)} · {block.revision.createdBy}
          {block.transcluded && (block.occurrence.watch ? ' · watched' : ' · pinned')}
        </span>
        <button title="deep fates" onClick={onToggleFates} className={fatesOpen ? 'active' : ''}>
          ☄
        </button>
        {block.transcluded && (
          <button title="sever this transclusion" onClick={sever}>
            ✂
          </button>
        )}
        <button disabled={!sameContainer(prevSibling)} onClick={moveUp} title="move up">
          ↑
        </button>
        <button disabled={!sameContainer(nextSibling)} onClick={moveDown} title="move down">
          ↓
        </button>
      </div>
    </div>
  );
}

// A chunk with no children opened as the focus: one editable surface for the
// chunk's own content (a floating promoted copy, a generated block, …).
function LeafDocView({
  state,
  ctx,
  docId,
  onError,
}: {
  state: WorkspaceGraph;
  ctx: TxCtx;
  docId: ChunkId;
  onError: (msg: string) => void;
}) {
  const text = currentText(state, docId);
  const [buffer, setBuffer] = useState(text);
  useEffect(() => setBuffer(text), [text]);
  return (
    <div className="block">
      <textarea
        value={buffer}
        rows={Math.min(buffer.split('\n').length + 2, 20)}
        onChange={(e) => setBuffer(e.target.value)}
        onBlur={async () => {
          if (buffer === text) return;
          try {
            await revise(ctx, { chunkId: docId, text: buffer });
          } catch (e) {
            onError(`edit: ${e instanceof Error ? e.message : String(e)}`);
            setBuffer(text);
          }
        }}
      />
    </div>
  );
}

function proposalBasisText(
  state: WorkspaceGraph,
  p: Proposal,
  chunkId: ChunkId | undefined,
  explicitRevisionIds?: string[],
): string | undefined {
  const ids = explicitRevisionIds ?? p.basisRevisionIds.filter(
    (revisionId) => state.revisions.get(revisionId)?.chunkId === chunkId,
  );
  if (ids.length === 0) return undefined;
  return ids
    .map((revisionId) => `${revisionId}\n${revisionText(state, revisionId)}`)
    .join('\n\n');
}

function summarizeChange(
  state: WorkspaceGraph,
  p: Proposal,
  ch: ProposedChange,
): { title: string; before?: string; beforeLabel?: string; after?: string; afterLabel?: string } {
  switch (ch.op) {
    case 'create':
      return { title: 'add new block', after: ch.text, afterLabel: 'proposed text' };
    case 'revise':
      return {
        title: 'revise block',
        before: proposalBasisText(state, p, ch.chunkId, ch.mergeParentRevisionIds),
        beforeLabel: 'recorded basis text',
        after: ch.text,
        afterLabel: 'proposed text',
      };
    case 'repin': {
      const occ = state.occurrences.get(ch.occurrenceId);
      const targetRevision = state.revisions.get(ch.revisionId);
      const targetChunkId = targetRevision?.chunkId ?? occ?.chunkId;
      return {
        title: 'update watched quote',
        before: proposalBasisText(state, p, targetChunkId),
        beforeLabel: 'recorded basis text',
        after: revisionText(state, ch.revisionId),
        afterLabel: 'proposed pinned text',
      };
    }
    case 'sever': {
      const occ = state.occurrences.get(ch.occurrenceId);
      const targetChunkId = occ?.chunkId ?? (p.targetChunkIds.length === 1 ? p.targetChunkIds[0] : undefined);
      return {
        title: 'remove block appearance',
        before: proposalBasisText(state, p, targetChunkId),
        beforeLabel: 'recorded basis text',
      };
    }
    case 'place':
      return { title: 'place block' };
    case 'relate':
      return { title: `relate (${ch.role})` };
  }
}

function ProposalCard({
  p,
  state,
  onAccept,
  onReject,
}: {
  p: Proposal;
  state: WorkspaceGraph;
  onAccept: () => void;
  onReject: () => void;
}) {
  const operation = state.operations.get(p.operationId)!;
  const stale = p.status === 'open' ? staleReason(state, p) : null;
  const freshness = p.status === 'open' ? (stale ? `stale · ${stale}` : 'fresh') : p.status;
  return (
    <div className={`proposal proposal-${p.status} ${stale ? 'proposal-stale' : ''}`}>
      <div className="proposal-heading">
        <span>{p.kind} · {p.createdBy}</span>
        <span className="proposal-status">{freshness}</span>
      </div>
      {p.note && <div className="proposal-note">{p.note}</div>}
      <details className="proposal-inspector" open={p.status === 'open'}>
        <summary>identity, inputs, basis, and targets</summary>
        <dl>
          <dt>proposal</dt><dd>{p.id}</dd>
          <dt>author</dt><dd>{p.createdBy}</dd>
          <dt>created</dt><dd>{p.createdAt}</dd>
          <dt>dispatcher</dt><dd>{operation.actorId}</dd>
          <dt>operation</dt><dd>{p.operationId}</dd>
          <dt>inputs</dt><dd>{operation.inputRevisionIds.join(', ') || 'none'}</dd>
          <dt>basis</dt><dd>{p.basisRevisionIds.join(', ') || 'none'}</dd>
          <dt>freshness</dt><dd>{p.freshnessRevisionIds?.join(', ') || 'not current-head-dependent'}</dd>
          <dt>structure</dt><dd>{p.freshnessStructure ? `${p.freshnessStructure.containers.length} container(s), ${p.freshnessStructure.placements.length} placement set(s)` : 'not structure-dependent'}</dd>
          <dt>targets</dt><dd>{p.targetChunkIds.join(', ') || 'none'}</dd>
          <dt>producer</dt><dd>{p.producer ? `${p.producer.id}@${p.producer.version}` : 'not recorded'}</dd>
          {p.producer?.implementation && <><dt>implementation</dt><dd>{p.producer.implementation}</dd></>}
          {p.producer?.receiptId && <><dt>provider receipt</dt><dd>{p.producer.receiptId}</dd></>}
          {p.resolution && (
            <>
              <dt>resolved</dt><dd>{p.resolution.by} · {p.resolution.at}</dd>
              <dt>resolution op</dt><dd>{p.resolution.operationId}</dd>
              {p.resolution.reason && <><dt>reason</dt><dd>{p.resolution.reason}</dd></>}
            </>
          )}
        </dl>
      </details>
      {p.payload.map((ch, i) => {
        const s = summarizeChange(state, p, ch);
        return (
          <div key={i} className="change">
            <div className="meta">{i + 1}. {s.title}</div>
            {s.before !== undefined && <><div className="meta change-label">{s.beforeLabel ?? 'before'}</div><pre className="before">{s.before}</pre></>}
            {s.after !== undefined && <><div className="meta change-label">{s.afterLabel ?? 'after'}</div><pre className="after">{s.after}</pre></>}
            <div className="meta change-label">exact proposed operation</div>
            <pre className="structure">{JSON.stringify(ch, null, 2)}</pre>
          </div>
        );
      })}
      {p.status === 'open' && (
        <div className="actions">
          <button onClick={onAccept} disabled={Boolean(stale)}>accept</button>
          <button onClick={onReject}>reject</button>
        </div>
      )}
    </div>
  );
}

function FatesPanel({
  state,
  indexes,
  chunkId,
  bindings,
  onFocusDoc,
}: {
  state: WorkspaceGraph;
  indexes: ReturnType<typeof buildIndexes>;
  chunkId: ChunkId;
  bindings: { docChunkId: string; relPath: string }[];
  onFocusDoc: (id: ChunkId) => void;
}) {
  const label = (id: ChunkId) => labelOf(state, bindings, id);
  const cameFrom = [...state.derivations.values()].filter((d) => d.childChunkId === chunkId);
  const wentTo = [...state.derivations.values()].filter((d) => state.revisions.get(d.sourceRevisionId)?.chunkId === chunkId);
  const appearances = occurrencesOfChunk(state, chunkId);
  const chunk = state.chunks.get(chunkId);
  const dupes = chunk ? duplicatesOf(state, indexes, state.revisions.get(chunk.currentRevisionId)!.blobHash).filter((c) => c !== chunkId) : [];
  const echoes = echoesOf(state, indexes, chunkId);

  return (
    <div className="fates">
      <div className="meta">deep fates — {label(chunkId)}</div>
      {cameFrom.length > 0 && (
        <div>
          came from:{' '}
          {cameFrom.map((d) => {
            const src = state.revisions.get(d.sourceRevisionId)!.chunkId;
            return (
              <button key={d.id} className="linkish" onClick={() => onFocusDoc(src)}>
                {label(src)} ({d.via})
              </button>
            );
          })}
        </div>
      )}
      {wentTo.length > 0 && (
        <div>
          became:{' '}
          {wentTo.map((d) => (
            <button key={d.id} className="linkish" onClick={() => onFocusDoc(d.childChunkId)}>
              {label(d.childChunkId)} ({d.via})
            </button>
          ))}
        </div>
      )}
      {appearances.length > 1 && (
        <div>
          appears in:{' '}
          {appearances.map((o) => (
            <button key={o.id} className="linkish" onClick={() => onFocusDoc(o.containerId)}>
              {label(o.containerId)}
              {o.mode === 'transclude' ? ' (transcluded)' : ''}
            </button>
          ))}
        </div>
      )}
      {dupes.length > 0 && (
        <div>
          identical content in:{' '}
          {dupes.map((c) => (
            <button key={c} className="linkish" onClick={() => onFocusDoc(c)}>
              {label(c)}
            </button>
          ))}
        </div>
      )}
      {echoes.length > 0 && (
        <div>
          echoes:
          {echoes.slice(0, 5).map((e, i) => (
            <div key={i} className="echo">
              “{e.text.length > 80 ? `${e.text.slice(0, 80)}…` : e.text}” also in{' '}
              {e.others.map((c) => (
                <button key={c} className="linkish" onClick={() => onFocusDoc(c)}>
                  {label(c)}
                </button>
              ))}
            </div>
          ))}
        </div>
      )}
      {cameFrom.length === 0 && wentTo.length === 0 && appearances.length <= 1 && dupes.length === 0 && echoes.length === 0 && (
        <div className="meta">no recorded fates yet — this material stands alone</div>
      )}
    </div>
  );
}

function AttachBox({
  state,
  ctx,
  indexes,
  docId,
  bindings,
  onError,
}: {
  state: WorkspaceGraph;
  ctx: TxCtx;
  indexes: ReturnType<typeof buildIndexes>;
  docId: ChunkId;
  bindings: { docChunkId: string; relPath: string }[];
  onError: (msg: string) => void;
}) {
  const [q, setQ] = useState('');
  const results = useMemo(() => (q.trim() ? searchChunks(state, indexes, q).filter((c) => c !== docId).slice(0, 6) : []), [q, state, indexes, docId]);
  return (
    <div className="attach">
      <input placeholder="attach: search a chunk to transclude (watched)…" value={q} onChange={(e) => setQ(e.target.value)} />
      {results.length > 0 && (
        <div className="attach-results">
          {results.map((c) => (
            <button
              key={c}
              className="linkish"
              onClick={() => {
                try {
                  transclude(ctx, { containerId: docId, sourceChunkId: c });
                  setQ('');
                } catch (e) {
                  onError(`transclude: ${e instanceof Error ? e.message : String(e)}`);
                }
              }}
            >
              {labelOf(state, bindings, c)}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
