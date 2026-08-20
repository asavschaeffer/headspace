import { useEffect, useMemo, useRef, useState } from 'react';
import type { SubstrateHook } from './App';
import { currentText, focusTarget, labelOf, leafBlocks, proposalsForDoc, revisionCount, type LeafBlock } from './client/helpers';
import { buildIndexes, duplicatesOf, echoesOf, searchChunks } from './index/indexes';
import { generateProposal } from './kernel/select';
import { isComposite, occurrencesOfChunk, revisionText, type SubstrateState } from './kernel/state';
import {
  acceptProposal,
  moveOccurrence,
  promoteCopy,
  promoteExtract,
  promoteSpanAnchor,
  rejectProposal,
  revise,
  severOccurrence,
  transclude,
  type TxCtx,
} from './kernel/tx';
import { METHOD_RAW } from './kernel/decompose';
import type { ChunkId, Proposal, ProposedChange, SpanAddress } from './kernel/types';

type Span = SpanAddress & { chunkId: ChunkId };

// The focused work surface: compose, promote, dispatch, integrate.
export function Star({
  sub,
  docId,
  onFocusDoc,
  onBack,
}: {
  sub: SubstrateHook;
  docId: ChunkId;
  onFocusDoc: (id: ChunkId) => void;
  onBack: () => void;
}) {
  const { state, bindings } = sub.ws!;
  const ctx = sub.ctx!;
  const [instruction, setInstruction] = useState('');
  const [notice, setNotice] = useState<string | null>(null);
  const [span, setSpan] = useState<Span | null>(null);
  const [fatesFor, setFatesFor] = useState<ChunkId | null>(null);
  const noticeTimer = useRef<number | null>(null);

  // Selection and fates belong to the doc they were made in.
  useEffect(() => {
    setSpan(null);
    setFatesFor(null);
    setInstruction('');
  }, [docId]);

  const blocks = useMemo(() => leafBlocks(state, docId), [docId, sub.version]); // eslint-disable-line react-hooks/exhaustive-deps
  const proposals = useMemo(() => proposalsForDoc(state, docId), [docId, sub.version]); // eslint-disable-line react-hooks/exhaustive-deps
  const indexes = useMemo(() => buildIndexes(state), [sub.version]); // eslint-disable-line react-hooks/exhaustive-deps
  const binding = bindings.find((b) => b.docChunkId === docId);

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
      flash(`${label}: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const dispatch = () =>
    guard('dispatch', async () => {
      const hits = instruction.trim() ? searchChunks(state, indexes, instruction).slice(0, 5) : [];
      await generateProposal(ctx, { focusChunkId: docId, instruction: instruction.trim() || 'continue this', searchHits: hits });
      setInstruction('');
    });

  const onAccept = (p: Proposal) =>
    guard('accept', async () => {
      const r = await acceptProposal(ctx, { proposalId: p.id });
      if (!r.applied) flash(`proposal superseded — ${r.reason}`);
    });

  const projectToFile = () =>
    guard('project', async () => {
      if (!binding) return;
      const r = await fetch('/api/project', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ relPath: binding.relPath }),
      });
      if (!r.ok) throw new Error(`server refused: ${r.status}`);
      flash(`projected to ${binding.relPath}`);
    });

  return (
    <div className="doc">
      <header>
        <button onClick={onBack}>← nebula</button>
        <h2>{labelOf(state, bindings, docId)}</h2>
        <span className="meta">
          v{revisionCount(state, docId)} · {blocks.length} blocks
        </span>
        {binding && <button onClick={projectToFile}>project → file</button>}
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

      <div className="dispatch">
        <input
          placeholder="instruct an agent…"
          value={instruction}
          onChange={(e) => setInstruction(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && dispatch()}
        />
        <button onClick={dispatch}>dispatch</button>
      </div>

      <AttachBox state={state} ctx={ctx} indexes={indexes} docId={docId} bindings={bindings} onError={flash} />

      {proposals.length > 0 && (
        <div className="proposals">
          <div className="meta">{proposals.length} open proposal(s)</div>
          {proposals.map((p) => (
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
  state: SubstrateState;
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
  // Swapping with a neighbor: push the previous sibling after me (up), or push
  // myself after the next sibling (down). Arrangement only — never a revision.
  const moveUp = () => moveOccurrence(ctx, { occurrenceId: prevSibling!.occurrence.id, at: { after: block.occurrence.id } });
  const moveDown = () => moveOccurrence(ctx, { occurrenceId: block.occurrence.id, at: { after: nextSibling!.occurrence.id } });

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
          v{revisionCount(state, block.chunkId)} · {block.revision.createdBy}
          {block.transcluded && (block.occurrence.watch ? ' · watched' : ' · pinned')}
        </span>
        <button title="deep fates" onClick={onToggleFates} className={fatesOpen ? 'active' : ''}>
          ☄
        </button>
        {block.transcluded && (
          <button
            title="sever this transclusion"
            onClick={() => severOccurrence(ctx, { occurrenceId: block.occurrence.id })}
          >
            ✂
          </button>
        )}
        <button disabled={!sameContainer(prevSibling)} onClick={() => moveUp()} title="move up">
          ↑
        </button>
        <button disabled={!sameContainer(nextSibling)} onClick={() => moveDown()} title="move down">
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
  state: SubstrateState;
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

function summarizeChange(state: SubstrateState, ch: ProposedChange): { title: string; before?: string; after?: string } {
  switch (ch.op) {
    case 'create':
      return { title: 'add new block', after: ch.text };
    case 'revise':
      return { title: 'revise block', before: currentText(state, ch.chunkId), after: ch.text };
    case 'repin': {
      const occ = state.occurrences.get(ch.occurrenceId);
      const before = occ && occ.pin !== 'current' ? revisionText(state, occ.pin) : undefined;
      return { title: 'update watched quote', before, after: revisionText(state, ch.revisionId) };
    }
    case 'sever': {
      const occ = state.occurrences.get(ch.occurrenceId);
      const text = occ ? currentText(state, occ.chunkId) : '(already gone)';
      return { title: 'remove block appearance', before: text };
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
  state: SubstrateState;
  onAccept: () => void;
  onReject: () => void;
}) {
  return (
    <div className="proposal">
      <span className="meta">
        {p.kind} · {p.createdBy}
        {p.note ? ` · ${p.note}` : ''}
      </span>
      {p.payload.map((ch, i) => {
        const s = summarizeChange(state, ch);
        if (!s.before && !s.after) return null;
        return (
          <div key={i} className="change">
            <div className="meta">{s.title}</div>
            {s.before !== undefined && <pre className="before">{s.before}</pre>}
            {s.after !== undefined && <pre className="after">{s.after}</pre>}
          </div>
        );
      })}
      <div className="actions">
        <button onClick={onAccept}>accept</button>
        <button onClick={onReject}>reject</button>
      </div>
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
  state: SubstrateState;
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
  state: SubstrateState;
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
