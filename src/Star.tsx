import { useEffect, useRef, useState } from 'react';
import { fork, reorder, updateText, type Chunk } from './kernel/chunk';
import { generate, reduce, select } from './kernel/ops';
import type { Store } from './kernel/store';

// The focused work surface: compose, dispatch, integrate.
// STUB: reveal-adjacent chunks, attach/connect material, multi-target
// context selection UI — select() currently takes the whole neighborhood.
export function Star({
  store,
  docId,
  commit,
  onFocusDoc,
  onBack,
}: {
  store: Store;
  docId: string;
  commit: (...cs: Chunk[]) => void;
  onFocusDoc: (id: string) => void;
  onBack: () => void;
}) {
  const doc = store.get(docId)!;
  const blocks = doc.children.flatMap((id) => store.get(id) ?? []);
  const [instruction, setInstruction] = useState('');
  const [proposal, setProposal] = useState<Chunk | null>(null);
  const [context, setContext] = useState<{ count: number; text: string } | null>(null);
  const proposalRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    proposalRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [proposal]);

  // rows is only the fallback where CSS field-sizing is unsupported
  const rows = (text: string) =>
    Math.min(text.split('\n').reduce((n, line) => n + 1 + Math.floor(line.length / 75), 0), 14);

  const dispatch = () => {
    const selection = select(store, docId);
    const reduced = reduce(selection);
    setContext({ count: selection.length, text: reduced });
    setProposal(generate(reduced, instruction || 'continue this'));
  };

  const accept = () => {
    if (!proposal) return;
    commit({ ...proposal, parent: doc.id }, { ...doc, children: [...doc.children, proposal.id] });
    setProposal(null);
  };

  const branch = () => {
    if (!proposal) return;
    const tag = `~branch-${Date.now().toString(36)}`;
    const forkedBlocks = blocks.map((b) => ({ ...fork(b, b.id + tag, 'human'), parent: doc.id + tag }));
    const prop = { ...proposal, parent: doc.id + tag };
    const forkedDoc = {
      ...fork(doc, doc.id + tag, 'human'),
      children: [...forkedBlocks.map((b) => b.id), prop.id],
    };
    commit(...forkedBlocks, prop, forkedDoc);
    setProposal(null);
    onFocusDoc(forkedDoc.id);
  };

  return (
    <div className="doc">
      <header>
        <button onClick={onBack}>← nebula</button>
        <h2>{doc.id}</h2>
        <span className="meta">
          v{doc.version} · {doc.provenance.author}
        </span>
      </header>

      {blocks.map((b, i) => (
        <div className="block" key={b.id}>
          <textarea value={b.text} rows={rows(b.text)} onChange={(e) => commit(updateText(b, e.target.value, 'human'))} />
          <div className="block-side">
            <span className="meta">
              v{b.version} · {b.provenance.author}
            </span>
            <button disabled={i === 0} onClick={() => commit(reorder(doc, i, i - 1))}>
              ↑
            </button>
            <button disabled={i === blocks.length - 1} onClick={() => commit(reorder(doc, i, i + 1))}>
              ↓
            </button>
          </div>
        </div>
      ))}

      <div className="dispatch">
        <input
          placeholder="instruct an agent…"
          value={instruction}
          onChange={(e) => setInstruction(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && dispatch()}
        />
        <button onClick={dispatch}>dispatch</button>
      </div>

      {context && (
        <details className="context">
          <summary>
            context: {context.count} chunks selected → {context.text.length} chars reduced
          </summary>
          <pre>{context.text}</pre>
        </details>
      )}

      {proposal && (
        <div className="proposal" ref={proposalRef}>
          <span className="meta">
            proposal · v{proposal.version} · {proposal.provenance.author}
          </span>
          <textarea
            value={proposal.text}
            rows={rows(proposal.text)}
            onChange={(e) => setProposal(updateText(proposal, e.target.value, 'human'))}
          />
          <div className="actions">
            <button onClick={accept}>accept</button>
            <button onClick={branch}>branch</button>
            <button onClick={() => setProposal(null)}>discard</button>
          </div>
        </div>
      )}
    </div>
  );
}
