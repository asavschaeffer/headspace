import type { ChunkId } from '../kernel/types';
import type { IngestionStatus, SourceKind } from '../host/ingestion';
import type { BindingInfo, SourceItemView } from './useSubstrate';

export const WORKSPACE_ROOT = '@workspace';
export type WorkspaceContainerId = typeof WORKSPACE_ROOT | string;

export interface WorkspaceCrumb {
  containerId: WorkspaceContainerId;
  label: string;
}

export interface WorkspaceSourceNode {
  key: string;
  sourceId: string;
  kind: SourceKind;
  path: string;
  label: string;
  status: IngestionStatus | 'observed' | 'missing';
  mediaType: string;
  sizeBytes: number;
  symlinkStatus: SourceItemView['observation']['symlink']['status'];
  adapterLabel?: string;
  docChunkId?: ChunkId;
  diagnostics: string[];
}

export function workspaceChildren(
  sources: SourceItemView[],
  containerId: WorkspaceContainerId,
): WorkspaceSourceNode[] {
  const parentSourceId = containerId === WORKSPACE_ROOT ? null : containerId;
  const nodes: WorkspaceSourceNode[] = [];
  for (const item of sources) {
    // `.` is the durable observation of the configured root. The UI already
    // has a virtual workspace root, so rendering it would create a loop.
    if (item.isWorkspaceRoot || item.parentSourceId !== parentSourceId) continue;
    const status = item.presence === 'missing' ? 'missing' : (item.lastResult?.status ?? 'observed');
    const adapter = item.lastResult?.adapter ?? item.representation?.adapter ?? null;
    const durableDiagnostics = item.representation?.warnings.map((entry) => entry.message) ?? [];
    const runDiagnostics = item.lastResult?.diagnostics.map((entry) => entry.message) ?? [];
    const diagnostics = [...new Set([...durableDiagnostics, ...runDiagnostics])];
    nodes.push({
      key: item.source.id,
      sourceId: item.source.id,
      kind: item.observation.kind,
      path: item.observation.relPath,
      label: item.name,
      status,
      mediaType: item.observation.mediaType,
      sizeBytes: item.observation.sizeBytes,
      symlinkStatus: item.observation.symlink.status,
      adapterLabel: adapter
        ? `${adapter.id}@${adapter.version}` +
          (adapter.provider
            ? ` via ${adapter.provider.identity}@${adapter.provider.implementationVersion}`
            : '')
        : undefined,
      docChunkId: item.representation?.rootChunkId,
      diagnostics:
        item.presence === 'missing'
          ? [`${item.observation.relPath} was not present in the latest ingestion run`, ...durableDiagnostics]
          : diagnostics,
    });
  }
  return nodes.sort(
    (a, b) =>
      Number(b.kind === 'directory') - Number(a.kind === 'directory') ||
      a.label.localeCompare(b.label) ||
      a.path.localeCompare(b.path),
  );
}

export function ancestorDirectoryIds(
  sources: SourceItemView[],
  documentChunkIds: ReadonlySet<ChunkId>,
): Set<string> {
  const lit = new Set<string>();
  const byId = new Map(sources.map((source) => [source.source.id, source]));
  for (const source of sources) {
    if (!source.representation || !documentChunkIds.has(source.representation.rootChunkId)) continue;
    let cursor = source.parentSourceId;
    const seen = new Set<string>();
    while (cursor && !seen.has(cursor)) {
      seen.add(cursor);
      lit.add(cursor);
      cursor = byId.get(cursor)?.parentSourceId ?? null;
    }
  }
  return lit;
}

export function containerExists(sources: SourceItemView[], containerId: WorkspaceContainerId): boolean {
  return (
    containerId === WORKSPACE_ROOT ||
    sources.some((item) => item.source.id === containerId && item.observation.kind === 'directory')
  );
}

export function parentContainer(
  sources: SourceItemView[],
  containerId: WorkspaceContainerId,
): WorkspaceContainerId {
  if (containerId === WORKSPACE_ROOT) return WORKSPACE_ROOT;
  return sources.find((item) => item.source.id === containerId)?.parentSourceId ?? WORKSPACE_ROOT;
}

export function containerLabel(
  sources: SourceItemView[],
  containerId: WorkspaceContainerId,
  workspaceLabel = 'workspace',
): string {
  if (containerId === WORKSPACE_ROOT) return workspaceLabel;
  return sources.find((item) => item.source.id === containerId)?.name ?? workspaceLabel;
}

export function workspaceCrumbs(
  sources: SourceItemView[],
  containerId: WorkspaceContainerId,
  workspaceLabel: string,
): WorkspaceCrumb[] {
  const crumbs: WorkspaceCrumb[] = [{ containerId: WORKSPACE_ROOT, label: workspaceLabel }];
  if (containerId === WORKSPACE_ROOT) return crumbs;
  const chain: SourceItemView[] = [];
  const seen = new Set<string>();
  let cursor: string | null = containerId;
  while (cursor && !seen.has(cursor)) {
    seen.add(cursor);
    const item = sources.find((candidate) => candidate.source.id === cursor);
    if (!item || item.isWorkspaceRoot) break;
    chain.unshift(item);
    cursor = item.parentSourceId;
  }
  for (const item of chain) crumbs.push({ containerId: item.source.id, label: item.name });
  return crumbs;
}

export function containerForDocument(
  sources: SourceItemView[],
  bindings: BindingInfo[],
  docChunkId: ChunkId,
): WorkspaceContainerId | null {
  const source = sources.find((item) => item.representation?.rootChunkId === docChunkId);
  if (source) return source.parentSourceId ?? WORKSPACE_ROOT;
  return bindings.some((binding) => binding.docChunkId === docChunkId) ? WORKSPACE_ROOT : null;
}
