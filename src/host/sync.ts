// Sync orchestration over the observable ingestion seam. New sources
// are adapted, known native sources reconcile, and watched sources raise
// update proposals. The report pairs summary metrics with the complete
// per-item ingestion result.

import { scanWatchedSources } from '../kernel/tx';
import { ingestWorkspace, type IngestionRunReport } from './ingestion';
import type { WorkspaceStore } from './store-fs';

export interface SyncReport {
  imported: string[];
  fastForwarded: string[];
  proposals: string[]; // proposal ids raised by reconciliation
  sourceUpdates: string[]; // proposal ids raised by watched-source scanning
  unchanged: number;
  ingestion: IngestionRunReport;
}

export async function syncWorkspace(
  ws: WorkspaceStore,
  opts: { contentDirs?: string[]; contentFiles?: string[] },
): Promise<SyncReport> {
  const ingestion = await ingestWorkspace(ws, opts);
  const representedFiles = ingestion.items.filter(
    (item) => item.observation.kind === 'file' && item.adapter !== null,
  );
  const report: SyncReport = {
    imported: representedFiles.filter((item) => item.status === 'imported').map((item) => item.observation.relPath),
    fastForwarded: representedFiles.filter((item) => item.status === 'updated').map((item) => item.observation.relPath),
    proposals: representedFiles.flatMap((item) => (item.proposalId ? [item.proposalId] : [])),
    sourceUpdates: scanWatchedSources(ws.ctxFor('adapter:filesystem')),
    unchanged: representedFiles.filter((item) => item.status === 'unchanged').length,
    ingestion,
  };
  ws.saveSnapshot();
  return report;
}
