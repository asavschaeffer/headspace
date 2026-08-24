// One authority for every host-owned workspace path. Headspace owns exactly
// one data directory and never probes, moves, or deletes other hidden state.

import { join } from 'node:path';

export const HEADSPACE_DATA_DIRNAME = '.headspace';

export interface WorkspaceDataPaths {
  workspaceRoot: string;
  dataDir: string;
  logPath: string;
  snapshotPath: string;
  lockPath: string;
  blobsDir: string;
  sidecarsDir: string;
  ingestionCatalogPath: string;
}

export function workspaceDataPaths(workspaceRoot: string): WorkspaceDataPaths {
  const dataDir = join(workspaceRoot, HEADSPACE_DATA_DIRNAME);

  return {
    workspaceRoot,
    dataDir,
    logPath: join(dataDir, 'log.jsonl'),
    snapshotPath: join(dataDir, 'snapshot.json'),
    lockPath: join(dataDir, 'lock'),
    blobsDir: join(dataDir, 'blobs'),
    sidecarsDir: join(dataDir, 'sidecars'),
    ingestionCatalogPath: join(dataDir, 'ingestion.json'),
  };
}
