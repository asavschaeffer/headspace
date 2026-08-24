import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { resolve } from 'node:path';
import { headspaceHostPlugin } from './src/host/api';

// One explicit environment seam selects the active local workspace. With no
// setting, the repository's design corpus remains the zero-setup demo.
const configuredWorkspace = process.env.HEADSPACE_WORKSPACE?.trim();
const workspaceRoot = configuredWorkspace ? resolve(configuredWorkspace) : process.cwd();

export default defineConfig({
  plugins: [
    react(),
    headspaceHostPlugin(
      configuredWorkspace
        ? { root: workspaceRoot, contentDirs: ['.'] }
        : { root: workspaceRoot, contentDirs: ['wiki'], contentFiles: ['headspace-brief.md'] },
    ),
  ],
});
