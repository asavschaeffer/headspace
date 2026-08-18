import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { substrateServer } from './src/host/api';

// The dev server hosts the workspace: the repo's own wiki is the demo corpus.
export default defineConfig({
  plugins: [react(), substrateServer({ contentDirs: ['wiki'], contentFiles: ['headspace-brief.md'] })],
});
