# Legacy Static Assets

The previous dashboard/document UI, LOD tooling, and worker-based texture
pipeline now live under `legacy/static/` (ignored from version control).
Those scripts and styles are kept only for reference and are **not** shipped
with the current Headspace experience.

If you need to resurrect the old interface or reuse one of the utilities
(e.g. `lod-manager.js`, `geometry-cache.js`), copy it back into `static/` and
re-wire it explicitly.
