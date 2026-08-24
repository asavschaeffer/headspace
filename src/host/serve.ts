// Release host: one local Node HTTP server owns both the durable substrate API
// and the built client. The API itself remains the same middleware used by
// Vite; this module only supplies its production transport and static shell.

import {
  createServer,
  type IncomingMessage,
  type Server,
  type ServerResponse,
} from 'node:http';
import { readFile, realpath, stat } from 'node:fs/promises';
import { realpathSync, statSync } from 'node:fs';
import { isIP } from 'node:net';
import { extname, isAbsolute, relative, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createSubstrateServer, type SubstrateServerOptions } from './api';

type Next = (error?: unknown) => void;
type Middleware = (req: IncomingMessage, res: ServerResponse, next: Next) => void;

export interface ReleaseServerOptions extends SubstrateServerOptions {
  /** Directory produced by `vite build`. Defaults to HEADSPACE_DIST or ./dist. */
  distDir?: string;
  /** Bind address. Headspace 0.1.0 accepts loopback addresses only. */
  host?: string;
  /** Port 0 asks the OS for an ephemeral port. Defaults to HEADSPACE_PORT or 4173. */
  port?: number;
}

export interface ReleaseServerAddress {
  host: string;
  port: number;
  url: string;
}

export interface ReleaseServer {
  readonly server: Server;
  readonly workspaceRoot: string;
  readonly distRoot: string;
  readonly address: ReleaseServerAddress | null;
  listen(): Promise<ReleaseServerAddress>;
  close(): Promise<void>;
}

const CONTENT_TYPES: Record<string, string> = {
  '.css': 'text/css; charset=utf-8',
  '.gif': 'image/gif',
  '.html': 'text/html; charset=utf-8',
  '.ico': 'image/x-icon',
  '.jpeg': 'image/jpeg',
  '.jpg': 'image/jpeg',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.txt': 'text/plain; charset=utf-8',
  '.wasm': 'application/wasm',
  '.webp': 'image/webp',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
};

function isWithin(root: string, candidate: string): boolean {
  const fromRoot = relative(root, candidate);
  return (
    fromRoot === '' ||
    (!isAbsolute(fromRoot) && fromRoot !== '..' && !fromRoot.startsWith(`..${sep}`))
  );
}

function canonicalDirectory(requested: string, label: string): string {
  let canonical: string;
  try {
    canonical = realpathSync(requested);
  } catch (error) {
    throw new Error(`${label} does not exist or cannot be resolved: ${requested}`, { cause: error });
  }
  if (!statSync(canonical).isDirectory()) {
    throw new Error(`${label} is not a directory: ${requested}`);
  }
  return canonical;
}

function requireConfinedIndex(distRoot: string): void {
  const requested = resolve(distRoot, 'index.html');
  let canonical: string;
  try {
    canonical = realpathSync(requested);
  } catch (error) {
    throw new Error(`built application entry does not exist: ${requested} (run the build first)`, {
      cause: error,
    });
  }
  if (!isWithin(distRoot, canonical) || !statSync(canonical).isFile()) {
    throw new Error(`built application entry is not a confined regular file: ${requested}`);
  }
}

function setReleaseSecurityHeaders(res: ServerResponse): void {
  res.setHeader('content-security-policy', "frame-ancestors 'none'");
  res.setHeader('referrer-policy', 'no-referrer');
  res.setHeader('x-content-type-options', 'nosniff');
  res.setHeader('x-frame-options', 'DENY');
}

function response(res: ServerResponse, status: number, message: string): void {
  if (res.writableEnded) return;
  const body = Buffer.from(message);
  res.statusCode = status;
  res.setHeader('content-type', 'text/plain; charset=utf-8');
  res.setHeader('content-length', String(body.byteLength));
  res.setHeader('x-content-type-options', 'nosniff');
  res.setHeader('cache-control', 'no-store');
  res.end(body);
}

function unbracketedHost(value: string): string {
  const normalized = value.trim().toLowerCase();
  return normalized.startsWith('[') && normalized.endsWith(']')
    ? normalized.slice(1, -1)
    : normalized;
}

function isLoopbackHost(value: string): boolean {
  const host = unbracketedHost(value);
  if (host === 'localhost' || host === '::1') return true;
  if (isIP(host) === 4) return host.split('.')[0] === '127';
  const mappedIpv4 = /^::ffff:(127(?:\.\d{1,3}){3})$/.exec(host)?.[1];
  return Boolean(mappedIpv4 && isIP(mappedIpv4) === 4);
}

function requestHostMatches(value: string | undefined, host: string, port: number): boolean {
  if (!value) return false;
  try {
    const supplied = new URL(`http://${value}`);
    const expected = new URL(`http://${urlHost(host)}:${port}`);
    if (
      supplied.username ||
      supplied.password ||
      supplied.pathname !== '/' ||
      supplied.search ||
      supplied.hash
    ) {
      return false;
    }
    const suppliedPort = supplied.port ? Number(supplied.port) : 80;
    const expectedPort = expected.port ? Number(expected.port) : 80;
    return supplied.hostname === expected.hostname && suppliedPort === expectedPort;
  } catch {
    return false;
  }
}

function apiRequestHasForeignOrigin(
  req: IncomingMessage,
  address: ReleaseServerAddress,
): boolean {
  const path = (req.url ?? '/').split('?', 1)[0];
  if (path !== '/api' && !path.startsWith('/api/')) return false;
  if (req.headers['sec-fetch-site'] === 'cross-site') return true;
  const origin = req.headers.origin;
  if (origin === undefined) return false; // Native/CLI clients do not send Origin.
  try {
    return new URL(origin).origin !== new URL(address.url).origin;
  } catch {
    return true;
  }
}

async function confinedRegularFile(root: string, candidate: string): Promise<string | null> {
  if (!isWithin(root, candidate)) return null;
  try {
    // Canonicalizing the existing target closes the symlink escape that a
    // lexical `resolve`/`relative` check alone would leave open.
    const canonical = await realpath(candidate);
    if (!isWithin(root, canonical)) return null;
    return (await stat(canonical)).isFile() ? canonical : null;
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === 'ENOENT' || code === 'ENOTDIR' || code === 'EACCES') return null;
    throw error;
  }
}

async function serveStatic(
  req: IncomingMessage,
  res: ServerResponse,
  distRoot: string,
): Promise<void> {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    res.setHeader('allow', 'GET, HEAD');
    response(res, 405, 'method not allowed');
    return;
  }

  const rawPath = (req.url ?? '/').split('?', 1)[0] || '/';
  let pathname: string;
  try {
    pathname = decodeURIComponent(rawPath);
  } catch {
    response(res, 400, 'invalid request path');
    return;
  }
  if (pathname.includes('\0')) {
    response(res, 400, 'invalid request path');
    return;
  }

  // Treat both slash styles as separators. That matters on Windows, where an
  // encoded backslash can otherwise turn into traversal after decoding.
  const pathSegments = pathname.split(/[\\/]+/);
  if (pathSegments.includes('..')) {
    response(res, 403, 'path is outside the built application');
    return;
  }
  const requested = pathname.replace(/^[\\/]+/, '') || 'index.html';
  const lexicalCandidate = resolve(distRoot, requested);
  if (!isWithin(distRoot, lexicalCandidate)) {
    response(res, 403, 'path is outside the built application');
    return;
  }

  let file = await confinedRegularFile(distRoot, lexicalCandidate);
  // Client-side routes get the application shell. Missing asset requests keep
  // their 404 so an absent bundle cannot be mistaken for HTML.
  if (!file && extname(pathname) === '') {
    file = await confinedRegularFile(distRoot, resolve(distRoot, 'index.html'));
  }
  if (!file) {
    response(res, 404, 'not found');
    return;
  }

  const body = await readFile(file);
  res.statusCode = 200;
  res.setHeader('content-type', CONTENT_TYPES[extname(file).toLowerCase()] ?? 'application/octet-stream');
  res.setHeader('content-length', String(body.byteLength));
  res.setHeader('x-content-type-options', 'nosniff');
  res.setHeader('cache-control', 'no-cache');
  res.end(req.method === 'HEAD' ? undefined : body);
}

function configuredPort(value: number | undefined): number {
  if (value !== undefined) {
    if (!Number.isInteger(value) || value < 0 || value > 65_535) {
      throw new Error(`invalid release server port: ${value}`);
    }
    return value;
  }
  const raw = process.env.HEADSPACE_PORT?.trim();
  if (!raw) return 4173;
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed < 0 || parsed > 65_535) {
    throw new Error(`HEADSPACE_PORT must be an integer between 0 and 65535, received ${raw}`);
  }
  return parsed;
}

function urlHost(host: string): string {
  return host.includes(':') && !host.startsWith('[') ? `[${host}]` : host;
}

/**
 * Create, but do not bind, the release server. Tests can choose port 0 and
 * production can inspect configuration before opening a socket.
 */
export function createReleaseServer(options: ReleaseServerOptions = {}): ReleaseServer {
  const configuredWorkspace = process.env.HEADSPACE_WORKSPACE?.trim();
  const requestedWorkspace = resolve(options.root ?? configuredWorkspace ?? process.cwd());
  const workspaceRoot = canonicalDirectory(requestedWorkspace, 'workspace root');
  const configuredDist = process.env.HEADSPACE_DIST?.trim();
  const requestedDist = resolve(options.distDir ?? configuredDist ?? resolve(process.cwd(), 'dist'));
  const distRoot = canonicalDirectory(requestedDist, 'built application directory');
  requireConfinedIndex(distRoot);

  const requestedHost = options.host?.trim() || process.env.HEADSPACE_HOST?.trim() || '127.0.0.1';
  const host = unbracketedHost(requestedHost);
  if (!isLoopbackHost(host)) {
    throw new Error(
      `Headspace 0.1.0 is loopback-only; refusing to bind ${requestedHost}. ` +
      'Use 127.0.0.1, localhost, or ::1.',
    );
  }
  const port = configuredPort(options.port);
  const middleware: Middleware[] = [];
  let runningAddress: ReleaseServerAddress | null = null;
  let closePromise: Promise<void> | null = null;
  let listenPromise: Promise<ReleaseServerAddress> | null = null;
  let terminal = false;

  const server = createServer((req, res) => {
    setReleaseSecurityHeaders(res);
    const address = runningAddress;
    if (!address) {
      response(res, 503, 'server is not ready');
      return;
    }
    if (!requestHostMatches(req.headers.host, address.host, address.port)) {
      response(res, 421, 'request authority is not served here');
      return;
    }
    if (apiRequestHasForeignOrigin(req, address)) {
      response(res, 403, 'cross-origin API requests are not allowed');
      return;
    }
    let cursor = 0;
    const next: Next = (error) => {
      if (error) {
        response(res, 500, 'request failed');
        return;
      }
      const handler = middleware[cursor++];
      if (handler) {
        try {
          handler(req, res, next);
        } catch {
          response(res, 500, 'request failed');
        }
        return;
      }
      void serveStatic(req, res, distRoot).catch(() => response(res, 500, 'request failed'));
    };
    next();
  });
  server.on('close', () => {
    runningAddress = null;
    terminal = true;
  });

  const usesDefaultScan = options.contentDirs === undefined && options.contentFiles === undefined;
  const api = createSubstrateServer({
    root: workspaceRoot,
    contentDirs: usesDefaultScan ? ['.'] : options.contentDirs,
    contentFiles: options.contentFiles,
    collaborators: options.collaborators,
  });
  const configure = api.plugin.configureServer;
  if (typeof configure !== 'function') {
    throw new Error('substrate API did not expose a server middleware hook');
  }
  (configure as unknown as (server: {
    httpServer: Server;
    middlewares: { use(handler: Middleware): void };
  }) => unknown)({
    httpServer: server,
    middlewares: { use: (handler) => middleware.push(handler) },
  });

  const runtime: ReleaseServer = {
    server,
    workspaceRoot,
    distRoot,
    get address() {
      return runningAddress;
    },
    listen: () => {
      if (terminal) return Promise.reject(new Error('release server is closed and cannot listen again'));
      if (runningAddress) return Promise.resolve(runningAddress);
      if (listenPromise) return listenPromise;

      const pending = (async (): Promise<ReleaseServerAddress> => {
        try {
          // Readiness includes opening, locking, replaying, and initially
          // ingesting the workspace. A broken workspace never gets a socket.
          await api.ready();
          if (terminal) throw new Error('release server closed while starting');
          await new Promise<void>((resolveListen, rejectListen) => {
            const onError = (error: Error): void => {
              server.off('listening', onListening);
              rejectListen(error);
            };
            const onListening = (): void => {
              server.off('error', onError);
              resolveListen();
            };
            server.once('error', onError);
            server.once('listening', onListening);
            server.listen(port, host);
          });
          const bound = server.address();
          if (!bound || typeof bound === 'string') {
            throw new Error('release server did not expose a TCP address');
          }
          if (!isLoopbackHost(bound.address)) {
            throw new Error(`Headspace 0.1.0 resolved ${host} to non-loopback address ${bound.address}`);
          }
          if (terminal) throw new Error('release server closed while starting');
          runningAddress = {
            host,
            port: bound.port,
            url: `http://${urlHost(host)}:${bound.port}`,
          };
          return runningAddress;
        } catch (error) {
          terminal = true;
          runningAddress = null;
          try {
            if (server.listening) {
              await new Promise<void>((resolveClose, rejectClose) => {
                server.close((closeError) => {
                  if (closeError) rejectClose(closeError);
                  else resolveClose();
                });
                server.closeIdleConnections?.();
              });
            } else {
              api.close();
            }
          } catch (cleanupError) {
            throw new AggregateError(
              [error, cleanupError],
              'release server failed to start and could not close cleanly',
            );
          }
          throw error;
        }
      })();
      listenPromise = pending;
      void pending.then(
        () => {
          if (listenPromise === pending) listenPromise = null;
        },
        () => {
          if (listenPromise === pending) listenPromise = null;
        },
      );
      return pending;
    },
    close: () => {
      if (closePromise) return closePromise;
      terminal = true;
      closePromise = (async (): Promise<void> => {
        await listenPromise?.catch(() => undefined);
        if (!server.listening) {
          api.close();
          runningAddress = null;
          return;
        }
        await new Promise<void>((resolveClose, rejectClose) => {
          server.close((error) => {
            if (error) rejectClose(error);
            else resolveClose();
          });
          server.closeIdleConnections?.();
        });
        runningAddress = null;
      })();
      return closePromise;
    },
  };
  return runtime;
}

/** Create and bind a release server. */
export async function startReleaseServer(options: ReleaseServerOptions = {}): Promise<ReleaseServer> {
  const runtime = createReleaseServer(options);
  await runtime.listen();
  return runtime;
}

function isDirectExecution(): boolean {
  const entry = process.argv[1];
  return Boolean(entry) && pathToFileURL(resolve(entry)).href === import.meta.url;
}

if (isDirectExecution()) {
  try {
    const runtime = await startReleaseServer();
    process.stdout.write(
      `Headspace ${runtime.address?.url} — workspace ${runtime.workspaceRoot}\n`,
    );
    let stopping = false;
    const stop = (): void => {
      if (stopping) return;
      stopping = true;
      void runtime.close().catch((error) => {
        process.exitCode = 1;
        process.stderr.write(`Headspace shutdown failed: ${String(error)}\n`);
      });
    };
    process.once('SIGINT', stop);
    process.once('SIGTERM', stop);
  } catch (error) {
    process.exitCode = 1;
    process.stderr.write(`Headspace failed to start: ${String(error)}\n`);
  }
}
