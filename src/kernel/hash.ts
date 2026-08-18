import type { Blob } from './types';

// Content addressing: SHA-256 over UTF-8 bytes of (mediaType + "\0" + text).
// WebCrypto is available in both the browser and Node >= 18, so the kernel
// stays dependency-free at the cost of hashing being async.

export async function sha256Hex(input: string): Promise<string> {
  const bytes = new TextEncoder().encode(input);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

export async function blobHashOf(mediaType: string, text: string): Promise<string> {
  return sha256Hex(`${mediaType}\0${text}`);
}

export async function makeBlob(mediaType: string, text: string): Promise<Blob> {
  return { hash: await blobHashOf(mediaType, text), mediaType, text };
}
