# Bindings

## Purpose

A binding records correspondence between a Substrate object and something
outside Substrate. It is the durable answer to: **what external thing does
this chunk represent or affect?**

A filesystem path alone is not stable enough to be identity. Files can move,
paths can be reused, and one file may correspond to several chunks.

## Working model

```ts
interface Binding {
  id: BindingId;
  chunkId: ChunkId;
  driver: DriverId;
  locator: ExternalLocator;
  observedVersion?: ExternalVersion;
}
```

`ExternalVersion` might be a content hash, file metadata, HTTP entity tag, or
provider revision. It supports conflict detection; it does not become chunk
identity.

## Responsibilities

- Locate an external object through the appropriate driver.
- Record the external version last observed or projected.
- Support moves and renamed external objects without redefining the chunk.
- Make conflicts and broken correspondence visible.

## Open questions

- Can one chunk have multiple bindings?
- Can one external object bind to multiple chunks?
- Does a binding target a chunk, a revision, or both?
- How are renamed files rediscovered reliably?
- Is binding history sedimentary in the same way as revision history?

