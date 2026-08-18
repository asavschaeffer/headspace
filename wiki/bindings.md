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

`observedVersion` is the external content hash last seen or projected. It
supports conflict detection — the driver compares it against the external
object's current state to notice divergence; it does not become chunk
identity.

## Accepted principles

### A binding targets a chunk

A binding names a continuing identity, not a historical state. Revisions come
and go beneath it; the binding records which chunk the external object
corresponds to, and projection works from that chunk's current revision.

### Cardinality is asymmetric

One chunk may carry multiple bindings — the same material exported to several
external targets. One external file binds to one doc chunk: the file's block
chunks ride the doc's sidecar rather than binding individually (see
[Drivers](drivers.md)).

### Renamed files are rediscovered by content

A dangling path is not yet broken correspondence. A renamed file is re-matched
by content hash and offered as a `reconciliation` proposal to rebind; the
rebind is an explicit acceptance, never a silent path update.

### Binding history is sedimentary

Binding changes — creation, rebind, removal — are recorded like everything
else: kept in history, out of the way, inspectable when reconciliation or
trust requires them.

## Responsibilities

- Locate an external object through the appropriate driver.
- Record the external version last observed or projected.
- Support moves and renamed external objects without redefining the chunk.
- Make conflicts and broken correspondence visible.

