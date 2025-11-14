# 🌌 Headspace Multiplayer Architecture

## Vision

A **shared 3D cosmos** where multiple users explore a single universe of interconnected diary entries. When user A writes a thought, all connected users watch it materialize in real-time in their cosmos view.

---

## Core Features

### 1. **Real-Time Presence** (Who's in the cosmos?)
- Show avatar/cursor for each connected user
- Display username or initial
- Show which entry they're currently viewing/writing
- Auto-hide after inactivity

### 2. **Live Entry Materialization** (Watch others write)
- When user A writes entry → all users see placeholder sphere
- As embeddings arrive → all users see shape morphing live
- Entry floats to position → all users see animation
- Optional: notification "User X just created an entry"

### 3. **User Colors & Identity**
- Each user has assigned color
- Their entries glow with their color
- Avatar/cursor uses their color
- Helps distinguish in crowded cosmos

### 4. **Shared Semantic Space**
- All users see same 3D positioning
- UMAP calculated per-document or global?
- Entries positioned by semantic similarity (all users see same positions)
- Clustering/nebulae visible to all

### 5. **User Authentication**
- Simple email/password or OAuth
- User ID in JWT token
- Track who created which entry
- Show author name/avatar on hover

---

## Technical Architecture

### Backend Changes

#### Database Schema Updates
```sql
-- Users table
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email VARCHAR UNIQUE NOT NULL,
  username VARCHAR NOT NULL,
  created_at TIMESTAMP DEFAULT NOW(),
  avatar_color VARCHAR DEFAULT '#667eea'
);

-- Update documents to track author
ALTER TABLE documents ADD COLUMN user_id UUID REFERENCES users(id);

-- Presence tracking
CREATE TABLE user_presence (
  user_id UUID PRIMARY KEY REFERENCES users(id),
  cosmos_connected BOOLEAN DEFAULT false,
  last_seen TIMESTAMP DEFAULT NOW(),
  current_chunk_id VARCHAR,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Entry notifications (optional)
CREATE TABLE entry_notifications (
  id SERIAL PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  entry_id VARCHAR,
  event_type VARCHAR, -- 'created', 'enriching', 'complete'
  created_at TIMESTAMP DEFAULT NOW()
);
```

#### New WebSocket Endpoints

```python
# Global enrichment broadcast
@router.websocket("/ws/cosmos")
async def cosmos_broadcast(websocket: WebSocket, user_id: str):
    """
    Main cosmos WebSocket - broadcasts:
    - New entries being created
    - Enrichment progress (chunk by chunk)
    - User presence/cursors
    - Chat (optional)
    """
    await websocket.accept()
    # Add user to broadcast manager

    try:
        while True:
            # Listen for user actions
            data = await websocket.receive_json()

            # Broadcast to all other users
            await broadcast_manager.broadcast(data)
    except:
        await broadcast_manager.disconnect(user_id)
```

#### Broadcast Manager
```python
# New service class
class CosmosEventBroadcaster:
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.user_cursors: Dict[str, dict] = {}

    async def connect(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        self.connections[user_id] = websocket
        await self.broadcast_presence()

    async def broadcast(self, event: dict):
        """Broadcast event to all connected users"""
        for user_id, ws in self.connections.items():
            try:
                await ws.send_json(event)
            except:
                pass

    async def broadcast_enrichment(self, doc_id: str, event: dict):
        """Broadcast enrichment event: chunk materialized"""
        await self.broadcast({
            "type": "chunk_enriched",
            "doc_id": doc_id,
            "event": event
        })
```

#### Enrichment Changes
Modify `enrich_document_background` to emit to broadcaster:

```python
async def enrich_document_background(processor, doc_id: str, user_id: str):
    # Emit to all users
    await enrichment_event_bus.emit(EnrichmentEvent(...))

    # Also notify broadcaster
    await cosmos_event_broadcaster.broadcast_enrichment(doc_id, event)
```

### Frontend Changes

#### New WebSocket Manager
```javascript
// cosmos-websocket.js
class CosmosWebSocket {
    connect(userId) {
        this.ws = new WebSocket(`ws://api/ws/cosmos/${userId}`);
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleCosmosEvent(data);
        };
    }

    handleCosmosEvent(data) {
        const { type, user_id, event } = data;

        switch(type) {
            case 'user_joined':
                this.renderUserCursor(user_id);
                break;
            case 'user_left':
                this.removeUserCursor(user_id);
                break;
            case 'entry_created':
                // Create placeholder for other user's entry
                this.createRemoteEntryPlaceholder(event);
                break;
            case 'chunk_enriched':
                // Update remote entry as chunks arrive
                this.updateRemoteEntryChunk(event);
                break;
            case 'entry_positioned':
                // Animate entry to final position
                this.animateEntryToPosition(event);
                break;
            case 'cursor_moved':
                // Update other user's cursor position
                this.updateUserCursor(user_id, event.position);
                break;
        }
    }
}
```

#### User Cursor/Avatar System
```javascript
class UserCursor {
    constructor(userId, color) {
        this.userId = userId;
        this.mesh = createCursorMesh(color);
        this.label = createUsernameLabel(userId);
    }

    updatePosition(pos) {
        // Show user's cursor position
        // Following their mouse or current selection
    }
}
```

#### Remote Entry Rendering
```javascript
function createRemoteEntryPlaceholder(event) {
    // Create placeholder for entry created by other user
    const { entry_id, user_id, user_color } = event;

    // Create sphere at random position near home
    const mesh = createPlaceholderGeometry(...);
    mesh.material.color.set(user_color);

    // Store reference
    remoteEntries.set(entry_id, {
        mesh,
        user_id,
        chunks: []
    });

    scene.add(mesh);
}

async function updateRemoteEntryChunk(event) {
    // As enrichment events arrive, morph remote user's planet
    const { entry_id, chunk_index, embedding, color } = event;
    const entry = remoteEntries.get(entry_id);

    if (entry) {
        // Animate their planet too
        const animator = new ShapeMorphingAnimator(entry.mesh, embedding);
        await animator.start();
    }
}
```

#### Shared Positioning
```javascript
// Both local and remote entries positioned using same algorithm
// This ensures everyone sees same 3D layout

function positionAllEntries() {
    const allEntries = [
        ...localEntries.values(),
        ...remoteEntries.values()
    ];

    // Calculate positions for all (local + remote)
    allEntries.forEach(entry => {
        const finalPos = calculatePositionFromNeighbors(
            entry.embedding,
            allEntries.filter(e => e.id !== entry.id)
        );
        animateToPosition(entry.mesh, finalPos);
    });
}
```

### Authentication Flow

```javascript
// Simple JWT auth
async function login(email, password) {
    const response = await fetch('/api/auth/login', {
        method: 'POST',
        body: JSON.stringify({ email, password })
    });

    const { token, user_id } = await response.json();
    localStorage.setItem('token', token);
    localStorage.setItem('user_id', user_id);

    // Connect to cosmos
    connectToCosmos(user_id, token);
}

function connectToCosmos(userId, token) {
    const ws = new WebSocket(`ws://api/ws/cosmos/${userId}`);
    ws.onopen = () => {
        ws.send(JSON.stringify({
            type: 'auth',
            token: token
        }));
    };
}
```

---

## Event Flow: User A Creates Entry

```
User A writes diary entry
    ↓
POST /api/documents → {id: doc123, user_id: user_A}
    ↓
Cosmos event broadcaster emits: "entry_created"
    ↓
[User A sees placeholder]    [User B sees placeholder]
[User C sees placeholder]    [User D sees placeholder]
    ↓
Backend enriches chunks (realtime)
    ↓
For each chunk:
    → Emit "chunk_enriched" event
    → Broadcast to all users
    ↓
[User A morph their entry] [User B morph User A's entry]
[User C morph User A's]    [User D morph User A's]
    ↓
All chunks done
    ↓
Calculate final position (same for everyone)
    ↓
[All users] Animate entry to final position
    ↓
✨ Entry materialized in shared cosmos
```

---

## Synchronization Strategy

### Timing Considerations
- **Entry created event** - sent immediately
- **Chunk enriched events** - streamed as they arrive
- **Position calculation** - done client-side (same algorithm = same positions)
- **No polling needed** - everything is event-driven

### Conflict Resolution
- **Entries can't conflict** - they're immutable once created
- **No concurrent editing** - each user has separate modal
- **Race conditions** - acceptable (both entries appear, both positioned)

### Bandwidth Optimization
- **Only send embedding dimension, not full vectors** - position calculated locally
- **Batch updates** - group small events
- **Presence culling** - hide distant cursors
- **LOD for cursor** - simple 2D indicators, not 3D meshes

---

## Phases

### Phase 1: Core Multiplayer (This Branch)
- [ ] User authentication (email/JWT)
- [ ] User presence tracking
- [ ] Real-time entry creation broadcast
- [ ] Live chunk materialization for remote entries
- [ ] Shared positioning algorithm
- [ ] Basic user cursors/indicators

### Phase 2: Polish & Features
- [ ] User profiles/avatars
- [ ] Entry authorship display
- [ ] Comment system
- [ ] Favorite/bookmark entries
- [ ] User stats/achievements

### Phase 3: Advanced
- [ ] Global UMAP clustering
- [ ] Semantic search across all entries
- [ ] Entry recommendations
- [ ] Collaborative playlists
- [ ] Time-based snapshots of cosmos

---

## Deployment Considerations

### Supabase
- ✅ Real-time subscriptions via PostgREST (optional alternative to WebSocket)
- ✅ Row-level security (each user sees their entries + public)
- ✅ Full-text search on entries
- ✅ Vector similarity search (pgvector)

### Render
- ✅ WebSocket support (long-lived connections)
- ✅ Horizontal scaling (load balancing)
- ✅ Environment variables for auth secrets

### Scaling
- **1-100 users**: Single server fine
- **100-1000 users**: Add Redis pub/sub for broadcasting
- **1000+ users**: Sharded WebSocket servers + message queue

---

## Example Commit Messages for This Branch

```
feat: Add user authentication (email/JWT)
feat: Implement cosmos WebSocket for real-time broadcasts
feat: Add user presence tracking and cursors
feat: Broadcast entry creation to all connected users
feat: Broadcast enrichment progress to all users
feat: Implement shared entry positioning algorithm
feat: Add remote entry rendering and morphing
feat: Add user color system for entry identification
```

---

## Testing Strategy

```python
# test_multiplayer.py
async def test_two_users_see_shared_entry():
    """Two users create cosmos, one writes entry, both see it materialize"""

async def test_concurrent_entry_creation():
    """Three users write simultaneously, all see three planets"""

async def test_positioning_consistency():
    """All users calculate same positions for all entries"""

async def test_user_presence():
    """User joins/leaves, others see cursor appear/disappear"""
```

---

This branch is **multiplayer-ready** without breaking anything from main. Ship it! 🚀
