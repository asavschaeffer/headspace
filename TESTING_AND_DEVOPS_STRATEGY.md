# Testing & DevOps Strategy for Headspace

## 🎯 Current State (Nov 14, 2025)

### The Refactoring in Progress
You're mid-refactor from a complex sidebar-based "Cosmic Knowledge System" to a pure immersive experience: **Headspace - Cosmic Diary**.

**What Changed:**
- ✅ `headspace.html` - Simplified UI focused on cosmos view only
- ✅ `headspace-main.js` - New entry point for the diary experience
- ✅ `headspace.css` - Immersive styling with modal, overlay, info panel
- ❌ `index-debug.html` - Deleted (was your working demo)
- ❌ `seed-documents.js` - Deleted (need to recreate)
- ❌ Original sidebar UI in `index.html` - Still there but may not be needed

**Branch:** `feature/visual-diary-experience` (dirty with uncommitted changes)

**Worktrees:**
- `2025-11-13-kdxr-mNnmu` - Python backend (embeddings, chunking, API)
- `2025-11-13-kxfj-W1A7a` - Similar backend structure
- Main branch has the frontend

---

## 🏗️ Architecture Overview

### Frontend Stack (Static/)
```
headspace.html (immersive diary UI)
    ↓
headspace-main.js (initialization, modal, document creation)
    ↓
cosmos-renderer.js (Three.js 3D visualization)
    ↓
enrichment-stream.js (real-time chunk processing)
    ↓
procedural-geometry.js (shape generation from embeddings)
```

### Backend Stack (Python, in worktrees)
```
headspace/main.py (Flask/FastAPI server)
    ↓
embeddings_engine.py (Gemini, OpenAI, Ollama)
    ↓
llm_chunker.py (intelligent text chunking)
    ↓
headspace_system.py (UMAP/HDBScan clustering)
    ↓
SQLite/Supabase (persistent storage)
```

### Data Flow
1. User writes thought in modal → `headspace-main.js` captures it
2. Frontend POSTs to `/api/documents` → Backend chunks + embeds
3. Backend returns embeddings + UMAP coordinates + cluster IDs
4. Frontend calls `/api/visualization` to get all 3D positions
5. `cosmos-renderer.js` animates chunks into 3D space
6. User hovers/clicks → `cosmos-info` panel updates

---

## 🧪 Testing Strategy (Layered Approach)

### Layer 1: **Isolated Feature Testing** (No Backend Required)
Test individual components in isolation using stub data.

**Setup:**
```bash
# Create test HTML entry points in static/
static/test/
  ├── test-cosmos-viewer.html        # Just Three.js + cosmos-renderer
  ├── test-shape-morphing.html       # Real-time embedding animation
  ├── test-home-planet.html          # Home planet with click handler
  ├── test-modal-experience.html     # Modal + form validation
  └── test-data/
      └── sample-visualization.json  # Mock /api/visualization response
```

**Example: `test-cosmos-viewer.html`**
```html
<!-- Load cosmos with mock data, no API calls -->
<script src="../js/three.min.js"></script>
<script src="../js/cosmos-renderer.js"></script>
<script>
  // Mock state with sample chunks/documents
  const state = { ... };
  initCosmos(); // Renders 3D scene
</script>
```

### Layer 2: **API Integration Testing** (Local Backend)
Test frontend + backend together.

**Setup:**
```bash
# Terminal 1: Start backend
cd worktree/2025-11-13-kdxr-mNnmu
python headspace/main.py  # Listens on localhost:8000

# Terminal 2: Serve frontend
cd main/headspace
python -m http.server 8001  # Serve static/ on :8001
```

**Test Sequence:**
```bash
# 1. Create test document via API
curl -X POST http://localhost:8000/api/documents \
  -H "Content-Type: application/json" \
  -d '{"title":"Test","content":"Paragraph 1\n\nParagraph 2","doc_type":"text"}'

# 2. Get visualization data
curl http://localhost:8000/api/visualization | jq .

# 3. Open http://localhost:8001/headspace.html
# → Should see planets appear and animate into place
```

### Layer 3: **E2E Testing** (Full User Journey)
1. Open headspace.html
2. Click "+ New Thought"
3. Fill form, click "Launch Planet"
4. Watch planet materialize in 3D
5. Hover over planet to see info
6. Click home planet to return to index.html

**Checklist:**
```
☐ Modal opens on "+ New Thought" click
☐ Form validation works (can't submit empty content)
☐ Signature field appends to content correctly
☐ API responds with document ID
☐ Visualization refreshes and shows new planet
☐ Shape morphs smoothly in real-time
☐ Planet animates from center → final position
☐ Hover shows chunk info in bottom-left panel
☐ Home planet is visible and clickable
☐ Click home planet → redirects to /index.html
```

---

## 🚀 Setting Up Your Workflow

### Option A: **Isolated Testing** (Fast, No Backend)
Best for UI/3D visualization work.

```bash
# 1. Create test files
mkdir -p static/test/test-data

# 2. Copy sample visualization response to test-data
curl http://localhost:8000/api/visualization > static/test/test-data/sample.json

# 3. Create test-cosmos-viewer.html (use mock state)

# 4. Open in browser
open http://localhost:8001/test/test-cosmos-viewer.html
```

### Option B: **Local Backend + Frontend** (Full Testing)
Best for integration testing.

```bash
# Terminal 1
cd ~/.cursor/worktrees/thoughts/mNnmu
python headspace/main.py

# Terminal 2
cd ~/Documents/thoughts
python -m http.server 8001

# Browser
open http://localhost:8001/headspace.html
```

### Option C: **Docker** (Production-like)
Best for deployment testing.

```bash
cd ~/.cursor/worktrees/thoughts/mNnmu
docker-compose up --build
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
```

---

## 📝 TODO: Complete the Refactoring

### 1. **Recreate Seed Documents** ✨
The old `seed-documents.js` was deleted. You need:
```javascript
// static/js/seed-documents.js
export const seedDocuments = [
  {
    title: "Home",
    content: "Portal back to the main index.",
    doc_type: "metadata"
  },
  {
    title: "Headspace Guide",
    content: "README for this cosmic diary...",
    doc_type: "guide"
  },
  {
    title: "Hagakure",
    content: "Extract or poetry excerpt...",
    doc_type: "creative"
  }
];
```

### 2. **Fix index.html** 🔧
Currently it has the old sidebar UI. You have two options:
- **Option A:** Keep it as a "dashboard" view with document list
- **Option B:** Simplify it to be a landing page → headspace.html

We recommend **Option B** for now:
```html
<!-- index.html simplified landing -->
<a href="/headspace.html">Enter Headspace</a>
```

### 3. **Test Form Validation** ✅
The modal in headspace-main.js has validation logic. Test:
```
☐ Empty content shows "Your thought needs substance"
☐ Signature field is optional
☐ Title defaults to "Untitled Thought"
```

### 4. **Fix Git Status** 📦
Decide what to commit:
```bash
# Current changes:
git status

# Option 1: Commit the refactoring
git add -A
git commit -m "feat: Simplify Headspace to pure cosmic diary experience"

# Option 2: Keep as WIP (don't commit yet)
git stash  # Save changes
git checkout main  # Go back to stable

# We recommend Option 1: commit the progress
```

---

## 🔍 Debugging Checklist

### If headspace.html shows "Initialization failed"
1. Check browser console: `F12` → Console tab
2. Look for errors in:
   - Three.js load? (CDN script)
   - headspace-main.js? (module import errors)
   - cosmos-renderer.js? (initialization error)
   - API calls? (fetch error to `/api/visualization`)

### If modal doesn't open
```javascript
// In browser console
document.getElementById('open-thought').click()
// If no error, issue is event listener registration
```

### If planets don't render
```javascript
// In console
const chunkMeshes = window.getChunkMeshes?.()
console.log('Chunk meshes:', chunkMeshes?.size)
// If undefined, cosmos-renderer didn't initialize
```

### If form submission fails
```javascript
// Mock the API to return sample document
fetch('/api/documents', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    title: 'Test',
    content: 'Test content',
    doc_type: 'text'
  })
}).then(r => r.json()).then(console.log)
```

---

## 📊 Worktree Usage Guide

### What's in Each Worktree?
```bash
# mNnmu = Python backend (embeddings + API)
cd ~/.cursor/worktrees/thoughts/mNnmu
ls  # See: embeddings_engine.py, headspace_system.py, main.py

# W1A7a = Similar backend (maybe different branch/feature?)
cd ~/.cursor/worktrees/thoughts/W1A7a
# Compare with mNnmu to see differences

# Main (~/Documents/thoughts) = Frontend
# This is where headspace.html, static/js/, etc. live
```

### Why Use Worktrees?
- **Isolation:** Backend + frontend can develop independently
- **Testing:** Switch between backend branches without affecting frontend
- **Deployment:** Each worktree can have different configs

### Recommended Workflow
```bash
# Frontend development (main repo)
cd ~/Documents/thoughts
# Work on headspace.html, headspace-main.js, etc.
# Test against worktree backend

# Backend development (worktree)
cd ~/.cursor/worktrees/thoughts/mNnmu
# Work on embeddings_engine.py, api endpoints, etc.
# Test with frontend from main repo

# Sync changes
cd ~/Documents/thoughts && git pull origin main
cd ~/.cursor/worktrees/thoughts/mNnmu && git pull origin <branch>
```

---

## 🎯 Next Steps (Priority Order)

### Phase 1: **Stabilize Current Code** (This Week)
1. ✅ Commit the headspace refactoring
2. ✅ Recreate seed-documents.js for initial planets
3. ✅ Test headspace.html with local backend
4. ✅ Fix any initialization errors

### Phase 2: **Complete the Vision** (Next Week)
1. Home planet with proper styling (foggy sphere + mountain peak)
2. Real-time embedding visualization as user types
3. Shape morphing animation
4. UMAP positioning for new documents
5. Nebulae around semantic clusters

### Phase 3: **Polish & Deploy** (Week After)
1. Responsive design for mobile
2. Accessibility (a11y) improvements
3. Performance optimization
4. Production deployment

---

## 📚 Reference Files

- **Test Guide:** `TESTING_GUIDE.md` (for API testing)
- **Features:** `FEATURES.md` (complete feature list)
- **Architecture:** README.md (overview)
- **Deployment:** `DEPLOYMENT_WORKFLOW.md` (for production)

---

## 🆘 Getting Help

If stuck on:
- **3D Visualization:** Look at `cosmos-renderer.js` (Three.js scene setup)
- **API Integration:** Look at `headspace-main.js` (fetch calls + error handling)
- **Embeddings:** Check worktree `embeddings_engine.py`
- **Testing:** Use the curl commands above or browser console

---

**Last Updated:** 2025-11-14
**Status:** In refactoring, tests needed
**Next Milestone:** Working headspace.html with local backend
