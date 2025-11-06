# Headspace Demo Planets & Storage Features

## ✅ Completed Features

### 1. Demo Planets/Chunks 🌱
- **Seed Data System**: Created `seed_data.py` that automatically populates demo content when database is empty
- **Demo Content Includes**:
  - 🏠 **Return Home Planet**: Clickable planet that links back to `/index.html`
  - 📖 **About Headspace**: Project information and features
  - ✨ **Poetry Demo 1**: "Cosmic Thoughts" 
  - ✨ **Poetry Demo 2**: "The Language of Stars"
  - 👋 **Welcome Message**: Instructions for users

- **Automatic Seeding**: Demo planets are created automatically on first run
- **Custom Positioning**: Each demo planet has a specific 3D position and color
- **Link Support**: Planets can have clickable links (used for home button)

### 2. Clickable Link Planets 🔗
- **Metadata Support**: Chunks can now have `link_url` and `is_external_link` in metadata
- **Frontend Integration**: 
  - Clicking a planet with a link navigates to that URL
  - Internal links navigate in same window
  - External links open in new tab
  - Info panel shows "🔗 Follow Link" button for link planets

### 3. Storage Mode Selector 💾☁️
- **UI Component**: Added storage mode selector in sidebar
- **Two Modes**:
  - 💾 **Local**: Stores data in SQLite database on your computer
  - ☁️ **Cloud**: Stores data in Supabase (shared across devices/users)

- **Features**:
  - Visual toggle buttons
  - Status indicator showing current mode
  - Automatic detection of cloud availability
  - LocalStorage persistence of user preference

### 4. Supabase Cloud Storage Integration ☁️
- **Backend Support**: Created `SupabaseStorage` class implementing cloud storage
- **Database Schema**: Created `supabase_schema.sql` with complete schema
- **Features**:
  - User-based data isolation (Row Level Security)
  - Full CRUD operations for documents, chunks, connections
  - Automatic user ID tracking
  - JSONB storage for metadata, embeddings, positions

- **Setup Required**:
  - Create Supabase project at https://supabase.com
  - Run `supabase_schema.sql` in SQL editor
  - Set environment variables:
    - `SUPABASE_URL`
    - `SUPABASE_KEY`
    - `STORAGE_MODE=cloud` (optional, auto-detects)

## 📁 Files Created/Modified

### New Files:
- `seed_data.py` - Demo content generator
- `headspace/services/storage_manager.py` - Storage mode management
- `headspace/services/supabase_storage.py` - Supabase backend
- `supabase_schema.sql` - Database schema for Supabase
- `static/js/storage-manager.js` - Frontend storage mode handling

### Modified Files:
- `headspace/main.py` - Added seed data initialization
- `headspace/api/routes.py` - Added storage status endpoint
- `static/headspace.html` - Added storage selector UI
- `static/js/cosmos-renderer.js` - Added link clicking support
- `static/js/main.js` - Integrated storage manager
- `static/css/components.css` - Added storage selector styles
- `requirements.txt` - Added supabase dependency

## 🚀 Usage

### Local Storage (Default)
1. No setup needed - works out of the box
2. Data stored in `headspace.db` SQLite file
3. Perfect for personal use
4. Data stays on your computer

### Cloud Storage (Supabase)
1. Create Supabase project: https://supabase.com
2. Run `supabase_schema.sql` in SQL editor
3. Get your project URL and anon key
4. Set environment variables:
   ```bash
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-anon-key
   ```
5. Click "☁️ Cloud" button in the UI
6. Your data syncs across devices!

## 🎨 Demo Experience

When users first visit:
1. Empty database triggers demo content creation
2. 5 demo planets appear in the cosmos:
   - Red planet at center: "Return Home" (links to index)
   - Turquoise planet: "About Headspace"
   - Green planets: Poetry demos
   - Light green planet: Welcome message
3. Users can click planets to explore
4. Users can add their own documents
5. Storage mode selector lets them choose local or cloud

## 🔮 Future Enhancements

- [ ] Real-time sync between local and cloud
- [ ] Export/import functionality
- [ ] Collaborative shared headspaces
- [ ] User authentication for cloud mode
- [ ] More demo content types
- [ ] Custom planet shapes based on content type

