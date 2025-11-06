# Render Deployment Guide

## Storage & Provider Configuration

### 📦 **Storage (Notes & Database)**

**Current Setup:**
- Uses SQLite database (`headspace.db`) 
- Stores documents in `documents/` folder
- Data is stored locally in files

**On Render:**
- ✅ **Persistent Disk** (RECOMMENDED): Configured via Render dashboard
  - Go to your service → Settings → Persistent Disk
  - Create disk: `headspace-disk`
  - Mount path: `/persistent`
  - Size: 1GB (increase as needed)
  - Database and documents will persist across deployments
  
- ⚠️ **Without Persistent Disk**: Data will be lost on every deployment
  - Ephemeral filesystem clears on restart
  - Only use for testing/demos

**Alternative Options:**
- Use external PostgreSQL database (Render PostgreSQL service)
- Use S3/object storage for documents
- Use external database service (Supabase, Neon, etc.)

---

### 🤖 **AI Providers**

#### **1. Gemini (RECOMMENDED for Production)**
- ✅ Works perfectly on Render
- ✅ Configured as primary provider in `loom_config.json`
- ✅ Uses API keys from environment variables
- **Setup:**
  1. Get API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
  2. Set `GEMINI_API_KEY` in Render dashboard → Environment
  3. No additional configuration needed

#### **2. OpenAI**
- ✅ Works perfectly on Render
- ✅ Configured as fallback provider
- **Setup:**
  1. Get API key from [OpenAI Platform](https://platform.openai.com/api-keys)
  2. Set `OPENAI_API_KEY` in Render dashboard → Environment
  3. No additional configuration needed

#### **3. Ollama**
- ⚠️ **Problem**: Ollama runs locally on your machine
- ❌ `http://host.docker.internal:11434` won't work on Render
- **Solutions:**

  **Option A: Disable Ollama (RECOMMENDED)**
  - Already configured to use Gemini first
  - Ollama is fallback only
  - Works without any Ollama setup

  **Option B: External Ollama Service**
  - Deploy Ollama on separate server/VPS
  - Set `OLLAMA_URL` environment variable to external URL
  - Example: `https://ollama.yourdomain.com`
  - More complex, requires separate hosting

  **Option C: Use Sentence Transformers**
  - Works without external services
  - Already in fallback chain for embeddings
  - Slower but functional

#### **4. Mock Provider**
- ✅ Always available as last resort
- ⚠️ Limited functionality (for testing only)

---

## Provider Fallback Chain

**LLM (Text Generation):**
1. Gemini (primary) ✅
2. OpenAI (fallback) ✅
3. Ollama (if configured) ⚠️
4. Mock (always works)

**Embeddings:**
1. Gemini (primary) ✅
2. Sentence Transformers (fallback) ✅
3. Ollama (if configured) ⚠️
4. Mock (always works)

---

## Environment Variables Needed

### Required:
- `GEMINI_API_KEY` - Get from [Google AI Studio](https://aistudio.google.com/app/apikey)

### Optional:
- `OPENAI_API_KEY` - Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- `OLLAMA_URL` - Only if using external Ollama service

### Auto-configured:
- `PORT` - Set automatically by Render
- `DATABASE_PATH` - Set to `/persistent/headspace.db` (if disk configured)
- `DOCUMENTS_FOLDER` - Set to `/persistent/documents` (if disk configured)

---

## Deployment Steps

1. **Push code to Git repository**
   ```bash
   git add .
   git commit -m "Configure for Render deployment"
   git push
   ```

2. **Connect Repository to Render**
   - Go to [render.com](https://render.com)
   - New → Web Service
   - Connect your repository
   - Render detects `render.yaml`

3. **Configure Persistent Disk** (IMPORTANT!)
   - Service → Settings → Persistent Disk
   - Create disk: `headspace-disk`
   - Mount path: `/persistent`
   - Size: 1GB minimum

4. **Set Environment Variables**
   - Service → Environment
   - Add `GEMINI_API_KEY` (your actual key)
   - Optionally add `OPENAI_API_KEY`

5. **Deploy**
   - Render auto-deploys on push
   - Or manually trigger deployment

6. **Configure Custom Domain** (for asaschaeffer.com)
   - Service → Settings → Custom Domains
   - Add `asaschaeffer.com`
   - Configure DNS as instructed

---

## Cost Considerations

**Render Free Tier:**
- ✅ 750 hours/month free
- ✅ Persistent disk available (paid)
- ⚠️ Spins down after inactivity

**Paid Tier:**
- Always-on service
- Better for production

**API Costs:**
- Gemini: Free tier available, then pay-per-use
- OpenAI: Pay-per-use
- Ollama: Free (if self-hosted)

---

## Troubleshooting

**Issue: Data lost after deployment**
- → Persistent disk not configured
- → Check disk mount path matches `/persistent`

**Issue: Gemini not working**
- → Check `GEMINI_API_KEY` is set correctly
- → Verify API key is valid

**Issue: Ollama connection failed**
- → Expected if no external Ollama service
- → App will automatically fallback to Gemini/OpenAI

**Issue: Database errors**
- → Check persistent disk is mounted
- → Verify `DATABASE_PATH` environment variable

---

## Recommended Production Setup

1. ✅ Use Gemini as primary provider (configured)
2. ✅ Configure persistent disk for data storage
3. ✅ Set up custom domain
4. ✅ Enable HTTPS (automatic on Render)
5. ✅ Monitor API usage/costs
6. ✅ Set up backups (export database periodically)

