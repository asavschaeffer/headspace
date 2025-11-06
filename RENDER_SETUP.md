# Render Service Setup Guide

## 🎯 Two-Environment Setup

### Option 1: Single Service (Simple)
Use one Render service, manually switch branches for testing:
- **Production**: Point to `main` branch
- **Testing**: Temporarily point to `develop` or feature branch

### Option 2: Two Services (Recommended)
Create separate services for staging and production:

---

## 📦 Production Service

**Service Name**: `headspace` (or `headspace-prod`)

**Configuration**:
- **Branch**: `main`
- **Auto-deploy**: ON
- **Environment**: Production
- **Database**: `/persistent/headspace.db`
- **URL**: `https://asaschaeffer.com/headspace.html`

**Environment Variables**:
```
GEMINI_API_KEY=<production_key>
SUPABASE_URL=<production_supabase_url>
SUPABASE_KEY=<production_supabase_key>
STORAGE_MODE=auto
```

---

## 🧪 Staging Service

**Service Name**: `headspace-staging`

**Configuration**:
- **Branch**: `develop`
- **Auto-deploy**: ON
- **Environment**: Staging
- **Database**: `/persistent/headspace-staging.db`
- **URL**: `https://headspace-staging.onrender.com/headspace.html`

**Environment Variables**:
```
GEMINI_API_KEY=<staging_key>
SUPABASE_URL=<staging_supabase_url>
SUPABASE_KEY=<staging_supabase_key>
STORAGE_MODE=auto
```

---

## 🔧 Setup Instructions

### Step 1: Create Production Service
1. Go to Render Dashboard
2. Click "New +" → "Web Service"
3. Connect your repository
4. Configure:
   - **Name**: `headspace`
   - **Branch**: `main`
   - **Root Directory**: (leave empty)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python -m headspace.main`
5. Add environment variables
6. Configure persistent disk
7. Save

### Step 2: Create Staging Service
1. Click "New +" → "Web Service"
2. Connect **same repository**
3. Configure:
   - **Name**: `headspace-staging`
   - **Branch**: `develop`
   - Same build/start commands
   - Different database path: `/persistent/headspace-staging.db`
4. Add staging environment variables
5. Configure persistent disk (separate from production)
6. Save

### Step 3: Configure Custom Domain (Production Only)
1. Go to production service → Settings
2. Click "Custom Domains"
3. Add: `asaschaeffer.com`
4. Follow DNS configuration instructions
5. Wait for SSL certificate (automatic)

---

## 🔄 Branch Testing Workflow

### Test Feature Branch on Render

1. **Push feature branch**:
   ```bash
   git push origin feature/my-feature
   ```

2. **In Render Dashboard**:
   - Go to staging service
   - Settings → Change "Branch" to `feature/my-feature`
   - Save (auto-deploys)

3. **Test**:
   - Visit staging URL
   - Verify feature works
   - Check logs for errors

4. **After testing**:
   - Change branch back to `develop`
   - Or merge feature to develop

---

## 📊 Service Comparison

| Feature | Production | Staging |
|---------|-----------|---------|
| Branch | `main` | `develop` |
| URL | `asaschaeffer.com` | `headspace-staging.onrender.com` |
| Database | `headspace.db` | `headspace-staging.db` |
| API Keys | Production | Staging/Test |
| Auto-deploy | ON | ON |
| Purpose | Live users | Testing |

---

## ✅ Verification Checklist

After setup, verify:

- [ ] Production service deploys from `main`
- [ ] Staging service deploys from `develop`
- [ ] Both services have persistent disks
- [ ] Environment variables set correctly
- [ ] Health checks pass (`/api/health`)
- [ ] Custom domain configured (production)
- [ ] SSL certificate active (production)
- [ ] Demo planets appear on both
- [ ] Storage selector works
- [ ] No console errors

---

## 🚨 Troubleshooting

**Issue**: Staging shows production data
- **Fix**: Use separate database paths and Supabase projects

**Issue**: Auto-deploy not working
- **Fix**: Check branch name matches exactly, verify webhook is set

**Issue**: Environment variables not loading
- **Fix**: Ensure variables are set in Render dashboard, not just `.env`

**Issue**: Database not persisting
- **Fix**: Verify persistent disk is mounted at `/persistent`

---

## 💡 Pro Tips

1. **Use different Supabase projects** for staging and production
2. **Monitor staging logs** before merging to main
3. **Test database migrations** on staging first
4. **Keep staging data separate** - don't use real user data
5. **Use feature flags** for gradual rollouts (future enhancement)

