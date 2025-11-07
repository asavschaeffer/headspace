# Quick Deployment Workflow Cheat Sheet

## 🚀 Simplified Testing Workflow

Skip local testing. Use Render for everything.

### Step 1: Create & Push Feature Branch
```bash
git checkout -b feature/your-feature-name
git add .
git commit -m "Your commit message"
git push origin feature/your-feature-name
```

### Step 2: Point Render to Feature Branch & Test
- Go to **Render Dashboard** → Service settings
- Change **"Branch"** to your feature branch name
- Save (auto-deploys in ~2-3 minutes)
- Test at: `https://your-service.onrender.com/headspace.html`

### Step 3: If Tests Pass → Merge & Deploy to Production
```bash
git checkout main
git merge feature/your-feature-name
git push origin main
```

Then in Render Dashboard:
- Change branch back to `main`
- Auto-deploys to production

### Step 4: If Tests Fail → Fix & Re-push
```bash
# Make fixes on your feature branch
git add .
git commit -m "fix: address issue"
git push origin feature/your-feature-name
```
Render auto-redeploys. Test again.

## 🔄 Render Service Configuration

### Production Service
- **Name**: `headspace`
- **Branch**: `main` (change this when testing features)
- **Auto-deploy**: ON

## 🚨 Quick Rollback

If something breaks on production:
- Render Dashboard → Deploys tab
- Find the last working deploy
- Click "Redeploy"

Or revert the commit:
```bash
git revert HEAD
git push origin main
```

