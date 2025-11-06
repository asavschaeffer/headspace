# Quick Deployment Workflow Cheat Sheet

## 🚀 Standard Feature Deployment

```bash
# 1. Create feature branch
git checkout develop
git pull origin develop
git checkout -b feature/new-feature

# 2. Develop & test locally
python -m headspace.main
# Test in browser: http://localhost:8000/headspace.html

# 3. Commit & push
git add .
git commit -m "feat: add new feature"
git push origin feature/new-feature

# 4. Create PR on GitHub
# Wait for review/approval

# 5. Test on staging (after merge to develop)
# Visit: https://headspace-staging.onrender.com

# 6. Merge to main (after staging tests pass)
git checkout main
git merge develop
git push origin main

# 7. Production auto-deploys
# Visit: https://asaschaeffer.com/headspace.html
```

## 🧪 Testing on Render Feature Branch

```bash
# 1. Push feature branch
git push origin feature/new-feature

# 2. In Render Dashboard:
#    - Go to Service Settings
#    - Change "Branch" to: feature/new-feature
#    - Save (auto-deploys)

# 3. Test feature branch URL
#    - https://headspace-xxxxx.onrender.com/headspace.html

# 4. After testing, change branch back to develop/main
```

## 🔄 Render Service Setup

### Production Service
- **Name**: `headspace`
- **Branch**: `main`
- **Auto-deploy**: ON
- **Environment**: Production

### Staging Service (Create separately)
- **Name**: `headspace-staging`
- **Branch**: `develop`
- **Auto-deploy**: ON
- **Environment**: Staging

## ✅ Pre-Deploy Checklist

- [ ] Code tested locally
- [ ] No console errors
- [ ] All features work
- [ ] Tests pass (if any)
- [ ] Code reviewed
- [ ] Staging tests pass
- [ ] Environment variables set
- [ ] Database migrations ready (if any)

## 🚨 Rollback

**Render Dashboard** → Service → Deploys → Find working version → Redeploy

OR

```bash
git revert HEAD
git push origin main
```

