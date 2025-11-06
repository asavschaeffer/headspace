# Professional Testing & Deployment Workflow

## 🌳 Git Branching Strategy

### Branch Structure
```
main (production)
  ├── develop (staging/integration)
  │   ├── feature/demo-planets
  │   ├── feature/storage-selector
  │   └── feature/supabase-integration
  └── hotfix/critical-bug
```

### Branch Types
- **`main`**: Production-ready code, always deployable
- **`develop`**: Integration branch for staging/testing
- **`feature/*`**: New features (e.g., `feature/demo-planets`)
- **`hotfix/*`**: Critical production fixes
- **`release/*`**: Preparing for production release

---

## 📋 Complete Workflow

### Step 1: Create Feature Branch
```bash
# Start from latest develop
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/demo-planets

# Make your changes...
git add .
git commit -m "feat: add demo planets and seed data"
```

### Step 2: Local Testing ✅

#### 2.1 Run Local Tests
```bash
# Run unit tests (if you have them)
pytest tests/

# Run linting
flake8 . --max-line-length=120
black --check .

# Type checking (if using mypy)
mypy headspace/
```

#### 2.2 Manual Local Testing
```bash
# Start local server
python -m headspace.main

# Test in browser
# - http://localhost:8000/headspace.html
# - Verify demo planets appear
# - Test link clicking
# - Test storage mode switching
# - Check console for errors
```

#### 2.3 Test Database Operations
```bash
# Test with fresh database
rm headspace.db
python -m headspace.main
# Verify seed data creates correctly

# Test with existing database
# Verify no duplicate seed data
```

### Step 3: Push to Remote & Create PR
```bash
# Push feature branch
git push origin feature/demo-planets

# Create Pull Request on GitHub/GitLab
# - Title: "feat: Add demo planets and seed data"
# - Description: Link to issue, describe changes
# - Request review from team
```

### Step 4: Render Staging Environment 🧪

#### 4.1 Setup Staging Service
In Render Dashboard:
1. Create **new Web Service** (or use existing staging service)
2. Name: `headspace-staging`
3. Connect to same repository
4. **Branch**: `develop` (or your feature branch)
5. Environment: `staging`
6. Auto-deploy: `ON` (deploys on push to branch)

#### 4.2 Configure Staging Environment Variables
```
GEMINI_API_KEY=your_staging_key
SUPABASE_URL=your_staging_supabase_url
SUPABASE_KEY=your_staging_key
STORAGE_MODE=auto
DATABASE_PATH=/persistent/headspace-staging.db
```

#### 4.3 Test on Staging
```bash
# After pushing to develop branch
# Render auto-deploys staging service

# Test staging URL
# https://headspace-staging.onrender.com/headspace.html

# Verify:
# - Demo planets appear
# - Links work
# - Storage selector works
# - No console errors
# - API endpoints respond
```

### Step 5: Point Render to Feature Branch (Optional)

For testing specific features before merging:

```bash
# In Render Dashboard → Service Settings
# Change "Branch" to: feature/demo-planets
# Save → Auto-deploys from that branch
```

**Use this for:**
- Testing feature in isolation
- Demo to stakeholders
- Final verification before merge

### Step 6: Merge to Develop

After staging tests pass:

```bash
# Merge feature branch to develop
git checkout develop
git pull origin develop
git merge feature/demo-planets
git push origin develop

# Staging auto-deploys from develop
```

### Step 7: Production Deployment 🚀

#### 7.1 Create Release Branch (Optional but Recommended)
```bash
git checkout develop
git pull origin develop
git checkout -b release/v1.2.0

# Final testing, version bumps, changelog
# Then merge to main
```

#### 7.2 Merge to Main
```bash
git checkout main
git pull origin main
git merge develop  # or merge release branch
git push origin main
```

#### 7.3 Production Render Service
In Render Dashboard:
1. Go to **production service** (`headspace` or `headspace-prod`)
2. **Branch**: `main`
3. **Manual Deploy** or wait for auto-deploy
4. Verify production URL works

---

## 🔧 Render Configuration

### Production Service (`render.yaml`)
```yaml
services:
  - type: web
    name: headspace
    env: python
    branch: main  # Production branch
    buildCommand: pip install -r requirements.txt
    startCommand: python -m headspace.main
    envVars:
      - key: GEMINI_API_KEY
        sync: false
      - key: SUPABASE_URL
        sync: false
      - key: SUPABASE_KEY
        sync: false
    healthCheckPath: /api/health
    autoDeploy: true  # Auto-deploy on push to main
```

### Staging Service (Manual Setup)
Create separate service in Render:
- Name: `headspace-staging`
- Branch: `develop`
- Environment: `staging`
- Different database path: `/persistent/headspace-staging.db`

---

## ✅ Testing Checklist

### Local Testing
- [ ] Code runs without errors
- [ ] All features work as expected
- [ ] No console errors/warnings
- [ ] Database operations work
- [ ] API endpoints respond correctly
- [ ] Frontend renders correctly
- [ ] Responsive design works

### Staging Testing
- [ ] Staging deployment succeeds
- [ ] All environment variables set
- [ ] Database persists correctly
- [ ] Demo planets appear
- [ ] Links work
- [ ] Storage mode switching works
- [ ] API health checks pass
- [ ] Performance is acceptable

### Production Pre-Deploy
- [ ] All staging tests passed
- [ ] Code reviewed and approved
- [ ] Changelog updated
- [ ] Version bumped (if applicable)
- [ ] Database migration plan (if needed)
- [ ] Rollback plan ready

### Production Post-Deploy
- [ ] Production deployment successful
- [ ] Health check passes
- [ ] Smoke test: visit homepage
- [ ] Verify demo planets
- [ ] Check error logs
- [ ] Monitor for 5-10 minutes

---

## 🚨 Rollback Procedure

If production has issues:

### Quick Rollback (Render)
1. Go to Render Dashboard → Service → Deploys
2. Find last working deployment
3. Click "Redeploy"
4. Service reverts to that version

### Git Rollback
```bash
# Revert last commit
git revert HEAD
git push origin main

# Or reset to previous commit (if no one else pulled)
git reset --hard HEAD~1
git push origin main --force  # ⚠️ Use carefully
```

---

## 📊 Recommended Workflow Summary

```
1. Create feature branch
   ↓
2. Develop & test locally
   ↓
3. Push branch & create PR
   ↓
4. Test on staging (develop branch)
   ↓
5. Optional: Test feature branch on Render
   ↓
6. Code review & approval
   ↓
7. Merge to develop
   ↓
8. Final staging tests
   ↓
9. Merge to main
   ↓
10. Production auto-deploys
   ↓
11. Monitor & verify
```

---

## 🔐 Environment Management

### Local Development
```bash
# .env file (not committed)
GEMINI_API_KEY=dev_key
SUPABASE_URL=local_test_url
STORAGE_MODE=local
```

### Staging (Render)
- Use staging API keys
- Staging Supabase project
- Test data only

### Production (Render)
- Production API keys
- Production Supabase project
- Real user data

---

## 🛠️ Tools & Commands

### Pre-commit Checks
```bash
# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

### Testing Commands
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=headspace --cov-report=html

# Lint code
flake8 . --max-line-length=120
black --check .

# Type check
mypy headspace/
```

### Database Testing
```bash
# Backup before testing
cp headspace.db headspace.db.backup

# Test fresh install
rm headspace.db
python -m headspace.main

# Restore backup
cp headspace.db.backup headspace.db
```

---

## 📝 Best Practices

1. **Never push directly to `main`** - Always use PRs
2. **Test locally first** - Catch issues early
3. **Use staging environment** - Test in production-like environment
4. **Monitor after deploy** - Watch logs for 10-15 minutes
5. **Keep commits small** - Easier to review and rollback
6. **Write descriptive commits** - Use conventional commits format
7. **Document breaking changes** - Update README/changelog
8. **Test rollback procedure** - Know how to revert quickly

---

## 🎯 Quick Reference

### Daily Workflow
```bash
# Start new feature
git checkout develop && git pull
git checkout -b feature/my-feature

# Work, test locally, commit
git add . && git commit -m "feat: description"
git push origin feature/my-feature

# Create PR, get review, merge to develop
# Staging auto-deploys

# After staging tests pass, merge to main
# Production auto-deploys
```

### Emergency Hotfix
```bash
# Create hotfix from main
git checkout main && git pull
git checkout -b hotfix/critical-bug

# Fix, test, commit
git add . && git commit -m "fix: critical bug"
git push origin hotfix/critical-bug

# Merge directly to main (skip develop for critical)
git checkout main
git merge hotfix/critical-bug
git push origin main

# Also merge back to develop
git checkout develop
git merge hotfix/critical-bug
git push origin develop
```

---

This workflow ensures:
- ✅ Code is tested at every stage
- ✅ Staging catches issues before production
- ✅ Easy rollback if needed
- ✅ Clear separation of environments
- ✅ Professional development practices

