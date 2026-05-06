# Deployment Action Plan: Render + Vercel

**Status:** ✅ Project is fully ready for deployment

---

## 📋 Pre-Deployment

**Duration:** 5 minutes

- [ ] **Merge review branch to main**
  - URL: https://github.com/harshitxix/DeepNexus/pull/new/review/all-current-changes-2026-05-06
  - Action: Create PR → Review → Merge

- [ ] **Create Render account** (if needed)
  - Go to https://render.com
  - Sign up with GitHub

- [ ] **Create Vercel account** (if needed)
  - Go to https://vercel.com
  - Sign up with GitHub

---

## 🚀 Phase 1: Deploy Backend on Render

**Duration:** 10 minutes | **Cost:** Free (or $7/mo for Starter)

```
1. Go to https://dashboard.render.com

2. Click "New +" → "Web Service"

3. Connect GitHub & select DeepNexus repo

4. Fill in settings:
   - Name: deepnexus-api
   - Root Directory: . (leave empty)
   - Runtime: Python 3.11
   - Build Command: pip install -r backend/requirements.txt
   - Start Command: cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

5. Add Environment Variables (click "Add"):
   PYTHONUNBUFFERED=true
   DEBUG=false
   ENVIRONMENT=production
   LOG_LEVEL=info
   CORS_ORIGINS=https://placeholder.vercel.app (update later)
   NVIDIA_API_KEY=<your-nvidia-api-key>

6. Click "Create Web Service"

7. Wait for build (3-5 minutes)

8. SAVE THE BACKEND URL when deployment finishes
   Example: https://deepnexus-api.onrender.com
```

---

## 🎨 Phase 2: Deploy Frontend on Vercel

**Duration:** 5 minutes | **Cost:** Free

```
1. Go to https://vercel.com/new

2. Click "Import Git Repository"

3. Search for and select DeepNexus

4. Fill in settings:
   - Project Name: deepnexus
   - Framework Preset: React
   - Root Directory: frontend
   - Build Command: npm run build
   - Install Command: npm install

5. Add Environment Variable:
   Name: VITE_API_BASE
   Value: https://deepnexus-api.onrender.com (from Phase 1)

6. Click "Deploy"

7. Wait for build (2-3 minutes)

8. SAVE THE FRONTEND URL when deployment finishes
   Example: https://deepnexus.vercel.app
```

---

## 🔗 Phase 3: Update CORS & Link Services

**Duration:** 2 minutes

```
1. Go back to Render Dashboard

2. Select deepnexus-api service

3. Go to "Environment"

4. Update CORS_ORIGINS to:
   CORS_ORIGINS=https://deepnexus.vercel.app (use your actual Vercel URL)

5. Click "Save" (automatic redeploy)

6. Wait for redeploy (~1 minute)
```

---

## ✅ Phase 4: Test Deployment

**Duration:** 5 minutes

**Test Backend:**
```bash
# Open terminal and run:
curl https://deepnexus-api.onrender.com/api/health

# Expected response:
{"status":"ok"}
```

**Test Frontend:**
1. Open https://deepnexus.vercel.app in browser
2. Open Developer Console (F12)
3. Try using a feature (Study Mode, Data Engine, etc.)
4. Check Network tab for API calls
5. Verify calls go to backend URL (not localhost)
6. Verify responses are 200 OK

**Test Console:**
- No CORS errors
- No 404 errors
- No red error messages

---

## 🔍 Troubleshooting

**If backend shows 502 Bad Gateway:**
1. Check Render logs (Render dashboard → select service → Logs)
2. Look for error messages
3. Common fixes:
   - Verify all env vars are set
   - Check requirements.txt is valid
   - Try manual redeploy: Manual Deploy → Deploy latest commit

**If frontend shows API errors:**
1. Check browser console for CORS errors
2. Verify VITE_API_BASE env var is correct in Vercel
3. Verify CORS_ORIGINS in Render backend includes frontend URL
4. Hard refresh browser (Ctrl+Shift+R or Cmd+Shift+R)

**If build fails:**
1. Check build logs in deployment platform
2. Render: https://dashboard.render.com → logs
3. Vercel: https://vercel.com → project → deployments → latest → logs

---

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| Backend first response | 30-60s (cold start, free tier) |
| API latency | 50-150ms |
| Frontend load time | 2-3s |
| Bundle size | 5.3MB uncompressed |

---

## 💾 What Gets Deployed

**Backend (Render):**
- FastAPI app with all 11 endpoints
- Python dependencies
- Local SQLite database (in process)
- Supports NVIDIA API, OpenCV, PyTorch

**Frontend (Vercel):**
- React SPA (Single Page App)
- All UI modules (Study Mode, Data Engine, Quiz, etc.)
- CSS and assets
- Global CDN distribution

---

## 🎯 Post-Deployment Checklist

After everything is live:

- [ ] Backend responds to health check
- [ ] Frontend loads without errors
- [ ] Study Mode quiz works
- [ ] Data Engine CSV upload works
- [ ] Scroll animations work
- [ ] Future modules tab displays correctly
- [ ] All API calls show correct backend URL
- [ ] No console errors
- [ ] Monitor backend logs for errors

---

## 📞 Support Resources

| Need | Resource |
|------|----------|
| Detailed guide | [DEPLOY_RENDER_VERCEL.md](./DEPLOY_RENDER_VERCEL.md) |
| Readiness report | [DEPLOYMENT_READINESS.md](./DEPLOYMENT_READINESS.md) |
| Render docs | https://render.com/docs |
| Vercel docs | https://vercel.com/docs |
| FastAPI | https://fastapi.tiangolo.com |
| React | https://react.dev |

---

## 🎓 Free Tier Notes

**Render Free Tier:**
- Services spin down after 15 min of no requests
- First request after spin-down takes 30-60s (cold start)
- Upgrade to Starter ($7/mo) to avoid spin-down

**Vercel Free Tier:**
- 100GB bandwidth/month (plenty for demo)
- Unlimited deployments
- Global CDN included

**Recommendations:**
1. Start on free tier for testing
2. Upgrade if you expect active daily users
3. Monitor resource usage in dashboards

---

**Estimated time to deployment:** 30 minutes  
**Status:** Ready to deploy now  
**Last updated:** May 6, 2026
