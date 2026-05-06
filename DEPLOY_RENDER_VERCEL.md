# DeepNexus Deployment Guide: Render + Vercel

Complete step-by-step guide to deploy DeepNexus backend on Render and frontend on Vercel.

---

## Prerequisites

- GitHub account with the DeepNexus repository
- Render account (https://render.com)
- Vercel account (https://vercel.com)
- Git CLI installed locally

---

## Phase 1: Prepare for Deployment

### 1.1 Commit all changes to main branch

```bash
cd C:\Users\harsh\OneDrive\Desktop\nn\DeepNexus
git add -A
git commit -m "Deploy: prepare for Render + Vercel"
git push origin main
```

Wait for your review branch to be merged into `main` first if needed.

---

## Phase 2: Deploy Backend on Render

### 2.1 Create Render Web Service

1. Go to https://dashboard.render.com
2. Click **"New +"** → **"Web Service"**
3. Select **"Deploy an existing repository"** → connect GitHub
4. Search and select `DeepNexus` repository

### 2.2 Configure Web Service

Fill in these fields:

| Field | Value |
|-------|-------|
| **Name** | `deepnexus-api` |
| **Root Directory** | `.` (leave empty) |
| **Runtime** | `Python 3.11` |
| **Build Command** | `pip install -r backend/requirements.txt` |
| **Start Command** | `python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000` |
| **Instance Type** | `Standard` (free tier available) |

### 2.3 Add Environment Variables

Click **"Environment"** and add these:

```
PYTHONUNBUFFERED=true
PYTHONPATH=/backend
DEBUG=false
ENVIRONMENT=production
LOG_LEVEL=info
NVIDIA_API_KEY=<your-nvidia-api-key>
CORS_ORIGINS=https://<your-vercel-frontend-url>.vercel.app
```

Replace `<your-vercel-frontend-url>` with your Vercel domain (you'll update this after frontend deploy).

### 2.4 Deploy

1. Scroll down and click **"Create Web Service"**
2. Wait 3-5 minutes for build to complete
3. Once deployed, copy the backend URL (e.g., `https://deepnexus-api.onrender.com`)

**⚠️ Note:** On Render's free tier, services spin down after 15 min of inactivity. For production, upgrade to Starter/Pro.

---

## Phase 3: Deploy Frontend on Vercel

### 3.1 Import Project to Vercel

1. Go to https://vercel.com/new
2. Select **"Import Git Repository"**
3. Search for `DeepNexus` and select it

### 3.2 Configure Build Settings

Set these fields:

| Field | Value |
|-------|-------|
| **Project Name** | `deepnexus` |
| **Framework Preset** | `React` |
| **Root Directory** | `frontend` |
| **Build Command** | `npm run build` |
| **Output Directory** | `dist` |
| **Install Command** | `npm install` |

### 3.3 Add Environment Variables

Click **"Environment Variables"** and add:

```
VITE_API_BASE=https://deepnexus-api.onrender.com
```

Use the Render backend URL from Phase 2.3.

### 3.4 Deploy

1. Click **"Deploy"**
2. Wait 2-3 minutes for build
3. Once complete, you'll get a Vercel URL (e.g., `https://deepnexus.vercel.app`)

---

## Phase 4: Update Backend CORS

### 4.1 Update Render Environment Variable

1. Go back to Render dashboard → select `deepnexus-api` service
2. Go to **"Environment"**
3. Update `CORS_ORIGINS` to:

```
CORS_ORIGINS=https://<your-vercel-url>.vercel.app
```

Example: `CORS_ORIGINS=https://deepnexus.vercel.app`

4. Click **"Save"** (automatic redeploy)

---

## Phase 5: Test Deployment

### 5.1 Health Check

```bash
curl https://deepnexus-api.onrender.com/api/health
```

Expected response:
```json
{"status": "ok"}
```

### 5.2 Test API from Frontend

1. Open https://deepnexus.vercel.app in browser
2. Open Developer Console (F12)
3. Try using any feature (e.g., Study Mode, Data Engine)
4. Check Network tab for `/api/*` requests
5. Verify they return 200 status

### 5.3 Check Logs

**Backend logs on Render:**
- Go to Render dashboard → select service → **"Logs"** tab

**Frontend logs on Vercel:**
- Go to Vercel dashboard → select project → **"Deployments"** → click latest → **"Logs"**

---

## Common Issues & Solutions

### Issue: CORS Error in Console

**Error:** `Access to XMLHttpRequest blocked by CORS policy`

**Solution:**
1. Verify `CORS_ORIGINS` env var in Render includes your Vercel URL
2. Redeploy Render service after updating
3. Hard refresh frontend (Ctrl+Shift+R / Cmd+Shift+R)

### Issue: 502 Bad Gateway from Render

**Cause:** Service crashed during build or runtime

**Solution:**
1. Check Render logs for error details
2. Verify all required packages in `backend/requirements.txt`
3. Check environment variables are set
4. Try rebuilding: go to **"Manual Deploy"** → **"Deploy latest commit"**

### Issue: Frontend Shows Blank Page or API Calls Fail

**Cause:** `VITE_API_BASE` not set or incorrect

**Solution:**
1. Verify env var in Vercel project settings
2. Check that URL includes `https://` and correct domain
3. Redeploy frontend: go to Vercel → **"Deployments"** → **"Redeploy"**

### Issue: Torch/Large Dependencies Failing on Render

**Cause:** Build exceeds time/memory limits (torch ~2GB)

**Solution:**
1. Upgrade to Render Starter ($7/month) for more resources
2. Or remove unused dependencies from `requirements.txt`
3. Use pre-built wheels: add to build command:
   ```
   pip install --prefer-binary -r backend/requirements.txt
   ```

---

## Post-Deployment

### Monitoring

- **Render Dashboard:** Check CPU/Memory usage, logs
- **Vercel Dashboard:** Check build time, edge function errors
- **Application:** Monitor user reports or errors

### Updates

To deploy new code:
1. Commit and push to `main` branch
2. Both Render and Vercel auto-redeploy on push
3. Check deployment status in respective dashboards

### Rollback

If deployment fails:
- **Render:** Go to **"Deployments"** → select previous → **"Redeploy"**
- **Vercel:** Go to **"Deployments"** → select previous → click **"..."** → **"Promote to Production"**

---

## Production Checklist

- [ ] Backend service running on Render
- [ ] Frontend deployed on Vercel
- [ ] Environment variables configured on both
- [ ] CORS properly configured
- [ ] Health check passing
- [ ] API calls working from frontend
- [ ] Nvidia API key set (if using Study Mode)
- [ ] Error logs checked
- [ ] Database migrations run (if applicable)
- [ ] SSL certificates valid (auto-handled by Render/Vercel)

---

## Security Notes

- Never commit `.env` files with secrets
- Use Render/Vercel environment variable UI for sensitive keys
- Set `DEBUG=false` in production
- Regularly rotate API keys
- Monitor logs for suspicious activity
- Use HTTPS only (enforced by both platforms)

---

## Support

- Render Docs: https://render.com/docs
- Vercel Docs: https://vercel.com/docs
- FastAPI Docs: https://fastapi.tiangolo.com
- React Docs: https://react.dev
