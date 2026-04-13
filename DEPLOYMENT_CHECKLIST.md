# DeepNexus Deployment Checklist

## Pre-Deployment

- [ ] All code committed to Git
- [ ] No hardcoded secrets or API keys
- [ ] Frontend builds without errors: `npm run build`
- [ ] Backend tests pass (if applicable)
- [ ] Environment variables configured
- [ ] Database migrations completed (if needed)

## Backend Deployment

### Docker Option
- [ ] Dockerfile created and tested locally
- [ ] `docker build -t deepnexus .` succeeds
- [ ] `docker run -p 8000:8000 deepnexus` works
- [ ] Health endpoint responds: `curl http://localhost:8000/api/health`
- [ ] Image pushed to registry (Docker Hub, ECR, etc.)

### Traditional Server Option
- [ ] Python 3.11+ installed on server
- [ ] `requirements.txt` installed: `pip install -r requirements.txt`
- [ ] System dependencies installed (libsm6, libxext6, libxrender-dev)
- [ ] Uvicorn or Gunicorn configured
- [ ] Systemd service file created
- [ ] Service enabled: `sudo systemctl enable deepnexus`

## Frontend Deployment

### Build
- [ ] Frontend builds: `npm run build`
- [ ] Build output in `frontend/dist` folder
- [ ] Static files are minified and optimized

### Hosting Options
- [ ] Environment variable `VITE_API_URL` set to backend URL
- [ ] Files uploaded to hosting (Vercel, Netlify, S3, etc.)
- [ ] Index.html configured for SPA routing

## Infrastructure

### Networking
- [ ] APIs accessible from frontend domain
- [ ] CORS headers properly configured
- [ ] SSL/TLS certificates installed
- [ ] Domain DNS records pointing to servers

### Reverse Proxy (if using Nginx)
- [ ] Nginx configuration updated
- [ ] Static files serving correctly
- [ ] API requests proxying to backend
- [ ] Gzip compression enabled
- [ ] Cache headers configured

## Security

- [ ] `.env` files not in version control
- [ ] Secrets stored in environment variables
- [ ] HTTPS enabled on all endpoints
- [ ] CORS properly restricted
- [ ] Input validation enabled
- [ ] Rate limiting configured

## Monitoring

- [ ] Health check endpoint accessible
- [ ] Logging configured
- [ ] Error tracking setup (Sentry, etc.)
- [ ] CPU/Memory monitoring in place
- [ ] Backup strategy for data

## Testing

- [ ] Frontend loads without errors
- [ ] API endpoints respond correctly
- [ ] Study mode content displays properly
- [ ] All modules load (Perceptron, MLP, CV, RNN, etc.)
- [ ] No console errors in browser
- [ ] Backend logs show normal operation

## Performance

- [ ] Frontend bundle size optimized
- [ ] Images and assets compressed
- [ ] Caching headers appropriate
- [ ] Database queries optimized (if applicable)
- [ ] Response times acceptable (<2s)

## Post-Deployment

- [ ] All URLs updated in documentation
- [ ] Team notified of deployment
- [ ] Backup created before deployment
- [ ] Rollback plan documented
- [ ] Monitor logs for errors
- [ ] Performance metrics collected

## Rollback Plan

If issues occur:
1. Revert application to previous working version
2. Check error logs for root cause
3. Fix issues locally
4. Test thoroughly before re-deploying
5. Document what went wrong and how to prevent it

---

## Quick Deployment Commands

### Docker Local Testing
```bash
docker build -t deepnexus:test .
docker run -p 8000:8000 -e VITE_API_URL=http://localhost:8000 deepnexus:test
```

### Docker Compose
```bash
docker-compose up --build
```

### Deploy to Production
```bash
# Build
docker build -t your-registry/deepnexus:latest .

# Push
docker push your-registry/deepnexus:latest

# Run on server
docker run -d -p 8000:8000 your-registry/deepnexus:latest
```

### Health Check
```bash
curl https://your-domain.com/api/health
```

---

## Support

Deployment issues? Check:
1. DEPLOYMENT.md for detailed guides
2. Backend logs: `docker logs <container-id>` or `journalctl -u deepnexus`
3. Frontend console errors: Open DevTools (F12) → Console
4. Network tab: Check API request URLs and responses
