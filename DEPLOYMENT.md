# DeepNexus Deployment Guide

This guide covers deploying DeepNexus (React frontend + FastAPI backend) to production.

## Project Structure
```
DeepNexus/
├── frontend/          # React + Vite SPA
│   ├── src/
│   ├── package.json
│   └── vite.config.js
└── backend/           # FastAPI server
    ├── app/
    ├── requirements.txt
    └── main.py
```

---

## Deployment Options

### **Option 1: Docker (Recommended for Full Control)**

#### 1.1 Create Docker configuration files

Create `Dockerfile` in project root:

```dockerfile
# Multi-stage build
FROM node:18-alpine as frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Backend setup
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend
COPY backend/ ./backend/

# Copy built frontend
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/backend

# Expose ports
EXPOSE 8000

# Start server
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create `.dockerignore`:
```
node_modules/
__pycache__/
.venv/
.git/
.github/
.env
.DS_Store
*.pyc
dist/
build/
.next/
```

#### 1.2 Build and run locally:
```bash
docker build -t deepnexus:latest .
docker run -p 8000:8000 deepnexus:latest
```

#### 1.3 Push to Docker Hub:
```bash
docker tag deepnexus:latest YOUR_DOCKERHUB_USERNAME/deepnexus:latest
docker push YOUR_DOCKERHUB_USERNAME/deepnexus:latest
```

---

### **Option 2: Vercel (Frontend) + Railway (Backend)**

#### Frontend on Vercel:

1. **Build the frontend only:**
   ```bash
   cd frontend
   npm run build
   ```

2. **Create `vercel.json` in frontend root:**
   ```json
   {
     "buildCommand": "npm run build",
     "outputDirectory": "dist",
     "env": {
       "VITE_API_URL": "@vite-api-url"
     }
   }
   ```

3. **Push to GitHub and deploy to Vercel:**
   - Go to https://vercel.com/new
   - Import your GitHub repo
   - Set `VITE_API_URL` environment variable (e.g., `https://deepnexus-backend.railway.app`)
   - Deploy

#### Backend on Railway:

1. **Create `railway.json` in backend root:**
   ```json
   {
     "deploy": {
       "builder": "dockerfile",
       "dockerfile": "Dockerfile.railway"
     }
   }
   ```

2. **Create `Dockerfile.railway` in backend:**
   ```dockerfile
   FROM python:3.11-slim
   
   RUN apt-get update && apt-get install -y \
       libsm6 libxext6 libxrender-dev \
       && rm -rf /var/lib/apt/lists/*
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   
   ENV PYTHONUNBUFFERED=1
   ENV PYTHONPATH=/app
   
   EXPOSE 8000
   CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

3. **Push to GitHub and deploy on Railway:**
   - Go to https://railway.app
   - Create new project → Deploy from GitHub
   - Select your repo
   - Set `PORT=8000` environment variable
   - Deploy

---

### **Option 3: AWS EC2 with Nginx Reverse Proxy**

#### 3.1 Launch EC2 Instance

1. Create Ubuntu 22.04 LTS instance
2. Configure security groups to allow ports 80, 443, 8000

#### 3.2 Setup on Server

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install Python dependencies
sudo apt install -y python3.11 python3.11-venv python3-pip
sudo apt install -y libsm6 libxext6 libxrender-dev

# Install Nginx
sudo apt install -y nginx

# Clone your repo
git clone https://github.com/YOUR_USERNAME/deepnexus.git
cd deepnexus
```

#### 3.3 Setup Backend

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install backend dependencies
cd backend
pip install -r requirements.txt
cd ..
```

#### 3.4 Setup Frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

#### 3.5 Configure Nginx

```bash
sudo nano /etc/nginx/sites-available/deepnexus
```

Add configuration:
```nginx
upstream backend {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        root /home/ubuntu/deepnexus/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api/ {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/deepnexus /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### 3.6 Run Backend with Systemd

```bash
sudo nano /etc/systemd/system/deepnexus.service
```

Add:
```ini
[Unit]
Description=DeepNexus Backend
After=network.target

[Service]
Type=notify
User=ubuntu
WorkingDirectory=/home/ubuntu/deepnexus/backend
Environment="PATH=/home/ubuntu/deepnexus/venv/bin"
Environment="PYTHONPATH=/home/ubuntu/deepnexus/backend"
ExecStart=/home/ubuntu/deepnexus/venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable deepnexus
sudo systemctl start deepnexus
```

#### 3.7 SSL Certificate (Let's Encrypt)

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

### **Option 4: Render (One-Click Deployment)**

#### 4.1 Prepare project structure

Create `render.yaml` in root:
```yaml
services:
  - type: web
    name: deepnexus-backend
    runtime: python
    buildCommand: "pip install -r backend/requirements.txt"
    startCommand: "python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"
    envVars:
      - key: PYTHONPATH
        value: /backend
      - key: PYTHONUNBUFFERED
        value: true
    routes:
      - path: /api
        destination: /api

  - type: static
    name: deepnexus-frontend
    buildCommand: "cd frontend && npm install && npm run build"
    startCommand: ""
    staticPublishPath: /dist
    routes:
      - path: /
        destination: /index.html
```

#### 4.2 Deploy

1. Push to GitHub
2. Go to render.com
3. Click "New +" → "Web Service"
4. Select your repo
5. Configure environment and deploy

---

### **Option 5: DigitalOcean App Platform**

Create `.do/app.yaml`:
```yaml
name: deepnexus
services:
  - name: frontend
    github:
      repo: YOUR_USERNAME/deepnexus
      branch: main
    build_command: "cd frontend && npm install && npm run build"
    http_port: 3000
    source_dir: frontend/dist

  - name: backend
    github:
      repo: YOUR_USERNAME/deepnexus
      branch: main
    build_command: "pip install -r backend/requirements.txt"
    run_command: "python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8080"
    http_port: 8080
    source_dir: backend
```

Deploy via DigitalOcean dashboard.

---

## Environment Variables

Create `.env` in frontend/.env.production:
```
VITE_API_URL=https://your-backend-domain.com
```

Create `.env` in backend (for production):
```
DEBUG=false
DATABASE_URL=postgresql://user:pass@db:5432/deepnexus
CORS_ORIGINS=https://your-frontend-domain.com
```

---

## Database Setup (if needed in future)

```bash
# PostgreSQL with Docker
docker run -d \
  --name deepnexus-db \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=deepnexus \
  -p 5432:5432 \
  postgres:15-alpine
```

---

## Performance Optimization

### Frontend:
```bash
npm run build
# Check bundle size
npm install -g vite
vite build --report
```

### Backend:
```bash
# Use Gunicorn for production
pip install gunicorn
gunicorn backend.app.main:app -w 4 -b 0.0.0.0:8000
```

---

## Monitoring & Logging

### Backend logs:
```bash
sudo journalctl -u deepnexus -f
```

### CPU/Memory monitoring:
```bash
# On server
htop
# In Docker
docker stats
```

---

## Recommended Deployment Path

**Beginner-Friendly:** Vercel (frontend) + Railway (backend)
**Full Control:** Docker on AWS EC2 or DigitalOcean
**Maximum Simplicity:** Render.com platform

---

## Domain Setup

1. Purchase domain (Namecheap, GoDaddy, etc.)
2. Point to your hosting provider's nameservers
3. Configure SSL/TLS certificates
4. Update API URLs in frontend .env

---

## Maintenance

```bash
# Update dependencies
cd frontend && npm update
cd ../backend && pip install --upgrade -r requirements.txt

# Backups
docker exec deepnexus-db pg_dump -U postgres deepnexus > backup.sql

# Health check
curl https://your-domain.com/api/health
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| 503 Service Unavailable | Check backend service logs |
| CORS errors | Verify frontend URL in backend CORS settings |
| Frontend 404 | Ensure Nginx try_files redirects to index.html |
| Memory issues | Increase instance RAM or enable swap |

