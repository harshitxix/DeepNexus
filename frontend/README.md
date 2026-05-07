# DeepNexus Frontend

This folder contains the Vite-based React SPA for DeepNexus.

For the full project overview, deployment setup, and architecture notes, see the root [README](../README.md).

## Local Development

```bash
cd frontend
npm install
npm run dev
```

## Production Build

```bash
cd frontend
npm run build
```

## Environment Variables

- `VITE_API_BASE` - Backend API base URL.
- For local development, the app can fall back to the Vite dev proxy.

## Deployment

This frontend is deployed on Vercel. The deployment config lives in [vercel.json](../vercel.json).
