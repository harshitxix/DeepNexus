<div align="center">

<br/>


#  DeepNexus

### *Learning Neural Networks, Visualized.*

<br/>

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-deep--nexus--three.vercel.app-6366f1?style=for-the-badge&logoColor=white)](https://deep-nexus-three.vercel.app/)
&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](https://opensource.org/licenses/MIT)
&nbsp;
[![React](https://img.shields.io/badge/React-19-38bdf8?style=for-the-badge&logo=react)](https://react.dev/)
&nbsp;
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)

<br/>

> An interactive deep learning and neural network visualization platform.
> Learn complex AI concepts through real-time visualizations, mathematical breakdowns, and hands-on experimentation.

<br/>

---

</div>

<br/>

## ✦ What is DeepNexus?

DeepNexus is an **educational visualization platform** for deep learning and neural networks. Instead of reading static textbooks, you interact — adjust weights, watch gradients flow, train networks in your browser, and see the math come alive. Built for students, educators, and curious minds.

<br/>

---

## ◈ Interactive Learning Modules

<br/>

| Module | Description |
|--------|-------------|
| 🔵 **Perceptron** | Single neuron modeling with activation functions and live decision boundaries |
| ➡️ **Forward Propagation** | Signal flow visualization through network layers, step by step |
| ⬅️ **Backward Propagation** | Gradient computation, weight updates, and vanishing gradient demos |
| 🧠 **Multilayer Perceptron** | Deep network architecture with live training curves |
| 👁️ **Computer Vision** | Image processing, convolution, and CNN concepts with visual demos |
| 🔄 **RNN & LSTM** | Sequence modeling with hidden state evolution and memory visualization |

<br/>

---

## ◈ Study Mode

Each module ships with a full **Study Mode** — a curated learning companion that includes:

- 📐 Mathematical formulas and step-by-step derivations
- 📊 Activation function comparisons — Sigmoid, ReLU, Tanh
- 🧩 Interactive graphs and visual explainers
- 🌍 Real-world use cases and applications
- 💻 Code examples and implementation guidance

<br/>

---

## ◈ Visualizations

```
┌─────────────────────────────────────────────────────────┐
│  Plotly Interactive Graphs   │  Network Signal Diagrams  │
│  Gradient Decay Curves       │  Decision Boundaries      │
│  Training Accuracy & Loss    │  Memory Persistence (RNN) │
└─────────────────────────────────────────────────────────┘
```

<br/>

---

## ◈ Tech Stack

<div align="center">

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 19 + Vite 8 |
| **Styling** | CSS3, GSAP, Motion |
| **Visualizations** | Plotly.js, React-Plotly |
| **Backend** | FastAPI + Uvicorn |
| **Computation** | NumPy, OpenCV |
| **Deployment** | Vercel (Frontend) + Render (Backend) |

</div>

<br/>

---

## ◈ Project Structure

```
DeepNexus/
│
├── frontend/                        ← React + Vite SPA
│   ├── src/
│   │   ├── App.jsx                  ← Main app component
│   │   ├── api.js                   ← API client
│   │   ├── components/              ← Reusable UI components
│   │   ├── ElasticSlider.jsx
│   │   ├── ShinyText.jsx
│   │   ├── StaggeredMenu.jsx
│   │   ├── PillNav.jsx
│   │   └── VariableProximity.jsx
│   ├── package.json
│   └── vite.config.js
│
├── backend/                         ← FastAPI Server
│   ├── app/
│   │   ├── main.py                  ← App entry point
│   │   ├── routers/                 ← API route handlers
│   │   │   ├── perceptron.py
│   │   │   ├── forward_prop.py
│   │   │   ├── backward_prop.py
│   │   │   ├── mlp.py
│   │   │   ├── opencv.py
│   │   │   └── rnn.py
│   │   ├── services/                ← Business logic
│   │   │   ├── perceptron_service.py
│   │   │   ├── forward_service.py
│   │   │   ├── backward_service.py
│   │   │   ├── mlp_service.py
│   │   │   └── rnn_service.py
│   │   └── schemas/                 ← Data models
│   └── requirements.txt
│
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
├── vercel.json
├── render.yaml
└── .github/workflows/deploy.yml
```

<br/>

---

## ◈ Getting Started

### Prerequisites

```
Node.js 18+   ·   Python 3.11+   ·   npm or yarn   ·   Git
```

### 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/deepnexus.git
cd DeepNexus
```

### 2 — Backend Setup

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate

# Install dependencies
cd backend
pip install -r requirements.txt
cd ..
```

### 3 — Frontend Setup

```bash
cd frontend
npm install
cd ..
```

<br/>

---

## ◈ Running Locally

### Option A — Separate Terminals

```bash
# Terminal 1 — Backend
cd backend
python -m uvicorn app.main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm run dev
```

> Open **http://localhost:5173** in your browser.

### Option B — Docker Compose

```bash
docker-compose up --build
```

> Open **http://localhost** (via Nginx reverse proxy).

### Option C — Single Docker Container

```bash
docker build -t deepnexus .
docker run -p 8000:8000 deepnexus
```

<br/>

---

## ◈ API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Server health check |
| `POST` | `/api/perceptron/predict` | Single neuron prediction |
| `POST` | `/api/forward/run` | Forward pass through layers |
| `POST` | `/api/backward/step` | Gradient computation + backprop |
| `POST` | `/api/mlp/summary` | Network config and parameter count |
| `POST` | `/api/mlp/train` | Train on AND / OR / XOR truth tables |
| `GET` | `/api/opencv/info` | Available CV operations |
| `POST` | `/api/opencv/process` | Image processing operations |
| `POST` | `/api/rnn/sentiment` | Sentiment analysis on text |
| `POST` | `/api/rnn/state` | Hidden state evolution tracking |

> Interactive Swagger docs available at **`http://localhost:8000/docs`**

<br/>

---

## ◈ Study Mode — Topics Covered

<details>
<summary><b>🔵 Perceptron</b></summary>
<br/>

- Biological inspiration and neuron model
- Activation functions: Sigmoid, ReLU, Tanh
- Decision boundaries and binary classification
- TensorFlow implementation walkthrough

</details>

<details>
<summary><b>➡️ Forward Propagation</b></summary>
<br/>

- Layer-by-layer signal flow breakdown
- Formula: `A⁽ⁿ⁾ = σ(W⁽ⁿ⁾A⁽ⁿ⁻¹⁾ + b⁽ⁿ⁾)`
- Pre-activation (Z) vs post-activation (A) values
- Practical implementation guidance

</details>

<details>
<summary><b>⬅️ Backward Propagation</b></summary>
<br/>

- Chain rule and gradient computation
- Error propagation through layers
- Weight update mechanisms
- Vanishing & exploding gradient problem
- Solutions: smart initialization, ReLU, batch normalization

</details>

<details>
<summary><b>🧠 Multilayer Perceptron</b></summary>
<br/>

- Fully connected network architecture
- Parameter growth analysis
- Loss functions: BCE, MSE, MAE
- Optimizers: SGD, Adam
- Training dynamics and convergence behavior

</details>

<details>
<summary><b>👁️ Computer Vision</b></summary>
<br/>

- Image transformations and enhancement
- Convolution, filtering, and feature extraction
- Edge detection, corner detection, SIFT
- Morphological operations
- Deep learning architectures: CNN, GAN, VAE, ViT

</details>

<details>
<summary><b>🔄 RNN & LSTM</b></summary>
<br/>

- Sequential processing and memory mechanisms
- Recurrent equations and network unrolling
- Backpropagation through time (BPTT)
- Gated architectures: LSTM, GRU
- NLP and time-series applications

</details>

<br/>

---

## ◈ Deployment

DeepNexus uses **Vercel** for the frontend and **Render** for the backend API.

```
Frontend → Vercel      (controlled via vercel.json)
Backend  → Render      (controlled via render.yaml)
Docker   → Optional    (for self-hosting or local container testing)
```

### Environment Variables

**Frontend** — create `frontend/.env.production`:
```env
VITE_API_URL=https://your-backend-domain.com
VITE_ENABLE_STUDY_MODE=true
```

**Backend** — create `backend/.env`:
```env
DEBUG=false
CORS_ORIGINS=https://your-frontend-domain.com
LOG_LEVEL=info
```

<br/>

---

## ◈ Troubleshooting

<details>
<summary><b>Frontend not connecting to backend</b></summary>
<br/>

```bash
# Confirm backend is running
curl http://localhost:8000/api/health

# Then update your frontend env
# frontend/.env → VITE_API_URL=http://localhost:8000
```

</details>

<details>
<summary><b>Docker build fails</b></summary>
<br/>

```bash
# Clear Docker cache and rebuild
docker system prune -a
docker build -t deepnexus .
```

</details>

<details>
<summary><b>Port already in use</b></summary>
<br/>

```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill it (Windows)
taskkill /PID <PID> /F
```

</details>

<br/>

---

## ◈ Roadmap

```
✅ 6 core interactive modules
✅ Study Mode with math and code
✅ FastAPI backend with Plotly visualizations
✅ Docker + CI/CD pipeline
✅ Vercel + Render deployment

⬜ GPU acceleration for larger networks
⬜ Database for saving training sessions
⬜ User authentication and profiles
⬜ Advanced visualization modes
⬜ Mobile-responsive design improvements
⬜ Multi-language support
⬜ Video tutorial integration
⬜ Jupyter notebook embedding
```

<br/>

---

## ◈ Contributing

Contributions are warmly welcome.

```bash
# 1. Fork the repository
# 2. Create your feature branch
git checkout -b feature/your-feature-name

# 3. Commit your changes
git commit -m "feat: add your feature"

# 4. Push and open a Pull Request
git push origin feature/your-feature-name
```

<br/>

---

## ◈ Learning Resources

- 📖 [Deep Learning Book](http://deeplearningbook.org/) — Goodfellow, Bengio, Courville
- ✍️ [Chris Olah's Blog](https://colah.github.io/) — Visual explanations of neural nets
- 🔢 [Backpropagation Derivation](https://medium.com/analytics-vidhya/understanding-backpropagation-85a5b4f72ae6)
- ⚡ [FastAPI Documentation](https://fastapi.tiangolo.com/)
- ⚛️ [React Documentation](https://react.dev/)

<br/>

---

## ◈ License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

<br/>

---

<div align="center">

```
╔══════════════════════════════════════════╗
║   Built for deep learning enthusiasts    ║
║   and students who learn by doing.       ║
╚══════════════════════════════════════════╝
```

[![Live Demo](https://img.shields.io/badge/🌐%20Try%20It%20Live-deep--nexus--three.vercel.app-6366f1?style=for-the-badge)](https://deep-nexus-three.vercel.app/)

<br/>

**DeepNexus** — *Learning Neural Networks, Visualized.*

Made with ❤️ for the open-source community

</div>
