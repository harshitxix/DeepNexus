# DeepNexus 🧠

An interactive deep learning and neural network visualization platform built with React, FastAPI, and Plotly. Learn neural networks through interactive modules, mathematical explanations, and real-time visualizations.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tech Stack](https://img.shields.io/badge/Stack-React%20%2B%20FastAPI-blue)](#tech-stack)

---

## 🚀 Features

### Interactive Learning Modules
- **Perceptron**: Single neuron modeling with activation functions and decision boundaries
- **Forward Propagation**: Signal flow visualization through network layers
- **Backward Propagation**: Gradient computation and weight update mechanisms
- **Multilayer Perceptron (MLP)**: Deep network architecture with training curves
- **Computer Vision**: Image processing and CNN concepts with visual demonstrations
- **RNN & LSTM**: Sequence modeling with hidden state evolution visualization

### Study Mode
Comprehensive learning materials for each topic including:
- Detailed mathematical formulas and equations
- Activation function comparisons (Sigmoid, ReLU, Tanh)
- Interactive graphs and visualizations
- Real-world applications and best practices
- Implementation guidance with code examples

### Advanced Visualizations
- **Plotly-based graphs** for interactive exploration
- **Network signal diagrams** showing data flow
- **Gradient decay visualization** demonstrating vanishing/exploding gradients
- **Training curves** tracking accuracy and loss
- **Decision boundaries** for classification tasks
- **Memory persistence graphs** for RNN/LSTM modules

---

## 📋 Project Structure

```
DeepNexus/
├── frontend/                          # React + Vite SPA
│   ├── src/
│   │   ├── App.jsx                   # Main app component
│   │   ├── App.css                   # Global styles
│   │   ├── api.js                    # API client
│   │   ├── components/               # Reusable UI components
│   │   ├── ElasticSlider.jsx         # Custom slider
│   │   ├── ShinyText.jsx             # Text animations
│   │   ├── StaggeredMenu.jsx         # Menu animations
│   │   ├── PillNav.jsx               # Navigation pills
│   │   ├── VariableProximity.jsx     # Proximity effects
│   │   └── assets/                   # Media files
│   ├── package.json
│   ├── vite.config.js
│   └── index.html
│
├── backend/                           # FastAPI server
│   ├── app/
│   │   ├── main.py                   # App entry point
│   │   ├── routers/                  # API route handlers
│   │   │   ├── perceptron.py
│   │   │   ├── forward_prop.py
│   │   │   ├── backward_prop.py
│   │   │   ├── mlp.py
│   │   │   ├── opencv.py
│   │   │   ├── rnn.py
│   │   │   └── health.py
│   │   ├── services/                 # Business logic
│   │   │   ├── perceptron_service.py
│   │   │   ├── forward_service.py
│   │   │   ├── backward_service.py
│   │   │   ├── mlp_service.py
│   │   │   ├── rnn_service.py
│   │   │   └── math_utils.py
│   │   ├── schemas/                  # Data models
│   │   └── __init__.py
│   ├── requirements.txt
│   └── legacy_streamlit/             # Archive (previous version)
│
├── Dockerfile                         # Docker containerization
├── docker-compose.yml                # Multi-container setup
├── nginx.conf                         # Reverse proxy config
├── vercel.json                        # Vercel deployment
├── render.yaml                        # Render deployment
├── .github/workflows/deploy.yml       # CI/CD automation
└── README.md                          # This file
```

### Deployment Notes

DeepNexus now uses **Render** for the backend API and **Vercel** for the frontend.

- `render.yaml` controls the backend deploy on Render.
- `vercel.json` controls the frontend deploy on Vercel.
- The Docker files are optional and only needed if you want to self-host, test containers locally, or keep a Docker-based deployment path.

If you are only using Render and Vercel, you do not need the Docker-based deployment path for normal production usage.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | React 19 + Vite 8 |
| **Styling** | CSS3 with animations (GSAP, Motion) |
| **Visualizations** | Plotly.js, React-Plotly |
| **Backend** | FastAPI, Uvicorn |
| **Computation** | NumPy, OpenCV |
| **CLI** | PowerShell / Bash |

---

## 📦 Installation

### Prerequisites
- **Node.js** 18+ (for frontend)
- **Python** 3.11+ (for backend)
- **npm** or **yarn** (for dependency management)
- **Git** (for version control)

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/deepnexus.git
cd DeepNexus
```

### Backend Setup

```bash
# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\Activate.ps1
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
cd backend
pip install -r requirements.txt
cd ..
```

### Frontend Setup

```bash
cd frontend
npm install
cd ..
```

---

## 🚀 Running Locally

### Option 1: Separate Terminals

**Terminal 1 - Backend:**
```bash
cd backend
$env:PYTHONPATH="$PWD"
python -m uvicorn app.main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

**Access:** http://localhost:5173

### Option 2: Docker Compose

```bash
docker-compose up --build
```

**Access:** http://localhost (Nginx proxy)

### Option 3: Docker (Single Container)

```bash
docker build -t deepnexus .
docker run -p 8000:8000 deepnexus
```

---

## 📡 API Endpoints

### Health Check
- `GET /api/health` - Server health status

### Perceptron Module
- `POST /api/perceptron/predict` - Single neuron prediction with activation

### Forward Propagation
- `POST /api/forward/run` - Forward pass through network layers

### Backward Propagation
- `POST /api/backward/step` - Gradient computation and backprop step

### MLP (Multilayer Perceptron)
- `POST /api/mlp/summary` - Network configuration and parameter analysis
- `POST /api/mlp/train` - Training on truth tables (AND, OR, XOR)

### Computer Vision
- `GET /api/opencv/info` - Available CV operations
- `POST /api/opencv/process` - Image processing operations

### RNN & LSTM
- `POST /api/rnn/sentiment` - Sentiment analysis on text input
- `POST /api/rnn/state` - Hidden state evolution tracking

### Swagger Documentation
- `http://localhost:8000/docs` - Interactive API documentation

---

## 🎓 Study Mode Content

The Study Mode provides comprehensive learning material for each topic:

### Perceptron
- Biological inspiration and model definition
- Activation functions (Sigmoid, ReLU, Tanh)
- Decision boundaries and classification
- Implementation with TensorFlow

### Forward Propagation
- Layer-by-layer signal flow
- Mathematical formulation: `A⁽ⁿ⁾ = σ(W⁽ⁿ⁾A⁽ⁿ⁻¹⁾ + b⁽ⁿ⁾)`
- Pre-activation (Z) and post-activation (A) values
- Practical implementation guidance

### Backward Propagation
- Chain rule and gradient computation
- Error propagation through layers
- Weight update mechanisms
- Vanishing/exploding gradient problem
- Solutions: initialization, ReLU, batch norm

### Multilayer Perceptron
- Fully connected architecture
- Parameter growth analysis
- Loss functions (BCE, MSE, MAE)
- Optimization algorithms (SGD, Adam)
- Training dynamics and convergence

### Computer Vision
- Image transformations and enhancement
- Convolution and filtering
- Feature extraction (edges, corners, SIFT)
- Morphological operations
- Deep learning models (CNN, GAN, VAE, ViT)

### RNN & LSTM
- Sequential processing and memory
- Recurrent equations and unrolling
- Backpropagation through time (BPTT)
- Gated architectures (LSTM, GRU)
- Applications in NLP and time series

---

## 🧪 Module Details

Each module provides:
1. **Interactive controls** for parameter adjustment
2. **Real-time visualizations** using Plotly
3. **Numerical outputs** showing computed values
4. **Mathematical explanations** of underlying algorithms
5. **Best practices** and common pitfalls

### Perceptron Example
```javascript
// Input: 2D features and weights
// Output: Binary classification with decision boundary
// Visualization: Sigmoid activation curve + contour plot
```

### MLP Example
```javascript
// Input: Network architecture (3 hidden layers)
// Output: Parameter count, training curves
// Visualization: Loss and accuracy over epochs
```

---

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t deepnexus:latest .
```

### Run Locally
```bash
docker run -p 8000:8000 deepnexus:latest
```

### Push to Docker Hub
```bash
docker tag deepnexus:latest YOUR_USERNAME/deepnexus:latest
docker push YOUR_USERNAME/deepnexus:latest
```

### Docker Compose
```bash
docker-compose up --build
```

---

## 🌐 Deployment Options

### Recommended: Vercel (Frontend) + Railway (Backend)
- **Setup time:** 15 minutes
- **Cost:** Free tier available
- **Scalability:** Excellent
- Details: See [DEPLOYMENT.md](./DEPLOYMENT.md)

### Docker on AWS/DigitalOcean
- **Setup time:** 30-45 minutes
- **Cost:** ~$5/month
- **Control:** Full server access
- Details: See [DEPLOYMENT.md](./DEPLOYMENT.md)

### Render.com (All-in-One)
- **Setup time:** 10 minutes
- **Cost:** Free tier available
- **Ease:** Very simple
- Details: See [DEPLOYMENT.md](./DEPLOYMENT.md)

### GitHub Actions CI/CD
Automated deployment on every push to main:
```bash
# Builds Docker image and pushes to registry
# See .github/workflows/deploy.yml
```

See [DEPLOYMENT.md](./DEPLOYMENT.md) for step-by-step guides and [DEPLOYMENT_CHECKLIST.md](./DEPLOYMENT_CHECKLIST.md) for pre/post-deployment verification.

---

## 🔧 Configuration

### Frontend Environment Variables
Create `frontend/.env.production`:
```env
VITE_API_URL=https://your-backend-domain.com
VITE_ENABLE_STUDY_MODE=true
```

### Backend Environment Variables
Create `backend/.env`:
```env
DEBUG=false
CORS_ORIGINS=https://your-frontend-domain.com
LOG_LEVEL=info
```

See `.env.example` files for complete configurations.

---

## 📚 Learning Resources

- [Deep Learning Book](http://deeplearningbook.org/) - Ian Goodfellow, Yoshua Bengio
- [Neural Networks Explained](https://colah.github.io/) - Chris Olah's Blog
- [Backpropagation Derivation](https://medium.com/analytics-vidhya/understanding-backpropagation-85a5b4f72ae6)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)

---

## 🤝 Contributing

Contributions welcome! Follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🐛 Troubleshooting

### Frontend not connecting to backend
```bash
# Check backend is running on port 8000
curl http://localhost:8000/api/health

# Update VITE_API_URL in frontend/.env
```

### Docker build fails
```bash
# Clear Docker cache
docker system prune -a

# Rebuild
docker build -t deepnexus .
```

### Port already in use
```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process (Windows)
taskkill /PID <PID> /F
```

See [DEPLOYMENT.md](./DEPLOYMENT.md) for more troubleshooting guides.

---

## 📧 Support & Contact

- GitHub Issues: [Report bugs](https://github.com/YOUR_USERNAME/deepnexus/issues)
- Email: your-email@example.com
- Documentation: See [DEPLOYMENT.md](./DEPLOYMENT.md) and [DEPLOYMENT_CHECKLIST.md](./DEPLOYMENT_CHECKLIST.md)

---

## 🎯 Roadmap

- [ ] GPU acceleration for larger networks
- [ ] Database for storing training sessions
- [ ] User authentication and profiles
- [ ] Advanced visualization modes
- [ ] Mobile responsive design improvements
- [ ] Multi-language support
- [ ] Video tutorials
- [ ] Jupyter notebook integration

---

## ✨ Highlights

- **Educational:** Learn deep learning through interactive visualization
- **Modern Stack:** React + FastAPI + Plotly for seamless experience
- **Production-Ready:** Docker, CI/CD, multiple deployment options
- **Comprehensive:** 6 major modules + study mode + API documentation
- **Open Source:** MIT licensed, community contributions welcome

---

Made with ❤️ for deep learning enthusiasts and students

**DeepNexus** - Learning Neural Networks, Visualized.
