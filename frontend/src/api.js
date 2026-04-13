const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000/api'

async function request(path, options = {}) {
  const isFormData = options.body instanceof FormData
  const baseHeaders = isFormData ? {} : { 'Content-Type': 'application/json' }
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { ...baseHeaders, ...(options.headers || {}) },
    ...options,
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`API ${res.status}: ${text}`)
  }

  return res.json()
}

export const api = {
  health: () => request('/health'),
  perceptronPredict: (payload) => request('/perceptron/predict', { method: 'POST', body: JSON.stringify(payload) }),
  forwardRun: (payload) => request('/forward/run', { method: 'POST', body: JSON.stringify(payload) }),
  backwardStep: (payload) => request('/backward/step', { method: 'POST', body: JSON.stringify(payload) }),
  mlpSummary: (payload) => request('/mlp/summary', { method: 'POST', body: JSON.stringify(payload) }),
  rnnInfo: () => request('/rnn/info'),
  rnnGenerateText: (payload) => request('/rnn/generate-text', { method: 'POST', body: JSON.stringify(payload) }),
  rnnAnalyzeEmotion: (payload) => request('/rnn/analyze-emotion', { method: 'POST', body: JSON.stringify(payload) }),
  cnnLabInfo: () => request('/cnn-lab/info'),
  cnnLabPredict: (formData) => request('/cnn-lab/predict', { method: 'POST', body: formData }),
  cnnLabFeatureMaps: (formData) => request('/cnn-lab/feature-maps', { method: 'POST', body: formData }),
  cnnLabGradCam: (formData) => request('/cnn-lab/grad-cam', { method: 'POST', body: formData }),
  hopfieldLibrary: (side = 10) => request(`/hopfield/library?side=${side}`),
  hopfieldRecall: (payload) => request('/hopfield/recall', { method: 'POST', body: JSON.stringify(payload) }),
}
