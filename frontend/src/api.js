// In production, VITE_API_BASE must be set to backend URL
// In development, fallback to relative /api (proxied by vite.config.js)
const PRIMARY_API_BASE = import.meta.env.VITE_API_BASE || '/api'
const API_BASES = PRIMARY_API_BASE === '/api'
  ? [PRIMARY_API_BASE]
  : [PRIMARY_API_BASE]

async function request(path, options = {}) {
  const isFormData = options.body instanceof FormData
  const baseHeaders = isFormData ? {} : { 'Content-Type': 'application/json' }

  let lastFetchError = null
  for (let i = 0; i < API_BASES.length; i += 1) {
    const base = API_BASES[i]
    try {
      const res = await fetch(`${base}${path}`, {
        headers: { ...baseHeaders, ...(options.headers || {}) },
        ...options,
      })

      if (!res.ok) {
        const text = await res.text()
        const error = new Error(`API ${res.status}: ${text}`)
        error.status = res.status

        if (res.status >= 500 && i < API_BASES.length - 1) {
          lastFetchError = error
          continue
        }

        throw error
      }

      return res.json()
    } catch (error) {
      lastFetchError = error

      // Retry only when request itself failed (network/CORS/connection).
      if (!(error instanceof TypeError) || i === API_BASES.length - 1) {
        throw error
      }
    }
  }

  throw lastFetchError || new Error('Unknown API request failure.')
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
  studyAssistantAsk: (payload) => request('/study-assistant/ask', { method: 'POST', body: JSON.stringify(payload) }),
  cnnLabInfo: () => request('/cnn-lab/info'),
  cnnLabPredict: (formData) => request('/cnn-lab/predict', { method: 'POST', body: formData }),
  cnnLabFeatureMaps: (formData) => request('/cnn-lab/feature-maps', { method: 'POST', body: formData }),
  cnnLabGradCam: (formData) => request('/cnn-lab/grad-cam', { method: 'POST', body: formData }),
  hopfieldLibrary: (side = 10) => request(`/hopfield/library?side=${side}`),
  hopfieldRecall: (payload) => request('/hopfield/recall', { method: 'POST', body: JSON.stringify(payload) }),
}
