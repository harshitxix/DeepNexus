import { Component, useEffect, useMemo, useRef, useState } from 'react'
import PlotlyImport from 'plotly.js-dist-min'
import animationVideo from './assets/animations/138962-770800093.mp4'
import ShinyText from './ShinyText'
import StaggeredMenu from './StaggeredMenu'
import ElasticSlider from './ElasticSlider'
import PillNav from './PillNav'
import { api } from './api'

const Plotly = PlotlyImport?.default || PlotlyImport

function Plot({ data, layout, config, style }) {
  const plotRef = useRef(null)

  useEffect(() => {
    if (!plotRef.current || !Plotly) return undefined
    Plotly.react(plotRef.current, data || [], layout || {}, config || {})
    return () => {
      if (plotRef.current) {
        Plotly.purge(plotRef.current)
      }
    }
  }, [data, layout, config])

  const mergedStyle = {
    borderRadius: '16px',
    overflow: 'hidden',
    border: '1px solid #d9dee8',
    background: '#ffffff',
    boxShadow: '0 10px 24px rgba(15, 23, 42, 0.08)',
    ...style,
  }

  return <div ref={plotRef} style={mergedStyle} />
}

const MODULES = [
  'Perceptron',
  'Neural Flow Engine',
  'CNN Lab',
  'RNN Lab',
  'Hopfield Network',
  'Study Mode',
]

const MODULE_DESCRIPTIONS = {
  'Perceptron': '',
  'Neural Flow Engine': 'Unified forward, loss, backprop, and weight-update workflow with interactive step controls.',
  'CNN Lab': 'Image classification, feature maps, and Grad-CAM explainability with pretrained CNNs.',
  'RNN Lab': 'Shared-input NLP lab with LSTM-style next-word generation and BERT-based emotion analysis.',
  'Hopfield Network': 'Mouse-draw character recall using associative memory over 0-9, a-z, and A-Z patterns.',
  'Study Mode': 'Concept map and guided checkpoints across all DeepNexus modules.',
}

const MENU_ITEMS = MODULES.map((moduleName) => ({ label: moduleName, moduleName }))
const GITHUB_URL = 'https://github.com/'

const ACTIVATIONS = {
  Sigmoid: (z) => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, z)))),
  ReLU: (z) => Math.max(0, z),
  Tanh: (z) => Math.tanh(z),
  Linear: (z) => z,
  Step: (z) => (z > 0 ? 1 : 0),
}

function randomIn(min, max) {
  return min + Math.random() * (max - min)
}

function makeRandomParams(layerSizes) {
  const weights = []
  const biases = []
  for (let i = 0; i < layerSizes.length - 1; i += 1) {
    const out = layerSizes[i + 1]
    const inn = layerSizes[i]
    weights.push(Array.from({ length: out }, () => Array.from({ length: inn }, () => randomIn(-1, 1))))
    biases.push(Array.from({ length: out }, () => randomIn(-0.2, 0.2)))
  }
  return { weights, biases }
}

function forwardDense(inputVec, params, hiddenActs, outputAct) {
  const { weights, biases } = params
  const layerZ = []
  const layerA = [inputVec.slice()]
  let aPrev = inputVec.slice()

  for (let l = 0; l < weights.length; l += 1) {
    const act = l === weights.length - 1 ? outputAct : hiddenActs[l]
    const z = weights[l].map((row, ri) => row.reduce((sum, w, ci) => sum + w * aPrev[ci], biases[l][ri]))
    const a = z.map((v) => ACTIVATIONS[act](v))
    layerZ.push(z)
    layerA.push(a)
    aPrev = a
  }

  return { layerZ, layerA }
}

function parameterCount(layerSizes) {
  let total = 0
  for (let i = 0; i < layerSizes.length - 1; i += 1) {
    total += layerSizes[i] * layerSizes[i + 1] + layerSizes[i + 1]
  }
  return total
}

function buildSignalMapFigure(layerSizes, labels, layerA = null, initialLayerA = null, flowMode = 'idle', flowPhase = 0) {
  const data = []
  const nLayers = layerSizes.length
  const xPos = Array.from({ length: nLayers }, (_, i) => (nLayers === 1 ? 0.5 : 0.08 + i * (0.84 / (nLayers - 1))))

  const layerNodes = layerSizes.map((count, i) => {
    const ys = count === 1 ? [0.5] : Array.from({ length: count }, (_, j) => 0.16 + j * (0.68 / (count - 1)))
    return ys.map((y) => ({ x: xPos[i], y }))
  })

  for (let l = 0; l < layerNodes.length - 1; l += 1) {
    for (const n1 of layerNodes[l]) {
      for (const n2 of layerNodes[l + 1]) {
        const edgeMid = (n1.x + n2.x) / 2
        const waveCenter = flowMode === 'forward' ? flowPhase : flowMode === 'backward' ? flowPhase : 0.5
        const distance = Math.abs(edgeMid - waveCenter)
        const pulse = flowMode === 'idle' ? 0 : Math.max(0, 1 - distance * 5.2)
        const alpha = flowMode === 'idle' ? 0.22 : 0.22 + pulse * 0.62
        const width = 1.1 + pulse * 2.6
        const rgb = flowMode === 'backward' ? '239,68,68' : flowMode === 'forward' ? '82,39,255' : '100,116,139'

        data.push({
          x: [n1.x, (n1.x + n2.x) / 2, n2.x],
          y: [n1.y, (n1.y + n2.y) / 2 + (n2.y > n1.y ? 0.07 : -0.07), n2.y],
          type: 'scatter',
          mode: 'lines',
          line: { color: `rgba(${rgb},${alpha})`, width },
          hoverinfo: 'skip',
          showlegend: false,
        })
      }
    }
  }

  if (flowMode !== 'idle') {
    const arrowY = 0.95
    const arrowX = Array.from({ length: Math.max(1, nLayers - 1) }, (_, i) => 0.15 + i * (0.7 / Math.max(1, nLayers - 2)))
    data.push({
      x: arrowX,
      y: arrowX.map(() => arrowY),
      type: 'scatter',
      mode: 'markers',
      marker: {
        symbol: flowMode === 'backward' ? 'triangle-left' : 'triangle-right',
        size: 10,
        color: flowMode === 'backward' ? '#ef4444' : '#5227ff',
      },
      hoverinfo: 'skip',
      showlegend: false,
    })
  }

  const colors = layerSizes.map((_, i) => {
    if (i === 0) return '#5227ff'
    if (i === layerSizes.length - 1) return '#9ca3af'
    return '#7c3aed'
  })

  for (let l = 0; l < layerNodes.length; l += 1) {
    const nodes = layerNodes[l]
    data.push({
      x: nodes.map((n) => n.x),
      y: nodes.map((n) => n.y),
      type: 'scatter',
      mode: 'markers+text',
      marker: { size: 38, color: colors[l], line: { color: 'rgba(255,255,255,0.85)', width: 1.4 } },
      customdata: nodes.map((_, idx) => {
        const initialVal = initialLayerA?.[l]?.[idx] ?? 0
        const currentVal = layerA?.[l]?.[idx] ?? 0
        return [initialVal, currentVal]
      }),
      text: nodes.map((_, idx) => {
        const base = l === 0 ? 'x' : l === layerNodes.length - 1 ? 'y' : 'h'
        const val = layerA?.[l]?.[idx]
        return val === undefined ? `${base}${idx + 1}` : `${base}${idx + 1}<br>${val.toFixed(3)}`
      }),
      textposition: 'middle center',
      textfont: { color: '#ffffff', size: 11, family: 'Courier New, monospace' },
      hovertemplate:
        '<b>%{text}</b><br>Initial value: %{customdata[0]:.4f}<br>Current value: %{customdata[1]:.4f}<extra></extra>',
      showlegend: false,
    })
  }

  return {
    data,
    layout: {
      title: { text: 'Neural Signal Map', x: 0.5 },
      xaxis: {
        title: 'Layer Progression (Input to Output)',
        range: [0, 1],
        showgrid: true,
        gridcolor: '#e5e7eb',
        showline: true,
        linecolor: '#94a3b8',
        ticks: 'outside',
        ticklen: 5,
        zeroline: false,
        tickformat: '.1f',
      },
      yaxis: {
        title: 'Neuron Position In Layer',
        range: [0, 1],
        showgrid: true,
        gridcolor: '#e5e7eb',
        showline: true,
        linecolor: '#94a3b8',
        ticks: 'outside',
        ticklen: 5,
        zeroline: false,
        tickformat: '.1f',
      },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      font: { color: '#0f172a' },
      margin: { l: 74, r: 24, t: 58, b: 64 },
      height: 450,
      annotations: layerNodes.map((nodes, i) => ({
        x: nodes[0].x,
        y: 0.06,
        text: `<b>${labels[i]}</b>`,
        showarrow: false,
        font: { size: 12, color: '#0f172a' },
      })).concat(
        flowMode === 'idle'
          ? []
          : [
              {
                x: 0.5,
                y: 1.06,
                xref: 'paper',
                yref: 'paper',
                text: `<b>${flowMode === 'backward' ? 'Backward Flow ←' : 'Forward Flow →'}</b>`,
                showarrow: false,
                font: { size: 12, color: flowMode === 'backward' ? '#ef4444' : '#5227ff' },
              },
            ]
      ),
    },
  }
}

function build3DBarTraces(values, labels, color = '#5227ff') {
  const traces = []
  const halfW = 0.28
  const halfD = 0.18

  values.forEach((val, idx) => {
    const x0 = idx + 1 - halfW
    const x1 = idx + 1 + halfW
    const y0 = -halfD
    const y1 = halfD
    const z0 = 0
    const z1 = val

    traces.push({
      type: 'mesh3d',
      x: [x0, x1, x1, x0, x0, x1, x1, x0],
      y: [y0, y0, y1, y1, y0, y0, y1, y1],
      z: [z0, z0, z0, z0, z1, z1, z1, z1],
      i: [0, 0, 0, 1, 1, 2, 4, 4, 5, 6, 3, 7],
      j: [1, 2, 3, 2, 5, 3, 5, 6, 6, 7, 7, 4],
      k: [2, 3, 1, 5, 6, 7, 6, 7, 4, 4, 0, 0],
      opacity: 0.86,
      color,
      flatshading: true,
      hovertemplate: `${labels[idx]}<br>Value: ${val.toFixed(6)}<extra></extra>`,
      showscale: false,
    })
  })

  traces.push({
    type: 'scatter3d',
    mode: 'text',
    x: values.map((_, i) => i + 1),
    y: values.map(() => 0.26),
    z: values.map((v) => v + Math.max(0.02, v * 0.04)),
    text: values.map((v) => v.toFixed(4)),
    textfont: { size: 10, color: '#0f172a' },
    hoverinfo: 'skip',
    showlegend: false,
  })

  return traces
}

function buildHopfieldHeatmap(pattern, side, title, accent = '#0f172a') {
  const grid = Array.from({ length: side }, (_, row) =>
    Array.from({ length: side }, (_, col) => {
      if (!pattern?.length) return 0
      return pattern[row * side + col] > 0 ? 1 : 0
    })
  )

  return {
    data: [
      {
        z: grid,
        type: 'heatmap',
        zmin: 0,
        zmax: 1,
        colorscale: [
          [0, '#f8fafc'],
          [1, accent],
        ],
        showscale: false,
        hoverinfo: 'skip',
      },
    ],
    layout: {
      title: { text: title, x: 0.5 },
      xaxis: { visible: false },
      yaxis: { visible: false, autorange: 'reversed' },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      margin: { l: 10, r: 10, t: 44, b: 10 },
      height: 260,
    },
  }
}

function AnimatedSignalMap({ layerSizes, layerLabels, layerA, initialLayerA, flowMode }) {
  const [flowPhase, setFlowPhase] = useState(flowMode === 'backward' ? 0.82 : flowMode === 'forward' ? 0.18 : 0.5)

  useEffect(() => {
    if (flowMode === 'idle') {
      setFlowPhase(0.5)
      return undefined
    }

    let rafId = 0
    let lastTs = 0
    const minFrameMs = 42

    const tick = (ts) => {
      if (!lastTs || ts - lastTs >= minFrameMs) {
        lastTs = ts
        setFlowPhase((prev) => {
          const delta = 0.02
          let next = flowMode === 'backward' ? prev - delta : prev + delta
          if (next > 1) next -= 1
          if (next < 0) next += 1
          return next
        })
      }
      rafId = window.requestAnimationFrame(tick)
    }

    rafId = window.requestAnimationFrame(tick)
    return () => window.cancelAnimationFrame(rafId)
  }, [flowMode])

  const fig = useMemo(
    () => buildSignalMapFigure(layerSizes, layerLabels, layerA, initialLayerA, flowMode, flowPhase),
    [layerSizes, layerLabels, layerA, initialLayerA, flowMode, flowPhase]
  )

  return <Plot data={fig.data} layout={fig.layout} style={{ width: '100%', height: '450px' }} config={{ displayModeBar: false }} />
}

class ModuleErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, message: '' }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, message: error?.message || 'Unknown module error' }
  }

  componentDidUpdate(prevProps) {
    if (prevProps.resetKey !== this.props.resetKey && this.state.hasError) {
      this.setState({ hasError: false, message: '' })
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-fallback">
          <h3>Module failed to render</h3>
          <p>{this.state.message}</p>
          <p>Switch module and return, or refresh the page to retry.</p>
        </div>
      )
    }
    return this.props.children
  }
}

function NeuralNetworkAnimation() {
  return (
    <video className="neural-network-video" autoPlay loop muted playsInline>
      <source src={animationVideo} type="video/mp4" />
      Your browser does not support the video tag.
    </video>
  )
}

function HomePage({ menuOpen, onToggleMenu, onMenuSelect }) {
  return (
    <div className="home-page">
      <NeuralNetworkAnimation />
      <div className="home-menu-anchor">
        <StaggeredMenu items={MENU_ITEMS} isOpen={menuOpen} onToggle={onToggleMenu} onItemClick={onMenuSelect} />
      </div>
      <div className={menuOpen ? 'hero-section home-content-shell menu-open' : 'hero-section home-content-shell'}>
        <ShinyText
          text="Build, Learn, and Experiment with Deep Learning - All in One Place"
          className="hero-title"
          color="#5227ff"
          shineColor="#8080ff"
          speed={3}
          spread={100}
        />
        <p className="hero-subtext">
          DeepNexus is an interactive deep learning workspace where you can run, visualize, and understand core neural
          network modules in one place.
        </p>
      </div>
    </div>
  )
}

function Section({ title, children }) {
  return (
    <section className="module-line-section">
      <h3>{title}</h3>
      {children}
    </section>
  )
}

function NumberField({ label, value, onChange, step = 0.1 }) {
  return (
    <label className="module-field">
      <span>{label}</span>
      <input type="number" step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  )
}

function PerceptronModule() {
  const [w1, setW1] = useState(0.5)
  const [w2, setW2] = useState(0.5)
  const [bias, setBias] = useState(0)
  const [threshold, setThreshold] = useState(0.5)
  const [x1, setX1] = useState(0.3)
  const [x2, setX2] = useState(0.3)
  const [activation, setActivation] = useState('Sigmoid')
  const [gate, setGate] = useState('AND')

  const z = w1 * x1 + w2 * x2 + bias
  const a = ACTIVATIONS[activation](z)
  const predictedClass = a >= threshold ? 1 : 0

  const xVals = Array.from({ length: 80 }, (_, i) => -1 + (i * 2) / 79)
  const yVals = Array.from({ length: 80 }, (_, i) => -1 + (i * 2) / 79)
  const zGrid = yVals.map((yy) => xVals.map((xx) => ACTIVATIONS[activation](w1 * xx + w2 * yy + bias)))

  const actX = Array.from({ length: 200 }, (_, i) => -5 + (i * 10) / 199)
  const actY = actX.map((v) => ACTIVATIONS[activation](v))

  const gatesData = {
    AND: [
      [0, 0, 0],
      [0, 1, 0],
      [1, 0, 0],
      [1, 1, 1],
    ],
    OR: [
      [0, 0, 0],
      [0, 1, 1],
      [1, 0, 1],
      [1, 1, 1],
    ],
  }

  const baselineConfig = useMemo(
    () => ({ w1: 0.5, w2: 0.5, bias: 0, threshold: 0.5, activation: 'Sigmoid' }),
    []
  )

  const computeGateMetrics = (config) => {
    const rows = gatesData[gate].map(([gx1, gx2, gy]) => {
      const raw = ACTIVATIONS[config.activation](config.w1 * gx1 + config.w2 * gx2 + config.bias)
      const gp = raw >= config.threshold ? 1 : 0
      return { gx1, gx2, gy, gp, raw }
    })

    const accuracy = (rows.filter((r) => r.gy === r.gp).length / rows.length) * 100
    const mse = rows.reduce((sum, r) => sum + (r.gy - r.raw) ** 2, 0) / rows.length

    return { rows, accuracy, mse }
  }

  const currentGate = computeGateMetrics({ w1, w2, bias, threshold, activation })
  const baselineGate = computeGateMetrics(baselineConfig)

  const gateRows = currentGate.rows
  const gateAccuracy = currentGate.accuracy

  const initialTestActivation = ACTIVATIONS[baselineConfig.activation](baselineConfig.w1 * x1 + baselineConfig.w2 * x2 + baselineConfig.bias)

  const computationRows = [
    ['1. Input', `x1=${x1.toFixed(4)}, x2=${x2.toFixed(4)}`],
    ['2. Weight sum', `w1*x1 + w2*x2 = ${(w1 * x1).toFixed(4)} + ${(w2 * x2).toFixed(4)}`],
    ['3. Add bias', `z + b = ${(z - bias).toFixed(4)} + ${bias.toFixed(4)}`],
    ['4. Activation', `a = ${activation}(${z.toFixed(4)}) = ${a.toFixed(4)}`],
    ['5. Decision', `pred = 1 if a >= ${threshold.toFixed(4)} else 0 = ${predictedClass}`],
  ]

  return (
    <>
      <Section title="how to begin">
        <div className="engine-guide">
          <p className="engine-guide-title">Quick Start For Perceptron</p>
          <p className="engine-guide-subtitle">
            Tune the <strong>weights</strong>, inspect the <strong>forward output</strong>, then validate behavior on
            <strong> logic gates</strong> and the <strong>decision boundary</strong>.
          </p>
          <div className="engine-guide-flow" role="list" aria-label="Perceptron quick guide">
            <div className="engine-guide-step" role="listitem">
              <p className="engine-step-head">Configure</p>
              <p className="engine-step-copy">Set w1, w2, bias, threshold, and activation.</p>
            </div>
            <div className="engine-guide-step" role="listitem">
              <p className="engine-step-head">Probe Input</p>
              <p className="engine-step-copy">Choose x1 and x2 to inspect z, a, and class output.</p>
            </div>
            <div className="engine-guide-step" role="listitem">
              <p className="engine-step-head">Visualize</p>
              <p className="engine-step-copy">Read boundary and activation shape to understand separability.</p>
            </div>
            <div className="engine-guide-step" role="listitem">
              <p className="engine-step-head">Validate</p>
              <p className="engine-step-copy">Compare predictions against AND/OR truth tables.</p>
            </div>
            <div className="engine-guide-step" role="listitem">
              <p className="engine-step-head">Summarize</p>
              <p className="engine-step-copy">Review improvements from initial to current configuration.</p>
            </div>
          </div>
        </div>
      </Section>

      <Section title="architecture and setup">
        <p className="module-note">
          A perceptron is a single-layer classifier where input features are linearly combined and passed through an
          activation. It provides a compact way to inspect how parameters shape classification.
        </p>
        <p className="module-note">
          <strong>Architecture</strong>: Input Layer (<strong>2</strong> neurons) {'→'} Output Layer (<strong>1</strong> neuron)
        </p>
        <p className="module-note formula-copy">z = w₁x₁ + w₂x₂ + b, a = f(z), ŷ = 1[a ≥ threshold]</p>
      </Section>

      <Section title="neuron parameters">
        <p className="module-note">
          Adjust weights and bias to control the decision surface, and use threshold to shift final class assignment.
        </p>
        <div className="slider-grid">
          <div>
            <p className="slider-label">Weight 1 (w1)</p>
            <ElasticSlider defaultValue={w1} startingValue={-2} maxValue={2} stepSize={0.1} isStepped onChange={setW1} />
          </div>
          <div>
            <p className="slider-label">Weight 2 (w2)</p>
            <ElasticSlider defaultValue={w2} startingValue={-2} maxValue={2} stepSize={0.1} isStepped onChange={setW2} />
          </div>
          <div>
            <p className="slider-label">Bias</p>
            <ElasticSlider defaultValue={bias} startingValue={-2} maxValue={2} stepSize={0.1} isStepped onChange={setBias} />
          </div>
          <div>
            <p className="slider-label">Decision Threshold</p>
            <ElasticSlider defaultValue={threshold} startingValue={0} maxValue={1} stepSize={0.05} isStepped onChange={setThreshold} />
          </div>
        </div>
        <p className="module-note">Neuron equation: z = {w1.toFixed(2)}*x1 + {w2.toFixed(2)}*x2 + {bias.toFixed(2)}</p>
      </Section>

      <Section title="test input and activation">
        <p className="module-note">
          Provide a sample input and activation function to inspect how raw linear output transforms into a prediction.
        </p>
        <div className="module-field-grid">
          <NumberField label="Input x1" value={x1} onChange={setX1} />
          <NumberField label="Input x2" value={x2} onChange={setX2} />
          <label className="module-field">
            <span>Activation</span>
            <select value={activation} onChange={(e) => setActivation(e.target.value)}>
              {Object.keys(ACTIVATIONS).map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>
          <div className="metric-inline">
            <p><strong>z:</strong> {z.toFixed(4)}</p>
            <p><strong>a:</strong> {a.toFixed(4)}</p>
            <p><strong>class:</strong> {predictedClass}</p>
          </div>
        </div>
      </Section>

      <Section title="visualizations">
        <p className="module-note">
          The boundary plot shows where class transitions happen, while the activation plot shows nonlinearity behavior.
        </p>
        <div className="plot-grid-2">
          <Plot
            data={[
              { x: xVals, y: yVals, z: zGrid, type: 'surface', colorscale: 'Viridis', opacity: 0.95 },
              {
                x: xVals,
                y: yVals,
                z: yVals.map(() => xVals.map(() => threshold)),
                type: 'surface',
                opacity: 0.28,
                showscale: false,
                colorscale: [[0, '#ef4444'], [1, '#ef4444']],
              },
            ]}
            layout={{
              title: { text: '3D Decision Boundary Surface', x: 0.5 },
              scene: {
                xaxis: { title: 'Input x1' },
                yaxis: { title: 'Input x2' },
                zaxis: { title: 'Activation Output' },
                camera: { eye: { x: 1.35, y: 1.1, z: 0.95 } },
              },
              margin: { t: 52, l: 20, r: 20, b: 20 },
            }}
            style={{ width: '100%', height: '360px' }}
            config={{ displayModeBar: false }}
          />

          <Plot
            data={[{ x: actX, y: actY, z: actX.map(() => 0), type: 'scatter3d', mode: 'lines+markers', line: { color: '#5227ff', width: 5 }, marker: { size: 2, color: '#5227ff' } }]}
            layout={{
              title: { text: `${activation} Activation (3D Curve)`, x: 0.5 },
              scene: {
                xaxis: { title: 'Pre-activation z' },
                yaxis: { title: 'Activation a' },
                zaxis: { title: 'Reference Axis' },
                camera: { eye: { x: 1.25, y: 1.2, z: 1.05 } },
              },
              margin: { t: 52, l: 20, r: 20, b: 20 },
            }}
            style={{ width: '100%', height: '360px' }}
            config={{ displayModeBar: false }}
          />
        </div>
      </Section>

      <Section title="computation breakdown">
        <p className="module-note">
          This stepwise trace explains exactly how the current output is produced from your latest parameter choices.
        </p>
        <table className="module-table">
          <thead>
            <tr><th>Step</th><th>Value</th></tr>
          </thead>
          <tbody>
            {computationRows.map(([step, value]) => (
              <tr key={step}><td>{step}</td><td>{value}</td></tr>
            ))}
          </tbody>
        </table>
      </Section>

      <Section title="logic gate evaluation">
        <p className="module-note">
          Evaluate the perceptron on a full truth table to see whether the chosen setup matches the target gate.
        </p>
        <div className="module-field-grid">
          <label className="module-field">
            <span>Gate</span>
            <select value={gate} onChange={(e) => setGate(e.target.value)}>
              <option value="AND">AND</option>
              <option value="OR">OR</option>
            </select>
          </label>
          <p className="module-note"><strong>Accuracy:</strong> {gateAccuracy.toFixed(1)}%</p>
        </div>
        <table className="module-table">
          <thead>
            <tr><th>x1</th><th>x2</th><th>true</th><th>pred</th></tr>
          </thead>
          <tbody>
            {gateRows.map((r, idx) => (
              <tr key={idx}><td>{r.gx1}</td><td>{r.gx2}</td><td>{r.gy}</td><td>{r.gp}</td></tr>
            ))}
          </tbody>
        </table>
      </Section>

      <Section title="run summary">
        <p className="module-note">
          <strong>Input & Configuration:</strong> Current setup uses w1=<strong>{w1.toFixed(4)}</strong>, w2=<strong>{w2.toFixed(4)}</strong>,
          b=<strong>{bias.toFixed(4)}</strong>, threshold=<strong>{threshold.toFixed(4)}</strong>, activation=<strong>{activation}</strong>,
          and gate=<strong>{gate}</strong>.
        </p>

        <p className="module-note">
          <strong>Flow of Data:</strong> The input pair ({x1.toFixed(3)}, {x2.toFixed(3)}) is transformed via linear combination
          into z=<strong>{z.toFixed(4)}</strong>, then mapped to a=<strong>{a.toFixed(4)}</strong>, yielding class <strong>{predictedClass}</strong>.
        </p>

        <p className="module-note">
          <strong>Learning Behavior:</strong> Compared to baseline, gate accuracy changed from
          <strong> {baselineGate.accuracy.toFixed(2)}%</strong> to <strong>{currentGate.accuracy.toFixed(2)}%</strong> ({(currentGate.accuracy - baselineGate.accuracy).toFixed(2)} pts),
          and gate MSE changed from <strong>{baselineGate.mse.toFixed(6)}</strong> to <strong>{currentGate.mse.toFixed(6)}</strong>
          ({(currentGate.mse - baselineGate.mse).toFixed(6)} delta).
        </p>

        <p className="module-note">
          <strong>Final Results:</strong> Test-input activation moved from <strong>{initialTestActivation.toFixed(4)}</strong> (baseline)
          to <strong>{a.toFixed(4)}</strong> (current), and logic-gate alignment now reflects the table below.
        </p>

        <table className="module-table">
          <thead>
            <tr><th>x1</th><th>x2</th><th>y_true</th><th>pred_initial</th><th>pred_final</th><th>|error_final|</th></tr>
          </thead>
          <tbody>
            {currentGate.rows.map((row, idx) => {
              const initialRow = baselineGate.rows[idx]
              return (
                <tr key={`summary-gate-${idx}`}>
                  <td>{row.gx1}</td>
                  <td>{row.gx2}</td>
                  <td>{row.gy}</td>
                  <td>{initialRow.gp}</td>
                  <td>{row.gp}</td>
                  <td>{Math.abs(row.gy - row.raw).toFixed(4)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </Section>
    </>
  )
}

function ForwardModule() {
  const [nInputs, setNInputs] = useState(2)
  const [nHiddenLayers, setNHiddenLayers] = useState(1)
  const [hiddenSizes, setHiddenSizes] = useState([2])
  const [inputs, setInputs] = useState([0.3, 0.45])
  const [sameHiddenActivation, setSameHiddenActivation] = useState(true)
  const [hiddenActivation, setHiddenActivation] = useState('Sigmoid')
  const [hiddenActivations, setHiddenActivations] = useState(['Sigmoid'])
  const [outputActivation, setOutputActivation] = useState('Linear')
  const [params, setParams] = useState(() => makeRandomParams([2, 2, 1]))
  const [result, setResult] = useState(null)

  useEffect(() => {
    setHiddenSizes((prev) => {
      const next = prev.slice(0, nHiddenLayers)
      while (next.length < nHiddenLayers) next.push(2)
      return next
    })
    setHiddenActivations((prev) => {
      const next = prev.slice(0, nHiddenLayers)
      while (next.length < nHiddenLayers) next.push(hiddenActivation)
      return next
    })
  }, [nHiddenLayers, hiddenActivation])

  useEffect(() => {
    setInputs((prev) => {
      const next = prev.slice(0, nInputs)
      while (next.length < nInputs) next.push(Number((0.3 + next.length * 0.15).toFixed(2)))
      return next
    })
  }, [nInputs])

  const layerSizes = useMemo(() => [nInputs, ...hiddenSizes, 1], [nInputs, hiddenSizes])
  const layerLabels = useMemo(() => ['Input', ...hiddenSizes.map((_, i) => `Hidden ${i + 1}`), 'Output'], [hiddenSizes])

  useEffect(() => {
    setParams(makeRandomParams(layerSizes))
    setResult(null)
  }, [layerSizes])

  const totalParams = useMemo(() => parameterCount(layerSizes), [layerSizes])

  const run = () => {
    const acts = sameHiddenActivation ? Array.from({ length: nHiddenLayers }, () => hiddenActivation) : hiddenActivations
    setResult(forwardDense(inputs, params, acts, outputActivation))
  }

  const fig = buildSignalMapFigure(layerSizes, layerLabels, result?.layerA)

  return (
    <>
      <Section title="network architecture">
        <div className="module-field-grid">
          <NumberField label="Input features" value={nInputs} onChange={(v) => setNInputs(Math.max(1, Math.min(8, Math.round(v))))} step={1} />
          <NumberField label="Hidden layers" value={nHiddenLayers} onChange={(v) => setNHiddenLayers(Math.max(1, Math.min(4, Math.round(v))))} step={1} />
          {hiddenSizes.map((size, i) => (
            <NumberField
              key={`hsize-${i}`}
              label={`Neurons in layer ${i + 1}`}
              value={size}
              onChange={(v) => setHiddenSizes((prev) => prev.map((p, idx) => (idx === i ? Math.max(1, Math.min(10, Math.round(v))) : p)))}
              step={1}
            />
          ))}
        </div>
        <p className="module-note">
          Input({nInputs}) {'->'} {hiddenSizes.map((h, i) => `H${i + 1}(${h})`).join(' -> ')} {'->'} Output(1) | Parameters: {totalParams}
        </p>
      </Section>

      <Section title="input values and activations">
        <div className="module-field-grid">
          {inputs.map((v, i) => (
            <NumberField key={i} label={`x${i + 1}`} value={v} onChange={(nv) => setInputs((prev) => prev.map((p, idx) => (idx === i ? nv : p)))} />
          ))}

          <label className="module-field checkbox-field">
            <span>Same activation for hidden layers</span>
            <input type="checkbox" checked={sameHiddenActivation} onChange={(e) => setSameHiddenActivation(e.target.checked)} />
          </label>

          <label className="module-field">
            <span>Hidden activation</span>
            <select value={hiddenActivation} onChange={(e) => setHiddenActivation(e.target.value)}>
              <option value="Sigmoid">Sigmoid</option>
              <option value="ReLU">ReLU</option>
              <option value="Tanh">Tanh</option>
              <option value="Linear">Linear</option>
            </select>
          </label>

          {!sameHiddenActivation
            ? hiddenActivations.map((act, i) => (
                <label className="module-field" key={`hact-${i}`}>
                  <span>Layer {i + 1} activation</span>
                  <select value={act} onChange={(e) => setHiddenActivations((prev) => prev.map((p, idx) => (idx === i ? e.target.value : p)))}>
                    <option value="Sigmoid">Sigmoid</option>
                    <option value="ReLU">ReLU</option>
                    <option value="Tanh">Tanh</option>
                    <option value="Linear">Linear</option>
                  </select>
                </label>
              ))
            : null}

          <label className="module-field">
            <span>Output activation</span>
            <select value={outputActivation} onChange={(e) => setOutputActivation(e.target.value)}>
              <option value="Sigmoid">Sigmoid</option>
              <option value="ReLU">ReLU</option>
              <option value="Tanh">Tanh</option>
              <option value="Linear">Linear</option>
            </select>
          </label>
        </div>

        <div className="module-action-row">
          <button className="module-action-btn" onClick={() => setParams(makeRandomParams(layerSizes))}>Randomize Weights</button>
          <button className="module-action-btn primary" onClick={run}>Run Forward Pass</button>
        </div>
      </Section>

      <Section title="signal map">
        <Plot data={fig.data} layout={fig.layout} style={{ width: '100%', height: '430px' }} config={{ displayModeBar: false }} />
      </Section>

      <Section title="layer outputs">
        {result ? (
          result.layerZ.map((zVals, idx) => (
            <div className="layer-plot-block" key={idx}>
              <h4>Layer {idx + 1}</h4>
              <div className="plot-grid-2">
                <Plot
                  data={[{ x: zVals.map((_, i) => `n${i + 1}`), y: zVals, type: 'bar', marker: { color: '#5227ff' } }]}
                  layout={{ title: `z values (Layer ${idx + 1})`, margin: { t: 42, l: 40, r: 20, b: 38 } }}
                  style={{ width: '100%', height: '300px' }}
                  config={{ displayModeBar: false }}
                />
                <Plot
                  data={[{ x: result.layerA[idx + 1].map((_, i) => `n${i + 1}`), y: result.layerA[idx + 1], type: 'bar', marker: { color: '#8a6cff' } }]}
                  layout={{ title: `a values (Layer ${idx + 1})`, margin: { t: 42, l: 40, r: 20, b: 38 } }}
                  style={{ width: '100%', height: '300px' }}
                  config={{ displayModeBar: false }}
                />
              </div>
            </div>
          ))
        ) : (
          <p className="module-note">Run forward pass to inspect z and a for each layer.</p>
        )}
      </Section>
    </>
  )
}

function BackwardModule() {
  const [x1, setX1] = useState(0.3)
  const [x2, setX2] = useState(0.7)
  const [yTrue, setYTrue] = useState(1)
  const [w1, setW1] = useState(0.5)
  const [w2, setW2] = useState(0.5)
  const [b, setB] = useState(0)
  const [lr, setLr] = useState(0.1)
  const [result, setResult] = useState(null)

  const run = () => {
    const z = w1 * x1 + w2 * x2 + b
    const a = ACTIVATIONS.Sigmoid(z)
    const loss = 0.5 * (a - yTrue) ** 2
    const dLda = a - yTrue
    const dadz = a * (1 - a)
    const dLdz = dLda * dadz
    const gradW1 = dLdz * x1
    const gradW2 = dLdz * x2
    const gradB = dLdz

    setResult({
      z,
      a,
      loss,
      dLda,
      dadz,
      dLdz,
      gradW1,
      gradW2,
      gradB,
      newW1: w1 - lr * gradW1,
      newW2: w2 - lr * gradW2,
      newB: b - lr * gradB,
    })
  }

  return (
    <>
      <Section title="parameter tuning">
        <div className="slider-grid">
          <div>
            <p className="slider-label">Input x1</p>
            <ElasticSlider defaultValue={x1} startingValue={-1} maxValue={1} isStepped stepSize={0.05} onChange={setX1} />
          </div>
          <div>
            <p className="slider-label">Input x2</p>
            <ElasticSlider defaultValue={x2} startingValue={-1} maxValue={1} isStepped stepSize={0.05} onChange={setX2} />
          </div>
          <div>
            <p className="slider-label">Weight w1</p>
            <ElasticSlider defaultValue={w1} startingValue={-2} maxValue={2} isStepped stepSize={0.05} onChange={setW1} />
          </div>
          <div>
            <p className="slider-label">Weight w2</p>
            <ElasticSlider defaultValue={w2} startingValue={-2} maxValue={2} isStepped stepSize={0.05} onChange={setW2} />
          </div>
          <div>
            <p className="slider-label">Bias</p>
            <ElasticSlider defaultValue={b} startingValue={-2} maxValue={2} isStepped stepSize={0.05} onChange={setB} />
          </div>
          <div>
            <p className="slider-label">Learning Rate</p>
            <ElasticSlider defaultValue={lr} startingValue={0.01} maxValue={1} isStepped stepSize={0.01} onChange={setLr} />
          </div>
          <div>
            <p className="slider-label">Target y</p>
            <ElasticSlider defaultValue={yTrue} startingValue={0} maxValue={1} isStepped stepSize={1} onChange={setYTrue} />
          </div>
        </div>
        <div className="module-action-row">
          <button className="module-action-btn primary" onClick={run}>Run Backprop Step</button>
        </div>
      </Section>

      <Section title="gradient visuals">
        {result ? (
          <>
            <p className="module-note">
              y_pred={result.a.toFixed(4)} | loss={result.loss.toFixed(4)} | new w1={result.newW1.toFixed(4)} | new w2={result.newW2.toFixed(4)} | new b={result.newB.toFixed(4)}
            </p>
            <Plot
              data={[{ x: ['|dL/dw1|', '|dL/dw2|', '|dL/db|'], y: [Math.abs(result.gradW1), Math.abs(result.gradW2), Math.abs(result.gradB)], type: 'bar', marker: { color: ['#ef4444', '#f97316', '#f59e0b'] } }]}
              layout={{ title: 'Gradient Magnitudes', margin: { t: 42, l: 40, r: 20, b: 38 } }}
              style={{ width: '100%', height: '320px' }}
              config={{ displayModeBar: false }}
            />

            <table className="module-table">
              <thead>
                <tr><th>Step</th><th>Expression</th><th>Value</th></tr>
              </thead>
              <tbody>
                <tr><td>1</td><td>z = w1*x1 + w2*x2 + b</td><td>{result.z.toFixed(6)}</td></tr>
                <tr><td>2</td><td>a = sigmoid(z)</td><td>{result.a.toFixed(6)}</td></tr>
                <tr><td>3</td><td>L = 0.5*(a-y)^2</td><td>{result.loss.toFixed(6)}</td></tr>
                <tr><td>4</td><td>dL/da = (a-y)</td><td>{result.dLda.toFixed(6)}</td></tr>
                <tr><td>5</td><td>da/dz = a*(1-a)</td><td>{result.dadz.toFixed(6)}</td></tr>
                <tr><td>6</td><td>dL/dz = dL/da * da/dz</td><td>{result.dLdz.toFixed(6)}</td></tr>
                <tr><td>7</td><td>dL/dw1 = dL/dz * x1</td><td>{result.gradW1.toFixed(6)}</td></tr>
                <tr><td>8</td><td>dL/dw2 = dL/dz * x2</td><td>{result.gradW2.toFixed(6)}</td></tr>
                <tr><td>9</td><td>dL/db = dL/dz</td><td>{result.gradB.toFixed(6)}</td></tr>
              </tbody>
            </table>
          </>
        ) : (
          <p className="module-note">Run the backprop step to inspect gradients and updates.</p>
        )}
      </Section>
    </>
  )
}

function NeuralFlowEngineModule() {
  const stageOrder = ['forward', 'loss', 'backward', 'update']
  const stageLabelMap = {
    forward: 'Forward Flow',
    loss: 'Loss Computation',
    backward: 'Backward Flow',
    update: 'Weight Update',
  }

  const datasets = {
    AND: [
      [0, 0, 0],
      [0, 1, 0],
      [1, 0, 0],
      [1, 1, 1],
    ],
    OR: [
      [0, 0, 0],
      [0, 1, 1],
      [1, 0, 1],
      [1, 1, 1],
    ],
    XOR: [
      [0, 0, 0],
      [0, 1, 1],
      [1, 0, 1],
      [1, 1, 0],
    ],
  }

  const differentiableActivations = ['Sigmoid', 'ReLU', 'Tanh', 'Linear']

  const [hiddenLayerCount, setHiddenLayerCount] = useState(1)
  const [neuronsPerHidden, setNeuronsPerHidden] = useState([4])
  const [activation, setActivation] = useState('ReLU')
  const [learningRate, setLearningRate] = useState(0.08)
  const [epochs, setEpochs] = useState(40)
  const [gate, setGate] = useState('XOR')

  const [params, setParams] = useState(() => makeRandomParams([2, 4, 1]))
  const [currentEpoch, setCurrentEpoch] = useState(0)
  const [currentStageIndex, setCurrentStageIndex] = useState(0)
  const [hasInitialized, setHasInitialized] = useState(false)
  const [running, setRunning] = useState(false)
  const [initialSnapshot, setInitialSnapshot] = useState(null)

  const [forwardState, setForwardState] = useState(null)
  const [lossState, setLossState] = useState(null)
  const [gradientState, setGradientState] = useState(null)
  const [updateState, setUpdateState] = useState(null)
  const [history, setHistory] = useState({ losses: [], accs: [] })

  useEffect(() => {
    setNeuronsPerHidden((prev) => {
      const next = prev.slice(0, hiddenLayerCount)
      while (next.length < hiddenLayerCount) next.push(4)
      return next
    })
  }, [hiddenLayerCount])

  const layerSizes = useMemo(() => [2, ...neuronsPerHidden.slice(0, hiddenLayerCount), 1], [neuronsPerHidden, hiddenLayerCount])
  const layerLabels = useMemo(() => ['Input', ...layerSizes.slice(1, -1).map((_, i) => `Hidden ${i + 1}`), 'Output'], [layerSizes])
  const rows = useMemo(() => datasets[gate], [gate])

  useEffect(() => {
    const freshParams = makeRandomParams(layerSizes)
    setRunning(false)
    setParams(freshParams)
    setCurrentEpoch(0)
    setCurrentStageIndex(0)
    setHasInitialized(true)
    setForwardState(null)
    setLossState(null)
    setGradientState(null)
    setUpdateState(null)
    setHistory({ losses: [], accs: [] })

    const freshEval = evaluateModel(freshParams)
    setInitialSnapshot({
      params: cloneParams(freshParams),
      loss: freshEval.mse,
      acc: freshEval.acc,
      predictions: freshEval.predictions,
    })
  }, [layerSizes, gate])

  const activationDerivative = (name, z, a) => {
    if (name === 'ReLU') return z > 0 ? 1 : 0
    if (name === 'Tanh') return 1 - a * a
    if (name === 'Linear') return 1
    return a * (1 - a)
  }

  const cloneParams = (source) => ({
    weights: source.weights.map((layer) => layer.map((nodeWeights) => nodeWeights.slice())),
    biases: source.biases.map((layerBiases) => layerBiases.slice()),
  })

  const resetEngine = () => {
    const freshParams = makeRandomParams(layerSizes)
    setRunning(false)
    setParams(freshParams)
    setCurrentEpoch(0)
    setCurrentStageIndex(0)
    setForwardState(null)
    setLossState(null)
    setGradientState(null)
    setUpdateState(null)
    setHistory({ losses: [], accs: [] })

    const freshEval = evaluateModel(freshParams)
    setInitialSnapshot({
      params: cloneParams(freshParams),
      loss: freshEval.mse,
      acc: freshEval.acc,
      predictions: freshEval.predictions,
    })
  }

  const runForwardStage = (activeParams = params) => {
    const hiddenActs = Array.from({ length: activeParams.weights.length - 1 }, () => activation)
    const previousSnapshot = forwardState?.sampleForDiagram?.cache?.layerA || null
    const sampleRuns = rows.map((r) => {
      const pass = forwardDense([r[0], r[1]], activeParams, hiddenActs, 'Sigmoid')
      return {
        x: [r[0], r[1]],
        yTrue: r[2],
        yPred: pass.layerA[pass.layerA.length - 1][0],
        cache: pass,
      }
    })
    const nextForward = {
      sampleRuns,
      sampleForDiagram: {
        ...sampleRuns[0],
        initialCache: previousSnapshot,
      },
    }
    setForwardState(nextForward)
    return nextForward
  }

  const runLossStage = (activeForward = forwardState) => {
    const n = activeForward.sampleRuns.length
    const mse = activeForward.sampleRuns.reduce((sum, s) => sum + (s.yTrue - s.yPred) ** 2, 0) / n
    const nextLoss = {
      mse,
      rows: activeForward.sampleRuns.map((s) => ({ x1: s.x[0], x2: s.x[1], yTrue: s.yTrue, yPred: s.yPred })),
    }
    setLossState(nextLoss)
    return nextLoss
  }

  const runBackwardStage = (activeForward = forwardState, activeParams = params) => {
    const gradW = activeParams.weights.map((layer) => layer.map((row) => row.map(() => 0)))
    const gradB = activeParams.biases.map((layer) => layer.map(() => 0))
    const n = activeForward.sampleRuns.length

    for (const sample of activeForward.sampleRuns) {
      let dLda = [2 * (sample.yPred - sample.yTrue)]

      for (let l = activeParams.weights.length - 1; l >= 0; l -= 1) {
        const zLayer = sample.cache.layerZ[l]
        const aLayer = sample.cache.layerA[l + 1]
        const aPrev = sample.cache.layerA[l]
        const actName = l === activeParams.weights.length - 1 ? 'Sigmoid' : activation

        const delta = zLayer.map((zv, j) => dLda[j] * activationDerivative(actName, zv, aLayer[j]))

        for (let j = 0; j < delta.length; j += 1) {
          gradB[l][j] += delta[j]
          for (let k = 0; k < aPrev.length; k += 1) {
            gradW[l][j][k] += delta[j] * aPrev[k]
          }
        }

        const dPrev = Array.from({ length: aPrev.length }, () => 0)
        for (let k = 0; k < aPrev.length; k += 1) {
          for (let j = 0; j < delta.length; j += 1) {
            dPrev[k] += activeParams.weights[l][j][k] * delta[j]
          }
        }
        dLda = dPrev
      }
    }

    for (let l = 0; l < gradW.length; l += 1) {
      for (let j = 0; j < gradW[l].length; j += 1) {
        gradB[l][j] /= n
        for (let k = 0; k < gradW[l][j].length; k += 1) gradW[l][j][k] /= n
      }
    }

    const layerGradientMagnitude = gradW.map((layer) => {
      const flat = layer.flat()
      const sumAbs = flat.reduce((sum, v) => sum + Math.abs(v), 0)
      return flat.length ? sumAbs / flat.length : 0
    })

    const strongestLayerIndex = layerGradientMagnitude.reduce(
      (best, val, idx) => (val > layerGradientMagnitude[best] ? idx : best),
      0
    )

    const nextGradients = {
      gradW,
      gradB,
      layerGradientMagnitude,
      strongestLayerIndex,
    }
    setGradientState(nextGradients)
    return nextGradients
  }

  const evaluateModel = (activeParams) => {
    const hiddenActs = Array.from({ length: activeParams.weights.length - 1 }, () => activation)
    const predictions = rows.map((r) => {
      const pass = forwardDense([r[0], r[1]], activeParams, hiddenActs, 'Sigmoid')
      return pass.layerA[pass.layerA.length - 1][0]
    })
    const mse = predictions.reduce((sum, pred, idx) => sum + (rows[idx][2] - pred) ** 2, 0) / rows.length
    const acc =
      (predictions.filter((pred, idx) => (pred >= 0.5 ? 1 : 0) === rows[idx][2]).length / rows.length) * 100
    return { mse, acc, predictions }
  }

  const parameterShiftStats = (fromParams, toParams) => {
    if (!fromParams || !toParams) return { meanWeightShift: 0, maxWeightShift: 0, meanBiasShift: 0 }

    let sumWeightShift = 0
    let weightCount = 0
    let maxWeightShift = 0
    let sumBiasShift = 0
    let biasCount = 0

    for (let l = 0; l < toParams.weights.length; l += 1) {
      for (let j = 0; j < toParams.weights[l].length; j += 1) {
        const biasDelta = Math.abs(toParams.biases[l][j] - fromParams.biases[l][j])
        sumBiasShift += biasDelta
        biasCount += 1
        for (let k = 0; k < toParams.weights[l][j].length; k += 1) {
          const delta = Math.abs(toParams.weights[l][j][k] - fromParams.weights[l][j][k])
          sumWeightShift += delta
          weightCount += 1
          if (delta > maxWeightShift) maxWeightShift = delta
        }
      }
    }

    return {
      meanWeightShift: weightCount ? sumWeightShift / weightCount : 0,
      maxWeightShift,
      meanBiasShift: biasCount ? sumBiasShift / biasCount : 0,
    }
  }

  const currentMetrics = useMemo(() => evaluateModel(params), [params, rows, activation])
  const shiftStats = useMemo(
    () => parameterShiftStats(initialSnapshot?.params, params),
    [initialSnapshot, params]
  )

  const runUpdateStage = (activeGradients = gradientState, activeParams = params) => {
    const oldParams = cloneParams(activeParams)
    const nextParams = cloneParams(activeParams)

    for (let l = 0; l < nextParams.weights.length; l += 1) {
      for (let j = 0; j < nextParams.weights[l].length; j += 1) {
        nextParams.biases[l][j] -= learningRate * activeGradients.gradB[l][j]
        for (let k = 0; k < nextParams.weights[l][j].length; k += 1) {
          nextParams.weights[l][j][k] -= learningRate * activeGradients.gradW[l][j][k]
        }
      }
    }

    const deltaByLayer = nextParams.weights.map((layer, l) => {
      let sumAbs = 0
      let count = 0
      for (let j = 0; j < layer.length; j += 1) {
        for (let k = 0; k < layer[j].length; k += 1) {
          sumAbs += Math.abs(layer[j][k] - oldParams.weights[l][j][k])
          count += 1
        }
      }
      return count ? sumAbs / count : 0
    })

    const evaluation = evaluateModel(nextParams)
    const nextEpoch = currentEpoch + 1

    setParams(nextParams)
    setHistory((prev) => ({ losses: [...prev.losses, evaluation.mse], accs: [...prev.accs, evaluation.acc] }))
    setUpdateState({
      deltaByLayer,
      sampleWeightBefore: oldParams.weights[0]?.[0]?.[0] ?? 0,
      sampleWeightAfter: nextParams.weights[0]?.[0]?.[0] ?? 0,
      latestLoss: evaluation.mse,
      latestAcc: evaluation.acc,
    })
    setCurrentEpoch(nextEpoch)

    if (nextEpoch >= epochs) setRunning(false)
    return nextEpoch
  }

  const performCurrentStage = () => {
    if (currentEpoch >= epochs) {
      setRunning(false)
      return
    }

    const stageKey = stageOrder[currentStageIndex]

    if (stageKey === 'forward') {
      runForwardStage()
      setCurrentStageIndex(1)
      return
    }

    if (stageKey === 'loss') {
      const forward = forwardState || runForwardStage()
      runLossStage(forward)
      setCurrentStageIndex(2)
      return
    }

    if (stageKey === 'backward') {
      const forward = forwardState || runForwardStage()
      runBackwardStage(forward)
      setCurrentStageIndex(3)
      return
    }

    const forward = forwardState || runForwardStage()
    const gradients = gradientState || runBackwardStage(forward)
    runUpdateStage(gradients)
    setForwardState(null)
    setLossState(null)
    setGradientState(null)
    setCurrentStageIndex(0)
  }

  const runOneEpoch = () => {
    if (currentEpoch >= epochs) {
      setRunning(false)
      return
    }
    const forward = runForwardStage()
    runLossStage(forward)
    const gradients = runBackwardStage(forward)
    runUpdateStage(gradients)
    setForwardState(null)
    setLossState(null)
    setGradientState(null)
    setCurrentStageIndex(0)
  }

  useEffect(() => {
    if (!running) return undefined
    const timer = setInterval(() => {
      performCurrentStage()
    }, 850)
    return () => clearInterval(timer)
  }, [running, currentStageIndex, currentEpoch, params, forwardState, gradientState])

  const signalFlowMode =
    currentStageIndex === 2 || currentStageIndex === 3
      ? 'backward'
      : currentStageIndex === 0 || currentStageIndex === 1
        ? 'forward'
        : 'idle'

  const guideSteps = [
    {
      key: 'configure',
      title: 'Configure',
      copy: 'Set layers, neurons, activation, learning rate, and dataset.',
    },
    {
      key: 'initialize',
      title: 'Initialize',
      copy: 'Reset the model to fresh random weights and biases.',
    },
    {
      key: 'forward',
      title: 'Forward + Evaluate',
      copy: 'Run prediction flow, then compute error with MSE loss.',
    },
    {
      key: 'backprop',
      title: 'Backprop + Update',
      copy: 'Trace gradients backward and apply updates with the learning rate.',
    },
    {
      key: 'iterate',
      title: 'Iterate + Visualize',
      copy: 'Repeat across epochs and watch charts for convergence trends.',
    },
  ]

  const activeGuideKey = (() => {
    const isIdleBeforeInit = !hasInitialized && !forwardState && !lossState && !gradientState && !updateState
    if (isIdleBeforeInit) return 'configure'

    const isIdleAfterInit =
      hasInitialized &&
      !history.losses.length &&
      !running &&
      currentStageIndex === 0 &&
      !forwardState &&
      !lossState &&
      !gradientState &&
      !updateState

    if (isIdleAfterInit) return 'initialize'
    if (history.losses.length > 0 && !running && currentStageIndex === 0 && !forwardState && !lossState && !gradientState && !updateState) return 'iterate'
    if (updateState || gradientState || currentStageIndex === 2 || currentStageIndex === 3) return 'backprop'
    if (lossState || (forwardState && currentStageIndex === 1)) return 'forward'
    if (forwardState || currentStageIndex === 0) return 'forward'
    return 'initialize'
  })()

  return (
    <>
      <Section title="engine setup">
        <div className="engine-guide">
          <p className="engine-guide-title">Fast Start Flow</p>
          <p className="engine-guide-subtitle">
            Move through the engine in this sequence and keep an eye on <strong>loss</strong>, <strong>gradients</strong>, and <strong>weight deltas</strong>.
          </p>
          <div className="engine-guide-flow" role="list" aria-label="Neural flow engine quick guide">
            {guideSteps.map((step, idx) => {
              const isActive = step.key === activeGuideKey
              const isDone = ['configure', 'initialize', 'forward', 'backprop'].includes(step.key) && step.key !== activeGuideKey && (history.losses.length > 0 || running || hasInitialized)
              return (
                <div
                  key={step.key}
                  className={isActive ? 'engine-guide-step active' : isDone ? 'engine-guide-step completed' : 'engine-guide-step'}
                  role="listitem"
                  style={{ '--step-delay': `${idx * 70}ms` }}
                >
                  <p className="engine-step-head">{step.title}</p>
                  <p className="engine-step-copy">{step.copy}</p>
                </div>
              )
            })}
          </div>
        </div>
        <div className="module-field-grid">
          <NumberField label="Hidden layers" value={hiddenLayerCount} onChange={(v) => setHiddenLayerCount(Math.max(1, Math.min(3, Math.round(v))))} step={1} />
          {neuronsPerHidden.slice(0, hiddenLayerCount).map((size, idx) => (
            <NumberField
              key={`hidden-size-${idx}`}
              label={`Neurons in H${idx + 1}`}
              value={size}
              onChange={(v) =>
                setNeuronsPerHidden((prev) => prev.map((p, i) => (i === idx ? Math.max(1, Math.min(12, Math.round(v))) : p)))
              }
              step={1}
            />
          ))}
          <label className="module-field">
            <span>Activation</span>
            <select value={activation} onChange={(e) => setActivation(e.target.value)}>
              {differentiableActivations.map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
          </label>
          <NumberField label="Learning rate" value={learningRate} onChange={(v) => setLearningRate(Math.max(0.001, Math.min(1, v)))} step={0.001} />
          <NumberField label="Epoch budget" value={epochs} onChange={(v) => setEpochs(Math.max(1, Math.min(500, Math.round(v))))} step={1} />
          <label className="module-field">
            <span>Dataset</span>
            <select value={gate} onChange={(e) => setGate(e.target.value)}>
              <option value="AND">AND</option>
              <option value="OR">OR</option>
              <option value="XOR">XOR</option>
            </select>
          </label>
        </div>
        <p className="module-note">
          <strong>Architecture</strong>: Input Layer (<strong>{layerSizes[0]}</strong> neurons)
          {layerSizes.slice(1, -1).map((size, idx) => (
            <span key={`arch-layer-${idx}`}>
              {' → '}Hidden Layer {idx + 1} (<strong>{size}</strong> neurons)
            </span>
          ))}
          {' → '}Output Layer (<strong>{layerSizes[layerSizes.length - 1]}</strong> neuron)
        </p>
        <div className="module-action-row">
          <button className="module-action-btn" onClick={resetEngine}>Initialize / Reset Model</button>
          <button className="module-action-btn" onClick={performCurrentStage}>Step Through Stage</button>
          <button className="module-action-btn" onClick={runOneEpoch}>Run One Epoch</button>
          <button className="module-action-btn primary" onClick={() => setRunning((prev) => !prev)}>
            {running ? 'Pause Training' : 'Resume Training'}
          </button>
        </div>
        <div className="engine-live-pills" role="status" aria-live="polite">
          <div className={running ? 'engine-live-pill stage is-running' : 'engine-live-pill stage'}>
            <span className="live-pill-label">Current Stage</span>
            <span key={`stage-${currentStageIndex}-${running ? 'run' : 'idle'}`} className="live-pill-value">
              {stageLabelMap[stageOrder[currentStageIndex]]}
            </span>
          </div>
          <div className={running ? 'engine-live-pill epoch is-running' : 'engine-live-pill epoch'}>
            <span className="live-pill-label">Epoch</span>
            <span key={`epoch-${currentEpoch}-${epochs}`} className="live-pill-value">
              {currentEpoch} / {epochs}
            </span>
          </div>
        </div>
      </Section>

      <Section title="Forward Flow">
        <p className="module-note">
          In the forward pass, input data moves through the network layer by layer, where it is transformed using weights,
          biases, and an activation function to produce outputs. Each layer's output becomes the input for the next,
          continuing until the final prediction is generated.
        </p>
        <p className="module-note formula-copy">a⁽ˡ⁾ = f(W⁽ˡ⁾ · a⁽ˡ⁻¹⁾ + b⁽ˡ⁾)</p>
        <AnimatedSignalMap
          layerSizes={layerSizes}
          layerLabels={layerLabels}
          layerA={forwardState?.sampleForDiagram?.cache?.layerA}
          initialLayerA={forwardState?.sampleForDiagram?.initialCache}
          flowMode={signalFlowMode}
        />
        {forwardState ? (
          <table className="module-table">
            <thead>
              <tr><th>x1</th><th>x2</th><th>y_true</th><th>y_pred</th></tr>
            </thead>
            <tbody>
              {forwardState.sampleRuns.map((s, i) => (
                <tr key={i}><td>{s.x[0]}</td><td>{s.x[1]}</td><td>{s.yTrue}</td><td>{s.yPred.toFixed(4)}</td></tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </Section>

      <Section title="Loss Computation">
        <p className="module-note">
          After the forward pass, the model's prediction is compared with the actual value to measure error. This
          difference is quantified using a loss function, which indicates how far the model is from the correct output.
        </p>
        <p className="module-note formula-copy">L = (1 / n) · Σ (y_true − y_pred)²</p>
        {lossState ? <p className="module-note"><strong>Current MSE Loss:</strong> {lossState.mse.toFixed(6)}</p> : <p className="module-note">Loss is computed after forward flow.</p>}
        <Plot
          data={[
            {
              x: history.losses.map((_, i) => i + 1),
              y: history.losses,
              type: 'scatter',
              mode: 'lines+markers',
              line: { color: '#5227ff', width: 2 },
              marker: { color: '#5227ff', size: 6 },
            },
          ]}
          layout={{
            title: { text: 'Loss Trend Across Epochs', x: 0.5 },
            xaxis: { title: 'Epoch Index', showline: true, linecolor: '#94a3b8', ticks: 'outside', ticklen: 5, showgrid: true, gridcolor: '#e5e7eb' },
            yaxis: { title: 'MSE Loss Value', showline: true, linecolor: '#94a3b8', ticks: 'outside', ticklen: 5, showgrid: true, gridcolor: '#e5e7eb' },
            margin: { t: 52, l: 64, r: 22, b: 52 },
          }}
          style={{ width: '100%', height: '340px' }}
          config={{ displayModeBar: false }}
        />
      </Section>

      <Section title="Backward Flow">
        <p className="module-note">
          In the backward pass, the error is propagated from the output layer back to the input layer to determine how
          each parameter contributed to the loss. Gradients are computed using the chain rule, and this process flows
          from right to left for learning.
        </p>
        <p className="module-note formula-copy">∂L/∂W = (∂L/∂a) · (∂a/∂z) · (∂z/∂W)</p>
        {gradientState ? (
          <>
            <Plot
              data={build3DBarTraces(
                gradientState.layerGradientMagnitude,
                gradientState.layerGradientMagnitude.map((_, i) => `Layer ${i + 1}`),
                '#ef4444'
              )}
              layout={{
                title: { text: '3D Gradient Strength by Layer', x: 0.5 },
                scene: {
                  xaxis: { title: 'Layer Index' },
                  yaxis: { title: 'Bar Depth Axis', range: [-0.4, 0.4] },
                  zaxis: { title: 'Mean |∂L/∂W|' },
                  camera: { eye: { x: 1.3, y: 1.15, z: 1.05 } },
                  aspectmode: 'cube',
                },
                uirevision: 'gradient-3d-fixed',
                margin: { t: 52, l: 20, r: 20, b: 20 },
              }}
              style={{ width: '100%', height: '340px' }}
              config={{ displayModeBar: false }}
            />
            <p className="module-note">Highest gradient contribution currently appears in Layer {gradientState.strongestLayerIndex + 1}.</p>
          </>
        ) : (
          <p className="module-note">Run backward stage to compute and visualize gradients.</p>
        )}
      </Section>

      <Section title="Weight Update">
        <p className="module-note">
          After gradients are computed, the model updates its weights to reduce the loss. Each weight is adjusted in
          the opposite direction of the gradient, scaled by the learning rate. This step applies learned corrections
          and shows how parameters change during training.
        </p>
        <p className="module-note formula-copy">W = W − η · (∂L/∂W)</p>
        {updateState ? (
          <>
            <table className="module-table">
              <thead>
                <tr><th>Layer</th><th>Mean |delta W|</th></tr>
              </thead>
              <tbody>
                {updateState.deltaByLayer.map((val, i) => (
                  <tr key={i}><td>Layer {i + 1}</td><td>{val.toFixed(6)}</td></tr>
                ))}
              </tbody>
            </table>
            <p className="module-note">
              Sample weight change: {updateState.sampleWeightBefore.toFixed(6)} {'->'} {updateState.sampleWeightAfter.toFixed(6)}
            </p>
            <p className="module-note">
              Output summary: Latest Loss = {updateState.latestLoss.toFixed(6)} | Latest Accuracy = {updateState.latestAcc.toFixed(2)}%
            </p>
          </>
        ) : (
          <p className="module-note">Run update stage to apply gradients and view weight deltas.</p>
        )}

        <div className="plot-grid-2">
          <Plot
            data={[
              {
                x: history.accs.map((_, i) => i + 1),
                y: history.accs,
                z: history.accs.map((v, i, arr) => (arr.length <= 1 ? 0 : i / (arr.length - 1))),
                type: 'scatter3d',
                mode: 'lines+markers',
                line: { color: '#2563eb', width: 4 },
                marker: { color: '#2563eb', size: 3 },
              },
            ]}
            layout={{
              title: { text: '3D Accuracy Trend', x: 0.5 },
              scene: {
                xaxis: { title: 'Epoch Index' },
                yaxis: { title: 'Accuracy (%)', range: [0, 105] },
                zaxis: { title: 'Training Progress' },
                camera: { eye: { x: 1.25, y: 1.1, z: 0.95 } },
              },
              margin: { t: 52, l: 20, r: 20, b: 20 },
            }}
            style={{ width: '100%', height: '320px' }}
            config={{ displayModeBar: false }}
          />
          <Plot
            data={[
              {
                x: history.losses.map((_, i) => i + 1),
                y: history.losses,
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#7c3aed', width: 2 },
                marker: { color: '#7c3aed', size: 6 },
              },
            ]}
            layout={{
              title: { text: 'Epoch Loop Progress (Loss)', x: 0.5 },
              xaxis: { title: 'Epoch Index', showline: true, linecolor: '#94a3b8', ticks: 'outside', ticklen: 5, showgrid: true, gridcolor: '#e5e7eb' },
              yaxis: { title: 'Loss Value', showline: true, linecolor: '#94a3b8', ticks: 'outside', ticklen: 5, showgrid: true, gridcolor: '#e5e7eb' },
              margin: { t: 52, l: 64, r: 22, b: 52 },
            }}
            style={{ width: '100%', height: '320px' }}
            config={{ displayModeBar: false }}
          />
        </div>
      </Section>

      <Section title="Run Summary">
        <p className="module-note">
          <strong>Input & Configuration:</strong> This run used <strong>{layerSizes.length - 1}</strong> trainable layers with
          architecture <strong>Input({layerSizes[0]})</strong>
          {layerSizes.slice(1, -1).map((size, idx) => (
            <span key={`summary-arch-${idx}`}>{' → '}<strong>H{idx + 1}({size})</strong></span>
          ))}
          {' → '}<strong>Output({layerSizes[layerSizes.length - 1]})</strong>, activation <strong>{activation}</strong> in hidden layers,
          output activation <strong>Sigmoid</strong>, learning rate <strong>{learningRate.toFixed(4)}</strong>, and dataset <strong>{gate}</strong>.
        </p>

        <p className="module-note">
          <strong>Flow of Data:</strong> The engine executed the cycle
          <strong> Forward → Loss → Backward → Update</strong> across <strong>{currentEpoch}</strong> completed epochs. Inputs were transformed
          layer by layer, error was computed, gradients were propagated backward, and parameter updates were applied each cycle.
        </p>

        <p className="module-note">
          <strong>Learning Behavior:</strong> Initial loss <strong>{(initialSnapshot?.loss ?? 0).toFixed(6)}</strong> changed to
          <strong> {currentMetrics.mse.toFixed(6)}</strong> ({(currentMetrics.mse - (initialSnapshot?.loss ?? 0)).toFixed(6)} delta).
          Initial accuracy <strong>{(initialSnapshot?.acc ?? 0).toFixed(2)}%</strong> changed to
          <strong> {currentMetrics.acc.toFixed(2)}%</strong> ({(currentMetrics.acc - (initialSnapshot?.acc ?? 0)).toFixed(2)} pts).
          Mean |ΔW| is <strong>{shiftStats.meanWeightShift.toFixed(6)}</strong>, max |ΔW| is <strong>{shiftStats.maxWeightShift.toFixed(6)}</strong>,
          and mean |Δb| is <strong>{shiftStats.meanBiasShift.toFixed(6)}</strong>.
        </p>

        <p className="module-note">
          <strong>Final Results:</strong> The current model state reflects learned parameter corrections and now produces predictions
          closer to expected outputs than at initialization.
        </p>

        <table className="module-table">
          <thead>
            <tr>
              <th>x1</th>
              <th>x2</th>
              <th>y_true</th>
              <th>y_pred_initial</th>
              <th>y_pred_final</th>
              <th>|error_final|</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, idx) => {
              const initialPred = initialSnapshot?.predictions?.[idx] ?? 0
              const finalPred = currentMetrics.predictions[idx] ?? 0
              return (
                <tr key={`summary-row-${idx}`}>
                  <td>{r[0]}</td>
                  <td>{r[1]}</td>
                  <td>{r[2]}</td>
                  <td>{initialPred.toFixed(4)}</td>
                  <td>{finalPred.toFixed(4)}</td>
                  <td>{Math.abs(r[2] - finalPred).toFixed(4)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </Section>
    </>
  )
}

function trainTinyMLP({ hiddenSize, lr, epochs, gate, hiddenAct }) {
  const datasets = {
    AND: [
      [0, 0, 0],
      [0, 1, 0],
      [1, 0, 0],
      [1, 1, 1],
    ],
    OR: [
      [0, 0, 0],
      [0, 1, 1],
      [1, 0, 1],
      [1, 1, 1],
    ],
    XOR: [
      [0, 0, 0],
      [0, 1, 1],
      [1, 0, 1],
      [1, 1, 0],
    ],
  }

  const rows = datasets[gate]
  const X = rows.map((r) => [r[0], r[1]])
  const Y = rows.map((r) => r[2])

  let W1 = Array.from({ length: hiddenSize }, () => [randomIn(-1, 1), randomIn(-1, 1)])
  let b1 = Array.from({ length: hiddenSize }, () => randomIn(-0.2, 0.2))
  let W2 = Array.from({ length: hiddenSize }, () => randomIn(-1, 1))
  let b2 = randomIn(-0.2, 0.2)

  const losses = []
  const accs = []
  let initialPreds = []
  let finalPreds = []

  for (let e = 0; e < epochs; e += 1) {
    let totalLoss = 0
    let correct = 0

    const dW1 = Array.from({ length: hiddenSize }, () => [0, 0])
    const db1 = Array.from({ length: hiddenSize }, () => 0)
    const dW2 = Array.from({ length: hiddenSize }, () => 0)
    let db2 = 0
    const predsThisEpoch = []

    for (let s = 0; s < X.length; s += 1) {
      const x = X[s]
      const y = Y[s]

      const z1 = W1.map((row, i) => row[0] * x[0] + row[1] * x[1] + b1[i])
      const a1 = z1.map((v) => ACTIVATIONS[hiddenAct](v))
      const z2 = a1.reduce((sum, v, i) => sum + v * W2[i], b2)
      const a2 = ACTIVATIONS.Sigmoid(z2)
      predsThisEpoch.push(a2)

      const eps = 1e-9
      totalLoss += -(y * Math.log(Math.max(eps, a2)) + (1 - y) * Math.log(Math.max(eps, 1 - a2)))

      if ((a2 >= 0.5 ? 1 : 0) === y) correct += 1

      const dLda2 = a2 - y
      const da2dz2 = a2 * (1 - a2)
      const dLdz2 = dLda2 * da2dz2

      for (let i = 0; i < hiddenSize; i += 1) {
        dW2[i] += dLdz2 * a1[i]
      }
      db2 += dLdz2

      for (let i = 0; i < hiddenSize; i += 1) {
        const dLda1 = dLdz2 * W2[i]
        const da1dz1 =
          hiddenAct === 'ReLU'
            ? (a1[i] > 0 ? 1 : 0)
            : hiddenAct === 'Tanh'
              ? 1 - a1[i] * a1[i]
              : hiddenAct === 'Linear'
                ? 1
                : a1[i] * (1 - a1[i])
        const dLdz1 = dLda1 * da1dz1

        dW1[i][0] += dLdz1 * x[0]
        dW1[i][1] += dLdz1 * x[1]
        db1[i] += dLdz1
      }
    }

    const n = X.length
    for (let i = 0; i < hiddenSize; i += 1) {
      W2[i] -= (lr * dW2[i]) / n
      b1[i] -= (lr * db1[i]) / n
      W1[i][0] -= (lr * dW1[i][0]) / n
      W1[i][1] -= (lr * dW1[i][1]) / n
    }
    b2 -= (lr * db2) / n

    losses.push(totalLoss / n)
    accs.push((correct / n) * 100)

    if (e === 0) initialPreds = predsThisEpoch.slice()
    finalPreds = predsThisEpoch.slice()
  }

  return { losses, accs, rows, initialPreds, finalPreds }
}

function MLPModule() {
  const [gate, setGate] = useState('XOR')
  const [hiddenSize, setHiddenSize] = useState(4)
  const [epochs, setEpochs] = useState(1000)
  const [lr, setLr] = useState(0.1)
  const [hiddenAct, setHiddenAct] = useState('Sigmoid')
  const [result, setResult] = useState(null)

  const totalParams = 2 * hiddenSize + hiddenSize + hiddenSize + 1
  const firstLoss = result?.losses?.[0] ?? null
  const finalLoss = result?.losses?.[result.losses.length - 1] ?? null
  const firstAcc = result?.accs?.[0] ?? null
  const finalAcc = result?.accs?.[result.accs.length - 1] ?? null

  return (
    <>
      <Section title="how to begin">
        <div className="engine-guide">
          <p className="engine-guide-title">Quick Start For MLP</p>
          <p className="engine-guide-subtitle">
            Configure model width and learning settings, train on a gate dataset, then inspect <strong>loss</strong>,
            <strong> accuracy</strong>, and <strong>prediction shifts</strong>.
          </p>
          <div className="engine-guide-flow" role="list" aria-label="MLP quick guide">
            <div className="engine-guide-step" role="listitem">
              <p className="engine-step-head">Choose Data</p>
              <p className="engine-step-copy">Select AND, OR, or XOR truth-table training set.</p>
            </div>
            <div className="engine-guide-step" role="listitem">
              <p className="engine-step-head">Configure</p>
              <p className="engine-step-copy">Set hidden neurons, epochs, learning rate, and activation.</p>
            </div>
            <div className="engine-guide-step" role="listitem">
              <p className="engine-step-head">Train</p>
              <p className="engine-step-copy">Run optimization to reduce error across epochs.</p>
            </div>
            <div className="engine-guide-step" role="listitem">
              <p className="engine-step-head">Analyze</p>
              <p className="engine-step-copy">Read loss and accuracy curves for convergence behavior.</p>
            </div>
            <div className="engine-guide-step" role="listitem">
              <p className="engine-step-head">Summarize</p>
              <p className="engine-step-copy">Compare initial and final predictions per sample.</p>
            </div>
          </div>
        </div>
      </Section>

      <Section title="data source">
        <p className="module-note">
          The selected truth table determines the nonlinear complexity the MLP must learn during training.
        </p>
        <label className="module-field">
          <span>Gate dataset</span>
          <select value={gate} onChange={(e) => setGate(e.target.value)}>
            <option value="AND">AND</option>
            <option value="OR">OR</option>
            <option value="XOR">XOR</option>
          </select>
        </label>
      </Section>

      <Section title="architecture and hyperparameters">
        <p className="module-note">
          This MLP uses one hidden layer where nonlinear activation helps capture patterns not solvable by a single
          linear boundary.
        </p>
        <p className="module-note">
          <strong>Architecture</strong>: Input Layer (<strong>2</strong> neurons) {'→'} Hidden Layer 1 (<strong>{hiddenSize}</strong> neurons)
          {' → '}Output Layer (<strong>1</strong> neuron)
        </p>
        <p className="module-note formula-copy">Parameters = 2H + H + H + 1 = <strong>{totalParams}</strong></p>
        <div className="slider-grid">
          <div>
            <p className="slider-label">Hidden neurons</p>
            <ElasticSlider defaultValue={hiddenSize} startingValue={1} maxValue={16} isStepped stepSize={1} onChange={setHiddenSize} />
          </div>
          <div>
            <p className="slider-label">Epochs</p>
            <ElasticSlider defaultValue={epochs} startingValue={100} maxValue={5000} isStepped stepSize={100} onChange={setEpochs} />
          </div>
          <div>
            <p className="slider-label">Learning Rate</p>
            <ElasticSlider defaultValue={lr} startingValue={0.01} maxValue={1} isStepped stepSize={0.01} onChange={setLr} />
          </div>
          <label className="module-field mlp-activation-field">
            <span>Hidden activation</span>
            <select value={hiddenAct} onChange={(e) => setHiddenAct(e.target.value)}>
              <option value="Sigmoid">Sigmoid</option>
              <option value="ReLU">ReLU</option>
              <option value="Tanh">Tanh</option>
              <option value="Linear">Linear</option>
            </select>
          </label>
        </div>

        <div className="module-action-row">
          <button className="module-action-btn primary" onClick={() => setResult(trainTinyMLP({ hiddenSize, lr, epochs, gate, hiddenAct }))}>
            Train MLP
          </button>
        </div>
      </Section>

      <Section title="training curves">
        <p className="module-note">
          Loss curve shows optimization progress, while accuracy curve shows how quickly predictions align with targets.
        </p>
        {result ? (
          <div className="plot-grid-2">
            <Plot
              data={[{ x: result.losses.map((_, i) => i + 1), y: result.losses, type: 'scatter', mode: 'lines+markers', line: { color: '#2563eb', width: 3 }, marker: { size: 5, color: '#2563eb' }, fill: 'tozeroy', fillcolor: 'rgba(37,99,235,0.08)' }]}
              layout={{ title: { text: 'Training Loss', x: 0.5 }, xaxis: { title: 'Epoch' }, yaxis: { title: 'Loss' }, margin: { t: 52, l: 56, r: 22, b: 44 } }}
              style={{ width: '100%', height: '320px' }}
              config={{ displayModeBar: false }}
            />
            <Plot
              data={[{ x: result.accs.map((_, i) => i + 1), y: result.accs, z: result.accs.map((v, i, arr) => (arr.length <= 1 ? 0 : i / (arr.length - 1))), type: 'scatter3d', mode: 'lines+markers', line: { color: '#5227ff', width: 4 }, marker: { size: 3, color: '#5227ff' } }]}
              layout={{
                title: { text: '3D Training Accuracy', x: 0.5 },
                scene: {
                  xaxis: { title: 'Epoch' },
                  yaxis: { title: 'Accuracy (%)', range: [0, 105] },
                  zaxis: { title: 'Training Progress' },
                  camera: { eye: { x: 1.25, y: 1.1, z: 0.95 } },
                },
                margin: { t: 52, l: 20, r: 20, b: 20 },
              }}
              style={{ width: '100%', height: '320px' }}
              config={{ displayModeBar: false }}
            />
          </div>
        ) : (
          <p className="module-note">Train the MLP to visualize loss and accuracy curves.</p>
        )}
      </Section>

      <Section title="truth table">
        <p className="module-note">
          Target labels define expected logic behavior for each input pair in the selected dataset.
        </p>
        <table className="module-table">
          <thead><tr><th>x1</th><th>x2</th><th>target</th></tr></thead>
          <tbody>
            {(result?.rows || []).map((r, i) => (
              <tr key={i}><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td></tr>
            ))}
          </tbody>
        </table>
      </Section>

      <Section title="run summary">
        {result ? (
          <>
            <p className="module-note">
              <strong>Input & Configuration:</strong> Dataset <strong>{gate}</strong>, hidden neurons <strong>{hiddenSize}</strong>,
              epochs <strong>{epochs}</strong>, learning rate <strong>{lr.toFixed(4)}</strong>, and hidden activation
              <strong> {hiddenAct}</strong>.
            </p>

            <p className="module-note">
              <strong>Flow of Data:</strong> Inputs moved through Input {'→'} Hidden {'→'} Output layers over
              <strong> {result.losses.length}</strong> epochs. At each epoch, forward outputs were compared to targets,
              errors were backpropagated, and parameters were updated.
            </p>

            <p className="module-note">
              <strong>Learning Behavior:</strong> Loss changed from <strong>{(firstLoss ?? 0).toFixed(6)}</strong> to
              <strong> {(finalLoss ?? 0).toFixed(6)}</strong> ({((finalLoss ?? 0) - (firstLoss ?? 0)).toFixed(6)} delta), and
              accuracy changed from <strong>{(firstAcc ?? 0).toFixed(2)}%</strong> to
              <strong> {(finalAcc ?? 0).toFixed(2)}%</strong> ({((finalAcc ?? 0) - (firstAcc ?? 0)).toFixed(2)} pts).
            </p>

            <p className="module-note">
              <strong>Final Results:</strong> The final predictions below represent the learned mapping after training,
              showing how each row moved from initial to final output.
            </p>

            <table className="module-table">
              <thead>
                <tr><th>x1</th><th>x2</th><th>target</th><th>y_pred_initial</th><th>y_pred_final</th><th>|error_final|</th></tr>
              </thead>
              <tbody>
                {result.rows.map((r, i) => {
                  const initPred = result.initialPreds?.[i] ?? 0
                  const finalPred = result.finalPreds?.[i] ?? 0
                  return (
                    <tr key={`mlp-summary-${i}`}>
                      <td>{r[0]}</td>
                      <td>{r[1]}</td>
                      <td>{r[2]}</td>
                      <td>{initPred.toFixed(4)}</td>
                      <td>{finalPred.toFixed(4)}</td>
                      <td>{Math.abs(r[2] - finalPred).toFixed(4)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </>
        ) : (
          <p className="module-note">Train the model once to generate a personalized summary from your current configuration.</p>
        )}
      </Section>
    </>
  )
}

function PerceptronAndMLPModule() {
  const [activeView, setActiveView] = useState('single-layer')

  const moduleItems = [
    { label: 'Single Layer', href: 'single-layer' },
    { label: 'MLP', href: 'mlp' },
  ]

  const moduleExplanation =
    activeView === 'mlp'
      ? 'Multilayer perceptron: adjust architecture and hyperparameters, then train to inspect loss and accuracy trends.'
      : 'Single-layer perceptron: tune weights, bias, and activation to observe decision boundary behavior.'

  return (
    <>
      <div className="study-pill-nav-wrap">
        <PillNav
          items={moduleItems}
          activeHref={activeView}
          onItemSelect={(item) => setActiveView(item.href)}
          baseColor="#ffffff"
          pillColor="#060010"
          hoveredPillTextColor="#060010"
          pillTextColor="#ffffff"
          initialLoadAnimation
        />
      </div>
      <p className="module-note">{moduleExplanation}</p>
      {activeView === 'mlp' ? <MLPModule /> : <PerceptronModule />}
    </>
  )
}

function createSampleImageDataURL() {
  const canvas = document.createElement('canvas')
  canvas.width = 400
  canvas.height = 300
  const ctx = canvas.getContext('2d')
  if (!ctx) return null

  ctx.fillStyle = '#ffffff'
  ctx.fillRect(0, 0, canvas.width, canvas.height)

  ctx.strokeStyle = '#ef4444'
  ctx.lineWidth = 3
  ctx.strokeRect(50, 50, 100, 100)

  ctx.strokeStyle = '#22c55e'
  ctx.beginPath()
  ctx.arc(250, 100, 40, 0, Math.PI * 2)
  ctx.stroke()

  ctx.strokeStyle = '#3b82f6'
  ctx.beginPath()
  ctx.ellipse(200, 250, 60, 30, 0, 0, Math.PI * 2)
  ctx.stroke()

  ctx.fillStyle = '#111827'
  ctx.font = '18px Segoe UI'
  ctx.fillText('Sample Image', 10, 285)

  return canvas.toDataURL('image/png')
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = src
  })
}

function clampByte(v) {
  return Math.max(0, Math.min(255, v))
}

function getGrayArray(imgData) {
  const { data, width, height } = imgData
  const out = new Float32Array(width * height)
  for (let i = 0; i < width * height; i += 1) {
    const di = i * 4
    out[i] = 0.299 * data[di] + 0.587 * data[di + 1] + 0.114 * data[di + 2]
  }
  return out
}

function sobelMagnitude(gray, width, height, axis = 'both') {
  const gxKernel = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
  const gyKernel = [-1, -2, -1, 0, 0, 0, 1, 2, 1]
  const out = new Uint8ClampedArray(width * height)

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      let gx = 0
      let gy = 0
      let k = 0
      for (let ky = -1; ky <= 1; ky += 1) {
        for (let kx = -1; kx <= 1; kx += 1) {
          const idx = (y + ky) * width + (x + kx)
          gx += gray[idx] * gxKernel[k]
          gy += gray[idx] * gyKernel[k]
          k += 1
        }
      }

      let val = 0
      if (axis === 'x') val = Math.abs(gx)
      else if (axis === 'y') val = Math.abs(gy)
      else val = Math.sqrt(gx * gx + gy * gy)

      out[y * width + x] = clampByte(val)
    }
  }
  return out
}

function toImageDataFromGray(gray, width, height) {
  const rgba = new Uint8ClampedArray(width * height * 4)
  for (let i = 0; i < width * height; i += 1) {
    const v = gray[i]
    const di = i * 4
    rgba[di] = v
    rgba[di + 1] = v
    rgba[di + 2] = v
    rgba[di + 3] = 255
  }
  return new ImageData(rgba, width, height)
}

function componentCount(binary, width, height) {
  const visited = new Uint8Array(width * height)
  const dirs = [-1, 0, 1]
  let count = 0

  for (let i = 0; i < binary.length; i += 1) {
    if (binary[i] === 0 || visited[i]) continue
    count += 1
    const stack = [i]
    visited[i] = 1

    while (stack.length) {
      const cur = stack.pop()
      const x = cur % width
      const y = Math.floor(cur / width)

      for (const dy of dirs) {
        for (const dx of dirs) {
          if (dx === 0 && dy === 0) continue
          const nx = x + dx
          const ny = y + dy
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue
          const ni = ny * width + nx
          if (!visited[ni] && binary[ni] > 0) {
            visited[ni] = 1
            stack.push(ni)
          }
        }
      }
    }
  }
  return count
}

async function processImage(source, operation, params) {
  const img = await loadImage(source)
  const canvas = document.createElement('canvas')
  canvas.width = img.width
  canvas.height = img.height
  const ctx = canvas.getContext('2d')
  if (!ctx) return null

  ctx.drawImage(img, 0, 0)
  let outData = ctx.getImageData(0, 0, canvas.width, canvas.height)
  let contourCount = null

  if (operation === 'Original') {
    // no-op
  } else if (operation === 'Grayscale') {
    const gray = getGrayArray(outData)
    outData = toImageDataFromGray(gray, canvas.width, canvas.height)
  } else if (operation === 'Blur') {
    const temp = document.createElement('canvas')
    temp.width = canvas.width
    temp.height = canvas.height
    const tctx = temp.getContext('2d')
    if (tctx) {
      tctx.filter = `blur(${Math.max(1, params.kernel / 2)}px)`
      tctx.drawImage(img, 0, 0)
      outData = tctx.getImageData(0, 0, canvas.width, canvas.height)
    }
  } else if (operation === 'Canny Edge Detection') {
    const gray = getGrayArray(outData)
    const mag = sobelMagnitude(gray, canvas.width, canvas.height, 'both')
    const edge = new Uint8ClampedArray(mag.length)
    for (let i = 0; i < mag.length; i += 1) {
      edge[i] = mag[i] >= params.cannyLow && mag[i] <= params.cannyHigh ? 255 : mag[i] > params.cannyHigh ? 255 : 0
    }
    outData = toImageDataFromGray(edge, canvas.width, canvas.height)
  } else if (operation === 'Sobel Edge Detection') {
    const gray = getGrayArray(outData)
    const sobel = sobelMagnitude(gray, canvas.width, canvas.height, params.sobelAxis)
    outData = toImageDataFromGray(sobel, canvas.width, canvas.height)
  } else if (operation === 'Binary Threshold') {
    const gray = getGrayArray(outData)
    const thr = new Uint8ClampedArray(gray.length)
    for (let i = 0; i < gray.length; i += 1) thr[i] = gray[i] >= params.threshold ? 255 : 0
    outData = toImageDataFromGray(thr, canvas.width, canvas.height)
  } else if (operation === 'Contour Detection') {
    const gray = getGrayArray(outData)
    const edges = sobelMagnitude(gray, canvas.width, canvas.height, 'both')
    const binary = new Uint8ClampedArray(edges.length)
    for (let i = 0; i < edges.length; i += 1) binary[i] = edges[i] >= 60 ? 255 : 0
    contourCount = componentCount(binary, canvas.width, canvas.height)

    const rgba = outData.data
    for (let i = 0; i < binary.length; i += 1) {
      if (!binary[i]) continue
      const di = i * 4
      rgba[di] = 34
      rgba[di + 1] = 197
      rgba[di + 2] = 94
      rgba[di + 3] = 255
    }
    outData = new ImageData(rgba, canvas.width, canvas.height)
  }

  ctx.putImageData(outData, 0, 0)
  const px = outData.data
  let min = 255
  let max = 0
  for (let i = 0; i < px.length; i += 4) {
    min = Math.min(min, px[i], px[i + 1], px[i + 2])
    max = Math.max(max, px[i], px[i + 1], px[i + 2])
  }

  return {
    src: canvas.toDataURL('image/png'),
    shape: `${canvas.height} x ${canvas.width} x 3`,
    dtype: 'uint8',
    min,
    max,
    contourCount,
  }
}

function CNNLabModule() {
  const [selectedModel, setSelectedModel] = useState('efficientnet_b0')
  const [uploadedFile, setUploadedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [predictions, setPredictions] = useState([])
  const [inputImage, setInputImage] = useState('')
  const [featureMaps, setFeatureMaps] = useState([])
  const [gradCam, setGradCam] = useState(null)
  const [includeGradCam, setIncludeGradCam] = useState(true)
  const [labInfo, setLabInfo] = useState(null)

  const modelNotes = {
    efficientnet_b0: {
      title: 'EfficientNet-B0',
      summary: 'Compound scaling balances depth, width, and resolution for strong accuracy per parameter.',
      params: '~5.3M params',
      speed: 'Moderate speed',
      strength: 'Best top-1 accuracy in this lab',
    },
    resnet50: {
      title: 'ResNet50',
      summary: 'Residual skip-connections stabilize deep optimization and keep strong transfer learning behavior.',
      params: '~25.6M params',
      speed: 'Balanced speed',
      strength: 'Reliable representation depth',
    },
    mobilenet_v2: {
      title: 'MobileNetV2',
      summary: 'Depthwise separable convolutions with inverted residual blocks favor latency efficiency.',
      params: '~3.5M params',
      speed: 'Fastest option',
      strength: 'Best for low-latency inference',
    },
  }

  const pipelineStages = [
    {
      key: 'input',
      title: '1. Input + Preprocessing',
      detail: 'Uploaded RGB image is resized to 224x224 and normalized with ImageNet mean/std so feature scales match pretrained expectations.',
      output: 'Tensor shape: 1 x 3 x 224 x 224',
    },
    {
      key: 'classification',
      title: '2. Classification Head',
      detail: 'Backbone features pass through the classifier and logits are transformed by softmax to estimate class probabilities.',
      output: 'Top-k classes with confidence',
    },
    {
      key: 'features',
      title: '3. Feature Map Probing',
      detail: 'Early, middle, and deep layers are sampled to visualize how representations evolve from edges to semantic patterns.',
      output: 'Layer-wise activation grids',
    },
    {
      key: 'gradcam',
      title: '4. Grad-CAM Attribution',
      detail: 'Gradients from the target class weight final convolution channels, producing a class-specific saliency heatmap overlay.',
      output: 'Heatmap + image overlay',
    },
  ]

  const selectedModelNote = modelNotes[selectedModel]

  const stageState = useMemo(
    () => ({
      input: Boolean(previewUrl),
      classification: predictions.length > 0,
      features: featureMaps.length > 0,
      gradcam: includeGradCam ? Boolean(gradCam) : false,
    }),
    [previewUrl, predictions, featureMaps, includeGradCam, gradCam]
  )

  const predictionChart = useMemo(() => {
    if (!predictions.length) return null

    const labels = predictions.map((pred, idx) => {
      const trimmed = pred.class_label.length > 28 ? `${pred.class_label.slice(0, 28)}...` : pred.class_label
      return `${idx + 1}. ${trimmed}`
    })
    const values = predictions.map((pred) => Number((pred.confidence * 100).toFixed(2)))

    return {
      data: [
        {
          type: 'bar',
          x: values,
          y: labels,
          orientation: 'h',
          marker: {
            color: ['#5227ff', '#2d6cf6', '#1fb6ff'],
            line: { color: '#ffffff', width: 1 },
          },
          text: values.map((v) => `${v.toFixed(2)}%`),
          textposition: 'inside',
          insidetextanchor: 'middle',
          hovertemplate: '%{y}<br>%{x:.2f}%<extra></extra>',
        },
      ],
      layout: {
        margin: { l: 140, r: 20, t: 10, b: 34 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        height: 250,
        xaxis: {
          title: 'Confidence (%)',
          range: [0, 100],
          gridcolor: '#e2e8f0',
          zeroline: false,
        },
        yaxis: {
          automargin: true,
        },
      },
      config: { displayModeBar: false },
    }
  }, [predictions])

  const topPrediction = predictions[0] || null

  const confidenceBand = useMemo(() => {
    if (!topPrediction) return null
    const c = topPrediction.confidence
    if (c >= 0.8) return 'high confidence'
    if (c >= 0.55) return 'moderate confidence'
    return 'low confidence'
  }, [topPrediction])

  const featureMapExplanation = useMemo(() => {
    if (!featureMaps.length) return null

    const stageNames = featureMaps.map((fm) => fm.stage)
    const uploadedName = uploadedFile?.name || 'your uploaded image'
    const modelTitle = selectedModelNote.title
    const mainLabel = topPrediction?.class_label || 'the predicted class'

    const confidenceMessage =
      confidenceBand === 'high confidence'
        ? `Because prediction confidence is high, these activations look relatively stable for ${mainLabel}.`
        : confidenceBand === 'moderate confidence'
          ? `Since confidence is moderate, you should expect mixed evidence across stages for ${mainLabel}.`
          : 'Low confidence means the network may be seeing competing cues, so feature responses can appear diffuse.'

    return {
      summary: `For ${uploadedName}, ${modelTitle} builds meaning progressively through ${stageNames.length} sampled stages: ${stageNames.join(', ')}.`,
      significance:
        'Feature maps reveal what the model keeps and discards while moving from local patterns (edges/textures) to higher-level object parts.',
      personalized: confidenceMessage,
      learnerTip:
        'If early maps are very clear but deep maps are noisy, the image has strong low-level structure but weaker class-specific semantics for this model.',
    }
  }, [featureMaps, uploadedFile, selectedModelNote.title, topPrediction, confidenceBand])

  const gradCamExplanation = useMemo(() => {
    if (!gradCam) return null

    const uploadedName = uploadedFile?.name || 'the uploaded image'
    const modelTitle = selectedModelNote.title
    const pct = (gradCam.confidence * 100).toFixed(2)

    const confidenceMeaning =
      gradCam.confidence >= 0.8
        ? 'The model is strongly committed to this decision, so highlighted regions are likely core evidence.'
        : gradCam.confidence >= 0.55
          ? 'The model has partial certainty, so highlighted regions are informative but not exclusively decisive.'
          : 'The model is uncertain, so attention may be spread and should be treated as exploratory evidence.'

    return {
      summary: `For ${uploadedName}, ${modelTitle} predicts ${gradCam.predicted_label} with ${pct}% confidence.`,
      significance:
        'Grad-CAM does not show where an object is located exactly; it shows which regions most increased the chosen class score.',
      personalized: confidenceMeaning,
      learnerTip:
        'Compare the overlay with your own intuition: if highlighted regions match meaningful parts of the subject, the model is using semantically relevant cues.',
    }
  }, [gradCam, uploadedFile, selectedModelNote.title])

  useEffect(() => {
    api.cnnLabInfo().then(setLabInfo).catch(() => setLabInfo(null))
  }, [])

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
  }, [previewUrl])

  const handleUpload = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    const url = URL.createObjectURL(file)
    setUploadedFile(file)
    setPreviewUrl(url)
    setError('')
    setPredictions([])
    setFeatureMaps([])
    setGradCam(null)
    setInputImage('')
  }

  const buildPayload = (extra = {}) => {
    const form = new FormData()
    form.append('image', uploadedFile)
    form.append('model_id', selectedModel)
    Object.entries(extra).forEach(([k, v]) => form.append(k, String(v)))
    return form
  }

  const runCnnLab = async () => {
    if (!uploadedFile) {
      setError('Upload an image first to run CNN Lab.')
      return
    }

    setLoading(true)
    setError('')
    try {
      const jobs = [
        api.cnnLabPredict(buildPayload({ top_k: 3 })),
        api.cnnLabFeatureMaps(buildPayload({ max_maps: 16 })),
      ]
      if (includeGradCam) {
        jobs.push(api.cnnLabGradCam(buildPayload()))
      }

      const [predictRes, mapsRes, gradRes] = await Promise.all(jobs)
      setPredictions(predictRes.top_predictions || [])
      setInputImage(predictRes.input_image || '')
      setFeatureMaps(mapsRes.feature_maps || [])
      setGradCam(includeGradCam ? gradRes : null)
    } catch (err) {
      setError(err?.message || 'CNN Lab inference failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <Section title="cnn lab setup">
        <div className="cnn-lab-grid">
          <div className="cnn-lab-panel">
            <p className="slider-label">Image Input</p>
            <label className="module-field">
              <span>Upload image</span>
              <input type="file" accept=".jpg,.jpeg,.png,.bmp,.webp" onChange={handleUpload} />
            </label>
            <p className="module-note">Input is resized to 224x224 and normalized using ImageNet statistics before inference.</p>
            {previewUrl ? <img src={previewUrl} alt="Uploaded" className="cnn-lab-image" /> : <p className="module-note">No image uploaded yet.</p>}
          </div>

          <div className="cnn-lab-panel">
            <p className="slider-label">Inference Controls</p>
            <label className="module-field">
              <span>Pretrained Model</span>
              <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
                <option value="efficientnet_b0">EfficientNet B0 (best accuracy)</option>
                <option value="resnet50">ResNet50 (balanced)</option>
                <option value="mobilenet_v2">MobileNetV2 (fast)</option>
              </select>
            </label>

            <label className="module-field checkbox-field">
              <span>Grad-CAM</span>
              <input type="checkbox" checked={includeGradCam} onChange={(e) => setIncludeGradCam(e.target.checked)} />
            </label>

            <div className="module-action-row">
              <button className="module-action-btn primary" onClick={runCnnLab} disabled={loading || !uploadedFile}>
                {loading ? 'Running CNN Lab...' : 'Run Classification + Visualization'}
              </button>
            </div>

            <div className="cnn-inference-insight">
              <p className="cnn-insight-title">Selected Model Snapshot</p>
              <p className="cnn-insight-model">{selectedModelNote.title}</p>
              <p className="module-note">{selectedModelNote.summary}</p>
              <div className="cnn-chip-row">
                <span className="cnn-chip">{selectedModelNote.params}</span>
                <span className="cnn-chip">{selectedModelNote.speed}</span>
                <span className="cnn-chip">{selectedModelNote.strength}</span>
              </div>
              <p className="module-note">
                {previewUrl
                  ? 'Image loaded. Pipeline is ready for inference and visual explainability outputs.'
                  : 'Upload an image to activate the full inference pipeline and output stages.'}
              </p>
            </div>

            <div className="cnn-stage-mini-grid">
              {pipelineStages.map((stage) => {
                const done = stageState[stage.key]
                return (
                  <div className={`cnn-stage-mini-card ${done ? 'done' : ''}`} key={stage.key}>
                    <span className="cnn-stage-mini-name">{stage.title.replace(/^\d+\.\s*/, '')}</span>
                    <span className="cnn-stage-mini-state">{done ? 'ready' : 'pending'}</span>
                  </div>
                )
              })}
            </div>

            {labInfo ? <p className="module-note">Backend supports: {labInfo.supported_models.join(', ')}</p> : null}
            {error ? <p className="module-note">{error}</p> : null}
          </div>
        </div>
      </Section>

      <Section title="what the lab does at each stage">
        <div className="cnn-stage-grid">
          {pipelineStages.map((stage) => {
            const done = stageState[stage.key]
            return (
              <div className="cnn-stage-card" key={stage.key}>
                <div className="cnn-stage-card-head">
                  <p className="cnn-stage-card-title">{stage.title}</p>
                  <span className={`cnn-stage-pill ${done ? 'done' : ''}`}>{done ? 'completed' : 'waiting'}</span>
                </div>
                <p className="module-note">{stage.detail}</p>
                <p className="cnn-stage-output">Output: {stage.output}</p>
              </div>
            )
          })}
        </div>
      </Section>

      <Section title="classification output">
        {predictions.length ? (
          <div className="cnn-lab-results-grid">
            <div className="cnn-lab-panel">
              <p className="slider-label">Top Predictions</p>
              {predictionChart ? (
                <Plot data={predictionChart.data} layout={predictionChart.layout} config={predictionChart.config} style={{ width: '100%', height: '260px' }} />
              ) : null}
              {predictions.map((pred, idx) => {
                const pct = Math.max(0, Math.min(100, pred.confidence * 100))
                return (
                  <div className="cnn-pred-row" key={`${pred.class_label}-${idx}`}>
                    <div className="cnn-pred-head">
                      <span>{idx + 1}. {pred.class_label}</span>
                      <strong>{pct.toFixed(2)}%</strong>
                    </div>
                    <div className="cnn-pred-bar-track">
                      <div className="cnn-pred-bar-fill" style={{ width: `${pct}%` }} />
                    </div>
                  </div>
                )
              })}
            </div>

            <div className="cnn-lab-panel">
              <p className="slider-label">Model Input (224x224)</p>
              {inputImage ? <img src={inputImage} alt="Model Input" className="cnn-lab-image" /> : <p className="module-note">Run the lab to preview normalized input source.</p>}
            </div>
          </div>
        ) : (
          <div className="cnn-lab-panel cnn-empty-pred-card">
            <p className="slider-label">Prediction Readiness</p>
            <p className="module-note">
              Run inference to populate probability bars, ranked class outputs, and decision confidence distribution.
            </p>
            <div className="cnn-placeholder-bars">
              {[74, 48, 29].map((value, idx) => (
                <div className="cnn-placeholder-row" key={`placeholder-${idx}`}>
                  <span>Top-{idx + 1}</span>
                  <div className="cnn-placeholder-track">
                    <div className="cnn-placeholder-fill" style={{ width: `${value}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </Section>

      <Section title="feature map visualization">
        {featureMaps.length ? (
          <>
            {featureMapExplanation ? (
              <div className="cnn-lab-panel cnn-explain-panel">
                <p className="slider-label">How to Read These Feature Maps</p>
                <p className="module-note">{featureMapExplanation.summary}</p>
                <p className="module-note"><strong>Significance:</strong> {featureMapExplanation.significance}</p>
                <p className="module-note"><strong>For this run:</strong> {featureMapExplanation.personalized}</p>
                <p className="module-note"><strong>Learner tip:</strong> {featureMapExplanation.learnerTip}</p>
              </div>
            ) : null}

            <div className="cnn-feature-grid">
              {featureMaps.map((fm) => (
                <div className="cnn-feature-card" key={fm.layer_name}>
                  <p className="opencv-image-title">{fm.stage} ({fm.layer_name})</p>
                  <img src={fm.grid_image} alt={`Feature map ${fm.layer_name}`} className="cnn-feature-image" />
                </div>
              ))}
            </div>
          </>
        ) : (
          <p className="module-note">Feature maps will appear after inference and illustrate early-to-deep convolutional processing.</p>
        )}
      </Section>

      <Section title="grad-cam interpretability">
        {includeGradCam ? (
          <>
            {gradCam ? (
              <div className="cnn-lab-results-grid">
                <div className="cnn-lab-panel">
                  <p className="slider-label">Grad-CAM Heatmap</p>
                  <img src={gradCam.heatmap_image} alt="Grad-CAM Heatmap" className="cnn-lab-image" />
                </div>
                <div className="cnn-lab-panel">
                  <p className="slider-label">Overlay on Image</p>
                  <img src={gradCam.overlay_image} alt="Grad-CAM Overlay" className="cnn-lab-image" />
                  <p className="module-note">
                    Focus class: <strong>{gradCam.predicted_label}</strong> ({(gradCam.confidence * 100).toFixed(2)}%)
                  </p>
                </div>
              </div>
            ) : null}

            {gradCam && gradCamExplanation ? (
              <div className="cnn-lab-panel cnn-explain-panel">
                <p className="slider-label">How to Interpret This Grad-CAM Result</p>
                <p className="module-note">{gradCamExplanation.summary}</p>
                <p className="module-note"><strong>Significance:</strong> {gradCamExplanation.significance}</p>
                <p className="module-note"><strong>For this run:</strong> {gradCamExplanation.personalized}</p>
                <p className="module-note"><strong>Learner tip:</strong> {gradCamExplanation.learnerTip}</p>
              </div>
            ) : (
              <p className="module-note">Enable and run Grad-CAM to visualize the regions that most influenced the model decision.</p>
            )}
          </>
        ) : (
          <p className="module-note">Grad-CAM is disabled. Enable it in setup to generate interpretability overlays.</p>
        )}
      </Section>
    </>
  )
}

const RNN_SAMPLE_TEXTS = [
  'deep learning models can',
  'I feel grateful and excited today',
  'this situation is frustrating and unfair',
  'the update sounds reasonable and balanced',
]

function RNNModule() {
  const [text, setText] = useState('deep learning models can')
  const [activeTab, setActiveTab] = useState('generate')
  const [topK, setTopK] = useState(5)
  const [generateWords, setGenerateWords] = useState(5)
  const [temperature, setTemperature] = useState(1)

  const [labInfo, setLabInfo] = useState(null)
  const [generationResult, setGenerationResult] = useState(null)
  const [emotionResult, setEmotionResult] = useState(null)

  const [genLoading, setGenLoading] = useState(false)
  const [emoLoading, setEmoLoading] = useState(false)
  const [genError, setGenError] = useState('')
  const [emoError, setEmoError] = useState('')

  const emotionPalette = {
    Happy: '#16a34a',
    Sad: '#2563eb',
    Angry: '#dc2626',
    Neutral: '#64748b',
  }

  useEffect(() => {
    api.rnnInfo().then(setLabInfo).catch(() => setLabInfo(null))
  }, [])

  const runTextGeneration = async () => {
    if (!text.trim()) {
      setGenError('Enter input text to generate next-word suggestions.')
      return
    }
    setGenLoading(true)
    setGenError('')
    try {
      const res = await api.rnnGenerateText({
        text,
        top_k: topK,
        generate_words: generateWords,
        temperature,
      })
      setGenerationResult(res)
    } catch (err) {
      setGenError(err?.message || 'Text generation failed')
    } finally {
      setGenLoading(false)
    }
  }

  const runEmotionAnalysis = async () => {
    if (!text.trim()) {
      setEmoError('Enter input text to analyze emotion.')
      return
    }
    setEmoLoading(true)
    setEmoError('')
    try {
      const res = await api.rnnAnalyzeEmotion({ text })
      setEmotionResult(res)
    } catch (err) {
      setEmoError(err?.message || 'Emotion analysis failed')
    } finally {
      setEmoLoading(false)
    }
  }

  const onTabChange = (tab) => {
    setActiveTab(tab)
    if (!text.trim()) return
    if (tab === 'generate') runTextGeneration()
    if (tab === 'emotion') runEmotionAnalysis()
  }

  const rankedEmotionProbabilities = useMemo(() => {
    if (!emotionResult?.probabilities?.length) return []
    return [...emotionResult.probabilities]
      .map((item) => ({
        ...item,
        percent: Number((item.probability * 100).toFixed(2)),
      }))
      .sort((a, b) => b.probability - a.probability)
  }, [emotionResult])

  const certaintyTier = useMemo(() => {
    if (!emotionResult) return ''
    if (emotionResult.confidence >= 0.8) return 'High certainty signal'
    if (emotionResult.confidence >= 0.6) return 'Moderate certainty signal'
    return 'Low certainty signal'
  }, [emotionResult])

  const topGapPercent = useMemo(() => {
    if (!rankedEmotionProbabilities.length) return 0
    const top = rankedEmotionProbabilities[0]?.percent || 0
    const second = rankedEmotionProbabilities[1]?.percent || 0
    return Math.max(0, top - second)
  }, [rankedEmotionProbabilities])

  const dominanceLabel = useMemo(() => {
    if (!rankedEmotionProbabilities.length) return ''
    if (topGapPercent >= 20) return 'Dominant class'
    if (topGapPercent >= 10) return 'Competitive class mix'
    return 'Ambiguous class boundary'
  }, [rankedEmotionProbabilities, topGapPercent])

  const normalizedEntropy = useMemo(() => {
    if (!rankedEmotionProbabilities.length) return 0
    const probs = rankedEmotionProbabilities.map((item) => item.probability).filter((p) => p > 0)
    if (!probs.length) return 0
    const entropy = -probs.reduce((sum, p) => sum + p * Math.log(p), 0)
    const maxEntropy = Math.log(rankedEmotionProbabilities.length)
    if (!Number.isFinite(entropy) || !Number.isFinite(maxEntropy) || maxEntropy <= 0) return 0
    return Math.max(0, Math.min(1, entropy / maxEntropy))
  }, [rankedEmotionProbabilities])

  const interpretationNote = useMemo(() => {
    if (!emotionResult) return ''
    if (emotionResult.confidence >= 0.8 && topGapPercent >= 15) {
      return 'The model shows a clear and stable class preference for this input.'
    }
    if (emotionResult.confidence >= 0.6) {
      return 'The model sees a leading class, but nearby classes still carry noticeable probability.'
    }
    return 'The class boundary is soft here. Rephrasing the sentence can help separate emotions more clearly.'
  }, [emotionResult, topGapPercent])

  const inputWordCount = useMemo(() => {
    if (!text.trim()) return 0
    return text.trim().split(/\s+/).length
  }, [text])

  const emotionModelLabel = useMemo(() => {
    const fromResult = typeof emotionResult?.model_name === 'string' ? emotionResult.model_name.trim() : ''
    const fromInfo = typeof labInfo?.bert_model_id === 'string' ? labInfo.bert_model_id.trim() : ''
    return fromResult || fromInfo || 'DistilRoBERTa'
  }, [emotionResult, labInfo])

  const emotionChart = useMemo(() => {
    if (!emotionResult?.probabilities?.length) return null
    return {
      data: [
        {
          type: 'bar',
          orientation: 'h',
          x: rankedEmotionProbabilities.map((p) => p.percent),
          y: rankedEmotionProbabilities.map((p) => p.label),
          marker: {
            color: rankedEmotionProbabilities.map((p) => emotionPalette[p.label] || '#6366f1'),
            line: { color: '#ffffff', width: 1.2 },
            opacity: 0.95,
          },
          text: rankedEmotionProbabilities.map((p) => `${p.percent.toFixed(2)}%`),
          textposition: 'inside',
          insidetextanchor: 'end',
          hovertemplate: '%{y}<br>%{x:.2f}% confidence<extra></extra>',
          cliponaxis: false,
        },
      ],
      layout: {
        title: { text: 'Emotion Probability Distribution', x: 0.5, font: { size: 16, color: '#0f172a' } },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#f8fafc',
        margin: { l: 70, r: 24, t: 56, b: 42 },
        font: { family: 'Segoe UI, sans-serif', color: '#1e293b' },
        xaxis: {
          title: 'Probability (%)',
          range: [0, 100],
          gridcolor: '#dbeafe',
          zerolinecolor: '#cbd5e1',
        },
        yaxis: {
          automargin: true,
          tickfont: { size: 12, color: '#1e293b' },
        },
      },
      config: { displayModeBar: false, responsive: true },
    }
  }, [emotionResult, rankedEmotionProbabilities])

  return (
    <div className="rnn-typography-scope">
      <Section title="rnn lab workspace">
        <div className="rnn-lab-shell">
          <label className="module-field span-2">
            <span>Shared input text</span>
            <textarea rows={4} value={text} onChange={(e) => setText(e.target.value)} placeholder="Enter text once. Switch tabs to run each system independently." />
          </label>

          <div className="sample-button-row">
            {RNN_SAMPLE_TEXTS.map((sample) => (
              <button key={sample} className="module-action-btn" onClick={() => setText(sample)}>{sample}</button>
            ))}
          </div>

          <div className="rnn-tab-row" role="tablist" aria-label="RNN tasks">
            <button className={`rnn-tab-btn ${activeTab === 'generate' ? 'active' : ''}`} onClick={() => onTabChange('generate')} role="tab" aria-selected={activeTab === 'generate'}>
              Text Generation
            </button>
            <button className={`rnn-tab-btn ${activeTab === 'emotion' ? 'active' : ''}`} onClick={() => onTabChange('emotion')} role="tab" aria-selected={activeTab === 'emotion'}>
              Emotion Analysis
            </button>
          </div>

          {labInfo ? <p className="module-note">{labInfo.lstm_model_type} + {labInfo.bert_model_id}</p> : null}
        </div>
      </Section>

      {activeTab === 'generate' ? (
        <div className="rnn-tab-panel" key="tab-generate">
          <Section title="text generation controls">
            <div className="slider-grid">
              <div>
                <p className="slider-label">Top suggestions</p>
                <ElasticSlider defaultValue={topK} startingValue={1} maxValue={5} isStepped stepSize={1} onChange={setTopK} />
              </div>
              <div>
                <p className="slider-label">Generate extra words</p>
                <ElasticSlider defaultValue={generateWords} startingValue={1} maxValue={10} isStepped stepSize={1} onChange={setGenerateWords} />
              </div>
              <div>
                <p className="slider-label">Temperature</p>
                <ElasticSlider defaultValue={temperature} startingValue={0.2} maxValue={2} isStepped stepSize={0.1} onChange={setTemperature} />
              </div>
            </div>
            <div className="module-action-row">
              <button className="module-action-btn primary" onClick={runTextGeneration} disabled={genLoading}>
                {genLoading ? 'Generating...' : 'Run Next-Word Prediction'}
              </button>
            </div>
            {genError ? <p className="module-note">{genError}</p> : null}
          </Section>

          <Section title="text generation output">
            {generationResult ? (
              <div className="cnn-lab-results-grid">
                <div className="cnn-lab-panel">
                  <p className="slider-label">Top Next-Word Suggestions</p>
                  {generationResult.suggestions.map((s, idx) => {
                    const pct = Math.max(0, Math.min(100, s.confidence * 100))
                    return (
                      <div className="cnn-pred-row" key={`${s.word}-${idx}`}>
                        <div className="cnn-pred-head">
                          <span>{idx + 1}. {s.word}</span>
                          <strong>{pct.toFixed(2)}%</strong>
                        </div>
                        <div className="cnn-pred-bar-track">
                          <div className="cnn-pred-bar-fill" style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    )
                  })}
                </div>

                <div className="cnn-lab-panel">
                  <p className="slider-label">Generated Text</p>
                  <p className="module-note"><strong>Input:</strong> {generationResult.input_text}</p>
                  <p className="module-note"><strong>Output:</strong> {generationResult.generated_text}</p>
                  <p className="module-note">
                    Temperature {temperature.toFixed(1)} controls randomness: lower values are conservative, higher values increase diversity.
                  </p>
                </div>
              </div>
            ) : (
              <p className="module-note">Run text generation to view top next-word suggestions and optional multi-word continuation.</p>
            )}
          </Section>
        </div>
      ) : (
        <div className="rnn-tab-panel" key="tab-emotion">
          <Section title="emotion analysis controls">
            <div className="module-action-row">
              <button className="module-action-btn primary" onClick={runEmotionAnalysis} disabled={emoLoading}>
                {emoLoading ? 'Analyzing...' : 'Run Analysis'}
              </button>
            </div>
            <p className="module-note">Mapped output classes: Happy, Sad, Angry, Neutral.</p>
            {emoError ? <p className="module-note">{emoError}</p> : null}
          </Section>

          <Section title="emotion analysis output">
            {emotionResult ? (
              <div className="emotion-results-grid">
                <div className="cnn-lab-panel">
                  <p className="slider-label">Predicted Emotion</p>
                  <div className="emotion-hero-chip" style={{ '--emotion-color': emotionPalette[emotionResult.predicted_emotion] || '#4f46e5' }}>
                    <span className="emotion-hero-label">Primary Class</span>
                    <strong>{emotionResult.predicted_emotion}</strong>
                    <span className="emotion-hero-score">{(emotionResult.confidence * 100).toFixed(2)}% confidence</span>
                  </div>

                  <div className="emotion-confidence-track" aria-label="Emotion confidence track">
                    <div className="emotion-confidence-fill" style={{ width: `${Math.max(0, Math.min(100, emotionResult.confidence * 100))}%`, background: emotionPalette[emotionResult.predicted_emotion] || '#4f46e5' }} />
                  </div>
                  <p className="module-note emotion-tier-note">{certaintyTier}</p>

                  <div className="emotion-metric-grid">
                    <div className="emotion-metric-card">
                      <span>Top Gap</span>
                      <strong>{topGapPercent.toFixed(2)}%</strong>
                    </div>
                    <div className="emotion-metric-card">
                      <span>Model</span>
                      <strong>{emotionModelLabel}</strong>
                    </div>
                    <div className="emotion-metric-card">
                      <span>Signal</span>
                      <strong>{dominanceLabel}</strong>
                    </div>
                  </div>

                  <div className="emotion-rank-list">
                    {rankedEmotionProbabilities.map((item, idx) => (
                      <div className="emotion-rank-row" key={`${item.label}-${idx}`}>
                        <div className="emotion-rank-head">
                          <span>{idx + 1}. {item.label}</span>
                          <strong>{item.percent.toFixed(2)}%</strong>
                        </div>
                        <div className="emotion-rank-track">
                          <div
                            className="emotion-rank-fill"
                            style={{ width: `${item.percent}%`, background: emotionPalette[item.label] || '#6366f1' }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="cnn-lab-panel emotion-plot-panel">
                  <p className="slider-label">Probability Ranking</p>
                  <div className="emotion-plot-shell">
                    {emotionChart ? (
                      <Plot data={emotionChart.data} layout={emotionChart.layout} config={emotionChart.config} style={{ width: '100%', height: '310px' }} />
                    ) : null}
                  </div>
                </div>

                <div className="cnn-lab-panel emotion-insight-panel">
                  <p className="slider-label">Interpretation</p>
                  <p className="module-note emotion-insight-copy">{interpretationNote}</p>
                  <div className="emotion-insight-grid">
                    <div className="emotion-insight-cell">
                      <span>Uncertainty Index</span>
                      <strong>{(normalizedEntropy * 100).toFixed(1)}%</strong>
                    </div>
                    <div className="emotion-insight-cell">
                      <span>Input Length</span>
                      <strong>{inputWordCount} words</strong>
                    </div>
                    <div className="emotion-insight-cell">
                      <span>Runner-up Class</span>
                      <strong>{rankedEmotionProbabilities[1]?.label || 'N/A'}</strong>
                    </div>
                    <div className="emotion-insight-cell">
                      <span>Runner-up Score</span>
                      <strong>{(rankedEmotionProbabilities[1]?.percent ?? 0).toFixed(2)}%</strong>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <p className="module-note">Run emotion analysis to view the predicted class and full probability distribution.</p>
            )}
          </Section>
        </div>
      )}

      <Section title="theory primer">
        <div className="rnn-theory-grid">
          <div className="rnn-theory-card">
            <p className="rnn-theory-title">Sequential Memory (RNN/LSTM)</p>
            <p className="module-note">
              The hidden state carries context across tokens, so the current prediction depends on previous words and not only the current token.
            </p>
            <p className="module-note formula-copy">h_t = f(W_x * x_t + W_h * h_(t-1) + b)</p>
          </div>

          <div className="rnn-theory-card">
            <p className="rnn-theory-title">Next-Token Distribution</p>
            <p className="module-note">
              The model outputs a probability distribution over candidate tokens. Top-k suggestions are the highest-probability next-word candidates.
            </p>
            <p className="module-note formula-copy">P(w_t | context) = softmax(z_t)</p>
          </div>

          <div className="rnn-theory-card">
            <p className="rnn-theory-title">Emotion Classification</p>
            <p className="module-note">
              DistilRoBERTa encodes sentence context and produces class probabilities, which are then mapped to Happy, Sad, Angry, and Neutral.
            </p>
            <p className="module-note formula-copy">y_hat = argmax_c P(c | text)</p>
          </div>
        </div>
      </Section>

      <Section title="learning notes">
        <div className="cnn-stage-grid">
          <div className="cnn-stage-card">
            <p className="cnn-stage-card-title">LSTM Next-Word Prediction</p>
            <p className="module-note">
              The LSTM reads sequence context and estimates which next token is most probable, showing top-ranked alternatives with confidence.
            </p>
          </div>
          <div className="cnn-stage-card">
            <p className="cnn-stage-card-title">BERT-based Emotion Analysis</p>
            <p className="module-note">
              DistilRoBERTa encodes semantic context and maps fine-grained emotions into learner-friendly categories: Happy, Sad, Angry, and Neutral.
            </p>
          </div>
        </div>
      </Section>
    </div>
  )
}

function HopfieldModule() {
  const side = 15
  const drawCanvasRef = useRef(null)
  const lastPointRef = useRef(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [noisePercent, setNoisePercent] = useState(12)
  const [recallSteps, setRecallSteps] = useState(900)
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.4)
  const [inputPattern, setInputPattern] = useState(null)
  const [noisyPattern, setNoisyPattern] = useState(null)
  const [recalledPattern, setRecalledPattern] = useState(null)
  const [result, setResult] = useState(null)
  const [libraryTotal, setLibraryTotal] = useState(0)
  const [libraryHint, setLibraryHint] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const libraryChips = useMemo(
    () => libraryHint.split(',').map((label) => label.trim()).filter(Boolean).slice(0, 12),
    [libraryHint]
  )

  const cleanedStats = useMemo(
    () => ({
      inputInk: inputPattern ? inputPattern.reduce((count, cell) => count + (cell > 0 ? 1 : 0), 0) : 0,
      noisyInk: noisyPattern ? noisyPattern.reduce((count, cell) => count + (cell > 0 ? 1 : 0), 0) : 0,
      recalledInk: recalledPattern ? recalledPattern.reduce((count, cell) => count + (cell > 0 ? 1 : 0), 0) : 0,
    }),
    [inputPattern, noisyPattern, recalledPattern]
  )

  const energyDelta = result ? result.energy_after - result.energy_before : 0
  const confidencePct = result ? result.confidence * 100 : 0
  const verdictLabel = result ? (result.is_unknown ? 'Unknown' : result.label) : 'Waiting'

  const paintDot = (evt) => {
    const canvas = drawCanvasRef.current
    if (!canvas) return null
    const rect = canvas.getBoundingClientRect()
    const x = evt.clientX - rect.left
    const y = evt.clientY - rect.top
    if (x < 0 || y < 0 || x > rect.width || y > rect.height) return null

    const ctx = canvas.getContext('2d')
    if (!ctx) return null
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    const px = x * scaleX
    const py = y * scaleY

    ctx.strokeStyle = '#0f172a'
    ctx.lineWidth = 11
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    const last = lastPointRef.current
    if (last) {
      ctx.beginPath()
      ctx.moveTo(last.x, last.y)
      ctx.lineTo(px, py)
      ctx.stroke()
    } else {
      ctx.fillStyle = '#0f172a'
      ctx.beginPath()
      ctx.arc(px, py, 5.5, 0, Math.PI * 2)
      ctx.fill()
    }

    lastPointRef.current = { x: px, y: py }
    return { x: px, y: py }
  }

  const clearDrawingCanvas = () => {
    const canvas = drawCanvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.strokeStyle = '#e2e8f0'
    ctx.lineWidth = 1
    const cell = canvas.width / side
    for (let i = 1; i < side; i += 1) {
      ctx.beginPath()
      ctx.moveTo(i * cell, 0)
      ctx.lineTo(i * cell, canvas.height)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(0, i * cell)
      ctx.lineTo(canvas.width, i * cell)
      ctx.stroke()
    }
  }

  const renderPatternToDrawingCanvas = (pattern) => {
    const canvas = drawCanvasRef.current
    if (!canvas || !pattern?.length) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const cell = canvas.width / side

    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    for (let r = 0; r < side; r += 1) {
      for (let c = 0; c < side; c += 1) {
        const v = pattern[r * side + c]
        ctx.fillStyle = v > 0 ? '#0f172a' : '#ffffff'
        ctx.fillRect(c * cell, r * cell, cell, cell)
      }
    }

    ctx.strokeStyle = '#e2e8f0'
    ctx.lineWidth = 1
    for (let i = 1; i < side; i += 1) {
      ctx.beginPath()
      ctx.moveTo(i * cell, 0)
      ctx.lineTo(i * cell, canvas.height)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(0, i * cell)
      ctx.lineTo(canvas.width, i * cell)
      ctx.stroke()
    }
  }

  const getDownsampledPattern = () => {
    const canvas = drawCanvasRef.current
    if (!canvas) return Array.from({ length: side * side }, () => 0)

    const offscreen = document.createElement('canvas')
    offscreen.width = side
    offscreen.height = side
    const octx = offscreen.getContext('2d')
    if (!octx) return Array.from({ length: side * side }, () => 0)
    octx.drawImage(canvas, 0, 0, side, side)

    const pixels = octx.getImageData(0, 0, side, side).data
    const out = []
    for (let i = 0; i < pixels.length; i += 4) {
      const r = pixels[i]
      const g = pixels[i + 1]
      const b = pixels[i + 2]
      const gray = 0.299 * r + 0.587 * g + 0.114 * b
      out.push(gray < 220 ? 1 : 0)
    }
    return out
  }

  const captureCurrentPattern = () => {
    const pattern = getDownsampledPattern()
    setInputPattern(pattern)
    setNoisyPattern(null)
    setRecalledPattern(null)
    setResult(null)
    return pattern
  }

  const addNoise = () => {
    const original = captureCurrentPattern()
    const cells = side * side
    const flips = Math.max(1, Math.round((noisePercent / 100) * cells))
    const noisy = original.slice()

    for (let i = 0; i < flips; i += 1) {
      const idx = Math.floor(Math.random() * cells)
      noisy[idx] = noisy[idx] ? 0 : 1
    }

    renderPatternToDrawingCanvas(noisy)
    setNoisyPattern(noisy)
    setResult(null)
    setRecalledPattern(null)
  }

  const recallPattern = async () => {
    setLoading(true)
    setError('')

    try {
      const pattern = getDownsampledPattern()
      setInputPattern(pattern)

      const response = await api.hopfieldRecall({
        pattern,
        side,
        steps: recallSteps,
        confidence_threshold: confidenceThreshold,
      })

      setResult(response)
      setRecalledPattern(response.recalled_pattern)
    } catch (err) {
      setError(err.message || 'Hopfield recall failed')
    } finally {
      setLoading(false)
    }
  }

  const clearAll = () => {
    clearDrawingCanvas()
    lastPointRef.current = null
    setResult(null)
    setError('')
    setInputPattern(null)
    setNoisyPattern(null)
    setRecalledPattern(null)
  }

  useEffect(() => {
    clearDrawingCanvas()

    api
      .hopfieldLibrary(side)
      .then((res) => {
        setLibraryTotal(res.total_patterns)
        setLibraryHint((res.labels || []).slice(0, 16).join(', '))
      })
      .catch(() => {
        setLibraryTotal(0)
        setLibraryHint('')
      })
  }, [])

  const inputPlot = useMemo(
    () => buildHopfieldHeatmap(inputPattern, side, 'Input Pattern', '#1e293b'),
    [inputPattern]
  )
  const noisyPlot = useMemo(
    () => buildHopfieldHeatmap(noisyPattern, side, 'Noisy Pattern', '#7c3aed'),
    [noisyPattern]
  )
  const recalledPlot = useMemo(
    () => buildHopfieldHeatmap(recalledPattern, side, 'Recalled Pattern', '#5227ff'),
    [recalledPattern]
  )

  const energyPlot = useMemo(() => {
    if (!result) return null

    return {
      data: [
        {
          x: ['Before', 'After'],
          y: [result.energy_before, result.energy_after],
          type: 'bar',
          marker: { color: ['#7c3aed', '#2563eb'] },
          text: [result.energy_before.toFixed(2), result.energy_after.toFixed(2)],
          textposition: 'outside',
          hoverinfo: 'skip',
        },
      ],
      layout: {
        title: { text: 'Energy Shift During Recall', x: 0.5 },
        xaxis: { title: 'State' },
        yaxis: { title: 'Energy', gridcolor: '#e5e7eb' },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        margin: { l: 48, r: 18, t: 48, b: 42 },
        height: 280,
      },
    }
  }, [result])

  return (
    <>
      <Section title="module overview">
        <div className="hopfield-hero">
          <div className="hopfield-hero-copy">
            <p className="module-note">
              Hopfield memory stores binary character templates and retrieves the closest stable pattern from a noisy drawing.
              This redesign keeps the module aligned with the rest of DeepNexus while making the recall pipeline easier to inspect.
            </p>
            <div className="hopfield-chip-list">
              <span className="hopfield-chip">Associative memory</span>
              <span className="hopfield-chip">Energy minimization</span>
              <span className="hopfield-chip">Noise recovery</span>
              <span className="hopfield-chip">Character OCR</span>
            </div>
          </div>
          <div className="hopfield-metrics-grid">
            <div className="hopfield-metric-card">
              <span className="hopfield-metric-label">Stored patterns</span>
              <strong>{libraryTotal || 62}</strong>
            </div>
            <div className="hopfield-metric-card">
              <span className="hopfield-metric-label">Canvas size</span>
              <strong>{side} × {side}</strong>
            </div>
            <div className="hopfield-metric-card">
              <span className="hopfield-metric-label">Confidence threshold</span>
              <strong>{confidenceThreshold.toFixed(2)}</strong>
            </div>
            <div className="hopfield-metric-card">
              <span className="hopfield-metric-label">Recall steps</span>
              <strong>{recallSteps}</strong>
            </div>
          </div>
        </div>
        <div className="engine-guide hopfield-guide">
          <p className="engine-guide-title">Recall Flow</p>
          <div className="engine-guide-flow hopfield-guide-flow" role="list" aria-label="Hopfield recall guide">
            <div className="engine-guide-step active" role="listitem">
              <p className="engine-step-head">Draw</p>
              <p className="engine-step-copy">Sketch a character on the canvas and keep the stroke bold.</p>
            </div>
            <div className="engine-guide-step" role="listitem">
              <p className="engine-step-head">Corrupt</p>
              <p className="engine-step-copy">Add noise to test how much distortion the memory can absorb.</p>
            </div>
            <div className="engine-guide-step" role="listitem">
              <p className="engine-step-head">Recall</p>
              <p className="engine-step-copy">Run associative retrieval and compare the reconstructed pattern.</p>
            </div>
          </div>
        </div>
      </Section>

      <Section title="pattern editor">
        <div className="hopfield-workbench">
          <div className="hopfield-canvas-card">
            <p className="slider-label">Draw here</p>
            <p className="module-note">
              Hold and drag to paint a character. The canvas uses a 15 × 15 view so the stored Hopfield templates and the user input stay comparable.
            </p>
            <canvas
              ref={drawCanvasRef}
              width={360}
              height={360}
              className="hopfield-draw-canvas"
              onMouseDown={(evt) => {
                setIsDrawing(true)
                setNoisyPattern(null)
                paintDot(evt)
              }}
              onMouseMove={(evt) => {
                if (!isDrawing) return
                paintDot(evt)
              }}
              onMouseUp={() => {
                setIsDrawing(false)
                lastPointRef.current = null
                captureCurrentPattern()
              }}
              onMouseLeave={() => {
                setIsDrawing(false)
                lastPointRef.current = null
                captureCurrentPattern()
              }}
            />
            <div className="module-action-row">
              <button className="module-action-btn primary" onClick={recallPattern} disabled={loading}>
                {loading ? 'Recalling...' : 'Recall Pattern'}
              </button>
              <button className="module-action-btn" onClick={clearAll}>Clear Canvas</button>
              <button className="module-action-btn" onClick={addNoise}>Add Noise</button>
            </div>
          </div>

          <div className="hopfield-control-card">
            <p className="slider-label">Tune recall</p>
            <div className="slider-grid hopfield-slider-grid">
              <div>
                <p className="slider-label">Noise percentage</p>
                <ElasticSlider
                  defaultValue={noisePercent}
                  startingValue={0}
                  maxValue={50}
                  stepSize={1}
                  isStepped
                  onChange={setNoisePercent}
                />
              </div>
              <div>
                <p className="slider-label">Recall steps</p>
                <ElasticSlider
                  defaultValue={recallSteps}
                  startingValue={20}
                  maxValue={1000}
                  stepSize={10}
                  isStepped
                  onChange={setRecallSteps}
                />
              </div>
              <div>
                <p className="slider-label">Confidence threshold</p>
                <ElasticSlider
                  defaultValue={confidenceThreshold}
                  startingValue={0}
                  maxValue={1}
                  stepSize={0.01}
                  isStepped
                  onChange={setConfidenceThreshold}
                />
              </div>
            </div>
            <div className="hopfield-library-card">
              <p className="module-note">
                Library sample: {libraryHint || '0, 1, 2, 3, a, b, c, A, B, C'}
              </p>
              <div className="hopfield-chip-list compact">
                {libraryChips.length ? libraryChips.map((label) => (
                  <span className="hopfield-chip soft" key={label}>{label}</span>
                )) : <span className="hopfield-chip soft">Loading library labels...</span>}
              </div>
              <p className="module-note">Use more steps for difficult draws and raise the threshold when you want stricter recognition.</p>
            </div>
          </div>
        </div>
      </Section>

      <Section title="pattern snapshots">
        <div className="hopfield-snapshot-grid">
          <div className="hopfield-snapshot-card">
            {inputPattern ? <Plot data={inputPlot.data} layout={inputPlot.layout} style={{ width: '100%', height: '260px' }} config={{ displayModeBar: false }} /> : <p className="module-note">Draw a pattern to capture the clean input snapshot.</p>}
            <p className="module-note">Ink cells: <strong>{cleanedStats.inputInk}</strong></p>
          </div>
          <div className="hopfield-snapshot-card">
            {noisyPattern ? <Plot data={noisyPlot.data} layout={noisyPlot.layout} style={{ width: '100%', height: '260px' }} config={{ displayModeBar: false }} /> : <p className="module-note">Add noise to preview the corrupted pattern.</p>}
            <p className="module-note">Ink cells: <strong>{cleanedStats.noisyInk}</strong></p>
          </div>
          <div className="hopfield-snapshot-card">
            {recalledPattern ? <Plot data={recalledPlot.data} layout={recalledPlot.layout} style={{ width: '100%', height: '260px' }} config={{ displayModeBar: false }} /> : <p className="module-note">Run recall to see the stable reconstructed pattern.</p>}
            <p className="module-note">Ink cells: <strong>{cleanedStats.recalledInk}</strong></p>
          </div>
        </div>
      </Section>

      <Section title="recall analysis">
        {error ? <p className="module-note">{error}</p> : null}
        {result ? (
          <>
            <div className="hopfield-result-grid">
              <div className="hopfield-result-callout">
                <p className="hopfield-result-label">Prediction</p>
                <h3>{verdictLabel}</h3>
                <p className="module-note">
                  Matched label: <strong>{result.matched_label}</strong> | Converged: <strong>{String(result.converged)}</strong>
                </p>
                <p className="module-note">
                  Confidence: <strong>{confidencePct.toFixed(2)}%</strong> | Threshold: <strong>{(confidenceThreshold * 100).toFixed(2)}%</strong>
                </p>
                <p className="module-note">
                  Energy delta: <strong>{energyDelta.toFixed(3)}</strong> | Iterations: <strong>{result.iterations}</strong>
                </p>
              </div>
              <div className="hopfield-result-summary">
                <div className={result.is_unknown ? 'hopfield-badge unknown' : 'hopfield-badge known'}>
                  {result.is_unknown ? 'Below threshold' : 'Accepted'}
                </div>
                <p className="module-note">
                  The Hopfield field settled into a lower-energy state and the classifier compared that stable pattern against the stored alphabet bank.
                </p>
                <p className="module-note">
                  Confidence is based on pattern agreement, while energy reflects the stability of the reconstructed state.
                </p>
              </div>
            </div>

            <div className="plot-grid-2">
              <Plot data={energyPlot.data} layout={energyPlot.layout} style={{ width: '100%', height: '280px' }} config={{ displayModeBar: false }} />
              <table className="module-table hopfield-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td>Label</td><td>{result.label}</td></tr>
                  <tr><td>Matched label</td><td>{result.matched_label}</td></tr>
                  <tr><td>Confidence</td><td>{(result.confidence * 100).toFixed(2)}%</td></tr>
                  <tr><td>Converged</td><td>{String(result.converged)}</td></tr>
                  <tr><td>Iterations</td><td>{result.iterations}</td></tr>
                  <tr><td>Energy before</td><td>{result.energy_before.toFixed(3)}</td></tr>
                  <tr><td>Energy after</td><td>{result.energy_after.toFixed(3)}</td></tr>
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <p className="module-note">Draw a character, optionally corrupt it, and then click Recall Pattern to inspect the associative memory response.</p>
        )}
      </Section>

      <Section title="how it works">
        <div className="hopfield-how-grid">
          <div className="hopfield-how-card">
            <p className="engine-step-head">Storage</p>
            <p className="module-note">The backend stores digit, lowercase, and uppercase character templates as binary patterns and builds a Hopfield weight matrix from them.</p>
          </div>
          <div className="hopfield-how-card">
            <p className="engine-step-head">Recall</p>
            <p className="module-note">The drawn pattern is converted to a bipolar state, then asynchronous updates move it toward the nearest stable memory.</p>
          </div>
          <div className="hopfield-how-card">
            <p className="engine-step-head">Decision</p>
            <p className="module-note">The module compares the recovered state against the memory bank, reports confidence, and flags low-certainty results as unknown.</p>
          </div>
        </div>
      </Section>
    </>
  )
}

function StudyModeModule() {
  const [activeTopic, setActiveTopic] = useState('perceptron_mlp')

  const studyItems = [
    { label: 'Perceptron + MLP', href: 'perceptron_mlp' },
    { label: 'Neural Flow Engine', href: 'neural_flow_engine' },
    { label: 'Hopfield Network', href: 'hopfield' },
    { label: 'CNN Lab', href: 'cnn_lab' },
    { label: 'RNN Lab', href: 'rnn' },
  ]

  // Helper to create activation function graphs
  const createActivationPlot = () => {
    const x = Array.from({ length: 300 }, (_, i) => -5 + (i * 10) / 299)
    const sigmoid = x.map(v => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, v)))))
    const relu = x.map(v => Math.max(0, v))
    const tanh = x.map(v => Math.tanh(v))
    
    return (
      <Plot
        data={[
          { x, y: sigmoid, mode: 'lines', name: 'Sigmoid', line: { width: 2 } },
          { x, y: relu, mode: 'lines', name: 'ReLU', line: { width: 2 } },
          { x, y: tanh, mode: 'lines', name: 'Tanh', line: { width: 2 } },
        ]}
        layout={{
          title: 'Activation Functions',
          xaxis: { title: 'z' },
          yaxis: { title: 'a' },
          plot_bgcolor: '#FFFFFF',
          paper_bgcolor: '#FFFFFF',
          font: { color: '#0F172A' },
          height: 330,
          margin: { l: 20, r: 20, t: 50, b: 20 },
        }}
        style={{ width: '100%', height: '350px' }}
        config={{ displayModeBar: false }}
      />
    )
  }

  // Perceptron decision surface
  const createPerceptronSurface = () => {
    const x = Array.from({ length: 120 }, (_, i) => -2.5 + (i * 5) / 119)
    const y = Array.from({ length: 120 }, (_, i) => -2.5 + (i * 5) / 119)
    const z = x.map(xi => y.map(yi => 1 / (1 + Math.exp(-(1.2 * xi - 0.9 * yi + 0.25)))))
    
    return (
      <Plot
        data={[
          {
            x: x,
            y: y,
            z: z,
            type: 'contour',
            colorscale: 'Viridis',
            contours: { showlines: false },
            colorbar: { title: 'p(y=1)' },
          },
          {
            x: x,
            y: y,
            z: z,
            type: 'contour',
            contours: { start: 0.5, end: 0.5, size: 1, coloring: 'none', showlines: true },
            line: { color: '#5227FF', width: 2 },
            showscale: false,
          },
        ]}
        layout={{
          title: 'Perceptron Decision Surface and Boundary',
          xaxis: { title: 'x1' },
          yaxis: { title: 'x2' },
          plot_bgcolor: '#FFFFFF',
          paper_bgcolor: '#FFFFFF',
          font: { color: '#0F172A' },
          height: 360,
          margin: { l: 20, r: 20, t: 50, b: 20 },
        }}
        style={{ width: '100%', height: '380px' }}
        config={{ displayModeBar: false }}
      />
    )
  }

  // Forward flow graph
  const createForwardFlow = () => {
    return (
      <Plot
        data={[
          {
            x: ['L1', 'L2', 'L3', 'L4'],
            y: [0.91, 0.44, 0.28, 0.13],
            name: 'Pre-activation Z',
            marker: { color: '#5227FF' },
            type: 'bar',
          },
          {
            x: ['L1', 'L2', 'L3', 'L4'],
            y: [0.71, 0.61, 0.57, 0.53],
            name: 'Post-activation A',
            marker: { color: '#9CA3AF' },
            type: 'bar',
          },
        ]}
        layout={{
          title: 'Layer-wise Forward Values (Z vs A)',
          xaxis: { title: 'Layer' },
          yaxis: { title: 'Value', range: [0, 1.1], gridcolor: '#E5E7EB' },
          plot_bgcolor: '#FFFFFF',
          paper_bgcolor: '#FFFFFF',
          font: { color: '#0F172A' },
          barmode: 'group',
          height: 320,
          margin: { l: 20, r: 20, t: 50, b: 20 },
        }}
        style={{ width: '100%', height: '350px' }}
        config={{ displayModeBar: false }}
      />
    )
  }

  // Gradient decay graph
  const createGradientDecay = () => {
    const depth = Array.from({ length: 8 }, (_, i) => i + 1)
    const stable = depth.map(d => Math.exp(-0.25 * d))
    const vanishing = depth.map(d => Math.exp(-0.9 * d))

    return (
      <Plot
        data={[
          {
            x: depth,
            y: stable,
            mode: 'lines+markers',
            name: 'Healthy Gradient',
            line: { width: 3, color: '#5227FF' },
          },
          {
            x: depth,
            y: vanishing,
            mode: 'lines+markers',
            name: 'Vanishing Gradient',
            line: { width: 3, color: '#DC2626', dash: 'dash' },
          },
        ]}
        layout={{
          title: 'Gradient Behavior Across Depth',
          xaxis: { title: 'Layer Depth' },
          yaxis: { title: '|Gradient|' },
          plot_bgcolor: '#FFFFFF',
          paper_bgcolor: '#FFFFFF',
          font: { color: '#0F172A' },
          height: 320,
          margin: { l: 20, r: 20, t: 50, b: 20 },
        }}
        style={{ width: '100%', height: '350px' }}
        config={{ displayModeBar: false }}
      />
    )
  }

  // Backprop weight update
  const createBackpropWeights = () => {
    return (
      <Plot
        data={[
          {
            x: ['w1,1', 'w1,2', 'w1,3', 'w2,1', 'w2,2', 'w2,3'],
            y: [-0.000945, -0.026775, -0.213385, -0.000555, -0.11466, -0.022184],
            type: 'bar',
            marker: {
              color: ['#DC2626', '#DC2626', '#DC2626', '#DC2626', '#DC2626', '#DC2626'].map((c, i) => i % 2 === 0 ? c : '#5227FF'),
            },
            text: ['-0.000945', '-0.026775', '-0.213385', '-0.000555', '-0.11466', '-0.022184'],
            textposition: 'outside',
          },
        ]}
        layout={{
          title: 'Example Weight Update Magnitudes (Delta w)',
          xaxis: { title: 'Weights' },
          yaxis: { title: 'Update Value' },
          plot_bgcolor: '#FFFFFF',
          paper_bgcolor: '#FFFFFF',
          font: { color: '#0F172A' },
          height: 340,
          margin: { l: 20, r: 20, t: 50, b: 20 },
        }}
        style={{ width: '100%', height: '360px' }}
        config={{ displayModeBar: false }}
      />
    )
  }

  // MLP parameter growth
  const createParameterGrowth = () => {
    const widths = [4, 8, 16, 32, 64]
    const params = widths.map(w => 2 * w + w * w + w + 1)

    return (
      <Plot
        data={[
          {
            x: widths,
            y: params,
            mode: 'lines+markers',
            line: { width: 3, color: '#5227FF' },
            marker: { size: 10 },
          },
        ]}
        layout={{
          title: 'MLP Parameter Growth With Hidden Width',
          xaxis: { title: 'Hidden Neurons' },
          yaxis: { title: 'Approx Parameters' },
          plot_bgcolor: '#FFFFFF',
          paper_bgcolor: '#FFFFFF',
          font: { color: '#0F172A' },
          height: 320,
          margin: { l: 20, r: 20, t: 50, b: 20 },
        }}
        style={{ width: '100%', height: '350px' }}
        config={{ displayModeBar: false }}
      />
    )
  }

  // MLP training profile
  const createMLPTraining = () => {
    const epochs = Array.from({ length: 10 }, (_, i) => i + 1)
    const trainAcc = [0.62, 0.74, 0.80, 0.84, 0.87, 0.89, 0.90, 0.91, 0.92, 0.93]
    const valAcc = [0.60, 0.71, 0.77, 0.81, 0.84, 0.85, 0.86, 0.86, 0.86, 0.86]

    return (
      <Plot
        data={[
          {
            x: epochs,
            y: trainAcc,
            mode: 'lines+markers',
            name: 'Train Accuracy',
            line: { color: '#5227FF', width: 3 },
          },
          {
            x: epochs,
            y: valAcc,
            mode: 'lines+markers',
            name: 'Validation Accuracy',
            line: { color: '#9CA3AF', width: 3, dash: 'dash' },
          },
        ]}
        layout={{
          title: 'MLP Training Profile (Accuracy vs Epoch)',
          xaxis: { title: 'Epoch' },
          yaxis: { title: 'Accuracy', range: [0.5, 1.0], gridcolor: '#E5E7EB' },
          plot_bgcolor: '#FFFFFF',
          paper_bgcolor: '#FFFFFF',
          font: { color: '#0F172A' },
          height: 340,
          margin: { l: 20, r: 20, t: 50, b: 20 },
        }}
        style={{ width: '100%', height: '360px' }}
        config={{ displayModeBar: false }}
      />
    )
  }

  // Convolution kernel
  const createKernelHeatmap = () => {
    const kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    
    return (
      <Plot
        data={[
          {
            z: kernel,
            type: 'heatmap',
            colorscale: 'RdBu',
            zmid: 0,
            text: kernel,
            texttemplate: '%{text}',
            textfont: { size: 14 },
            showscale: true,
          },
        ]}
        layout={{
          title: 'Edge-Detection Convolution Kernel',
          xaxis: { visible: false },
          yaxis: { visible: false },
          plot_bgcolor: '#FFFFFF',
          paper_bgcolor: '#FFFFFF',
          font: { color: '#0F172A' },
          height: 320,
          margin: { l: 20, r: 20, t: 50, b: 20 },
        }}
        style={{ width: '100%', height: '350px' }}
        config={{ displayModeBar: false }}
      />
    )
  }

  // CV task complexity map
  const createCVTaskMap = () => {
    return (
      <Plot
        data={[
          {
            r: [0.45, 0.72, 0.88, 0.45],
            theta: ['Classification', 'Detection', 'Segmentation', 'Classification'],
            fill: 'toself',
            type: 'scatterpolar',
            line: { color: '#5227FF', width: 3 },
            marker: { color: '#7C3AED', size: 8 },
          },
        ]}
        layout={{
          title: 'CNN Lab Task Complexity Map',
          polar: { radialaxis: { range: [0, 1], visible: true } },
          paper_bgcolor: '#FFFFFF',
          plot_bgcolor: '#FFFFFF',
          font: { color: '#0F172A' },
          height: 340,
          margin: { l: 20, r: 20, t: 50, b: 20 },
          showlegend: false,
        }}
        style={{ width: '100%', height: '360px' }}
        config={{ displayModeBar: false }}
      />
    )
  }

  // RNN memory
  const createRNNMemory = () => {
    const t = Array.from({ length: 10 }, (_, i) => i + 1)
    const shortMemory = t.map(ti => Math.exp(-0.55 * ti))
    const gatedMemory = t.map(ti => Math.exp(-0.20 * ti))

    return (
      <Plot
        data={[
          {
            x: t,
            y: shortMemory,
            mode: 'lines+markers',
            name: 'Vanilla RNN Memory',
            line: { width: 3, color: '#DC2626' },
          },
          {
            x: t,
            y: gatedMemory,
            mode: 'lines+markers',
            name: 'LSTM/GRU Memory',
            line: { width: 3, color: '#5227FF' },
          },
        ]}
        layout={{
          title: 'How Memory Persists Across Time Steps',
          xaxis: { title: 'Time Step' },
          yaxis: { title: 'Information Retained' },
          plot_bgcolor: '#FFFFFF',
          paper_bgcolor: '#FFFFFF',
          font: { color: '#0F172A' },
          height: 320,
          margin: { l: 20, r: 20, t: 50, b: 20 },
        }}
        style={{ width: '100%', height: '350px' }}
        config={{ displayModeBar: false }}
      />
    )
  }

  // RNN unrolling
  const createRNNUnrolling = () => {
    return (
      <Plot
        data={[
          {
            x: ['t-2', 't-1', 't', 't+1'],
            y: [0.22, 0.47, 0.63, 0.71],
            mode: 'lines+markers+text',
            text: ['h=0.22', 'h=0.47', 'h=0.63', 'h=0.71'],
            textposition: 'top center',
            line: { color: '#5227FF', width: 3 },
            marker: { size: 11, color: '#7C3AED' },
          },
        ]}
        layout={{
          title: 'RNN Unrolling: Hidden State Across Time',
          xaxis: { title: 'Time Step' },
          yaxis: { title: 'Hidden State Magnitude', range: [0, 1], gridcolor: '#E5E7EB' },
          plot_bgcolor: '#FFFFFF',
          paper_bgcolor: '#FFFFFF',
          font: { color: '#0F172A' },
          height: 320,
          margin: { l: 20, r: 20, t: 50, b: 20 },
        }}
        style={{ width: '100%', height: '350px' }}
        config={{ displayModeBar: false }}
      />
    )
  }

  // Deep learning impact
  const createDLImpact = () => {
    const domains = ['Vision', 'NLP', 'Speech', 'Healthcare', 'Finance', 'Robotics']
    const impact = [92, 88, 81, 76, 73, 69]
    const colors = ['#5227FF', '#7C3AED', '#9CA3AF', '#D1D5DB', '#E5E7EB', '#F3F4F6']

    return (
      <Plot
        data={[
          {
            x: domains,
            y: impact,
            type: 'bar',
            marker: { color: colors },
            text: impact.map(v => `${v}%`),
            textposition: 'outside',
          },
        ]}
        layout={{
          title: 'Where Deep Learning Creates Strong Impact',
          xaxis: { title: 'Domain' },
          yaxis: { title: 'Relative Impact Score', range: [0, 100], gridcolor: '#E5E7EB' },
          plot_bgcolor: '#FFFFFF',
          paper_bgcolor: '#FFFFFF',
          font: { color: '#0F172A' },
          height: 350,
          margin: { l: 20, r: 20, t: 50, b: 20 },
        }}
        style={{ width: '100%', height: '370px' }}
        config={{ displayModeBar: false }}
      />
    )
  }

  const createHopfieldEnergyCurve = () => {
    const iterations = Array.from({ length: 12 }, (_, i) => i)
    const energy = [0, -1.5, -3.2, -4.8, -5.9, -6.7, -7.3, -7.8, -8.0, -8.0, -8.0, -8.0]

    return (
      <Plot
        data={[
          {
            x: iterations,
            y: energy,
            mode: 'lines+markers',
            line: { width: 3, color: '#5227FF' },
            marker: { size: 7, color: '#7C3AED' },
            name: 'Energy E',
          },
        ]}
        layout={{
          title: 'Hopfield Energy Convergence',
          xaxis: { title: 'Iteration' },
          yaxis: { title: 'Energy (Lyapunov)', gridcolor: '#E5E7EB' },
          plot_bgcolor: '#FFFFFF',
          paper_bgcolor: '#FFFFFF',
          font: { color: '#0F172A' },
          height: 330,
          margin: { l: 20, r: 20, t: 50, b: 20 },
        }}
        style={{ width: '100%', height: '350px' }}
        config={{ displayModeBar: false }}
      />
    )
  }

  const createHopfieldCapacityPlot = () => {
    const neurons = [100, 225, 400, 625, 900]
    const capacity = neurons.map((n) => Math.floor(0.15 * n))

    return (
      <Plot
        data={[
          {
            x: neurons,
            y: capacity,
            type: 'bar',
            marker: { color: ['#5227FF', '#6D4CFF', '#7C3AED', '#8B5CF6', '#A78BFA'] },
            text: capacity.map((c) => `${c}`),
            textposition: 'outside',
          },
        ]}
        layout={{
          title: 'Approximate Discrete Hopfield Capacity (0.15n)',
          xaxis: { title: 'Neuron Count (n)' },
          yaxis: { title: 'Max Storable Patterns', gridcolor: '#E5E7EB' },
          plot_bgcolor: '#FFFFFF',
          paper_bgcolor: '#FFFFFF',
          font: { color: '#0F172A' },
          height: 330,
          margin: { l: 20, r: 20, t: 50, b: 20 },
        }}
        style={{ width: '100%', height: '350px' }}
        config={{ displayModeBar: false }}
      />
    )
  }

  const content = {
    perceptron_mlp: {
      title: 'Perceptron + MLP',
      sections: [
        {
          heading: 'Unified Topic Overview',
          text: 'This track combines single-layer and multi-layer perceptrons into one learning path. A perceptron builds the intuition for weighted sums and decision boundaries, while MLP extends the same idea with hidden layers to model nonlinear patterns.',
        },
        {
          heading: 'Single Layer Perceptron',
          text: 'The single-layer perceptron is a linear classifier for binary decisions. It computes a weighted sum plus bias, applies an activation, and updates parameters using prediction error.',
          formula: 'z = w^T x + b\n\hat{y} = f(z)\n\text{Decision boundary: } w^T x + b = 0',
        },
        {
          heading: 'Multilayer Perceptron (MLP)',
          text: 'MLP stacks fully connected hidden layers on top of perceptron logic. Nonlinear activations (ReLU, Sigmoid, Tanh) allow representation of complex relationships that are not linearly separable.',
          formula: 'a^{(l)} = f(W^{(l)}a^{(l-1)} + b^{(l)})\n\mathcal{L}_{MSE} = \frac{1}{N}\sum_i (y_i - \hat{y}_i)^2\nw \leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w}',
        },
        {
          heading: 'When to Use Which',
          text: 'Use a single-layer perceptron for linearly separable binary tasks and fast interpretability. Use MLP when interactions are nonlinear and you need deeper feature transformations.',
        },
      ],
      graphs: [
        { title: 'Activation Functions', render: createActivationPlot },
        { title: 'Perceptron Decision Boundary', render: createPerceptronSurface },
        { title: 'MLP Training Profile', render: createMLPTraining },
      ],
    },
    neural_flow_engine: {
      title: 'Neural Flow Engine',
      sections: [
        {
          heading: 'Unified Training Cycle',
          text: 'Neural Flow Engine merges forward propagation and backward propagation into one continuous training pipeline: Forward pass -> Loss computation -> Backward gradients -> Parameter update.',
        },
        {
          heading: 'Forward Stage',
          text: 'Inputs move layer-by-layer and each neuron computes weighted combinations and activations. The final layer produces the prediction used for loss.',
          formula: 'a^{(l)} = f(W^{(l)}a^{(l-1)} + b^{(l)})',
        },
        {
          heading: 'Loss + Backward Stage',
          text: 'Loss quantifies prediction error, then backpropagation applies chain rule from output layer to input layer to compute each gradient.',
          formula: '\mathcal{L} = \frac{1}{N}\sum_i (y_i - \hat{y}_i)^2\n\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T',
        },
        {
          heading: 'Update Stage + Stability',
          text: 'Gradients update weights using learning rate control. Monitoring gradient magnitudes and loss trends helps detect vanishing/exploding behavior and overfitting risk.',
          formula: 'W \leftarrow W - \eta \frac{\partial \mathcal{L}}{\partial W}',
        },
      ],
      graphs: [
        { title: 'Layer-wise Forward Values', render: createForwardFlow },
        { title: 'Gradient Stability', render: createGradientDecay },
        { title: 'Weight Update Magnitudes', render: createBackpropWeights },
      ],
    },
    hopfield: {
      title: 'Hopfield Network',
      sections: [
        {
          heading: 'Overview',
          text: 'A Hopfield Network is a recurrent neural network used for associative memory. It stores patterns and recalls complete patterns from partial or noisy inputs by iteratively updating neurons until convergence.',
        },
        {
          heading: 'Core Properties',
          text: '• Single-layer fully connected recurrent topology\n• Auto-associative memory (input and output dimensions are the same)\n• Symmetric weights with no self-connections\n• Converges toward stable states via energy minimization',
          formula: 'w_{ij} = w_{ji},\; w_{ii} = 0',
        },
        {
          heading: 'Types of Hopfield Network',
          text: 'Discrete Hopfield Network uses finite neuron states (binary 0/1 or bipolar -1/+1) and updates in discrete steps. Continuous Hopfield Network uses continuous outputs in [0,1] and evolves through differential equations for smoother optimization dynamics.',
        },
        {
          heading: 'Architecture and Recall Dynamics',
          text: 'Each neuron connects to all others, keeps state, and is updated (typically asynchronously) based on the weighted network input. The network converges when no neuron state changes over a full sweep.',
          formula: 'y^{in}_i = \sum_{j=1}^{n} w_{ij} y_j\n\text{Binary: } y_i = \begin{cases}1,& y_i^{in}>\theta_i\\y_i,& y_i^{in}=\theta_i\\0,& y_i^{in}<\theta_i\end{cases}\n\text{Bipolar: } y_i = \begin{cases}+1,& y_i^{in}\ge 0\\-1,& y_i^{in}<0\end{cases}',
        },
        {
          heading: 'Training Rule (Hebbian Storage)',
          text: 'Hopfield training stores memory patterns into the weight matrix by outer products. Binary patterns are commonly converted to bipolar form before storage.',
          formula: 'w_{ij} = \sum_{p=1}^{P} s_i^{(p)} s_j^{(p)},\; i \ne j\n\text{Binary to bipolar: } (2s_i^{(p)}-1)',
        },
        {
          heading: 'Energy Function and Convergence',
          text: 'The Lyapunov energy decreases or remains constant during updates, guaranteeing convergence to a stable local minimum in the discrete asynchronous case.',
          formula: 'E = -\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n} w_{ij}x_i x_j + \sum_{i=1}^{n}\theta_i x_i\n\frac{dE}{dt} \le 0\;\text{(continuous case)}',
        },
        {
          heading: 'Implementation Roadmap',
          text: '1. Prepare and preprocess patterns (binary/bipolar).\n2. Train weights with Hebbian outer products and remove diagonal terms.\n3. Run asynchronous recall iterations.\n4. Track energy until convergence.\n5. Compare train, noisy input, and recalled output.\n6. Evaluate recall quality using Hamming distance and similarity scores.',
        },
      ],
      graphs: [
        { title: 'Hopfield Energy Descent', render: createHopfieldEnergyCurve },
        { title: 'Memory Capacity Trend', render: createHopfieldCapacityPlot },
      ],
    },
    cnn_lab: {
      title: 'CNN Lab Concepts',
      sections: [
        {
          heading: 'Overview',
          text: 'CNN Lab focuses on image classification and interpretability using pretrained convolutional neural networks. It combines predictive performance with visual explanations through feature maps and Grad-CAM.',
        },
        {
          heading: 'Basics',
          text: '• Introduction: goal is machine understanding of visual scenes\n• Image Processing: enhancement, filtering, denoising, transformation\n• Image Representation and Pixels: images as numeric grids/channels\n• Image Analysis and Manipulation: extracting structure and editing content',
        },
        {
          heading: 'Key Concepts',
          text: '1. Image Transformation: Changes spatial or intensity structure. Geometric transforms (rotate/scale/translate/warp) align viewpoints, intensity transforms adjust brightness, Fourier-domain transforms isolate periodic noise.\n\n2. Image Enhancement: Improves visual quality. Histogram equalization exposes hidden details, sharpening emphasizes boundaries, color correction reduces illumination bias.\n\n3. Noise Reduction: Median filtering for salt-and-pepper, bilateral filtering preserves edges, wavelet denoising removes multi-scale noise.\n\n4. Morphological Operations: Erosion/dilation shrink/expand foreground, opening removes noise blobs, closing fills holes, morphological gradient highlights boundaries.\n\n5. Feature Extraction: Edge detectors (Canny/Sobel), corner detectors, SIFT/SURF/ORB, HOG descriptors.',
          formula: '(I*K)(i,j) = Σₘ Σₙ I(i-m,j-n)K(m,n)\nIoU = Area(B_pred ∩ B_gt) / Area(B_pred ∪ B_gt)',
        },
        {
          heading: 'Deep Learning for CNN Lab',
          text: '• Convolutional Neural Networks (CNNs): Convolutional layers, pooling layers, fully connected layers\n• Generative Adversarial Networks (GANs): DCGAN, cGAN, CycleGAN, SRGAN, StyleGAN\n• Variational Autoencoders (VAEs): VAE, Denoising Autoencoder, Convolutional Autoencoder\n• Vision Transformers (ViT): ViT, Swin Transformer, CvT\n• Vision-Language Models: CLIP, ALIGN, BLIP',
        },
      ],
      graphs: [
        { title: 'Convolution Kernel', render: createKernelHeatmap },
        { title: 'Task Complexity Map', render: createCVTaskMap },
      ],
    },
    rnn: {
      title: 'RNN Lab',
      sections: [
        {
          heading: 'Overview',
          text: 'Recurrent Neural Networks (RNNs) are designed for sequential and temporal data. They retain information from previous steps using hidden state memory, making them useful where order and context matter.',
        },
        {
          heading: 'Why RNNs',
          text: '• Designed for sequential and temporal inputs\n• Maintains memory of previous inputs\n• Common in NLP, forecasting, and speech tasks\n\nWhen predicting the next word in a sentence, we use previous words as context. RNNs work similarly by passing hidden state from one time step to the next.',
        },
        {
          heading: 'Core Architecture',
          text: '• Recurrent neurons: units with hidden state that carry sequence memory\n• Unfolding (unrolling): expansion over time steps for learning dependencies',
          formula: 'h = σ(U·X + W·h_{t-1} + B)\nY = O(V·h + C)\nY = f(X,h,W,U,V,B,C)\nh_t = f(h_{t-1}, x_t)\nh_t = tanh(W_{hh}h_{t-1} + W_{xh}x_t)\ny_t = W_{hy}h_t',
        },
        {
          heading: 'Backpropagation Through Time (BPTT)',
          text: 'Because sequence states depend on earlier states, gradients are propagated backward through time steps. This enables learning of long-range dependencies.',
          formula: '∂L(θ)/∂W = ∂L(θ)/∂h₃ · ∂h₃/∂W\n∂L(θ)/∂W = ∂L(θ)/∂h₃ · Σₖ ∂h₃/∂hₖ · ∂hₖ/∂W',
        },
        {
          heading: 'RNN Variants',
          text: '• Vanilla RNN: simple, short dependency learning\n• Bidirectional RNN: forward + backward context\n• LSTM: input/forget/output gates for long memory\n• GRU: lighter gated version, often faster\n\nLSTMs and GRUs solve the vanishing gradient problem present in vanilla RNNs.',
        },
        {
          heading: 'Applications',
          text: '• Time-series prediction\n• NLP and language modeling\n• Speech recognition\n• Video and sequence understanding',
        },
      ],
      graphs: [
        { title: 'Memory Retention', render: createRNNMemory },
        { title: 'Hidden State Evolution', render: createRNNUnrolling },
      ],
    },
  }

  const selected = content[activeTopic]

  return (
    <>
      <div className="study-pill-nav-wrap">
        <PillNav
          items={studyItems}
          activeHref={activeTopic}
          onItemSelect={(item) => setActiveTopic(item.href)}
          baseColor="#ffffff"
          pillColor="#060010"
          hoveredPillTextColor="#060010"
          pillTextColor="#ffffff"
          initialLoadAnimation
        />
      </div>

      {/* Sections with Study Content */}
      {selected.sections.map((section, idx) => (
        <Section key={idx} title={section.heading}>
          <p className="module-note" style={{ whiteSpace: 'pre-wrap' }}>
            {section.text}
          </p>
          {section.formula && (
            <div style={{ background: '#f0f0f0', padding: '12px', borderRadius: '4px', margin: '12px 0', fontFamily: 'monospace', fontSize: '12px', whiteSpace: 'pre-wrap' }}>
              {section.formula}
            </div>
          )}
        </Section>
      ))}

      {/* Graphs */}
      {selected.graphs && selected.graphs.map((graph, idx) => (
        <Section key={`graph-${idx}`} title={graph.title}>
          {graph.render()}
        </Section>
      ))}

      <Section title="practice checklist">
        <ol className="module-steps">
          <li>Read the theory section carefully for the selected module.</li>
          <li>Review the mathematical formulas to understand the equations.</li>
          <li>Interpret each graph before changing parameters in the module view.</li>
          <li>Cross-reference the study content with hands-on experiments in other modules.</li>
          <li>Document key insights and implementation notes for future reference.</li>
        </ol>
      </Section>
    </>
  )
}

function ModulePage({ activeModule, moduleDescription }) {
  const content = () => {
    if (activeModule === 'Perceptron') return <PerceptronAndMLPModule />
    if (activeModule === 'Neural Flow Engine') return <NeuralFlowEngineModule />
    if (activeModule === 'CNN Lab') return <CNNLabModule />
    if (activeModule === 'RNN Lab') return <RNNModule />
    if (activeModule === 'Hopfield Network') return <HopfieldModule />
    return <StudyModeModule />
  }

  return (
    <main className="main-panel module-page-simple">
      <header className="module-header-line">
        <div>
          <h1>{activeModule}</h1>
          {moduleDescription ? <p>{moduleDescription}</p> : null}
        </div>
        <span className="badge">DeepNexus</span>
      </header>

      <div className="module-plain-layout">
        <ModuleErrorBoundary resetKey={activeModule}>{content()}</ModuleErrorBoundary>
      </div>
    </main>
  )
}

function App() {
  const [activeModule, setActiveModule] = useState(null)
  const [menuOpen, setMenuOpen] = useState(false)

  const moduleDescription = useMemo(() => (activeModule ? MODULE_DESCRIPTIONS[activeModule] : ''), [activeModule])

  const handleMenuSelect = (item) => {
    setMenuOpen(false)
    if (item.moduleName === '__HOME__') {
      setActiveModule(null)
      return
    }
    setActiveModule(item.moduleName)
  }

  const goHome = () => {
    setMenuOpen(false)
    setActiveModule(null)
  }

  if (activeModule === null) {
    return (
      <div className="app-wrapper">
        <header className="navbar">
          <div className="navbar-brand">DeepNexus</div>
          <a className="github-link" href={GITHUB_URL} target="_blank" rel="noreferrer" aria-label="View on Github" data-tooltip="View on Github">
            <svg viewBox="0 0 24 24" aria-hidden="true" className="github-icon">
              <path
                fill="currentColor"
                d="M12 .5C5.65.5.5 5.66.5 12.03c0 5.1 3.3 9.43 7.88 10.96.58.1.79-.25.79-.56 0-.28-.01-1.01-.02-1.99-3.2.7-3.88-1.54-3.88-1.54-.53-1.35-1.3-1.72-1.3-1.72-1.06-.73.08-.71.08-.71 1.18.08 1.8 1.21 1.8 1.21 1.04 1.8 2.74 1.28 3.4.98.1-.76.41-1.28.74-1.58-2.55-.29-5.24-1.28-5.24-5.67 0-1.25.45-2.28 1.18-3.09-.12-.29-.51-1.46.11-3.04 0 0 .97-.31 3.16 1.18a10.9 10.9 0 0 1 5.76 0c2.19-1.49 3.16-1.18 3.16-1.18.62 1.58.23 2.75.11 3.04.74.81 1.18 1.84 1.18 3.09 0 4.4-2.69 5.38-5.26 5.67.42.36.79 1.06.79 2.14 0 1.55-.01 2.8-.01 3.18 0 .31.21.67.8.56A11.54 11.54 0 0 0 23.5 12.03C23.5 5.66 18.35.5 12 .5Z"
              />
            </svg>
          </a>
        </header>
        <HomePage menuOpen={menuOpen} onToggleMenu={() => setMenuOpen((prev) => !prev)} onMenuSelect={handleMenuSelect} />
      </div>
    )
  }

  const moduleMenuItems = [{ label: 'Home', moduleName: '__HOME__' }, ...MENU_ITEMS]

  return (
    <div className="module-shell">
      <div className="module-menu-anchor">
        <StaggeredMenu
          items={moduleMenuItems}
          isOpen={menuOpen}
          onToggle={() => setMenuOpen((prev) => !prev)}
          onItemClick={handleMenuSelect}
          toggleVariant="triangle"
          activeItemKey={activeModule || '__HOME__'}
        />
      </div>

      <div className={menuOpen ? 'module-content-shell menu-open' : 'module-content-shell'}>
        <ModulePage activeModule={activeModule} moduleDescription={moduleDescription} />
      </div>
    </div>
  )
}

export default App
