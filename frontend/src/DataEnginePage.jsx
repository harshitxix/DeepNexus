import { useEffect, useMemo, useRef, useState } from 'react'
import PlotlyImport from 'plotly.js-dist-min'

const API_BASE = '/api/data-engine'
const Plotly = PlotlyImport?.default || PlotlyImport

function toNumeric(value) {
  const num = Number(value)
  return Number.isFinite(num) ? num : null
}

function NeuralOrbitLoader() {
  return (
    <div className="neural-orbit-loader" aria-hidden="true">
      <div className="neural-orbit-core">
        <svg viewBox="0 0 100 100" className="neural-orbit-svg">
          <ellipse cx="50" cy="50" rx="40" ry="18" stroke="url(#orbit-grad)" strokeWidth="1" fill="none" />
          <ellipse
            cx="50"
            cy="50"
            rx="25"
            ry="40"
            stroke="url(#orbit-grad)"
            strokeWidth="2"
            fill="none"
            transform="rotate(45 50 50)"
          />
          <defs>
            <linearGradient id="orbit-grad">
              <stop offset="0%" stopColor="#6366f1" />
              <stop offset="100%" stopColor="#8b5cf6" />
            </linearGradient>
          </defs>
        </svg>

        <div className="neural-orbit-spin-wrap">
          <div className="neural-orbit-neuron-wrap">
            <div className="neural-orbit-neuron" />
          </div>
        </div>
      </div>
    </div>
  )
}

function Plot({ data, layout, config, style }) {
  const plotRef = useRef(null)

  useEffect(() => {
    if (!plotRef.current || !Plotly) return undefined
    Plotly.react(plotRef.current, data || [], layout || {}, config || {})
    return () => {
      if (plotRef.current) Plotly.purge(plotRef.current)
    }
  }, [data, layout, config])

  return <div ref={plotRef} style={style} />
}

const DataEnginePage = () => {
  const [step, setStep] = useState('upload')
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [selectedTarget, setSelectedTarget] = useState('')
  const [features, setFeatures] = useState(null)
  const [analysis, setAnalysis] = useState(null)
  const [sessionId, setSessionId] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [resultTab, setResultTab] = useState('analysis')
  const [conversation, setConversation] = useState([])
  const [userQuestion, setUserQuestion] = useState('')
  const [showFullReport, setShowFullReport] = useState(false)
  const [assistantThinking, setAssistantThinking] = useState(false)
  const [thinkingDotCount, setThinkingDotCount] = useState(1)
  const fileInputRef = useRef(null)

  useEffect(() => {
    if (!assistantThinking) {
      setThinkingDotCount(1)
      return undefined
    }
    const id = window.setInterval(() => {
      setThinkingDotCount((prev) => (prev % 3) + 1)
    }, 350)
    return () => window.clearInterval(id)
  }, [assistantThinking])

  // Basic markdown-ish -> HTML converter for common blocks (headings, bold, code, tables)
  const markdownToHtml = (raw) => {
    if (!raw) return ''
    // escape HTML
    let s = String(raw)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')

    // code blocks ```...```
    s = s.replace(/```([\s\S]*?)```/g, (m, code) => `<pre class="de-code"><code>${code.replace(/&/g,'&amp;').replace(/</g,'&lt;')}</code></pre>`)

    // tables: contiguous lines with | -> convert to table
    s = s.replace(/(^\|.*\|\s*$\n?(?:^\|.*\|\s*$\n?)+)/gm, (block) => {
      const lines = block.trim().split('\n').filter(Boolean)
      if (lines.length < 2) return block
      const header = lines[0].split('|').map((c) => c.trim()).filter(Boolean)
      // skip separator line if exists (---|---)
      let rowStart = 1
      if (/^\s*\|?\s*:-?/.test(lines[1]) || /---/.test(lines[1])) rowStart = 2
      const rows = lines.slice(rowStart).map((ln) => ln.split('|').map((c) => c.trim()).filter(Boolean))
      const th = header.map((h) => `<th>${h}</th>`).join('')
      const trs = rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join('')}</tr>`).join('')
      return `<table class="de-table"><thead><tr>${th}</tr></thead><tbody>${trs}</tbody></table>`
    })

    // headings
    s = s.replace(/^###\s*(.*)$/gm, '<h4>$1</h4>')
    s = s.replace(/^##\s*(.*)$/gm, '<h3>$1</h3>')
    s = s.replace(/^#\s*(.*)$/gm, '<h2>$1</h2>')

    // blockquotes
    s = s.replace(/^>\s*(.*)$/gm, '<blockquote>$1</blockquote>')

    // bold **text**
    s = s.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')

    // inline code `x`
    s = s.replace(/`([^`]+)`/g, '<code>$1</code>')

    // horizontal rule
    s = s.replace(/^---$/gm, '<hr/>')

    // bullet lists
    s = s.replace(/(?:^|\n)(-\s+.+(?:\n-\s+.+)+)/g, (block) => {
      const items = block
        .trim()
        .split('\n')
        .map((ln) => ln.replace(/^-\s+/, '').trim())
        .filter(Boolean)
      return `\n<ul>${items.map((item) => `<li>${item}</li>`).join('')}</ul>`
    })

    // paragraphs: split by double-newline
    const parts = s.split(/\n{2,}/g).map((p) => p.trim()).filter(Boolean)
    return parts
      .map((p) => {
        const isBlock = /^<(table|h2|h3|h4|pre|blockquote|hr|ul)/i.test(p)
        return isBlock ? p : `<p>${p.replace(/\n/g, '<br/>')}</p>`
      })
      .join('')
  }

  const cleanGeneratedText = (raw) => {
    if (!raw) return ''
    return String(raw).replace(/Dataset\s+Overview\s*[\-–]\s*Quick\s+Reference\s+for\s+Data\s*[- ]?Science\s+Teams/gi, 'Dataset Overview')
  }

  const quickOverviewMd = useMemo(() => {
    const rows = preview?.rows ?? analysis?.rows ?? 0
    const columns = preview?.columns ?? analysis?.columns ?? 0
    const totalNulls = (preview?.columns_info || []).reduce((sum, col) => sum + (col.null_count || 0), 0)

    const duplicateIssue = (analysis?.analysis?.recommendations?.data_quality_issues || []).find((issue) =>
      /duplicate/i.test(issue)
    )

    const numerical = features?.numerical_features?.length ?? analysis?.features?.numerical_features?.length ?? 0
    const categorical = features?.categorical_features?.length ?? analysis?.features?.categorical_features?.length ?? 0
    const datetime = features?.datetime_features?.length ?? analysis?.features?.datetime_features?.length ?? 0
    const suggestedModels = analysis?.analysis?.recommendations?.suggested_models || []
    const modelText = suggestedModels.length ? suggestedModels.join(', ') : 'No model list returned yet'
    const problemType = analysis?.analysis?.recommendations?.primary_problem_type || 'Not identified yet'

    const qualityIssues = analysis?.analysis?.recommendations?.data_quality_issues || []
    const qualityText = qualityIssues.length ? qualityIssues.join('; ') : 'No issues reported'

    const overviewLines = [
      '## Dataset Overview',
      '',
      `- **Dimensions:** ${rows} rows x ${columns} columns`,
      `- **Total null values:** ${totalNulls}`,
    ]
    if (duplicateIssue) {
      overviewLines.push(`- **Duplicate rows:** ${duplicateIssue}`)
    }
    overviewLines.push(
      `- **Detected features:** Numerical ${numerical}, Categorical ${categorical}, Datetime ${datetime}`,
      '',
      '## Modeling Direction',
      '',
      `- **Problem type:** ${problemType}`,
      `- **Suggested models:** ${modelText}`,
      '',
      '## Future Scope (Data Handling)',
      '',
      `- **Data-quality checks:** ${qualityText}`,
      '- **Before training:** validate dtypes, remove or treat duplicates if present, and confirm target consistency.',
      '- **Next iteration:** expand feature engineering and rerun analysis after data cleaning for stronger model selection.',
    )

    return overviewLines.join('\n')
  }, [preview, analysis, features])

  const numericColumns = useMemo(() => {
    if (!preview?.columns_info) return []
    return preview.columns_info
      .filter((col) => (col.type === 'integer' || col.type === 'float') && col.name !== selectedTarget)
      .map((col) => col.name)
  }, [preview, selectedTarget])

  const vizCharts = useMemo(() => {
    if (!preview?.preview_data?.length || !preview?.columns_info?.length) return []

    const rows = preview.preview_data
    const columnsInfo = preview.columns_info
    const numericCols = columnsInfo.filter((c) => c.type === 'integer' || c.type === 'float').map((c) => c.name)
    const categoricalCols = columnsInfo
      .filter((c) => c.type === 'categorical' || c.type === 'text')
      .map((c) => c.name)
    const datetimeCols = columnsInfo.filter((c) => c.type === 'datetime').map((c) => c.name)

    const charts = []

    if (selectedTarget) {
      const counts = new Map()
      rows.forEach((row) => {
        const key = String(row[selectedTarget] ?? 'Missing')
        counts.set(key, (counts.get(key) || 0) + 1)
      })
      const labels = [...counts.keys()].slice(0, 12)
      const values = labels.map((k) => counts.get(k) || 0)

      charts.push({
        title: `Target Distribution: ${selectedTarget}`,
        description: 'Shows frequency of target values in the preview sample.',
        data: [
          {
            type: 'bar',
            x: labels,
            y: values,
            marker: { color: '#5227ff', line: { color: '#33217a', width: 1 } },
            hovertemplate: `%{x}<br>Count: %{y}<extra></extra>`,
          },
        ],
        layout: {
          title: { text: `Target Distribution (${selectedTarget})`, font: { size: 16 } },
          xaxis: { title: { text: selectedTarget }, tickangle: -20 },
          yaxis: { title: { text: 'Count' } },
          paper_bgcolor: 'white',
          plot_bgcolor: '#f8faff',
          margin: { l: 70, r: 20, t: 60, b: 80 },
        },
      })
    }

    if (numericCols.length) {
      const n = numericCols[0]
      const vals = rows.map((r) => toNumeric(r[n])).filter((v) => v !== null)
      if (vals.length) {
        charts.push({
          title: `Distribution of ${n}`,
          description: 'Histogram for the primary numeric feature.',
          data: [
            {
              type: 'histogram',
              x: vals,
              marker: { color: '#0ea5e9' },
              nbinsx: 15,
              hovertemplate: `${n}: %{x}<br>Count: %{y}<extra></extra>`,
            },
          ],
          layout: {
            title: { text: `Histogram: ${n}`, font: { size: 16 } },
            xaxis: { title: { text: n } },
            yaxis: { title: { text: 'Count' } },
            paper_bgcolor: 'white',
            plot_bgcolor: '#f8faff',
            margin: { l: 70, r: 20, t: 60, b: 60 },
          },
        })
      }
    }

    if (numericCols.length >= 2) {
      const xCol = numericCols[0]
      const yCol = numericCols[1]
      const points = rows
        .map((r) => ({ x: toNumeric(r[xCol]), y: toNumeric(r[yCol]) }))
        .filter((p) => p.x !== null && p.y !== null)

      if (points.length) {
        charts.push({
          title: `${xCol} vs ${yCol}`,
          description: 'Scatter plot to inspect correlation and spread.',
          data: [
            {
              type: 'scatter',
              mode: 'markers',
              x: points.map((p) => p.x),
              y: points.map((p) => p.y),
              marker: {
                size: 9,
                color: '#16a34a',
                line: { color: 'white', width: 1 },
                opacity: 0.85,
              },
              hovertemplate: `${xCol}: %{x}<br>${yCol}: %{y}<extra></extra>`,
            },
          ],
          layout: {
            title: { text: `Scatter: ${xCol} vs ${yCol}`, font: { size: 16 } },
            xaxis: { title: { text: xCol } },
            yaxis: { title: { text: yCol } },
            paper_bgcolor: 'white',
            plot_bgcolor: '#f8faff',
            margin: { l: 70, r: 20, t: 60, b: 60 },
          },
        })
      }
    }

    if (datetimeCols.length && numericCols.length) {
      const dtCol = datetimeCols[0]
      const nCol = numericCols[0]
      const series = rows
        .map((r) => ({ x: r[dtCol], y: toNumeric(r[nCol]) }))
        .filter((p) => p.x && p.y !== null)
      if (series.length) {
        charts.push({
          title: `${nCol} Over Time (${dtCol})`,
          description: 'Time-series trend using datetime and numeric columns.',
          data: [
            {
              type: 'scatter',
              mode: 'lines+markers',
              x: series.map((p) => p.x),
              y: series.map((p) => p.y),
              line: { color: '#7c3aed', width: 2.5 },
              marker: { size: 6, color: '#7c3aed' },
              hovertemplate: `${dtCol}: %{x}<br>${nCol}: %{y}<extra></extra>`,
            },
          ],
          layout: {
            title: { text: `Time Series: ${nCol} by ${dtCol}`, font: { size: 16 } },
            xaxis: { title: { text: dtCol } },
            yaxis: { title: { text: nCol } },
            paper_bgcolor: 'white',
            plot_bgcolor: '#f8faff',
            margin: { l: 70, r: 20, t: 60, b: 70 },
          },
        })
      }
    }

    if (categoricalCols.length && numericCols.length) {
      const cCol = categoricalCols[0]
      const nCol = numericCols[0]
      const grouped = new Map()
      rows.forEach((r) => {
        const cat = String(r[cCol] ?? 'Missing')
        const val = toNumeric(r[nCol])
        if (val === null) return
        if (!grouped.has(cat)) grouped.set(cat, [])
        grouped.get(cat).push(val)
      })

      const entries = [...grouped.entries()].slice(0, 8)
      if (entries.length) {
        charts.push({
          title: `${nCol} by ${cCol}`,
          description: 'Box plot comparing numeric spread across categories.',
          data: entries.map(([name, vals]) => ({
            type: 'box',
            name,
            y: vals,
            boxpoints: 'outliers',
            marker: { color: '#f59e0b' },
            hovertemplate: `${cCol}: ${name}<br>${nCol}: %{y}<extra></extra>`,
          })),
          layout: {
            title: { text: `Box Plot: ${nCol} by ${cCol}`, font: { size: 16 } },
            xaxis: { title: { text: cCol } },
            yaxis: { title: { text: nCol } },
            paper_bgcolor: 'white',
            plot_bgcolor: '#f8faff',
            margin: { l: 70, r: 20, t: 60, b: 70 },
          },
        })
      }
    }

    return charts
  }, [preview, selectedTarget])

  const handleFileSelect = async (event) => {
    const selected = event.target.files?.[0]
    if (!selected) return

    if (!selected.name.toLowerCase().endsWith('.csv')) {
      setError('Only CSV files are allowed for now.')
      return
    }

    setFile(selected)
    setError('')
    setLoading(true)

    try {
      const formData = new FormData()
      formData.append('file', selected)

      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail || 'Upload failed.')
      }

      const payload = await response.json()
      setPreview(payload)
      setSessionId(payload.session_id)
      setSelectedTarget(payload.suggested_target_column || '')
      setStep('preview')
    } catch (uploadError) {
      setError(uploadError.message || 'Upload failed.')
    } finally {
      setLoading(false)
    }
  }

  const handleConfirmTargetAndDetectFeatures = async () => {
    if (!selectedTarget) {
      setError('Please confirm a target column to continue.')
      return
    }

    setLoading(true)
    setError('')

    try {
      const response = await fetch(
        `${API_BASE}/detect-features?session_id=${encodeURIComponent(sessionId)}&target_column=${encodeURIComponent(selectedTarget)}`,
        { method: 'POST' }
      )

      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail || 'Feature detection failed.')
      }

      const payload = await response.json()
      setFeatures(payload)
      setStep('confirm')
    } catch (featureError) {
      setError(featureError.message || 'Feature detection failed.')
    } finally {
      setLoading(false)
    }
  }

  const handleRunPredefinedTasks = async () => {
    if (!sessionId) {
      setError('No session found. Please upload a CSV first.')
      return
    }
    console.log('Starting analysis with sessionId:', sessionId, 'target:', selectedTarget)
    setLoading(true)
    setError('')
    setStep('analyzing')

    try {
      const response = await fetch(
        `${API_BASE}/analyze?session_id=${encodeURIComponent(sessionId)}&target_column=${encodeURIComponent(selectedTarget)}`,
        { method: 'POST' }
      )

      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        console.error('Analysis failed:', body)
        throw new Error(body.detail || 'Analysis failed.')
      }

      const payload = await response.json()
      setAnalysis(payload)
      setResultTab('analysis')
      setStep('results')
    } catch (analysisError) {
      setError(analysisError.message || 'Analysis failed.')
      setStep('confirm')
    } finally {
      setLoading(false)
    }
  }

  const handleAskAssistant = async () => {
    if (!analysis?.analysis?.analysis_id || !userQuestion.trim()) return

    const nextConversation = [...conversation, { role: 'user', content: userQuestion.trim() }]
    setConversation(nextConversation)
    setUserQuestion('')
    setLoading(true)
    setAssistantThinking(true)
    setError('')

    try {
      const response = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: nextConversation[nextConversation.length - 1].content,
          analysis_id: analysis.analysis.analysis_id,
          conversation_history: nextConversation.slice(0, -1),
        }),
      })

      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail || 'Assistant request failed.')
      }

      const payload = await response.json()
      setConversation((prev) => [...prev, { role: 'assistant', content: payload.answer }])
    } catch (assistantError) {
      setError(assistantError.message || 'Assistant request failed.')
    } finally {
      setAssistantThinking(false)
      setLoading(false)
    }
  }

  const handleDownloadReport = async () => {
    if (!analysis?.analysis?.analysis_id) return

    try {
      const response = await fetch(`${API_BASE}/download-report/${analysis.analysis.analysis_id}`)
      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail || 'Report download failed.')
      }

      const payload = await response.json()
      const anchor = document.createElement('a')
      anchor.setAttribute('href', `data:text/plain;charset=utf-8,${encodeURIComponent(payload.content)}`)
      anchor.setAttribute('download', payload.filename)
      anchor.style.display = 'none'
      document.body.appendChild(anchor)
      anchor.click()
      document.body.removeChild(anchor)
    } catch (downloadError) {
      setError(downloadError.message || 'Report download failed.')
    }
  }

  const handleStartOver = () => {
    setStep('upload')
    setFile(null)
    setPreview(null)
    setSelectedTarget('')
    setFeatures(null)
    setAnalysis(null)
    setSessionId('')
    setError('')
    setResultTab('analysis')
    setConversation([])
    setUserQuestion('')
    setShowFullReport(false)
  }

  const renderUpload = () => (
    <section style={styles.panel}>
      <h2 style={styles.sectionHeading}>Upload Dataset</h2>
      <p style={styles.muted}>CSV format only</p>
      <div
        style={styles.dropZone}
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault()
          handleFileSelect({ target: { files: event.dataTransfer.files } })
        }}
      >
        <p style={styles.dropTitle}>Drop CSV here or click to browse</p>
        <p style={styles.dropSubtitle}>Initial tasks will run after target confirmation.</p>
      </div>
      <input ref={fileInputRef} type="file" accept=".csv" style={{ display: 'none' }} onChange={handleFileSelect} />
      {file ? <p style={styles.fileLabel}>Selected file: {file.name}</p> : null}
    </section>
  )

  const renderPreview = () => (
    <section style={styles.panel}>
      <h2 style={styles.sectionHeading}>Preview and Confirm Target</h2>
      <p style={styles.muted}>Rows: {preview?.rows || 0} | Columns: {preview?.columns || 0}</p>

      <div style={styles.previewTableShell}>
        <table style={styles.table}>
          <thead>
            <tr>
              {preview?.preview_data?.[0] ? Object.keys(preview.preview_data[0]).map((key) => <th key={key} style={styles.th}>{key}</th>) : null}
            </tr>
          </thead>
          <tbody>
            {(preview?.preview_data || []).map((row, rowIndex) => (
              <tr key={rowIndex}>
                {Object.values(row).map((value, colIndex) => (
                  <td key={`${rowIndex}-${colIndex}`} style={styles.td}>{String(value).slice(0, 40)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={styles.targetRow}>
        <label htmlFor="target-select" style={styles.label}>Target column (auto-detected, editable):</label>
        <select id="target-select" style={styles.select} value={selectedTarget} onChange={(event) => setSelectedTarget(event.target.value)}>
          <option value="">Select target</option>
          {(preview?.columns_info || []).map((col) => (
            <option key={col.name} value={col.name}>{col.name} ({col.type})</option>
          ))}
        </select>
        {preview?.suggested_target_column ? <p style={styles.suggested}>Suggested target: {preview.suggested_target_column}</p> : null}
      </div>

      <div style={styles.buttonRow}>
        <button type="button" style={styles.primaryButton} disabled={loading} onClick={handleConfirmTargetAndDetectFeatures}>
          {loading ? 'Detecting features...' : 'Confirm target and detect features'}
        </button>
        <button type="button" style={styles.secondaryButton} onClick={handleStartOver}>Start over</button>
      </div>
    </section>
  )

  const renderFeatureConfirmation = () => (
    <section style={styles.panel}>
      <h2 style={styles.sectionHeading}>Feature Review</h2>
      <p style={styles.muted}>Confirm detected features, then run predefined analysis tasks.</p>
      <div style={styles.featureColumns}>
        <div style={styles.featureColumn}>
          <h3 style={styles.featureTitle}>Numerical</h3>
          {(features?.numerical_features || []).map((name) => <div key={name} style={styles.featureItem}>{name}</div>)}
        </div>
        <div style={styles.featureColumn}>
          <h3 style={styles.featureTitle}>Categorical</h3>
          {(features?.categorical_features || []).map((name) => <div key={name} style={styles.featureItem}>{name}</div>)}
        </div>
        <div style={styles.featureColumn}>
          <h3 style={styles.featureTitle}>Datetime</h3>
          {(features?.datetime_features || []).map((name) => <div key={name} style={styles.featureItem}>{name}</div>)}
        </div>
      </div>
      <div style={styles.buttonRow}>
        <button type="button" style={styles.primaryButton} disabled={loading} onClick={handleRunPredefinedTasks}>
          {loading ? 'Running predefined tasks...' : 'Run predefined tasks'}
        </button>
        <button type="button" style={styles.secondaryButton} onClick={handleStartOver}>Start over</button>
      </div>
    </section>
  )

  const renderAnalysisTab = () => (
    <div style={styles.tabPane}>
      <div style={styles.resultsCard}>
        <div style={styles.summaryText} dangerouslySetInnerHTML={{ __html: markdownToHtml(quickOverviewMd) }} />
      </div>

      <div style={styles.summaryActions}>
        <button type="button" style={styles.togglePrimary} onClick={() => setShowFullReport((s) => !s)}>{showFullReport ? 'Hide full report' : 'Show full report'}</button>
      </div>

      {showFullReport ? (
        <div>
          <section style={styles.reportSection}>
            <h3 style={styles.reportTitle}>Dataset understanding</h3>
            <div dangerouslySetInnerHTML={{ __html: markdownToHtml(cleanGeneratedText(analysis.analysis.understanding)) }} />
          </section>
          <section style={styles.reportSection}>
            <h3 style={styles.reportTitle}>Problem type</h3>
            <div dangerouslySetInnerHTML={{ __html: markdownToHtml(cleanGeneratedText(analysis.analysis.problem_type)) }} />
          </section>
          <section style={styles.reportSection}>
            <h3 style={styles.reportTitle}>Feature engineering</h3>
            <div dangerouslySetInnerHTML={{ __html: markdownToHtml(cleanGeneratedText(analysis.analysis.feature_engineering)) }} />
          </section>
          <section style={styles.reportSection}>
            <h3 style={styles.reportTitle}>Preprocessing</h3>
            <div dangerouslySetInnerHTML={{ __html: markdownToHtml(cleanGeneratedText(analysis.analysis.preprocessing)) }} />
          </section>
          <section style={styles.reportSection}>
            <h3 style={styles.reportTitle}>Model recommendations</h3>
            <div dangerouslySetInnerHTML={{ __html: markdownToHtml(cleanGeneratedText(analysis.analysis.model_recommendations)) }} />
          </section>
          <section style={styles.reportSection}>
            <h3 style={styles.reportTitle}>Priority recommendations</h3>
            <div dangerouslySetInnerHTML={{ __html: markdownToHtml([
              `- **Primary problem type:** ${analysis.analysis.recommendations.primary_problem_type || 'Not identified'}`,
              `- **Suggested models:** ${(analysis.analysis.recommendations.suggested_models || []).join(', ') || 'None'}`,
              `- **Data quality issues:** ${(analysis.analysis.recommendations.data_quality_issues || []).join('; ') || 'None'}`,
            ].join('\n')) }} />
          </section>
        </div>
      ) : null}

      <div style={{ marginTop: 10 }}>
        <button type="button" style={styles.secondaryButton} onClick={handleDownloadReport}>Download report</button>
      </div>
    </div>
  )

  const renderVisualizationsTab = () => {
    return (
      <div style={styles.tabPane}>
        {vizCharts.length ? vizCharts.map((chart, idx) => (
          <section key={`${chart.title}-${idx}`} style={styles.reportSection}>
            <h3 style={styles.reportTitle}>{chart.title}</h3>
            <p style={styles.muted}>{chart.description}</p>
            <Plot
              data={chart.data}
              layout={chart.layout}
              style={{ width: '100%', height: '380px' }}
              config={{ responsive: true, displayModeBar: true, scrollZoom: true }}
            />
          </section>
        )) : (
          <section style={styles.reportSection}>
            <h3 style={styles.reportTitle}>No relevant visualizations available</h3>
            <p style={styles.reportText}>
              Upload a dataset with at least one numeric, categorical, or datetime column to generate interactive charts.
            </p>
          </section>
        )}
      </div>
    )
  }

  const renderAssistantTab = () => (
    <div style={styles.tabPane}>
      <section style={styles.reportSection}>
        <h3 style={styles.reportTitle}>Interactive AI assistant</h3>
        <p style={styles.muted}>Ask for additional experiments, checks, and model strategies.</p>
        <div style={styles.chatLog}>
          {conversation.length === 0 ? <p style={styles.muted}>No messages yet.</p> : null}
          {conversation.map((msg, idx) => (
            <div key={`${msg.role}-${idx}`} style={msg.role === 'user' ? styles.userBubble : styles.assistantBubble}>
              {msg.role === 'assistant' ? (
                <div
                  style={styles.assistantRichText}
                  dangerouslySetInnerHTML={{ __html: markdownToHtml(cleanGeneratedText(msg.content)) }}
                />
              ) : (
                msg.content
              )}
            </div>
          ))}
          {assistantThinking ? (
            <div style={styles.assistantBubble}>
              <div style={styles.thinkingRow}>
                <span style={styles.thinkingDot} />
                <span style={{ ...styles.thinkingDot, animationDelay: '120ms' }} />
                <span style={{ ...styles.thinkingDot, animationDelay: '240ms' }} />
              </div>
              <p style={styles.thinkingLabel}>NexusAI is thinking{'.'.repeat(thinkingDotCount)}</p>
            </div>
          ) : null}
        </div>
        <div style={styles.chatInputRow}>
          <input
            style={styles.input}
            type="text"
            value={userQuestion}
            placeholder="Ask follow-up questions about your data"
            onChange={(event) => setUserQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') handleAskAssistant()
            }}
          />
          <button type="button" style={styles.primaryButtonCompact} disabled={loading || assistantThinking} onClick={handleAskAssistant}>
            {assistantThinking ? 'Thinking...' : 'Send'}
          </button>
        </div>
      </section>
    </div>
  )

  const renderResults = () => (
    <section style={{ ...styles.panel, ...styles.resultsPanel }}>
      <h2 style={styles.sectionHeading}>Data Engine Results</h2>
      <div style={styles.tabHeader}>
        <button type="button" style={resultTab === 'analysis' ? styles.activeTabButton : styles.tabButton} onClick={() => setResultTab('analysis')}>Analysis</button>
        <button type="button" style={resultTab === 'viz' ? styles.activeTabButton : styles.tabButton} onClick={() => setResultTab('viz')}>Visualizations</button>
        <button type="button" style={resultTab === 'assistant' ? styles.activeTabButton : styles.tabButton} onClick={() => setResultTab('assistant')}>Assistant</button>
      </div>
      <div style={styles.tabBody}>
        {resultTab === 'analysis' ? renderAnalysisTab() : null}
        {resultTab === 'viz' ? renderVisualizationsTab() : null}
        {resultTab === 'assistant' ? renderAssistantTab() : null}
      </div>
      <div style={{ ...styles.buttonRow, marginTop: '20px', borderTop: '1px solid rgba(148, 163, 184, 0.12)', paddingTop: '20px' }}>
        <button type="button" style={styles.secondaryButton} onClick={handleStartOver}>Start over</button>
      </div>
    </section>
  )

  // inject minimal report CSS once
  useMemo(() => {
    try {
      if (!document.getElementById('de-report-styles')) {
        const s = document.createElement('style')
        s.id = 'de-report-styles'
        s.innerHTML = `
          .de-table{width:100%;border-collapse:collapse;margin:8px 0}
          .de-table th,.de-table td{padding:8px;border:1px solid rgba(148,163,184,0.12);text-align:left}
          .de-code{background:#0f172a;color:#fff;padding:12px;border-radius:8px;overflow:auto}
          .de-code code{white-space:pre-wrap}
          blockquote{border-left:4px solid rgba(82,39,255,0.18);padding:8px 12px;background:rgba(82,39,255,0.04)}
          @keyframes deThinkingPulse{0%,80%,100%{opacity:.28;transform:scale(.85)}40%{opacity:1;transform:scale(1)}}
        `
        document.head.appendChild(s)
      }
    } catch (e) {
      // ignore
    }
    return null
  }, [])

  return (
    <main style={styles.page}>
      <div style={styles.shell}>
        <header style={styles.hero}>
          <div style={styles.heroCopy}>
            <p style={styles.kicker}>DeepNexus Data Engine</p>
            <h1 style={styles.title}>CSV workflows with target confirmation, feature review, and AI-driven analysis.</h1>
            <p style={styles.subtitle}>
              Keep the flow structured: upload a CSV, verify the suggested target, inspect detected features, then run
              analysis and assistant follow-ups without leaving the page.
            </p>
            <div style={styles.heroPills}>
              <span style={styles.pill}>CSV only</span>
              <span style={styles.pill}>Target confirmation</span>
              <span style={styles.pill}>AI recommendations</span>
              <span style={styles.pill}>Downloadable report</span>
            </div>
          </div>

          <div style={styles.heroActions}>
            <div style={styles.heroStats}>
              <div style={styles.statBox}>
                <span style={styles.statLabel}>Current stage</span>
                <strong style={styles.statValue}>{step}</strong>
              </div>
              <div style={styles.statBox}>
                <span style={styles.statLabel}>Rows previewed</span>
                <strong style={styles.statValue}>{preview?.rows || 0}</strong>
              </div>
              <div style={styles.statBox}>
                <span style={styles.statLabel}>Detected features</span>
                <strong style={styles.statValue}>{features ? (features.numerical_features?.length || 0) + (features.categorical_features?.length || 0) + (features.datetime_features?.length || 0) : 0}</strong>
              </div>
            </div>
          </div>
        </header>

        <section style={styles.workflowRail}>
          <div style={styles.workflowStep}><span style={styles.workflowIndex}>01</span><span>Upload CSV</span></div>
          <div style={styles.workflowStep}><span style={styles.workflowIndex}>02</span><span>Confirm target</span></div>
          <div style={styles.workflowStep}><span style={styles.workflowIndex}>03</span><span>Review features</span></div>
          <div style={styles.workflowStep}><span style={styles.workflowIndex}>04</span><span>Run analysis</span></div>
        </section>

        {step === 'upload' ? renderUpload() : null}
        {step === 'preview' ? renderPreview() : null}
        {step === 'confirm' ? renderFeatureConfirmation() : null}
        {step === 'analyzing' ? (
          <section style={styles.panel}>
            <h2 style={styles.sectionHeading}>Running predefined tasks</h2>
            <p style={styles.muted}>Preparing understanding, problem type, feature engineering, preprocessing, and model recommendations.</p>
            <div style={{ display: 'flex', justifyContent: 'center', minHeight: '200px', alignItems: 'center' }}>
              <NeuralOrbitLoader />
            </div>
          </section>
        ) : null}
        {step === 'results' && analysis ? renderResults() : null}

        {error ? <p style={styles.error}>{error}</p> : null}
      </div>
    </main>
  )
}

const styles = {
  page: {
    minHeight: '100vh',
    padding: '24px',
    background:
      'radial-gradient(circle at 10% 0%, rgba(82, 39, 255, 0.22), transparent 34%), radial-gradient(circle at 90% 10%, rgba(14, 165, 233, 0.18), transparent 28%), linear-gradient(180deg, #eef4ff 0%, #e8eefc 100%)',
    color: '#1f2937',
  },
  shell: {
    maxWidth: '1080px',
    margin: '0 auto',
  },
  hero: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1.5fr) minmax(280px, 0.8fr)',
    gap: '18px',
    alignItems: 'stretch',
    marginBottom: '18px',
  },
  heroCopy: {
    padding: '10px 0 0 0',
  },
  kicker: {
    margin: 0,
    color: '#4f46e5',
    fontSize: '0.82rem',
    fontWeight: 800,
    letterSpacing: '0.16em',
    textTransform: 'uppercase',
  },
  title: {
    margin: '10px 0 12px 0',
    fontSize: 'clamp(2.1rem, 4vw, 3.6rem)',
    lineHeight: 1.05,
    color: '#172033',
  },
  subtitle: {
    margin: 0,
    color: '#475569',
    lineHeight: 1.7,
    maxWidth: '68ch',
  },
  heroPills: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '10px',
    marginTop: '18px',
  },
  pill: {
    border: '1px solid rgba(82, 39, 255, 0.16)',
    background: 'rgba(255,255,255,0.72)',
    color: '#4338ca',
    borderRadius: '999px',
    padding: '8px 12px',
    fontSize: '0.86rem',
    fontWeight: 700,
  },
  heroActions: {
    display: 'grid',
    gap: '12px',
    alignContent: 'start',
  },
  heroStats: {
    display: 'grid',
    gap: '10px',
  },
  statBox: {
    padding: '14px 16px',
    borderRadius: '18px',
    background: 'rgba(255,255,255,0.56)',
    border: '1px solid rgba(148, 163, 184, 0.18)',
    boxShadow: 'none',
  },
  statLabel: {
    display: 'block',
    color: '#64748b',
    fontSize: '0.78rem',
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
    marginBottom: '4px',
  },
  statValue: {
    fontSize: '1.15rem',
    color: '#111827',
  },
  workflowRail: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, minmax(0, 1fr))',
    gap: '12px',
    marginBottom: '18px',
  },
  workflowStep: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    padding: '14px 16px',
    borderRadius: '18px',
    background: 'rgba(255,255,255,0.55)',
    border: '1px solid rgba(148, 163, 184, 0.16)',
    color: '#24324b',
    boxShadow: 'none',
  },
  workflowIndex: {
    display: 'inline-grid',
    placeItems: 'center',
    width: '32px',
    height: '32px',
    borderRadius: '999px',
    color: '#fff',
    background: 'linear-gradient(135deg, #5227ff 0%, #0ea5e9 100%)',
    fontSize: '0.78rem',
    fontWeight: 800,
  },
  panel: {
    margin: '0 0 18px 0',
    padding: '18px 0 0 0',
    borderTop: '1px solid rgba(148, 163, 184, 0.2)',
    background: 'transparent',
    backdropFilter: 'none',
    borderRadius: 0,
    boxShadow: 'none',
    animation: 'fadeIn 320ms ease-out',
  },
  sectionHeading: {
    marginTop: 0,
    color: '#172033',
  },
  muted: {
    color: '#64748b',
    fontSize: '14px',
  },
  dropZone: {
    marginTop: '14px',
    border: '1px dashed rgba(82, 39, 255, 0.3)',
    borderRadius: '16px',
    padding: '36px 24px',
    cursor: 'pointer',
    background: 'rgba(255,255,255,0.55)',
  },
  dropTitle: {
    margin: 0,
    fontSize: '18px',
    color: '#0f172a',
  },
  dropSubtitle: {
    marginTop: '6px',
    marginBottom: 0,
    color: '#64748b',
  },
  fileLabel: {
    marginTop: '12px',
    color: '#059669',
  },
  previewTableShell: {
    marginTop: '14px',
    maxHeight: '260px',
    overflow: 'auto',
    border: '1px solid rgba(148, 163, 184, 0.25)',
    borderRadius: '10px',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: '12px',
  },
  th: {
    textAlign: 'left',
    padding: '10px',
    background: 'rgba(82, 39, 255, 0.08)',
    color: '#1e293b',
    borderBottom: '1px solid rgba(148, 163, 184, 0.22)',
    position: 'sticky',
    top: 0,
  },
  td: {
    padding: '10px',
    borderBottom: '1px solid rgba(148, 163, 184, 0.12)',
    color: '#334155',
  },
  targetRow: {
    marginTop: '16px',
  },
  label: {
    display: 'block',
    marginBottom: '8px',
  },
  select: {
    width: '100%',
    padding: '10px 12px',
    borderRadius: '8px',
    border: '1px solid rgba(148, 163, 184, 0.35)',
    background: '#ffffff',
    color: '#111827',
  },
  suggested: {
    marginTop: '8px',
    color: '#4338ca',
  },
  primaryButton: {
    marginTop: '16px',
    padding: '11px 18px',
    border: 0,
    borderRadius: '8px',
    background: 'linear-gradient(100deg, #5227ff, #0ea5e9)',
    color: 'white',
    fontWeight: 700,
    cursor: 'pointer',
  },
  secondaryButton: {
    marginTop: '14px',
    padding: '10px 16px',
    borderRadius: '8px',
    border: '1px solid rgba(148, 163, 184, 0.35)',
    background: 'transparent',
    color: '#1e293b',
    cursor: 'pointer',
  },
  buttonRow: {
    display: 'flex',
    gap: '10px',
    flexWrap: 'wrap',
  },
  primaryButtonCompact: {
    padding: '10px 14px',
    border: 0,
    borderRadius: '8px',
    background: 'linear-gradient(100deg, #5227ff, #0ea5e9)',
    color: '#fff',
    fontWeight: 700,
    cursor: 'pointer',
  },
  featureColumns: {
    display: 'grid',
    gap: '14px',
    gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
  },
  featureColumn: {
    borderLeft: '3px solid #5227ff',
    paddingLeft: '10px',
  },
  featureTitle: {
    margin: '0 0 8px 0',
  },
  featureItem: {
    marginBottom: '6px',
    color: '#334155',
  },
  tabHeader: {
    display: 'flex',
    gap: '10px',
    borderBottom: '1px solid rgba(148, 163, 184, 0.2)',
    paddingBottom: '10px',
    flexWrap: 'wrap',
  },
  tabButton: {
    flex: '1 1 140px',
    padding: '8px 14px',
    borderRadius: '999px',
    border: '1px solid rgba(82, 39, 255, 0.18)',
    background: 'rgba(255,255,255,0.76)',
    color: '#334155',
    cursor: 'pointer',
    whiteSpace: 'nowrap',
  },
  activeTabButton: {
    flex: '1 1 140px',
    padding: '8px 14px',
    borderRadius: '999px',
    border: '1px solid #5227ff',
    background: 'linear-gradient(135deg, rgba(82, 39, 255, 0.14), rgba(14, 165, 233, 0.12))',
    color: '#111827',
    cursor: 'pointer',
    whiteSpace: 'nowrap',
  },
  tabBody: {
    paddingTop: '12px',
  },
  tabPane: {
    animation: 'fadeIn 220ms ease-out',
  },
  reportSection: {
    borderLeft: '2px solid rgba(82, 39, 255, 0.45)',
    paddingLeft: '12px',
    marginBottom: '18px',
  },
  resultsPanel: {
    margin: '0 0 18px 0',
    padding: '18px',
    borderRadius: '12px',
    background: '#ffffff',
    border: '1px solid rgba(148, 163, 184, 0.12)',
    boxShadow: '0 6px 18px rgba(15, 23, 42, 0.06)'
  },
  resultsCard: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    padding: '12px',
    borderRadius: '10px',
    background: 'linear-gradient(180deg, #fafbfd, #ffffff)',
    border: '1px solid rgba(148, 163, 184, 0.06)',
  },
  summaryText: {
    color: '#0f172a',
    fontSize: '14px',
    lineHeight: 1.4,
  },
  toggleButton: {
    padding: '8px 12px',
    borderRadius: '8px',
    border: '1px solid rgba(82,39,255,0.12)',
    background: 'transparent',
    color: '#4338ca',
    cursor: 'pointer'
  },
  summaryActions: {
    display: 'flex',
    justifyContent: 'center',
    marginTop: '10px'
  },
  togglePrimary: {
    padding: '10px 16px',
    borderRadius: '10px',
    border: '0',
    background: 'linear-gradient(90deg,#5227ff,#0ea5e9)',
    color: '#fff',
    fontWeight: 700,
    cursor: 'pointer'
  },
  reportTitle: {
    margin: '0 0 6px 0',
    color: '#111827',
  },
  reportText: {
    margin: 0,
    whiteSpace: 'pre-wrap',
    lineHeight: 1.6,
    color: '#1f2937',
  },
  issueList: {
    marginTop: '8px',
    color: '#1f2937',
    paddingLeft: '18px',
  },
  barChart: {
    display: 'grid',
    gap: '8px',
  },
  barRow: {
    display: 'grid',
    gridTemplateColumns: '180px 1fr 40px',
    gap: '10px',
    alignItems: 'center',
  },
  barLabel: {
    color: '#334155',
    fontSize: '12px',
  },
  barTrack: {
    height: '10px',
    borderRadius: '999px',
    background: 'rgba(82, 39, 255, 0.09)',
    overflow: 'hidden',
  },
  barFill: {
    height: '100%',
    background: 'linear-gradient(90deg, #5227ff, #0ea5e9)',
    transition: 'width 420ms ease',
  },
  barCount: {
    textAlign: 'right',
    color: '#4f46e5',
    fontSize: '12px',
  },
  sparkSvg: {
    width: '100%',
    maxWidth: '700px',
    background: 'rgba(255,255,255,0.82)',
    borderRadius: '10px',
    border: '1px solid rgba(148, 163, 184, 0.25)',
  },
  chatLog: {
    maxHeight: '300px',
    overflowY: 'auto',
    border: '1px solid rgba(148, 163, 184, 0.25)',
    borderRadius: '10px',
    padding: '10px',
    background: 'rgba(255, 255, 255, 0.82)',
    marginBottom: '10px',
  },
  userBubble: {
    background: 'rgba(82, 39, 255, 0.1)',
    borderLeft: '3px solid #5227ff',
    padding: '8px 10px',
    marginBottom: '8px',
    borderRadius: '6px',
  },
  assistantBubble: {
    background: 'rgba(14, 165, 233, 0.1)',
    borderLeft: '3px solid #0ea5e9',
    padding: '8px 10px',
    marginBottom: '8px',
    borderRadius: '6px',
  },
  assistantRichText: {
    lineHeight: 1.55,
    color: '#0f172a',
  },
  thinkingRow: {
    display: 'flex',
    gap: '6px',
    alignItems: 'center',
    marginBottom: '6px',
  },
  thinkingDot: {
    width: '8px',
    height: '8px',
    borderRadius: '999px',
    background: '#0ea5e9',
    animation: 'deThinkingPulse 1s infinite ease-in-out',
  },
  thinkingLabel: {
    margin: 0,
    color: '#0f172a',
    fontSize: '13px',
    fontWeight: 600,
  },
  chatInputRow: {
    display: 'flex',
    gap: '8px',
  },
  input: {
    flex: 1,
    padding: '10px 12px',
    border: '1px solid rgba(148, 163, 184, 0.35)',
    borderRadius: '8px',
    background: '#ffffff',
    color: '#111827',
  },
  error: {
    maxWidth: '1080px',
    margin: '14px auto 0 auto',
    padding: '10px 12px',
    borderRadius: '8px',
    color: '#991b1b',
    background: 'rgba(239, 68, 68, 0.12)',
    border: '1px solid rgba(239, 68, 68, 0.35)',
  },
  loadingContainer: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: '20px',
    minHeight: '100px',
  },
}

export default DataEnginePage
