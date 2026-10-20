import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import CompareSlider from './components/CompareSlider.jsx'

const WsiViewer = lazy(() => import('./components/WsiViewer.jsx'))

const API_BASE = import.meta.env.VITE_API_BASE || ''
const WSI_RE = /\.(svs|ome\.tiff?|tiff?)$/i

export default function App() {
  const [file, setFile] = useState(null)
  const [direction, setDirection] = useState('he2dapi')
  const [originalUrl, setOriginalUrl] = useState(null)
  const [resultUrl, setResultUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [viewerOpen, setViewerOpen] = useState(false)
  const [health, setHealth] = useState(null)
  const urls = useRef([])

  const revokeAll = useCallback(() => {
    urls.current.forEach((u) => URL.revokeObjectURL(u))
    urls.current = []
  }, [])

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setHealth)
      .catch(() => setHealth(null))
    return revokeAll
  }, [revokeAll])

  const onDrop = useCallback(
    (accepted) => {
      const f = accepted[0]
      if (!f) return
      revokeAll()
      setFile(f)
      setResultUrl(null)
      setError(null)
      const src = URL.createObjectURL(f)
      urls.current.push(src)
      setOriginalUrl(src)
    },
    [revokeAll],
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: false,
    maxSize: 500 * 1024 * 1024,
    accept: {
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/tiff': ['.tif', '.tiff', '.ome.tif', '.ome.tiff'],
      'application/octet-stream': ['.svs'],
    },
  })

  const translate = async () => {
    if (!file || loading) return
    setLoading(true)
    setError(null)
    const endpoint = WSI_RE.test(file.name) ? '/translate/wsi' : '/translate'
    const body = new FormData()
    body.append('file', file)
    try {
      const res = await fetch(`${API_BASE}${endpoint}?direction=${encodeURIComponent(direction)}`, {
        method: 'POST',
        body,
      })
      if (!res.ok) {
        let detail = `${res.status} ${res.statusText}`
        try {
          detail = (await res.json()).detail || detail
        } catch {
          /* non-JSON error */
        }
        throw new Error(detail)
      }
      const blob = await res.blob()
      const out = URL.createObjectURL(blob)
      urls.current.push(out)
      setResultUrl(out)
    } catch (e) {
      setError(e.message || 'Translation failed')
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    revokeAll()
    setFile(null)
    setOriginalUrl(null)
    setResultUrl(null)
    setError(null)
  }

  return (
    <div className="app">
      <header>
        <div className="brand">
          <span className="dot" />
          <h1>H&amp;E / DAPI Virtual Staining</h1>
        </div>
        <span className="badge">research demo · not for diagnosis</span>
      </header>

      <main>
        <section className="panel">
          <div className="controls">
            <div className="segmented" role="group" aria-label="Translation direction">
              <button
                className={direction === 'he2dapi' ? 'active' : ''}
                onClick={() => setDirection('he2dapi')}
              >
                H&amp;E → DAPI
              </button>
              <button
                className={direction === 'dapi2he' ? 'active' : ''}
                onClick={() => setDirection('dapi2he')}
              >
                DAPI → H&amp;E
              </button>
            </div>
            <span className="hint">
              patch: PNG/JPG/TIFF · whole slide: SVS/OME-TIFF · max 500 MB upload
            </span>
          </div>

          <div
            {...getRootProps()}
            className={`dropzone${isDragActive ? ' drag' : ''}${file ? ' has-file' : ''}`}
          >
            <input {...getInputProps()} />
            {file ? (
              <div className="file-info">
                <strong>{file.name}</strong>
                <span>{(file.size / 1024 / 1024).toFixed(1)} MB</span>
              </div>
            ) : (
              <div className="file-info">
                <strong>Drop an image here</strong>
                <span>or click to choose a file</span>
              </div>
            )}
          </div>

          <div className="actions">
            <button className="primary" onClick={translate} disabled={!file || loading}>
              {loading ? 'Translating…' : 'Translate'}
            </button>
            <button onClick={reset} disabled={!file || loading}>
              Clear
            </button>
            {resultUrl && (
              <>
                <a className="button" href={resultUrl} download={`synthetic_${direction}.png`}>
                  Download PNG
                </a>
                <button onClick={() => setViewerOpen(true)}>Open WSI viewer</button>
              </>
            )}
          </div>

          {error && <p className="error">{error}</p>}
        </section>

        {resultUrl && (
          <section className="panel result">
            <div className="result-head">
              <h2>Result</h2>
              <span className="badge warn">SYNTHETIC</span>
            </div>
            <CompareSlider before={originalUrl} after={resultUrl} />
            <p className="disclaimer">
              Predicted {direction === 'he2dapi' ? 'DAPI' : 'H&amp;E'} is synthetic and must not be
              used for clinical diagnosis.
            </p>
          </section>
        )}

        <section className="panel status">
          <h2>System</h2>
          <p>
            {health
              ? `backend online · ${(health.models || []).join(', ')}`
              : 'backend offline (start it with `uvicorn src.backend.main:app --port 8000`)'}
          </p>
        </section>
      </main>

      <footer>H&amp;E ↔ DAPI · Pix2Pix + CycleGAN · CPU-friendly tiled inference</footer>

      {viewerOpen && resultUrl && (
        <Suspense fallback={null}>
          <WsiViewer src={resultUrl} onClose={() => setViewerOpen(false)} />
        </Suspense>
      )}
    </div>
  )
}
