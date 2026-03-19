import { useState, useCallback } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const EXAMPLES = [
  'free beginner crochet baby hat',
  'most popular Petite Knit sweaters',
  'advanced lace knitting shawl',
  'top rated free adult sock pattern in fingering weight',
  'bulky beginner knit sweater top down',
]

const PROVIDERS = [
  { id: 'anthropic', label: 'Claude (Anthropic)', placeholder: 'sk-ant-...', link: 'https://console.anthropic.com/settings/keys' },
  { id: 'openai',    label: 'GPT-4o (OpenAI)',    placeholder: 'sk-...',     link: 'https://platform.openai.com/api-keys' },
  { id: 'gemini',    label: 'Gemini (Google)',     placeholder: 'AIza...',    link: 'https://aistudio.google.com/app/apikey' },
]

const RESULT_OPTIONS = [5, 10, 15, 20, 25]

function DifficultyDots({ average }) {
  const level = !average ? 0 : average < 1.5 ? 1 : average < 2.5 ? 2 : average < 3.5 ? 3 : average < 4.5 ? 4 : 5
  const label = ['', 'beginner', 'easy', 'intermediate', 'advanced', 'expert'][level]
  return level > 0 ? (
    <div className="difficulty">
      <div className="dots">{[1,2,3,4,5].map(i => <span key={i} className={`dot ${i <= level ? 'filled' : ''}`} />)}</div>
      <span className="diff-label">{label}</span>
    </div>
  ) : null
}

function PatternCard({ pattern }) {
  const photo = pattern.photos?.[0]
  const imgSrc = photo?.medium2_url || photo?.medium_url || photo?.small_url || ''
  const designer = pattern.pattern_author?.name || pattern.designer?.name || ''
  const craft = pattern.craft?.name?.toLowerCase() || ''
  const rating = pattern.rating_average ? Number(pattern.rating_average).toFixed(1) : null
  const link = `https://www.ravelry.com/patterns/library/${pattern.permalink}`
  return (
    <a className="card" href={link} target="_blank" rel="noopener noreferrer">
      <div className="card-img-wrap">
        {imgSrc ? <img src={imgSrc} alt={pattern.name} loading="lazy" /> : <div className="img-placeholder">🧶</div>}
      </div>
      <div className="card-body">
        <div className="card-name">{pattern.name}</div>
        {designer && <div className="card-designer">by {designer}</div>}
        <div className="card-meta">
          {pattern.free && <span className="badge badge-free">free</span>}
          {craft && <span className="badge badge-craft">{craft}</span>}
          {rating && <span className="rating">★ {rating}</span>}
        </div>
        <DifficultyDots average={pattern.difficulty_average} />
      </div>
    </a>
  )
}

function RagPills({ params, ragContext }) {
  if (!params) return null

  const ragResolved = []
  const llmInferred = []

  // RAG-resolved entries
  if (ragContext?.designer) {
    ragResolved.push(`${ragContext.designer.display_name} → id:${ragContext.designer.designer_id}`)
  }
  if (ragContext?.categories?.length) {
    ragContext.categories.forEach(c => ragResolved.push(`pc: ${c.pc}`))
  }
  if (ragContext?.attributes?.length) {
    const pa = ragContext.attributes.map(a => a.pa).join('+')
    ragResolved.push(`pa: ${pa}`)
  }
  if (ragContext?.fit_params?.length) {
    const fit = ragContext.fit_params.map(f => f.api_value).join('+')
    ragResolved.push(`fit: ${fit}`)
  }
  if (ragContext?.fiber) {
    ragResolved.push(`fiber: ${ragContext.fiber.api_value}`)
  }
  if (ragContext?.needle_size) {
    ragResolved.push(`needle: US${ragContext.needle_size.us} / ${ragContext.needle_size.mm}mm`)
  }
  if (ragContext?.parameters?.sort) {
    ragResolved.push(`sort: ${ragContext.parameters.sort}`)
  }
  if (ragContext?.parameters?.availability) {
    ragResolved.push(`availability: ${ragContext.parameters.availability}`)
  }
  if (ragContext?.parameters?.difficulty) {
    ragResolved.push(`difficulty: ${ragContext.parameters.difficulty}`)
  }

  // LLM-inferred entries (params not covered by RAG)
  const ragPcValues = ragContext?.categories?.map(c => c.pc) || []
  const ragFitValues = ragContext?.fit_params?.map(f => f.api_value) || []
  const ragSortSet = !!ragContext?.parameters?.sort
  const ragAvailSet = !!ragContext?.parameters?.availability
  const ragDiffSet = !!ragContext?.parameters?.difficulty

  const llmFields = {
    query: params.query,
    craft: params.craft,
    weight: params.weight,
    ...(!ragSortSet && params.sort && params.sort !== 'best' ? { sort: params.sort } : {}),
    ...(!ragAvailSet && params.availability ? { availability: params.availability } : {}),
    ...(!ragDiffSet && params.difficulty ? { difficulty: params.difficulty } : {}),
    ...(!ragPcValues.length && params.pc ? { pc: params.pc } : {}),
    ...(!ragFitValues.length && params.fit ? { fit: params.fit } : {}),
    ...(params.pa && !ragContext?.attributes?.length ? { pa: params.pa } : {}),
    ...(params.colors ? { colors: params.colors } : {}),
    ...(params.ratings ? { ratings: params.ratings } : {}),
  }

  Object.entries(llmFields).forEach(([k, v]) => {
    if (v) llmInferred.push(`${k}: ${v}`)
  })

  if (!ragResolved.length && !llmInferred.length) return null

  return (
    <div className="param-pills-wrap">
      {ragResolved.length > 0 && (
        <div className="pill-row">
          {ragResolved.map((label, i) => (
            <span key={i} className="pill pill-rag" title="Resolved from RAG index">
              🗄 {label}
            </span>
          ))}
        </div>
      )}
      {llmInferred.length > 0 && (
        <div className="pill-row">
          {llmInferred.map((label, i) => (
            <span key={i} className="pill pill-llm">
              {label}
            </span>
          ))}
        </div>
      )}
      {ragResolved.length === 0 && llmInferred.length > 0 && (
        <span className="fully-inferred">fully LLM inferred</span>
      )}
    </div>
  )
}

function CredsBox({ title, saved, badge, onToggle, isOpen, children }) {
  return (
    <div className="creds-box">
      <button className="creds-toggle" onClick={onToggle}>
        <span>{title}</span>
        <span className={`cred-badge ${saved ? 'saved' : 'unsaved'}`}>{badge}</span>
      </button>
      {isOpen && <div className="creds-inner">{children}</div>}
    </div>
  )
}

export default function App() {
  const [ravUser, setRavUser] = useState('')
  const [ravPass, setRavPass] = useState('')
  const [ravSaved, setRavSaved] = useState(false)
  const [ravOpen, setRavOpen] = useState(true)

  const [provider, setProvider] = useState('anthropic')
  const [llmKey, setLlmKey] = useState('')
  const [llmSaved, setLlmSaved] = useState(false)
  const [llmOpen, setLlmOpen] = useState(true)

  const [query, setQuery] = useState('')
  const [pageSize, setPageSize] = useState(5)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [statusMsg, setStatusMsg] = useState('')
  const [results, setResults] = useState(null)
  const [parsedParams, setParsedParams] = useState(null)
  const [ragContext, setRagContext] = useState(null)

  const currentProvider = PROVIDERS.find(p => p.id === provider)

  const saveRavelry = () => {
    if (!ravUser || !ravPass) { setError('Enter both Ravelry username and password.'); return }
    setRavSaved(true); setRavOpen(false); setError('')
  }

  const saveLLM = () => {
    if (!llmKey) { setError('Enter an API key.'); return }
    setLlmSaved(true); setLlmOpen(false); setError('')
  }

  const handleSearch = useCallback(async () => {
    if (!query.trim()) return
    if (!ravSaved) { setError('Save your Ravelry credentials first.'); return }
    if (!llmSaved) { setError('Save your LLM API key first.'); return }

    setLoading(true); setError(''); setResults(null)
    setParsedParams(null); setRagContext(null)
    setStatusMsg(`Searching with ${currentProvider.label}...`)

    try {
      const res = await fetch(`${API_BASE}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          ravelry_username: ravUser,
          ravelry_password: ravPass,
          llm_provider: provider,
          llm_api_key: llmKey,
          page_size: pageSize,
        }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail || 'Search failed')
      }

      setStatusMsg('Loading patterns...')
      const data = await res.json()
      setParsedParams(data.params)
      setRagContext(data.rag_context)
      setResults(data.patterns)
      setStatusMsg(`${data.patterns.length} patterns found`)
    } catch (e) {
      setError(e.message); setStatusMsg('')
    } finally {
      setLoading(false)
    }
  }, [query, pageSize, ravUser, ravPass, ravSaved, provider, llmKey, llmSaved, currentProvider])

  return (
    <div className="app">
      <header className="header">
        <h1>Ravelry Pattern Search</h1>
        <p>Describe what you're looking for in plain English</p>
      </header>

      {/* Ravelry credentials */}
      <CredsBox title="Ravelry API credentials" saved={ravSaved} badge={ravSaved ? 'saved ✓' : 'not set'} isOpen={ravOpen} onToggle={() => setRavOpen(o => !o)}>
        <div className="creds-form">
          <input type="text" placeholder="API username" value={ravUser} onChange={e => setRavUser(e.target.value)} autoComplete="off" />
          <input type="password" placeholder="API password" value={ravPass} onChange={e => setRavPass(e.target.value)} />
          <a href="https://www.ravelry.com/pro/n-a-85/apps" target="_blank" rel="noopener noreferrer" className="get-key-link standalone">Get Ravelry API credentials →</a>
          <button className="btn-primary" onClick={saveRavelry}>Save</button>
        </div>
      </CredsBox>

      {/* LLM credentials */}
      <CredsBox title="AI provider" saved={llmSaved} badge={llmSaved ? `${currentProvider.label} ✓` : 'not set'} isOpen={llmOpen} onToggle={() => setLlmOpen(o => !o)}>
        <div className="creds-form llm-form">
          <div className="provider-tabs">
            {PROVIDERS.map(p => (
              <button key={p.id} className={`provider-tab ${provider === p.id ? 'active' : ''}`} onClick={() => { setProvider(p.id); setLlmSaved(false) }}>
                {p.label}
              </button>
            ))}
          </div>
          <div className="llm-key-row">
            <input type="password" placeholder={currentProvider.placeholder} value={llmKey} onChange={e => { setLlmKey(e.target.value); setLlmSaved(false) }} autoComplete="off" />
            <a href={currentProvider.link} target="_blank" rel="noopener noreferrer" className="get-key-link">Get key →</a>
          </div>
          <button className="btn-primary" onClick={saveLLM}>Save</button>
        </div>
      </CredsBox>

      {/* Search */}
      <div className="search-area">
        <div className="search-row">
          <input className="search-input" type="text" placeholder="e.g. most popular Petite Knit top-down sweaters" value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleSearch()} />
          <div className="results-select-wrap">
            <label className="results-label">Results</label>
            <select className="results-select" value={pageSize} onChange={e => setPageSize(Number(e.target.value))}>
              {RESULT_OPTIONS.map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </div>
          <button className="btn-search" onClick={handleSearch} disabled={loading}>{loading ? '...' : 'Search'}</button>
        </div>
        <div className="examples">
          {EXAMPLES.map(ex => <button key={ex} className="example-chip" onClick={() => setQuery(ex)}>{ex}</button>)}
        </div>
      </div>

      {error && <div className="error-msg">{error}</div>}
      <RagPills params={parsedParams} ragContext={ragContext} />
      {statusMsg && <div className="status-bar">{statusMsg}</div>}

      {results !== null && (
        results.length === 0
          ? <div className="empty">No patterns found. Try a different search.</div>
          : <div className="grid">{results.map(p => <PatternCard key={p.id} pattern={p} />)}</div>
      )}
      {results === null && !loading && <div className="empty">Enter a search above to discover patterns</div>}
    </div>
  )
}
