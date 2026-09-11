import { useEffect, useRef, useState } from 'react'
import Plotly from 'plotly.js-basic-dist-min'
import createPlotlyComponent from 'react-plotly.js/factory'
import { api } from './api'
import type { Example, Job, Koi, ModelCards, VetResult } from './api'
import './index.css'

const Plot = createPlotlyComponent(Plotly)

const STAGE_LABELS: Record<string, string> = {
  ephemeris: 'Look up period and epoch',
  fetch: 'Fetch Kepler light curve',
  fold: 'Detrend, fold and bin',
  classify: 'Run CNN and baseline',
}

type Page = 'vet' | 'model'

export default function App() {
  const [page, setPage] = useState<Page>('vet')
  return (
    <div className="app">
      <header>
        <h1>TransitHunter</h1>
        <p>Is this dip in a Kepler star's brightness a planet or a false positive?</p>
        <nav>
          <button className={page === 'vet' ? 'active' : ''} onClick={() => setPage('vet')}>Vet a star</button>
          <button className={page === 'model' ? 'active' : ''} onClick={() => setPage('model')}>Model card</button>
        </nav>
      </header>
      {page === 'vet' ? <VetPage /> : <ModelPage />}
      <footer>
        Data: NASA Exoplanet Archive KOI table and Kepler light curves from MAST. Labels shown as
        "catalogue" are NASA's disposition and are never a model input.
      </footer>
    </div>
  )
}

function VetPage() {
  const [kepidText, setKepidText] = useState('11904151')
  const [examples, setExamples] = useState<Example[]>([])
  const [kois, setKois] = useState<Koi[]>([])
  const [koi, setKoi] = useState<string>('')
  const [job, setJob] = useState<Job | null>(null)
  const [error, setError] = useState<string | null>(null)
  const timer = useRef<number | null>(null)

  useEffect(() => {
    api.examples().then(setExamples).catch(() => undefined)
  }, [])

  const kepid = Number.parseInt(kepidText, 10)

  useEffect(() => {
    if (!Number.isFinite(kepid)) return
    api.kois(kepid).then((rows) => {
      setKois(rows)
      setKoi(rows[0]?.kepoi_name ?? '')
    }).catch(() => setKois([]))
  }, [kepid])

  const stopPolling = () => {
    if (timer.current !== null) window.clearInterval(timer.current)
    timer.current = null
  }

  const run = async () => {
    if (!Number.isFinite(kepid)) return
    setError(null)
    setJob(null)
    stopPolling()
    try {
      const started = await api.startVet(kepid, koi || undefined)
      const poll = async () => {
        const j = await api.job(started.job_id)
        setJob(j)
        if (j.stage === 'done' || j.stage === 'error') stopPolling()
      }
      await poll()
      timer.current = window.setInterval(poll, 800)
    } catch (e) {
      setError(String(e))
    }
  }

  useEffect(() => stopPolling, [])

  // A verdict belongs to the star it was run for. Hide it once the selection
  // moves on, rather than leaving the last result under a new KIC ID. Derived
  // rather than cleared on change: the candidate list arrives asynchronously,
  // so an effect on [kepid, koi] would wipe a job started before it landed.
  const stale = job !== null && (job.kepid !== kepid || (job.kepoi_name ?? '') !== koi)

  return (
    <main>
      <section className="controls">
        <label>
          KIC ID
          <input value={kepidText} onChange={(e) => setKepidText(e.target.value.trim())} inputMode="numeric" />
        </label>
        {kois.length > 0 && (
          <label>
            Candidate on this star
            <select value={koi} onChange={(e) => setKoi(e.target.value)}>
              {kois.map((k) => (
                <option key={k.kepoi_name} value={k.kepoi_name}>
                  {k.kepoi_name}{k.kepler_name ? ` (${k.kepler_name})` : ''}, {k.koi_period.toFixed(2)} d
                </option>
              ))}
            </select>
          </label>
        )}
        {kois.length === 0 && Number.isFinite(kepid) && (
          <span className="hint">No catalogue entry: a Box Least Squares search will find the period.</span>
        )}
        <button className="primary" onClick={run} disabled={!Number.isFinite(kepid) || (job !== null && job.stage !== 'done' && job.stage !== 'error')}>
          Vet
        </button>
      </section>
      <section className="examples">
        {examples.map((ex) => (
          <button key={ex.kepid} onClick={() => setKepidText(String(ex.kepid))} title={ex.note}>
            {ex.name}
          </button>
        ))}
      </section>
      {error && <p className="error">{error}</p>}
      {job && !stale && <Stages job={job} />}
      {job?.error && !stale && <p className="error">{job.error}</p>}
      {job?.result && !stale && <Result r={job.result} />}
    </main>
  )
}

function Stages({ job }: { job: Job }) {
  const idx = job.stages.indexOf(job.stage)
  const finished = job.stage === 'done'
  return (
    <ol className="stages">
      {job.stages.map((s, i) => {
        const state = finished || i < idx ? 'done' : i === idx ? 'active' : 'pending'
        return (
          <li key={s} className={state}>
            <span className="dot" />
            {STAGE_LABELS[s] ?? s}
          </li>
        )
      })}
    </ol>
  )
}

function Result({ r }: { r: VetResult }) {
  const planet = r.verdict === 'PLANET'
  const agree = r.catalogue_disposition
    ? (r.catalogue_disposition === 'CONFIRMED') === planet
    : null
  const totalS = Object.values(r.seconds).reduce((a, b) => a + b, 0)
  return (
    <section className="result">
      <div className="verdict-row">
        <div className={`verdict ${planet ? 'planet' : 'fp'}`}>
          <span className="label">CNN verdict</span>
          <strong>{r.verdict}</strong>
          <Gauge value={r.probability} threshold={r.threshold} />
          <span className="small">P(planet) {r.probability.toFixed(3)}, threshold {r.threshold.toFixed(2)}</span>
        </div>
        <dl className="facts">
          <dt>Candidate</dt>
          <dd>{r.ephemeris.kepoi_name ?? 'found by BLS'} on KIC {r.kepid}</dd>
          <dt>Period</dt>
          <dd>{r.ephemeris.period.toFixed(4)} days</dd>
          <dt>Duration</dt>
          <dd>{r.ephemeris.duration_hours.toFixed(2)} hours</dd>
          <dt>Depth</dt>
          <dd>{r.ephemeris.depth_ppm != null ? `${Math.round(r.ephemeris.depth_ppm)} ppm` : 'n/a'}</dd>
          <dt>Transits seen</dt>
          <dd>{r.n_transits_seen} in {r.n_points.toLocaleString()} cadences</dd>
          <dt>Baseline GBM</dt>
          <dd>{r.baseline_probability != null ? `P(planet) ${r.baseline_probability.toFixed(3)}` : 'not run (no catalogue features)'}</dd>
          <dt>NASA catalogue</dt>
          <dd>
            {r.catalogue_disposition ?? 'no entry'}
            {agree !== null && <span className={agree ? 'ok' : 'bad'}> {agree ? 'agrees' : 'disagrees'}</span>}
          </dd>
          <dt>Time</dt>
          <dd>{totalS.toFixed(1)} s{r.cached_light_curve ? ' (light curve cached)' : ' (fetched from MAST)'}</dd>
        </dl>
      </div>
      <div className="plots">
        <FoldedPlot title="Global view: whole orbit" y={r.global_view} xlo={-0.5} xhi={0.5} />
        <FoldedPlot title="Local view: zoom on the transit" y={r.local_view} xlo={-2} xhi={2} xlabel="transit durations from centre" />
      </div>
    </section>
  )
}

function Gauge({ value, threshold }: { value: number; threshold: number }) {
  return (
    <div className="gauge" aria-label={`probability ${value.toFixed(3)}`}>
      <div className="fill" style={{ width: `${Math.round(value * 100)}%` }} />
      <div className="mark" style={{ left: `${Math.round(threshold * 100)}%` }} />
    </div>
  )
}

function FoldedPlot({ title, y, xlo, xhi, xlabel = 'orbital phase' }: {
  title: string; y: number[]; xlo: number; xhi: number; xlabel?: string
}) {
  const x = y.map((_, i) => xlo + ((xhi - xlo) * (i + 0.5)) / y.length)
  return (
    <Plot
      data={[{ x, y, type: 'scatter', mode: 'lines', line: { width: 1.2, color: '#2563eb' } }]}
      layout={{
        title: { text: title, font: { size: 14 } },
        margin: { l: 50, r: 10, t: 40, b: 40 },
        height: 300,
        xaxis: { title: { text: xlabel }, zeroline: false },
        yaxis: { title: { text: 'normalised flux' }, zeroline: false },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%' }}
      useResizeHandler
    />
  )
}

function ModelPage() {
  const [cards, setCards] = useState<ModelCards | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    api.models().then(setCards).catch((e) => setError(String(e)))
  }, [])
  if (error) return <main><p className="error">{error}</p></main>
  if (!cards) return <main><p>Loading model cards...</p></main>
  const rows: [string, (m: NonNullable<ModelCards['baseline']>['test']) => string][] = [
    ['PR-AUC', (m) => m.pr_auc.toFixed(3)],
    ['ROC-AUC', (m) => m.roc_auc.toFixed(3)],
    ['Precision', (m) => m.precision.toFixed(3)],
    ['Recall', (m) => m.recall.toFixed(3)],
    ['F1', (m) => m.f1.toFixed(3)],
    ['Accuracy', (m) => m.accuracy.toFixed(3)],
    ['Threshold', (m) => m.threshold.toFixed(3)],
    ['Confusion (tn, fp, fn, tp)', (m) => `${m.tn}, ${m.fp}, ${m.fn}, ${m.tp}`],
  ]
  return (
    <main className="model">
      <p>
        Held-out test set, split by star so no light curve appears on both sides. The threshold was
        chosen on the validation set to maximise F1 and then frozen. PR-AUC is the headline number
        because the classes are not balanced.
      </p>
      <table>
        <thead>
          <tr><th>Metric</th><th>Gradient boosting baseline</th><th>Two-branch CNN</th></tr>
        </thead>
        <tbody>
          {rows.map(([name, f]) => (
            <tr key={name}>
              <td>{name}</td>
              <td>{cards.baseline ? f(cards.baseline.test) : 'n/a'}</td>
              <td>{f(cards.cnn.test)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h3>Inputs</h3>
      <p>
        <strong>Baseline:</strong> {cards.baseline ? cards.baseline.features.join(', ') : 'n/a'}. Catalogue
        measurements only; vetting flags and scores are excluded by an allowlist enforced in tests.
      </p>
      <p>
        <strong>CNN:</strong> a 2001-bin global view of the folded orbit and a 201-bin local view of the
        transit, following Shallue and Vanderburg (2018). Channels {String(cards.cnn.config.global_channels)} global,{' '}
        {String(cards.cnn.config.local_channels)} local, hidden {String(cards.cnn.config.hidden)}, dropout{' '}
        {String(cards.cnn.config.dropout)}.
      </p>
      <p className="small">MLflow runs: CNN {cards.cnn.mlflow_run_id}{cards.baseline ? `, baseline ${cards.baseline.mlflow_run_id}` : ''}</p>
    </main>
  )
}
