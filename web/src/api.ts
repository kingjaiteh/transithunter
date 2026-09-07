export interface Ephemeris {
  period: number
  t0: number
  duration_hours: number
  source: 'catalogue' | 'bls'
  kepoi_name: string | null
  depth_ppm: number | null
}

export interface VetResult {
  kepid: number
  ephemeris: Ephemeris
  verdict: 'PLANET' | 'FALSE POSITIVE'
  probability: number
  threshold: number
  baseline_probability: number | null
  catalogue_disposition: string | null
  global_view: number[]
  local_view: number[]
  n_points: number
  n_transits_seen: number
  seconds: Record<string, number>
  cached_light_curve: boolean
}

export interface Job {
  id: string
  kepid: number
  kepoi_name: string | null
  stage: string
  stages: string[]
  result: VetResult | null
  error: string | null
}

export interface Koi {
  kepoi_name: string
  kepler_name: string | null
  koi_disposition: string
  koi_period: number
  koi_depth: number
  koi_duration: number
}

export interface Example {
  kepid: number
  name: string
  note: string
}

export interface Metrics {
  pr_auc: number
  roc_auc: number
  threshold: number
  precision: number
  recall: number
  f1: number
  accuracy: number
  tn: number
  fp: number
  fn: number
  tp: number
}

export interface ModelCards {
  cnn: { config: Record<string, unknown>; threshold: number; test: Metrics; mlflow_run_id: string }
  baseline: { threshold: number; features: string[]; test: Metrics; mlflow_run_id: string } | null
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`/api${path}`)
  if (!r.ok) throw new Error(`${path}: ${r.status}`)
  return r.json()
}

export const api = {
  examples: () => get<Example[]>('/examples'),
  models: () => get<ModelCards>('/models'),
  kois: (kepid: number) => get<Koi[]>(`/kois/${kepid}`),
  job: (id: string) => get<Job>(`/jobs/${id}`),
  async startVet(kepid: number, koi?: string): Promise<{ job_id: string; stage: string; cached: boolean }> {
    const q = koi ? `?koi=${encodeURIComponent(koi)}` : ''
    const r = await fetch(`/api/vet/${kepid}${q}`, { method: 'POST' })
    if (!r.ok) throw new Error(`vet: ${r.status}`)
    return r.json()
  },
}
