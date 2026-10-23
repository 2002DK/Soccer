// Client for the FastAPI backend (backend/app/main.py)

const API_URL = (import.meta.env.VITE_API_URL ?? 'http://localhost:8000').replace(/\/$/, '')

export type Outcome = 'H' | 'D' | 'A'

export interface TeamsResponse {
  season: string
  teams: string[]
  data_through: string
}

export interface FormMatch {
  date: string
  opponent: string
  venue: 'H' | 'A'
  goals_for: number
  goals_against: number
  outcome: 'W' | 'D' | 'L'
}

export interface TeamStats {
  team: string
  season: string
  position: number
  played: number
  won: number
  drawn: number
  lost: number
  points: number
  goals_for: number
  goals_against: number
  goal_difference: number
  goals_for_per_game: number | null
  goals_against_per_game: number | null
  xg_for: number | null
  xg_against: number | null
  form: FormMatch[]
}

export interface Prediction {
  home_team: string
  away_team: string
  date: string
  probabilities: { home_win: number; draw: number; away_win: number }
  prediction: Outcome
  model: string
}

export interface ModelResult {
  model: string
  feature_set: 'base' | 'xg' | 'v2'
  n_test: number
  test_seasons: string
  accuracy: number
  f1_macro: number
  precision_macro: number
  recall_macro: number
  log_loss: number
  predicted_share: Record<Outcome, number>
  confusion_matrix: number[][]
  baseline: { always_home_accuracy: number; class_prior_log_loss: number }
}

export interface ModelsResponse {
  production: {
    model: string
    description?: string
    feature_set: 'base' | 'xg' | 'v2'
    n_features: number
    trained_on: { matches: number; first_season: string; last_match: string }
  }
  comparison: ModelResult[]
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(API_URL + path, init)
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      // body wasn't JSON; keep the status text
    }
    throw new Error(detail)
  }
  return res.json()
}

export const api = {
  teams: () => request<TeamsResponse>('/teams'),
  stats: (team: string) => request<TeamStats>(`/teams/${encodeURIComponent(team)}/stats`),
  predict: (homeTeam: string, awayTeam: string) =>
    request<Prediction>('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam }),
    }),
  models: () => request<ModelsResponse>('/models'),
}
