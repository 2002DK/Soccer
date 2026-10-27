// Client for the FastAPI backend (backend/app/main.py). Every call is scoped to a league.

const API_URL = (import.meta.env.VITE_API_URL ?? 'http://localhost:8000').replace(/\/$/, '')

export type Outcome = 'H' | 'D' | 'A'
export type Probabilities = { home_win: number; draw: number; away_win: number }

export interface League {
  code: string
  name: string
  season: string
  data_through: string
}

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
  elo: { rating: number; rank: number; season_change: number } | null
}

export interface Driver {
  driver: string
  label: string
  detail: string
  effect: Probabilities
  points: number // percentage points on the most likely outcome
}

export interface Scorelines {
  expected_goals: { home: number; away: number }
  scorelines: { home: number; away: number; probability: number }[]
  over_2_5: number
  both_teams_score: number
  grid: number[][] // P(home i, away j), rows are home goals
}

/** What every prediction has, whether made on request or stored for a fixture */
export interface MatchPrediction {
  home_team: string
  away_team: string
  probabilities: Probabilities
  prediction: Outcome
  why: Driver[]
  scorelines: Scorelines | null
}

export interface Prediction extends MatchPrediction {
  date: string
  model: string
}

export interface Fixture extends MatchPrediction {
  id: string
  kickoff: string // UTC, ISO 8601
}

export interface Upcoming {
  generated_at?: string
  data_through?: string
  model?: string
  window_days?: number
  fixtures: Fixture[]
}

export interface Rating {
  rank: number
  team: string
  elo: number
  season_change: number
  trend: { date: string | null; elo: number }[]
}

export interface RatingsResponse {
  season: string
  ratings: Rating[]
}

export interface Scores {
  accuracy: number | null
  log_loss: number | null
  brier: number | null
}

export interface TrackedMatch {
  fixture_id: string
  kickoff: string
  home_team: string
  away_team: string
  frozen_at: string
  probabilities: Probabilities
  predicted: Outcome
  top_scoreline: string
  result: Outcome | null
  score: string | null
  correct: boolean | null
}

export interface Accuracy {
  tracking_since: string | null
  frozen: number
  scored: number
  awaiting_result?: number
  missed?: number
  model?: Scores
  baselines?: { elo_only: Scores; always_home: { accuracy: number | null } }
  by_week?: ({ week_from: string; matches: number } & Scores)[]
  matches: TrackedMatch[]
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

export interface CalibrationBand {
  outcome: Outcome
  band: string
  n: number
  predicted: number
  actual: number
  standard_error: number
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
  v2: {
    production: { model: string; calibrated: boolean }
    results: (Omit<ModelResult, 'feature_set' | 'baseline'> & {
      calibrated: boolean
      calibration: CalibrationBand[]
      mean_predicted: Record<Outcome, number>
      actual_share: Record<Outcome, number>
    })[]
  } | null
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

const scoped = (league: string, path: string) => `/leagues/${encodeURIComponent(league)}${path}`

export const api = {
  leagues: () => request<League[]>('/leagues'),
  teams: (league: string) => request<TeamsResponse>(scoped(league, '/teams')),
  stats: (league: string, team: string) =>
    request<TeamStats>(scoped(league, `/teams/${encodeURIComponent(team)}/stats`)),
  predict: (league: string, homeTeam: string, awayTeam: string) =>
    request<Prediction>(scoped(league, '/predict'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam }),
    }),
  fixtures: (league: string) => request<Upcoming>(scoped(league, '/fixtures')),
  ratings: (league: string) => request<RatingsResponse>(scoped(league, '/ratings')),
  accuracy: (league: string) => request<Accuracy>(scoped(league, '/accuracy')),
  models: (league: string) => request<ModelsResponse>(scoped(league, '/models')),
}
