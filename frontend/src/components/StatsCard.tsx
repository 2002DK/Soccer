import type { TeamStats } from '../api'
import { ordinal, signed } from '../format'

const OUTCOME_NAME = { W: 'Win', D: 'Draw', L: 'Loss' }

export function StatsCard({ stats }: { stats: TeamStats }) {
  const xgPerGame = (total: number | null) =>
    total === null || stats.played === 0 ? '–' : (total / stats.played).toFixed(2)

  return (
    <div className="stats">
      <div className="stats-headline">
        <span className="stats-position">{ordinal(stats.position)}</span>
        <span className="stats-record">
          {stats.points} pts · {stats.won}W {stats.drawn}D {stats.lost}L
        </span>
      </div>

      <dl className="stats-grid">
        {stats.elo && (
          <div>
            <dt>Elo rating</dt>
            <dd>
              {Math.round(stats.elo.rating)} <span className="muted">({ordinal(stats.elo.rank)}, {signed(stats.elo.season_change, 0)} this season)</span>
            </dd>
          </div>
        )}
        <div>
          <dt>Scored / game</dt>
          <dd>{stats.goals_for_per_game?.toFixed(2) ?? '–'}</dd>
        </div>
        <div>
          <dt>Conceded / game</dt>
          <dd>{stats.goals_against_per_game?.toFixed(2) ?? '–'}</dd>
        </div>
        <div>
          <dt>xG / game</dt>
          <dd>{xgPerGame(stats.xg_for)}</dd>
        </div>
        <div>
          <dt>xG against / game</dt>
          <dd>{xgPerGame(stats.xg_against)}</dd>
        </div>
      </dl>

      <div className="form">
        <span className="form-label">Form</span>
        {stats.form.length === 0 ? (
          <span className="muted">No games yet</span>
        ) : (
          <ol className="form-list" aria-label="Last five results, oldest first">
            {stats.form.map((m) => {
              const detail = `${OUTCOME_NAME[m.outcome]} ${m.goals_for}–${m.goals_against} ${m.venue === 'H' ? 'v' : 'at'} ${m.opponent}`
              return (
                <li key={m.date} className={`form-chip form-${m.outcome}`} title={detail} aria-label={detail}>
                  {m.outcome}
                </li>
              )
            })}
          </ol>
        )}
      </div>
    </div>
  )
}
