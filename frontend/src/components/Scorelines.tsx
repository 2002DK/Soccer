import type { MatchPrediction } from '../api'
import { pct } from '../format'

/** Expected goals, the most likely scores, goal markets and (optionally) the full score grid. */
export function Scorelines({ prediction, grid = false }: { prediction: MatchPrediction; grid?: boolean }) {
  const sc = prediction.scorelines
  if (!sc) return null
  const { home_team, away_team } = prediction

  return (
    <div className="scorelines">
      <h3>Scorelines</h3>
      <dl className="stats-grid">
        <div>
          <dt>Expected goals</dt>
          <dd>{sc.expected_goals.home.toFixed(1)} – {sc.expected_goals.away.toFixed(1)}</dd>
        </div>
        <div>
          <dt>Most likely scores</dt>
          <dd>
            {sc.scorelines.map((s) => `${s.home}–${s.away} (${pct(s.probability)})`).join(', ')}
          </dd>
        </div>
        <div>
          <dt>Over 2.5 goals</dt>
          <dd>{pct(sc.over_2_5)}</dd>
        </div>
        <div>
          <dt>Both teams score</dt>
          <dd>{pct(sc.both_teams_score)}</dd>
        </div>
      </dl>
      {grid && <ScoreGrid grid={sc.grid} home={home_team} away={away_team} />}
    </div>
  )
}

function ScoreGrid({ grid, home, away }: { grid: number[][]; home: string; away: string }) {
  const max = Math.max(...grid.flat())
  const caption = `Chance of each score (${home} goals down, ${away} goals across)`
  return (
    <>
      <p className="grid-caption">{caption}</p>
      <div className="table-wrap">
      <table className="score-grid" aria-label={caption}>
        <thead>
          <tr>
            <th scope="col"><span className="visually-hidden">{home} goals</span></th>
            {grid[0].map((_, j) => <th key={j} scope="col">{j}</th>)}
          </tr>
        </thead>
        <tbody>
          {grid.map((row, i) => (
            <tr key={i}>
              <th scope="row">{i}</th>
              {row.map((p, j) => (
                <td
                  key={j}
                  // Shade by chance relative to the likeliest score (one hue, light to dark)
                  style={{ '--shade': `${Math.round((p / max) * 85)}%` } as React.CSSProperties}
                  title={`${home} ${i}–${j} ${away}: ${pct(p, 1)}`}
                  className={i > j ? 'grid-home' : i < j ? 'grid-away' : 'grid-draw'}
                >
                  {p >= 0.005 ? pct(p) : ''}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      </div>
    </>
  )
}
