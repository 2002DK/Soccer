import type { MatchPrediction } from '../api'
import { pct } from '../format'

/** Home / draw / away as one bar. `compact` drops the legend for use in lists. */
export function ProbabilityBar({ prediction, compact = false }: { prediction: MatchPrediction; compact?: boolean }) {
  const { probabilities: p, home_team, away_team } = prediction
  const segments = [
    { key: 'home', label: `${home_team} win`, value: p.home_win },
    { key: 'draw', label: 'Draw', value: p.draw },
    { key: 'away', label: `${away_team} win`, value: p.away_win },
  ]

  return (
    <div className={compact ? 'prob prob-compact' : 'prob'}>
      <div className="prob-bar" role="img" aria-label={segments.map((s) => `${s.label} ${pct(s.value)}`).join(', ')}>
        {segments.map((s) => (
          <div
            key={s.key}
            className={`prob-seg prob-${s.key}`}
            style={{ flexGrow: s.value }}
            title={`${s.label}: ${pct(s.value, 1)}`}
          >
            {/* Label inside only when the segment is wide enough to hold it */}
            {s.value >= 0.12 && <span>{pct(s.value)}</span>}
          </div>
        ))}
      </div>
      {!compact && (
        <ul className="prob-legend">
          {segments.map((s) => (
            <li key={s.key}>
              <span className={`swatch prob-${s.key}`} aria-hidden="true" />
              <span className="prob-legend-label">{s.label}</span>
              <strong>{pct(s.value)}</strong>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
