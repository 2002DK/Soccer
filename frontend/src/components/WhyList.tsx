import type { Prediction } from '../api'

export function WhyList({ prediction }: { prediction: Prediction }) {
  const { prediction: outcome, home_team, away_team } = prediction
  // Effects under 1 point are negligible and can tip either way for second-order reasons
  const why = prediction.why.filter((d) => Math.abs(d.points) >= 1)
  const hidden = prediction.why.length - why.length
  if (why.length === 0) return null
  const outcomeName = { H: `${home_team} win`, D: 'a draw', A: `${away_team} win` }[outcome]

  return (
    <div className="why">
      <h3>Why</h3>
      <p className="why-lead">Effect of each factor on the chance of {outcomeName}:</p>
      <ul className="why-list">
        {why.map((d) => (
          <li key={d.driver}>
            <span className="why-points">
              {d.points > 0 ? '+' : d.points < 0 ? '−' : ''}
              {Math.abs(d.points).toFixed(1)} pts
            </span>
            <span>
              <strong>{d.label}</strong>
              <span className="why-detail">{d.detail}</span>
            </span>
          </li>
        ))}
      </ul>
      {hidden > 0 && <p className="why-note">Factors under 1 point are left out.</p>}
    </div>
  )
}
