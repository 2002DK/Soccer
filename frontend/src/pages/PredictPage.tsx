import { useState } from 'react'
import { api, type Prediction, type TeamsResponse } from '../api'
import { ProbabilityBar } from '../components/ProbabilityBar'
import { StatsCard } from '../components/StatsCard'
import { ErrorMessage, Loading } from '../components/Status'
import { longDate } from '../format'
import { useLoad } from '../hooks'

export function PredictPage({ teams }: { teams: TeamsResponse }) {
  const [home, setHome] = useState('')
  const [away, setAway] = useState('')
  const [requested, setRequested] = useState<[string, string] | null>(null)

  const prediction = useLoad<Prediction>(
    requested && (() => api.predict(...requested)),
    [requested?.[0], requested?.[1]],
  )
  // A prediction is only shown for the teams it was made for
  const showPrediction = requested && requested[0] === home && requested[1] === away

  const canPredict = home !== '' && away !== '' && home !== away

  return (
    <>
      <section className="intro">
        <h1>Who wins?</h1>
        <p>
          Pick two {teams.season} Premier League teams to see their season so far and the model's odds
          of a home win, draw or away win.
        </p>
      </section>

      <section className="matchup" aria-label="Choose teams">
        <TeamColumn side="Home" team={home} other={away} teams={teams.teams} onChange={setHome} />
        <div className="versus" aria-hidden="true">v</div>
        <TeamColumn side="Away" team={away} other={home} teams={teams.teams} onChange={setAway} />
      </section>

      <div className="predict-action">
        <button className="button" disabled={!canPredict} onClick={() => setRequested([home, away])}>
          Predict
        </button>
        {!canPredict && <span className="muted">Choose a home and an away team</span>}
      </div>

      {showPrediction && prediction && (
        <section className="card result" aria-live="polite">
          {prediction.status === 'loading' && <Loading label="Running the model…" />}
          {prediction.status === 'error' && <ErrorMessage error={prediction.error} />}
          {prediction.status === 'ready' && (
            <>
              <h2>{headline(prediction.data)}</h2>
              <ProbabilityBar prediction={prediction.data} />
              <p className="footnote">
                {prediction.data.model} trained on Premier League matches since 2006-07, using results up to{' '}
                {longDate(teams.data_through)}.
              </p>
            </>
          )}
        </section>
      )}
    </>
  )
}

function TeamColumn(props: {
  side: 'Home' | 'Away'
  team: string
  other: string
  teams: string[]
  onChange: (team: string) => void
}) {
  const { side, team, other, teams, onChange } = props
  const stats = useLoad(team ? () => api.stats(team) : null, [team])
  const id = `team-${side.toLowerCase()}`

  return (
    <div className="card team">
      <label htmlFor={id} className="team-label">
        {side} team
      </label>
      <select id={id} className="select" value={team} onChange={(e) => onChange(e.target.value)}>
        <option value="">Select a team</option>
        {teams.map((t) => (
          <option key={t} value={t} disabled={t === other}>
            {t}
          </option>
        ))}
      </select>

      {stats?.status === 'loading' && <Loading label="Loading stats…" />}
      {stats?.status === 'error' && <ErrorMessage error={stats.error} />}
      {stats?.status === 'ready' && <StatsCard stats={stats.data} />}
    </div>
  )
}

function headline({ probabilities: p, home_team, away_team }: Prediction): string {
  const ranked = [
    { text: `${home_team} win`, value: p.home_win },
    { text: 'Draw', value: p.draw },
    { text: `${away_team} win`, value: p.away_win },
  ].sort((a, b) => b.value - a.value)
  // Within 5 points of the runner-up the model has no real favourite
  return ranked[0].value - ranked[1].value < 0.05 ? 'Too close to call' : `Most likely: ${ranked[0].text}`
}
