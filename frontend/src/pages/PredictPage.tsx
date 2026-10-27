import { useState } from 'react'
import { api, type Prediction, type TeamsResponse } from '../api'
import { ProbabilityBar } from '../components/ProbabilityBar'
import { Scorelines } from '../components/Scorelines'
import { StatsCard } from '../components/StatsCard'
import { ErrorMessage, Loading } from '../components/Status'
import { WhyList } from '../components/WhyList'
import { longDate } from '../format'
import { headline } from '../headline'
import { useLoad } from '../hooks'
import { useLeague } from '../league'

/** Custom match: any two teams in the league, predicted on request. */
export function PredictPage() {
  const league = useLeague()
  const teams = useLoad(() => api.teams(league.code), [league.code])

  if (!teams || teams.status === 'loading') return <Loading label="Loading teams…" />
  if (teams.status === 'error') return <ErrorMessage error={teams.error} />
  return <CustomMatch key={league.code} teams={teams.data} />
}

function CustomMatch({ teams }: { teams: TeamsResponse }) {
  const league = useLeague()
  const [home, setHome] = useState('')
  const [away, setAway] = useState('')
  const [requested, setRequested] = useState<[string, string] | null>(null)

  const prediction = useLoad<Prediction>(
    requested && (() => api.predict(league.code, ...requested)),
    [league.code, requested?.[0], requested?.[1]],
  )
  // A prediction is only shown for the teams it was made for
  const showPrediction = requested && requested[0] === home && requested[1] === away
  const canPredict = home !== '' && away !== '' && home !== away

  return (
    <>
      <section className="intro">
        <h1>Custom match</h1>
        <p>
          Pick any two {teams.season} {league.name} teams to see their season so far and the model's odds of a home
          win, draw or away win, as if they played next.
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
              <WhyList prediction={prediction.data} />
              <Scorelines prediction={prediction.data} />
              <p className="footnote">
                Model: {prediction.data.model}, using {league.name} results up to {longDate(teams.data_through)}.
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
  const league = useLeague()
  const stats = useLoad(team ? () => api.stats(league.code, team) : null, [league.code, team])
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
