import { Link, useParams } from 'react-router-dom'
import { api, type Fixture } from '../api'
import { ProbabilityBar } from '../components/ProbabilityBar'
import { Scorelines } from '../components/Scorelines'
import { StatsCard } from '../components/StatsCard'
import { ErrorMessage, Loading } from '../components/Status'
import { WhyList } from '../components/WhyList'
import { kickoffDay, kickoffTime, longDate } from '../format'
import { headline } from '../headline'
import { useLoad } from '../hooks'
import { useLeague } from '../league'

export function MatchPage() {
  const league = useLeague()
  const { id } = useParams()
  const upcoming = useLoad(() => api.fixtures(league.code), [league.code])

  if (!upcoming || upcoming.status === 'loading') return <Loading label="Loading match…" />
  if (upcoming.status === 'error') return <ErrorMessage error={upcoming.error} />

  const fixture = upcoming.data.fixtures.find((f) => f.id === id)
  if (!fixture) {
    return (
      <section className="card">
        <p>
          This match is no longer in the next 7 days, most likely because it has kicked off. See{' '}
          <Link to="../accuracy">how its prediction did</Link> or <Link to="../fixtures">the current fixtures</Link>.
        </p>
      </section>
    )
  }
  return <MatchDetail fixture={fixture} model={upcoming.data.model} dataThrough={upcoming.data.data_through} />
}

function MatchDetail({ fixture: f, model, dataThrough }: { fixture: Fixture; model?: string; dataThrough?: string }) {
  const league = useLeague()
  return (
    <>
      <p className="breadcrumb"><Link to="../fixtures">← All fixtures</Link></p>
      <section className="intro">
        <h1>{f.home_team} v {f.away_team}</h1>
        <p>{kickoffDay(f.kickoff)}, {kickoffTime(f.kickoff)} (your time)</p>
      </section>

      <section className="card result">
        <h2>{headline(f)}</h2>
        <ProbabilityBar prediction={f} />
        <WhyList prediction={f} />
        <Scorelines prediction={f} grid />
        {model && dataThrough && (
          <p className="footnote">
            Model: {model}, using {league.name} results up to {longDate(dataThrough)}. Predictions are refreshed every
            morning.
          </p>
        )}
      </section>

      <section className="matchup" aria-label="Both teams this season">
        <TeamCard side="Home" team={f.home_team} />
        <div className="versus" aria-hidden="true">v</div>
        <TeamCard side="Away" team={f.away_team} />
      </section>
    </>
  )
}

function TeamCard({ side, team }: { side: string; team: string }) {
  const league = useLeague()
  const stats = useLoad(() => api.stats(league.code, team), [league.code, team])
  return (
    <div className="card team">
      <span className="team-label">{side}</span>
      <h3 className="team-name">{team}</h3>
      {(!stats || stats.status === 'loading') && <Loading label="Loading stats…" />}
      {stats?.status === 'error' && <ErrorMessage error={stats.error} />}
      {stats?.status === 'ready' && <StatsCard stats={stats.data} />}
    </div>
  )
}
