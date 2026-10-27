import { Link } from 'react-router-dom'
import { api, type Fixture } from '../api'
import { ProbabilityBar } from '../components/ProbabilityBar'
import { ErrorMessage, Loading } from '../components/Status'
import { kickoffDay, kickoffTime } from '../format'
import { useLoad } from '../hooks'
import { useLeague } from '../league'

export function FixturesPage() {
  const league = useLeague()
  const upcoming = useLoad(() => api.fixtures(league.code), [league.code])

  return (
    <>
      <section className="intro">
        <h1>{league.name}: the next 7 days</h1>
        <p>
          Every match coming up, with the model's chances and most likely score. Times are in your time zone. Pick a
          match for the reasons behind the prediction and both teams' form.
        </p>
      </section>

      {(!upcoming || upcoming.status === 'loading') && <Loading label="Loading fixtures…" />}
      {upcoming?.status === 'error' && <ErrorMessage error={upcoming.error} />}
      {upcoming?.status === 'ready' && <FixtureList fixtures={upcoming.data.fixtures} />}
    </>
  )
}

function FixtureList({ fixtures }: { fixtures: Fixture[] }) {
  if (fixtures.length === 0) {
    return (
      <section className="card">
        <p>No matches in the next 7 days. Try <Link to="../custom">a custom match</Link> instead.</p>
      </section>
    )
  }

  // Group by the viewer's local day
  const days = new Map<string, Fixture[]>()
  for (const f of fixtures) {
    const day = kickoffDay(f.kickoff)
    days.set(day, [...(days.get(day) ?? []), f])
  }

  return (
    <>
      <ul className="prob-legend fixture-key" aria-label="Bar colours">
        <li><span className="swatch prob-home" aria-hidden="true" /> Home win</li>
        <li><span className="swatch prob-draw" aria-hidden="true" /> Draw</li>
        <li><span className="swatch prob-away" aria-hidden="true" /> Away win</li>
      </ul>
      {[...days].map(([day, list]) => (
        <section key={day} className="fixture-day" aria-label={day}>
          <h2>{day}</h2>
          <ul className="fixture-list">
            {list.map((f) => <FixtureCard key={f.id} fixture={f} />)}
          </ul>
        </section>
      ))}
    </>
  )
}

function FixtureCard({ fixture: f }: { fixture: Fixture }) {
  const top = f.scorelines?.scorelines[0]
  return (
    <li>
      <Link className="card fixture" to={`../match/${f.id}`}>
        <span className="fixture-time">{kickoffTime(f.kickoff)}</span>
        <span className="fixture-teams">
          <span>{f.home_team}</span>
          <span className="muted">v</span>
          <span>{f.away_team}</span>
        </span>
        <ProbabilityBar prediction={f} compact />
        {top && <span className="fixture-score muted">Likeliest score {top.home}–{top.away}</span>}
      </Link>
    </li>
  )
}
