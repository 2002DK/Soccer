import { Link } from 'react-router-dom'
import { api, type Accuracy, type Outcome, type TrackedMatch } from '../api'
import { ErrorMessage, Loading } from '../components/Status'
import { kickoffDay, kickoffTime, longDate, pct } from '../format'
import { useLoad } from '../hooks'
import { useLeague } from '../league'

export function AccuracyPage() {
  const league = useLeague()
  const accuracy = useLoad(() => api.accuracy(league.code), [league.code])

  return (
    <>
      <section className="intro">
        <h1>Live accuracy</h1>
        <p>
          How the model does on real matches. Each prediction is frozen before kick-off and saved in a log that is never
          edited, so these scores can't be tidied up afterwards.
        </p>
      </section>

      {(!accuracy || accuracy.status === 'loading') && <Loading label="Loading results…" />}
      {accuracy?.status === 'error' && <ErrorMessage error={accuracy.error} />}
      {accuracy?.status === 'ready' && <Report data={accuracy.data} />}
    </>
  )
}

function Report({ data }: { data: Accuracy }) {
  if (data.frozen === 0) {
    return (
      <section className="card">
        <p>
          No predictions frozen yet. Each match is frozen by the last daily run before it kicks off, so the first ones
          appear about a day before the next match. Meanwhile, see <Link to="../fixtures">the upcoming fixtures</Link>.
        </p>
      </section>
    )
  }

  const m = data.model
  const elo = data.baselines?.elo_only
  const home = data.baselines?.always_home
  return (
    <>
      {data.scored > 0 && m && (
        <section className="tiles" aria-label="Scores so far">
          <Tile label="Model accuracy" value={pct(m.accuracy ?? 0, 1)} note={`${data.scored} matches`} />
          <Tile label="Elo-only accuracy" value={elo?.accuracy == null ? '–' : pct(elo.accuracy, 1)} note="baseline" />
          <Tile label="Always pick home" value={home?.accuracy == null ? '–' : pct(home.accuracy, 1)} note="baseline" />
          <Tile
            label="Log loss"
            value={m.log_loss?.toFixed(3) ?? '–'}
            note={elo?.log_loss != null ? `Elo-only ${elo.log_loss.toFixed(3)}; lower is better` : 'lower is better'}
          />
        </section>
      )}

      <section className="card">
        <p className="lead-line">
          {data.frozen} predictions frozen since {data.tracking_since ? longDate(data.tracking_since.slice(0, 10)) : '–'};{' '}
          {data.scored} scored, {data.awaiting_result ?? 0} awaiting a result
          {data.missed ? `, ${data.missed} missed (no run before kick-off)` : ''}.
          {data.scored > 0 && data.scored < 100 && ' Early days: with this few matches, a few points either way is luck.'}
        </p>

        {data.by_week && data.by_week.length > 0 && (
          <>
            <h2>By week</h2>
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th scope="col">Week from</th>
                    <th scope="col" className="num">Matches</th>
                    <th scope="col" className="num">Accuracy</th>
                    <th scope="col" className="num">Log loss</th>
                  </tr>
                </thead>
                <tbody>
                  {[...data.by_week].reverse().map((w) => (
                    <tr key={w.week_from}>
                      <th scope="row">{longDate(w.week_from)}</th>
                      <td className="num">{w.matches}</td>
                      <td className="num">{w.accuracy == null ? '–' : pct(w.accuracy, 0)}</td>
                      <td className="num">{w.log_loss?.toFixed(3) ?? '–'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}

        <h2>Every prediction</h2>
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th scope="col">Kick-off</th>
                <th scope="col">Match</th>
                <th scope="col">Predicted</th>
                <th scope="col">Result</th>
              </tr>
            </thead>
            <tbody>
              {data.matches.map((t) => <TrackedRow key={t.fixture_id} match={t} />)}
            </tbody>
          </table>
        </div>
      </section>
    </>
  )
}

function Tile({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="card tile">
      <span className="tile-label">{label}</span>
      <span className="tile-value">{value}</span>
      <span className="tile-note">{note}</span>
    </div>
  )
}

const OUTCOME_TEXT = (o: Outcome, t: TrackedMatch) => ({ H: `${t.home_team} win`, D: 'Draw', A: `${t.away_team} win` })[o]
const OUTCOME_PROB = { H: 'home_win', D: 'draw', A: 'away_win' } as const

function TrackedRow({ match: t }: { match: TrackedMatch }) {
  const chance = t.probabilities[OUTCOME_PROB[t.predicted]]
  return (
    <tr>
      <td>{kickoffDay(t.kickoff).replace(/^\w+ /, '')}, {kickoffTime(t.kickoff)}</td>
      <th scope="row">{t.home_team} v {t.away_team}</th>
      <td>{OUTCOME_TEXT(t.predicted, t)} <span className="muted">({pct(chance)})</span></td>
      <td>
        {t.result == null ? (
          <span className="muted">Awaiting result</span>
        ) : (
          <span className={t.correct ? 'hit' : 'miss'}>
            <span aria-hidden="true">{t.correct ? '✓' : '✗'}</span> {t.score}
            <span className="visually-hidden">{t.correct ? ', correct' : ', wrong'}</span>
          </span>
        )}
      </td>
    </tr>
  )
}
