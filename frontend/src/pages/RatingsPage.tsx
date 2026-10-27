import { api, type Rating } from '../api'
import { ErrorMessage, Loading } from '../components/Status'
import { signed } from '../format'
import { useLoad } from '../hooks'
import { useLeague } from '../league'

export function RatingsPage() {
  const league = useLeague()
  const ratings = useLoad(() => api.ratings(league.code), [league.code])

  return (
    <>
      <section className="intro">
        <h1>Team strength ratings</h1>
        <p>
          Elo ratings: every team gains points for beating expectations and loses them for falling short, more for
          bigger wins. 1500 is an average side; a 100-point gap makes the stronger team clearly more likely to win.
        </p>
      </section>

      {(!ratings || ratings.status === 'loading') && <Loading label="Loading ratings…" />}
      {ratings?.status === 'error' && <ErrorMessage error={ratings.error} />}
      {ratings?.status === 'ready' && (
        <section className="card">
          <h2>{ratings.data.season}</h2>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th scope="col" className="num">#</th>
                  <th scope="col">Team</th>
                  <th scope="col" className="num">Elo</th>
                  <th scope="col" className="num">This season</th>
                  <th scope="col">Trend</th>
                </tr>
              </thead>
              <tbody>
                {ratings.data.ratings.map((r) => (
                  <tr key={r.team}>
                    <td className="num">{r.rank}</td>
                    <th scope="row">{r.team}</th>
                    <td className="num">{Math.round(r.elo)}</td>
                    <td className="num">{signed(r.season_change, 0)}</td>
                    <td><Sparkline rating={r} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="footnote">
            Trend: the rating after each match this season, starting from where the summer reset left it. Each line has
            its own scale, so compare the shape, and use the numbers to compare teams.
          </p>
        </section>
      )}
    </>
  )
}

const W = 120
const H = 28
const PAD = 4

function Sparkline({ rating }: { rating: Rating }) {
  const values = rating.trend.map((t) => t.elo)
  if (values.length < 2) return <span className="muted">–</span>
  const lo = Math.min(...values)
  const hi = Math.max(...values)
  const x = (i: number) => PAD + (i / (values.length - 1)) * (W - 2 * PAD)
  const y = (v: number) => (hi === lo ? H / 2 : PAD + (1 - (v - lo) / (hi - lo)) * (H - 2 * PAD))
  const points = values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')
  const last = values.length - 1

  return (
    <svg
      className="sparkline"
      width={W}
      height={H}
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={`${rating.team}: ${Math.round(values[0])} at the start of the season, ${Math.round(values[last])} now`}
    >
      <title>{`${Math.round(values[0])} → ${Math.round(values[last])} over ${last} matches`}</title>
      <polyline points={points} fill="none" />
      <circle cx={x(last)} cy={y(values[last])} r={3} />
    </svg>
  )
}
