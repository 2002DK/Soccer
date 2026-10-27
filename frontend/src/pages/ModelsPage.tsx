import { useState } from 'react'
import { api, type CalibrationBand, type ModelResult, type ModelsResponse } from '../api'
import { ErrorMessage, Loading } from '../components/Status'
import { pct, signed } from '../format'
import { useLoad } from '../hooks'
import { useLeague } from '../league'

const FEATURE_SET_NAME = { base: 'V1, without xG', xg: 'V1, with xG', v2: 'V2' }
const OUTCOMES = ['Home win', 'Draw', 'Away win']

export function ModelsPage() {
  const league = useLeague()
  const models = useLoad(() => api.models(league.code), [league.code])

  if (!models || models.status === 'loading') return <Loading label="Loading model results…" />
  if (models.status === 'error') return <ErrorMessage error={models.error} />
  return <ModelsReport data={models.data} />
}

function ModelsReport({ data }: { data: ModelsResponse }) {
  const { production, comparison } = data
  const first = comparison[0]
  const isProduction = (r: ModelResult) => r.model === production.model && r.feature_set === production.feature_set
  const chosen = comparison.find(isProduction)!
  const best = Math.max(...comparison.map((r) => r.accuracy))
  const worst = Math.min(...comparison.map((r) => r.accuracy))
  const lowestLogLoss = Math.min(...comparison.map((r) => r.log_loss))
  const neverDraws = comparison.every((r) => r.predicted_share.D === 0)

  return (
    <>
      <section className="intro">
        <h1>How the models compare</h1>
        <p>
          {comparison.length} models and feature sets were tested on {first.n_test} matches from{' '}
          {first.test_seasons} that none of them saw during training. All land between {pct(worst, 1)} and{' '}
          {pct(best, 1)} accuracy, against {pct(first.baseline.always_home_accuracy, 1)} for always picking the home
          team.
        </p>
      </section>

      <section className="card">
        <h2>Accuracy on the test seasons</h2>
        <AccuracyChart results={comparison} isProduction={isProduction} />
      </section>

      <section className="card">
        <h2>All scores</h2>
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Features</th>
                <th scope="col" className="num">Accuracy</th>
                <th scope="col" className="num">Precision</th>
                <th scope="col" className="num">Recall</th>
                <th scope="col" className="num">F1</th>
                <th scope="col" className="num">Log loss</th>
              </tr>
            </thead>
            <tbody>
              {[...comparison]
                .sort((a, b) => a.log_loss - b.log_loss)
                .map((r) => (
                  <tr key={r.model + r.feature_set} className={isProduction(r) ? 'row-chosen' : undefined}>
                    <th scope="row">
                      {r.model}
                      {isProduction(r) && <span className="badge">In use</span>}
                    </th>
                    <td>{FEATURE_SET_NAME[r.feature_set]}</td>
                    <td className="num">{pct(r.accuracy, 1)}</td>
                    <td className="num">{r.precision_macro.toFixed(3)}</td>
                    <td className="num">{r.recall_macro.toFixed(3)}</td>
                    <td className="num">{r.f1_macro.toFixed(3)}</td>
                    <td className="num">{r.log_loss.toFixed(4)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
        <p className="footnote">
          Sorted by log loss, which scores the probabilities themselves (lower is better). Precision, recall and F1
          are averaged over the three outcomes. "With xG" models train on 2014-15 onwards, when xG data starts;
          the others on 2006-07 onwards. V2 adds Elo ratings, the Dixon-Coles scoreline model and blends of the
          two.
        </p>
      </section>

      <section className="card">
        <h2>Why {production.model}?</h2>
        <p>
          {production.description ?? production.model}.{' '}
          {chosen.log_loss === lowestLogLoss ? 'It has the lowest log loss of all the models' : `Its log loss is ${chosen.log_loss.toFixed(4)}`},
          so its probabilities are the most trustworthy, and those are what the predictor shows. Its accuracy is{' '}
          {pct(chosen.accuracy, 1)}: accuracy only counts whether the single most likely result happened, so
          models within a point or two of each other are effectively tied. It is now trained on all{' '}
          {production.trained_on.matches.toLocaleString()} matches since {production.trained_on.first_season}.
        </p>
        <CalibrationSection data={data} />
        <h3>Predicted vs actual results</h3>
        <ConfusionMatrix matrix={chosen.confusion_matrix} />
        {neverDraws && (
          <p className="footnote">
            No model ever picks a draw: a draw is rarely the single most likely result, even when it has a 25–30%
            chance. That keeps F1 below 0.39, but the predictor still shows each draw probability.
          </p>
        )}
      </section>
    </>
  )
}

const AXIS_MIN = 0.4
const AXIS_MAX = 0.55
const TICKS = [0.4, 0.45, 0.5, 0.55]
const toX = (v: number) => `${((v - AXIS_MIN) / (AXIS_MAX - AXIS_MIN)) * 100}%`

function AccuracyChart({ results, isProduction }: { results: ModelResult[]; isProduction: (r: ModelResult) => boolean }) {
  const [hovered, setHovered] = useState<ModelResult | null>(null)
  const baseline = results[0].baseline.always_home_accuracy
  const models = [...new Set(results.map((r) => r.model))]
  const sets = [...new Set(results.map((r) => r.feature_set))]

  return (
    <div className="dotplot">
      <ul className="legend" aria-hidden="true">
        {sets.map((s) => <li key={s}><span className={`dot dot-${s}`} /> {FEATURE_SET_NAME[s]}</li>)}
        <li><span className="legend-baseline" /> Always pick home</li>
      </ul>

      <div className="dotplot-body">
        {models.map((model) => (
          <div className="dotplot-row" key={model}>
            <div className="dotplot-label">{model}</div>
            <div className="dotplot-track">
              {TICKS.map((t) => <span key={t} className="dotplot-grid" style={{ left: toX(t) }} />)}
              <span className="dotplot-baseline" style={{ left: toX(baseline) }} />
              {results
                .filter((r) => r.model === model)
                .map((r) => (
                  <button
                    key={r.feature_set}
                    type="button"
                    className="dotplot-hit"
                    style={{ left: toX(r.accuracy) }}
                    aria-label={`${model}, ${FEATURE_SET_NAME[r.feature_set]}: ${pct(r.accuracy, 1)} accuracy`}
                    onMouseEnter={() => setHovered(r)}
                    onMouseLeave={() => setHovered(null)}
                    onFocus={() => setHovered(r)}
                    onBlur={() => setHovered(null)}
                  >
                    <span className={`dot dot-${r.feature_set}`} />
                    {isProduction(r) && <span className="dotplot-tag">{pct(r.accuracy, 1)}</span>}
                    {hovered === r && (
                      <span className="tooltip" role="tooltip">
                        <strong>{model}</strong>
                        <span>{FEATURE_SET_NAME[r.feature_set]}</span>
                        <span>Accuracy {pct(r.accuracy, 1)}</span>
                        <span>Log loss {r.log_loss.toFixed(4)}</span>
                      </span>
                    )}
                  </button>
                ))}
            </div>
          </div>
        ))}

        <div className="dotplot-row dotplot-axis" aria-hidden="true">
          <div className="dotplot-label" />
          <div className="dotplot-track">
            {TICKS.map((t) => <span key={t} className="dotplot-tick" style={{ left: toX(t) }}>{pct(t)}</span>)}
          </div>
        </div>
      </div>
    </div>
  )
}

function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  return (
    <div className="table-wrap">
      <table className="table confusion">
        <thead>
          <tr>
            <th scope="col">Actual ↓ · Predicted →</th>
            {OUTCOMES.map((o) => <th key={o} scope="col" className="num">{o}</th>)}
            <th scope="col" className="num">Total</th>
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => {
            const total = row.reduce((a, b) => a + b, 0)
            return (
              <tr key={OUTCOMES[i]}>
                <th scope="row">{OUTCOMES[i]}</th>
                {row.map((n, j) => (
                  <td key={j} className={`num${i === j ? ' correct' : ''}`}>
                    {n} <span className="muted">({pct(n / total)})</span>
                  </td>
                ))}
                <td className="num">{total}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function CalibrationSection({ data }: { data: ModelsResponse }) {
  const v2 = data.v2
  if (!v2) return null
  const chosen = v2.results.find((r) => r.model === v2.production.model && r.calibrated === v2.production.calibrated)
  if (!chosen) return null
  const bands = chosen.calibration.filter((b) => b.outcome === 'H')
  const lean = chosen.mean_predicted.H - chosen.actual_share.H
  return (
    <>
      <h3>Are the home-win chances honest?</h3>
      <p>
        Test matches are grouped by the home-win chance the model gave them, and each group's prediction is compared
        with how often the home side actually won. On average the model gave home teams {pct(chosen.mean_predicted.H, 1)}{' '}
        against {pct(chosen.actual_share.H, 1)} that won ({signed(lean * 100)} points). Bars show the range random noise
        alone could produce (two standard errors); a dot whose bar crosses the diagonal is in line with the results.
      </p>
      <CalibrationChart bands={bands} />
    </>
  )
}

const CW = 320
const CH = 320
const M = { top: 12, right: 12, bottom: 40, left: 44 }
const AX = 0.8 // both axes run from 0 to 80%

function CalibrationChart({ bands }: { bands: CalibrationBand[] }) {
  const [hovered, setHovered] = useState<CalibrationBand | null>(null)
  const x = (v: number) => M.left + (v / AX) * (CW - M.left - M.right)
  const y = (v: number) => CH - M.bottom - (Math.min(v, AX) / AX) * (CH - M.top - M.bottom)
  const ticks = [0, 0.2, 0.4, 0.6, 0.8]

  return (
    <div className="calibration">
      <svg
        viewBox={`0 0 ${CW} ${CH}`}
        className="calibration-svg"
        role="img"
        aria-label="Predicted home-win chance against actual home-win rate, by probability band"
      >
        {ticks.map((t) => (
          <g key={t}>
            <line className="cal-grid" x1={x(t)} x2={x(t)} y1={y(0)} y2={y(AX)} />
            <line className="cal-grid" x1={x(0)} x2={x(AX)} y1={y(t)} y2={y(t)} />
            <text className="cal-tick" x={x(t)} y={y(0) + 16} textAnchor="middle">{pct(t)}</text>
            <text className="cal-tick" x={x(0) - 6} y={y(t) + 4} textAnchor="end">{pct(t)}</text>
          </g>
        ))}
        <line className="cal-diagonal" x1={x(0)} y1={y(0)} x2={x(AX)} y2={y(AX)} />
        <text className="cal-axis" x={(x(0) + x(AX)) / 2} y={CH - 6} textAnchor="middle">
          Predicted home-win chance
        </text>
        <text className="cal-axis" transform={`translate(12 ${(y(0) + y(AX)) / 2}) rotate(-90)`} textAnchor="middle">
          Home sides that won
        </text>
        {bands.map((b) => (
          <g
            key={b.band}
            className="cal-point"
            tabIndex={0}
            onMouseEnter={() => setHovered(b)}
            onMouseLeave={() => setHovered(null)}
            onFocus={() => setHovered(b)}
            onBlur={() => setHovered(null)}
          >
            <line
              className="cal-whisker"
              x1={x(b.predicted)}
              x2={x(b.predicted)}
              y1={y(Math.max(0, b.actual - 2 * b.standard_error))}
              y2={y(b.actual + 2 * b.standard_error)}
            />
            <circle cx={x(b.predicted)} cy={y(b.actual)} r={5} />
            <circle className="cal-hit" cx={x(b.predicted)} cy={y(b.actual)} r={14} />
          </g>
        ))}
      </svg>
      <p className="cal-readout" aria-live="polite">
        {hovered
          ? `Band ${hovered.band.replace('-', '–')}: ${hovered.n} matches, predicted ${pct(hovered.predicted, 1)}, home side won ${pct(hovered.actual, 1)}`
          : 'Hover or tab to a dot for its numbers.'}
      </p>
    </div>
  )
}
