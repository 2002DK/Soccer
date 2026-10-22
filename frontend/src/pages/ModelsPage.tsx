import { useState } from 'react'
import { api, type ModelResult, type ModelsResponse } from '../api'
import { ErrorMessage, Loading } from '../components/Status'
import { pct } from '../format'
import { useLoad } from '../hooks'

const FEATURE_SET_NAME = { base: 'Without xG', xg: 'With xG' }
const MODEL_ORDER = ['Logistic Regression', 'Random Forest', 'XGBoost']
const OUTCOMES = ['Home win', 'Draw', 'Away win']

export function ModelsPage() {
  const models = useLoad(() => api.models(), [])

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

  return (
    <>
      <section className="intro">
        <h1>How the models compare</h1>
        <p>
          Three models were each trained twice, with and without xG, and tested on {first.n_test} matches from{' '}
          {first.test_seasons} that none of them saw during training. All six land between {pct(worst, 1)} and{' '}
          {pct(best, 1)} accuracy, about 9 points better than always picking the home team (
          {pct(first.baseline.always_home_accuracy, 1)}).
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
          the others on 2006-07 onwards.
        </p>
      </section>

      <section className="card">
        <h2>Why {production.model}?</h2>
        <p>
          It ties for the best accuracy and has the lowest log loss, so its probabilities are the most
          trustworthy, and those are what the predictor shows. It is now trained on all{' '}
          {production.trained_on.matches.toLocaleString()} matches since {production.trained_on.first_season} using{' '}
          {production.n_features} features.
        </p>
        <h3>Predicted vs actual results</h3>
        <ConfusionMatrix matrix={chosen.confusion_matrix} />
        <p className="footnote">
          No model ever picks a draw: a draw is rarely the single most likely result, even when it has a 25–30%
          chance. That keeps F1 near 0.38, but the predictor still shows each draw probability.
        </p>
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

  return (
    <div className="dotplot">
      <ul className="legend" aria-hidden="true">
        <li><span className="dot dot-base" /> Without xG</li>
        <li><span className="dot dot-xg" /> With xG</li>
        <li><span className="legend-baseline" /> Always pick home</li>
      </ul>

      <div className="dotplot-body">
        {MODEL_ORDER.map((model) => (
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
