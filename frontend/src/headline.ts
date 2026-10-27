import type { MatchPrediction } from './api'

/** "Most likely: Arsenal win", or "Too close to call" when the top two are within 5 points. */
export function headline({ probabilities: p, home_team, away_team }: MatchPrediction): string {
  const ranked = [
    { text: `${home_team} win`, value: p.home_win },
    { text: 'Draw', value: p.draw },
    { text: `${away_team} win`, value: p.away_win },
  ].sort((a, b) => b.value - a.value)
  return ranked[0].value - ranked[1].value < 0.05 ? 'Too close to call' : `Most likely: ${ranked[0].text}`
}
