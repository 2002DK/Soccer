import { createContext, useContext } from 'react'
import type { League } from './api'

/** The league chosen in the URL (/#/<code>/...), available to every page under it. */
export const LeagueContext = createContext<League | null>(null)

export function useLeague(): League {
  const league = useContext(LeagueContext)
  if (!league) throw new Error('useLeague must be used inside a league route')
  return league
}
