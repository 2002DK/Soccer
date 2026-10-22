import { useEffect, useState } from 'react'

export type Loadable<T> =
  | { status: 'loading' }
  | { status: 'error'; error: string }
  | { status: 'ready'; data: T }

/** Run `load` whenever `deps` change; results from stale calls are dropped. */
export function useLoad<T>(load: (() => Promise<T>) | null, deps: unknown[]): Loadable<T> | null {
  const [state, setState] = useState<Loadable<T> | null>(null)

  // eslint-disable-next-line react-hooks/exhaustive-deps -- deps come from the caller
  useEffect(() => {
    if (!load) {
      setState(null)
      return
    }
    let current = true
    setState({ status: 'loading' })
    load()
      .then((data) => current && setState({ status: 'ready', data }))
      .catch((e: Error) => current && setState({ status: 'error', error: e.message }))
    return () => {
      current = false
    }
    // deps are the caller's, so the linter can't check them here
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return state
}

/** False until `ms` after mount — used to explain a slow (sleeping) free-tier server. */
export function useSlowAfter(ms = 4000): boolean {
  const [slow, setSlow] = useState(false)
  useEffect(() => {
    const id = setTimeout(() => setSlow(true), ms)
    return () => clearTimeout(id)
  }, [ms])
  return slow
}
