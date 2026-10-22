import { useSlowAfter } from '../hooks'

export function Loading({ label }: { label: string }) {
  const slow = useSlowAfter()
  return (
    <div className="status" role="status">
      <span className="spinner" aria-hidden="true" />
      <span>
        {label}
        {slow && (
          <span className="status-note">
            The server sleeps when idle and can take up to a minute to wake up.
          </span>
        )}
      </span>
    </div>
  )
}

export function ErrorMessage({ error }: { error: string }) {
  return (
    <div className="status status-error" role="alert">
      Something went wrong: {error}
    </div>
  )
}
