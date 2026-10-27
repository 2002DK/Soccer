export const pct = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`

export function ordinal(n: number): string {
  const suffix = n % 100 >= 11 && n % 100 <= 13 ? 'th' : ({ 1: 'st', 2: 'nd', 3: 'rd' } as Record<number, string>)[n % 10] ?? 'th'
  return `${n}${suffix}`
}

export function longDate(iso: string): string {
  return new Date(`${iso}T00:00:00`).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })
}

/** Kick-off in the viewer's own time zone, e.g. "15:00". */
export function kickoffTime(iso: string): string {
  return new Date(iso).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })
}

/** The viewer's local calendar day of a UTC time, e.g. "Saturday 31 October". */
export function kickoffDay(iso: string): string {
  return new Date(iso).toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'long' })
}

/** "+21.8" / "−3.0" with a real minus sign. */
export function signed(x: number, digits = 1): string {
  return `${x > 0 ? '+' : x < 0 ? '−' : ''}${Math.abs(x).toFixed(digits)}`
}
