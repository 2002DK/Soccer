import { Navigate, NavLink, Outlet, Route, Routes, useNavigate, useParams } from 'react-router-dom'
import { api, type League } from './api'
import { ErrorMessage, Loading } from './components/Status'
import { longDate } from './format'
import { useLoad } from './hooks'
import { LeagueContext } from './league'
import { AccuracyPage } from './pages/AccuracyPage'
import { FixturesPage } from './pages/FixturesPage'
import { MatchPage } from './pages/MatchPage'
import { ModelsPage } from './pages/ModelsPage'
import { PredictPage } from './pages/PredictPage'
import { RatingsPage } from './pages/RatingsPage'

const DEFAULT_LEAGUE = 'epl'

export default function App() {
  const leagues = useLoad(() => api.leagues(), [])

  if (!leagues || leagues.status === 'loading') return <Shell><Loading label="Loading…" /></Shell>
  if (leagues.status === 'error') return <Shell><ErrorMessage error={leagues.error} /></Shell>

  return (
    <Routes>
      <Route path="/:league" element={<LeagueLayout leagues={leagues.data} />}>
        <Route index element={<Navigate to="fixtures" replace />} />
        <Route path="fixtures" element={<FixturesPage />} />
        <Route path="match/:id" element={<MatchPage />} />
        <Route path="custom" element={<PredictPage />} />
        <Route path="ratings" element={<RatingsPage />} />
        <Route path="accuracy" element={<AccuracyPage />} />
        <Route path="models" element={<ModelsPage />} />
      </Route>
      {/* V1 links: /#/ was the predictor, /#/models the model page */}
      <Route path="/" element={<Navigate to={`/${DEFAULT_LEAGUE}/fixtures`} replace />} />
      <Route path="*" element={<Navigate to={`/${DEFAULT_LEAGUE}/fixtures`} replace />} />
    </Routes>
  )
}

function LeagueLayout({ leagues }: { leagues: League[] }) {
  const { league: code } = useParams()
  const navigate = useNavigate()
  const league = leagues.find((l) => l.code === code)
  if (code === 'models') return <Navigate to={`/${DEFAULT_LEAGUE}/models`} replace />
  if (!league) return <Navigate to={`/${DEFAULT_LEAGUE}/fixtures`} replace />

  return (
    <LeagueContext.Provider value={league}>
      <Shell
        nav={
          <>
            <select
              className="league-select"
              aria-label="League"
              value={league.code}
              onChange={(e) => navigate(`/${e.target.value}/fixtures`)}
            >
              {leagues.map((l) => <option key={l.code} value={l.code}>{l.name}</option>)}
            </select>
            <nav className="nav" aria-label="Pages">
              <NavLink to="fixtures">Fixtures</NavLink>
              <NavLink to="custom">Custom match</NavLink>
              <NavLink to="ratings">Ratings</NavLink>
              <NavLink to="accuracy">Accuracy</NavLink>
              <NavLink to="models">Models</NavLink>
            </nav>
          </>
        }
        footer={<span>{league.name} results up to {longDate(league.data_through)}. </span>}
      >
        <Outlet />
      </Shell>
    </LeagueContext.Provider>
  )
}

function Shell({ children, nav, footer }: { children: React.ReactNode; nav?: React.ReactNode; footer?: React.ReactNode }) {
  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <span className="brand">Football Predictor</span>
          {nav}
        </div>
      </header>
      <main className="main">{children}</main>
      <footer className="footer">
        {footer}
        Data from football-data.co.uk and Understat.{' '}
        <a href="https://github.com/2002DK/Soccer">Source on GitHub</a>
      </footer>
    </div>
  )
}
