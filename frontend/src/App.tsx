import { NavLink, Route, Routes } from 'react-router-dom'
import { api } from './api'
import { ErrorMessage, Loading } from './components/Status'
import { longDate } from './format'
import { useLoad } from './hooks'
import { ModelsPage } from './pages/ModelsPage'
import { PredictPage } from './pages/PredictPage'

export default function App() {
  const teams = useLoad(() => api.teams(), [])

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <span className="brand">PL Predictor</span>
          <nav className="nav">
            <NavLink to="/" end>Predict</NavLink>
            <NavLink to="/models">Models</NavLink>
          </nav>
        </div>
      </header>

      <main className="main">
        <Routes>
          <Route
            path="/"
            element={
              !teams || teams.status === 'loading' ? <Loading label="Loading teams…" />
              : teams.status === 'error' ? <ErrorMessage error={teams.error} />
              : <PredictPage teams={teams.data} />
            }
          />
          <Route path="/models" element={<ModelsPage />} />
        </Routes>
      </main>

      <footer className="footer">
        {teams?.status === 'ready' && <span>Results up to {longDate(teams.data.data_through)}. </span>}
        Data from football-data.co.uk and Understat.{' '}
        <a href="https://github.com/2002DK/Soccer">Source on GitHub</a>
      </footer>
    </div>
  )
}
