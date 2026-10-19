import { useState } from 'react'
import './styles/App.css'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8004'

type Prediction = {
  churn: string
  churn_probability: number
  retention_probability: number
  risk_level: 'low' | 'medium' | 'high'
  model: string
}

const CONTRACTS = ['Month-to-month', 'One year', 'Two year']
const PAYMENTS = ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']
const INTERNET = ['DSL', 'Fiber optic', 'No']
const SUPPORT = ['Yes', 'No', 'No internet service']

const INITIAL = {
  tenure: 12,
  MonthlyCharges: 70,
  TotalCharges: 840,
  Contract: CONTRACTS[0],
  PaymentMethod: PAYMENTS[0],
  InternetService: INTERNET[1],
  OnlineSecurity: SUPPORT[1],
  TechSupport: SUPPORT[1],
}

export default function App() {
  const [form, setForm] = useState(INITIAL)
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const update = (field: keyof typeof INITIAL, value: string | number) =>
    setForm(prev => ({ ...prev, [field]: value }))

  const predict = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      if (!response.ok) {
        const detail = await response.json().catch(() => null)
        throw new Error(detail?.detail ?? `Request failed (${response.status})`)
      }
      setPrediction(await response.json())
    } catch (err) {
      setPrediction(null)
      setError(err instanceof Error ? err.message : 'Could not reach the API')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="logo">📉</div>
        <h1>Customer Churn Risk</h1>
        <p>Score an account before the renewal conversation</p>
      </header>

      <main>
        <section className="card">
          <h2>Account profile</h2>
          <div className="grid">
            <label>
              Tenure (months)
              <input
                type="number"
                min={0}
                max={120}
                value={form.tenure}
                onChange={e => update('tenure', Number(e.target.value))}
              />
            </label>
            <label>
              Monthly charges ($)
              <input
                type="number"
                min={0}
                step="0.01"
                value={form.MonthlyCharges}
                onChange={e => update('MonthlyCharges', Number(e.target.value))}
              />
            </label>
            <label>
              Total charges ($)
              <input
                type="number"
                min={0}
                step="0.01"
                value={form.TotalCharges}
                onChange={e => update('TotalCharges', Number(e.target.value))}
              />
            </label>
            <label>
              Contract
              <select value={form.Contract} onChange={e => update('Contract', e.target.value)}>
                {CONTRACTS.map(option => <option key={option}>{option}</option>)}
              </select>
            </label>
            <label>
              Payment method
              <select value={form.PaymentMethod} onChange={e => update('PaymentMethod', e.target.value)}>
                {PAYMENTS.map(option => <option key={option}>{option}</option>)}
              </select>
            </label>
            <label>
              Internet service
              <select value={form.InternetService} onChange={e => update('InternetService', e.target.value)}>
                {INTERNET.map(option => <option key={option}>{option}</option>)}
              </select>
            </label>
            <label>
              Online security
              <select value={form.OnlineSecurity} onChange={e => update('OnlineSecurity', e.target.value)}>
                {SUPPORT.map(option => <option key={option}>{option}</option>)}
              </select>
            </label>
            <label>
              Tech support
              <select value={form.TechSupport} onChange={e => update('TechSupport', e.target.value)}>
                {SUPPORT.map(option => <option key={option}>{option}</option>)}
              </select>
            </label>
          </div>

          <button onClick={predict} disabled={loading}>
            {loading ? 'Scoring…' : 'Predict churn risk'}
          </button>

          {error && <p className="error">{error}</p>}

          {prediction && (
            <div className={`result ${prediction.risk_level}`}>
              <div className="result-icon">
                {prediction.risk_level === 'high' ? '⚠️' : prediction.risk_level === 'medium' ? '🔔' : '✅'}
              </div>
              <div className="result-body">
                <h3>
                  {prediction.risk_level.toUpperCase()} risk · {prediction.churn === 'Yes' ? 'likely to churn' : 'likely to stay'}
                </h3>
                <div className="meter">
                  <div className="meter-fill" style={{ width: `${prediction.churn_probability * 100}%` }} />
                </div>
                <p>
                  Churn probability {(prediction.churn_probability * 100).toFixed(1)}% · retention{' '}
                  {(prediction.retention_probability * 100).toFixed(1)}%
                </p>
                <p className="meta">Model: {prediction.model}</p>
              </div>
            </div>
          )}
        </section>

        <section className="info-card">
          <h3>How to read this</h3>
          <p>
            Tenure, contract type and payment method carry the most weight. Scores come from a model trained on
            synthetic accounts, so use them to explore the interface rather than to make retention decisions.
          </p>
        </section>
      </main>
    </div>
  )
}
